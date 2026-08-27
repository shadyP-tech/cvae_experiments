from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router import (
    development_model,
    development_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.evidence_builder import (
    RawSourceEvidence,
    build_outer_development_evidence,
    build_target_prediction_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.model_freeze import (
    AdaptiveUtilityExactBFallback,
    AdaptiveUtilityRoute,
    EXACT_UTILITY_TIE_REASON,
    FROZEN_MODEL_PUBLICATION_STATUS,
    FULL_ROUTER_ROLE,
    FULL_ROUTER_SCHEMA,
    FrozenAdaptiveUtilityModel,
    FrozenPrelabelRouter,
    INVALID_UTILITY_EVIDENCE_REASON,
    MISSING_UTILITY_EVIDENCE_REASON,
    PREDICTED_UTILITY_POLICY_ID,
    PREDICTED_UTILITY_SEMANTICS,
    freeze_adaptive_utility_model,
    freeze_full_prelabel_router,
    replay_frozen_adaptive_utility_model,
    replay_full_prelabel_router,
    route_frozen_predicted_utility_or_exact_b,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.generation.contracts import (
    CLAIM_SCOPE,
    EXPECTED_BANK_LOCK_HASH,
    SOURCE_BUDGET_PER_CLASS,
    SOURCE_STREAM_NAMESPACE,
    TOTAL_PER_CLASS,
    GenerationLock,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.sceptre import build_candidate_menu
from midogpp_thesis.cvae.routing.sceptre.contracts import RAW_ROUTE_POLICY_ID
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.uncertainty import (
    FIXED_ACCEPTANCE_PROBABILITY,
    DirichletBootstrapConfig,
)


RAW_SOURCE_RECEIPT = "5" * 64
HistoricalUtilityCell = development_surface.HistoricalUtilityCell
SourceInnerDevelopmentSurface = development_surface.SourceInnerDevelopmentSurface
fit_nested_lodo_pairwise_ranker = development_model.fit_nested_lodo_pairwise_ranker


def _generation_lock() -> GenerationLock:
    expert_locks = [
        {
            "source_center": center,
            "training_seed": training_seed,
            "expert_lock_hash": stable_hash(
                {"source_center": center, "training_seed": training_seed}
            ),
        }
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
        "claim_scope": CLAIM_SCOPE,
        "bank": {
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_locks": expert_locks,
            "candidate_sources_by_target": {
                target: list(legal_routing_sources(target)) for target in CENTERS
            },
        },
        "generation": {
            "training_seeds": list(TRAINING_SEEDS),
            "generation_seeds": list(GENERATION_SEEDS),
            "source_stream_namespace": SOURCE_STREAM_NAMESPACE,
            "max_source_block_per_class": TOTAL_PER_CLASS,
            "equal_union_source_budget_per_class": SOURCE_BUDGET_PER_CLASS,
            "total_per_class": TOTAL_PER_CLASS,
        },
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    return GenerationLock(payload)


def _surface() -> SourceInnerDevelopmentSurface:
    index = {center: ordinal for ordinal, center in enumerate(CENTERS)}
    cells = []
    for query in CENTERS:
        for candidate in CENTERS:
            if query == candidate:
                continue
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    bacc = (
                        0.55
                        + 0.01 * index[candidate]
                        - 0.001 * abs(index[query] - index[candidate])
                        + 0.00001 * (training_seed + generation_seed)
                    )
                    cells.append(
                        HistoricalUtilityCell(
                            query_center=query,
                            candidate_center=candidate,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            bacc=bacc,
                            macro_f1=bacc,
                        )
                    )
    return SourceInnerDevelopmentSurface(
        cells=tuple(cells),
        utility_lock_sha256="1" * 64,
        utility_table_sha256="2" * 64,
        case_confusions_sha256="3" * 64,
        amendment_sha256="4" * 64,
    )


def _raw_evidence() -> tuple[RawSourceEvidence, ...]:
    index = {center: ordinal for ordinal, center in enumerate(CENTERS)}
    rows = []
    for query in CENTERS:
        for candidate in CENTERS:
            if query == candidate:
                continue
            base = float(index[candidate] + abs(index[query] - index[candidate]))
            rows.append(
                RawSourceEvidence(
                    query_center=query,
                    candidate_center=candidate,
                    training_replica_proxy_energy={
                        seed: base + 0.001 * seed for seed in TRAINING_SEEDS
                    },
                    predictive_entropy=0.2 + 0.01 * index[candidate],
                    vote_disagreement=0.1 + 0.01 * index[query],
                )
            )
    return tuple(rows)


def _partition():
    identities = []
    for center_index, center in enumerate(CENTERS):
        case_count = 25 if center_index < 2 else 24
        identities.extend(
            CaseIdentity(
                target_center=center,
                case_id=f"case-{center}-{ordinal:03d}",
                sample_id=f"sample-{center}-{ordinal:03d}",
            )
            for ordinal in range(case_count)
        )
    return build_three_role_partition(tuple(identities))


@pytest.fixture(scope="module")
def frozen_fixture() -> SimpleNamespace:
    target = "2"
    raw = _raw_evidence()
    surface = _surface()
    lock = _generation_lock()
    fits = []
    outer_by_target = {}
    menus = []
    models = []
    for center in CENTERS:
        outer = build_outer_development_evidence(
            raw,
            outer_target=center,
            raw_source_receipt_hash=RAW_SOURCE_RECEIPT,
        )
        fit = fit_nested_lodo_pairwise_ranker(
            surface.for_outer_target(center),
            outer,
            alphas=(1.0,),
        )
        menu = build_candidate_menu(lock, center)
        models.append(
            freeze_adaptive_utility_model(
                fit,
                generation_lock=lock,
                candidate_menu=menu,
            )
        )
        fits.append(fit)
        outer_by_target[center] = outer
        menus.append(menu)
    target_index = CENTERS.index(target)
    fit = fits[target_index]
    menu = menus[target_index]
    frozen = models[target_index]
    target_evidence = build_target_prediction_evidence(
        raw,
        target_center=target,
        raw_source_receipt_hash=RAW_SOURCE_RECEIPT,
    )
    partition = _partition()
    dirichlet_config = DirichletBootstrapConfig(draw_count=64, rng_seed=11)
    full_router = freeze_full_prelabel_router(
        tuple(models),
        generation_lock=lock,
        partition=partition,
        dirichlet_config=dirichlet_config,
    )
    return SimpleNamespace(
        raw=raw,
        fit=fit,
        fits=tuple(fits),
        models=tuple(models),
        outer_evidence=outer_by_target[target],
        lock=lock,
        menu=menu,
        menus=tuple(menus),
        frozen=frozen,
        target_evidence=target_evidence,
        partition=partition,
        dirichlet_config=dirichlet_config,
        full_router=full_router,
    )


def _zero_coefficient_freeze(
    frozen: FrozenAdaptiveUtilityModel,
) -> FrozenAdaptiveUtilityModel:
    coefficients = (0.0,) * len(frozen.coefficients)
    training_keys = sorted(
        (query, candidate)
        for query in frozen.training_query_centers
        for candidate in frozen.candidate_sources
        if query != candidate
    )
    training_receipt = canonical_hash(
        {
            "schema_version": "sceptre_pairwise_utility_model_v1",
            "outer_target": frozen.outer_target,
            "candidate_centers": list(frozen.candidate_sources),
            "feature_names": list(frozen.feature_names),
            "feature_means": list(frozen.feature_means),
            "feature_scales": list(frozen.feature_scales),
            "coefficients": list(coefficients),
            "alpha": frozen.selected_alpha,
            "training_query_centers": list(frozen.training_query_centers),
            "training_keys": [list(key) for key in training_keys],
            "parent_exclusion_receipt_hash": (
                frozen.parent_exclusion_receipt_sha256
            ),
            "evidence_transform_receipt_hash": (
                frozen.evidence_transform_receipt_sha256
            ),
        }
    )
    return replace(
        frozen,
        coefficients=coefficients,
        training_receipt_sha256=training_receipt,
        model_sha256="",
    )


def test_freeze_is_canonical_deterministic_and_binds_all_receipts(
    frozen_fixture: SimpleNamespace,
) -> None:
    first = frozen_fixture.frozen
    second = freeze_adaptive_utility_model(
        frozen_fixture.fit,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )

    assert first == second
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert len(first.model_sha256) == 64
    assert first.parent_exclusion_receipt_sha256 == (
        frozen_fixture.fit.final_model.parent_exclusion_receipt_hash
    )
    assert first.evidence_transform_receipt_sha256 == (
        frozen_fixture.fit.final_model.evidence_receipt_hash
    )
    assert first.outer_evidence_receipt_sha256 == (
        frozen_fixture.fit.outer_evidence_receipt_hash
    )
    assert first.reconstruct_model() == frozen_fixture.fit.final_model
    assert FrozenAdaptiveUtilityModel.from_canonical_bytes(
        first.to_canonical_bytes()
    ) == first
    payload = first.to_payload()
    assert payload["higher_is_better"] is True
    assert payload["route_time_labels_consumed"] is False
    assert payload["publication_status"] == FROZEN_MODEL_PUBLICATION_STATUS


def test_freeze_rejects_tampering_and_noncanonical_serialization(
    frozen_fixture: SimpleNamespace,
) -> None:
    payload = frozen_fixture.frozen.to_payload()
    model = dict(payload["model"])
    coefficients = list(model["coefficients"])
    coefficients[0] += 0.1
    model["coefficients"] = coefficients
    payload["model"] = model

    with pytest.raises(ProtocolError, match="receipt does not replay"):
        FrozenAdaptiveUtilityModel.from_payload(payload)
    pretty = json.dumps(
        frozen_fixture.frozen.to_payload(),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    with pytest.raises(ProtocolError, match="not canonical"):
        FrozenAdaptiveUtilityModel.from_canonical_bytes(pretty)


def test_replay_rejects_changed_model_or_target_identity(
    frozen_fixture: SimpleNamespace,
) -> None:
    receipt = replay_frozen_adaptive_utility_model(
        frozen_fixture.frozen,
        frozen_fixture.fit,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    assert len(receipt.receipt_sha256) == 64
    assert receipt.parent_exclusion_receipt_sha256 == (
        frozen_fixture.frozen.parent_exclusion_receipt_sha256
    )

    changed_model = replace(
        frozen_fixture.fit.final_model,
        coefficients=(
            frozen_fixture.fit.final_model.coefficients[0] + 0.1,
            *frozen_fixture.fit.final_model.coefficients[1:],
        ),
    )
    with pytest.raises(ProtocolError, match="model receipt does not replay"):
        replay_frozen_adaptive_utility_model(
            frozen_fixture.frozen,
            replace(frozen_fixture.fit, final_model=changed_model),
            generation_lock=frozen_fixture.lock,
            candidate_menu=frozen_fixture.menu,
        )
    with pytest.raises(ProtocolError, match="candidate target differ"):
        replay_frozen_adaptive_utility_model(
            frozen_fixture.frozen,
            frozen_fixture.fit,
            generation_lock=frozen_fixture.lock,
            candidate_menu=build_candidate_menu(frozen_fixture.lock, "3"),
        )


def test_higher_is_better_route_is_distinct_from_core_energy_semantics(
    frozen_fixture: SimpleNamespace,
) -> None:
    decision = route_frozen_predicted_utility_or_exact_b(
        frozen_fixture.frozen,
        frozen_fixture.target_evidence,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )

    assert isinstance(decision, AdaptiveUtilityRoute)
    assert decision.selected_source_center in frozen_fixture.menu.candidate_sources
    payload = decision.to_payload()
    assert payload["higher_is_better"] is True
    assert payload["score_semantics"] == PREDICTED_UTILITY_SEMANTICS
    assert payload["policy_id"] == PREDICTED_UTILITY_POLICY_ID
    assert payload["policy_id"] != RAW_ROUTE_POLICY_ID


def test_exact_prediction_tie_preserves_full_set_and_falls_back_to_b(
    frozen_fixture: SimpleNamespace,
) -> None:
    tied = _zero_coefficient_freeze(frozen_fixture.frozen)
    decision = route_frozen_predicted_utility_or_exact_b(
        tied,
        frozen_fixture.target_evidence,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )

    assert isinstance(decision, AdaptiveUtilityExactBFallback)
    assert decision.reason == EXACT_UTILITY_TIE_REASON
    assert decision.winner_sources == frozen_fixture.menu.candidate_sources
    assert decision.ranking_sha256 is not None
    assert decision.to_payload()["control_id"] == "B"
    assert decision.to_payload()["fake_tie_breaking"] is False


def test_missing_invalid_or_unbound_evidence_falls_back_deterministically(
    frozen_fixture: SimpleNamespace,
) -> None:
    missing_first = route_frozen_predicted_utility_or_exact_b(
        frozen_fixture.frozen,
        None,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    missing_second = route_frozen_predicted_utility_or_exact_b(
        frozen_fixture.frozen,
        None,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    assert isinstance(missing_first, AdaptiveUtilityExactBFallback)
    assert missing_first.reason == MISSING_UTILITY_EVIDENCE_REASON
    assert missing_first.decision_sha256 == missing_second.decision_sha256

    bare_rows = frozen_fixture.target_evidence.rows
    invalid = route_frozen_predicted_utility_or_exact_b(
        frozen_fixture.frozen,
        bare_rows,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    assert isinstance(invalid, AdaptiveUtilityExactBFallback)
    assert invalid.reason == INVALID_UTILITY_EVIDENCE_REASON

    drifted = route_frozen_predicted_utility_or_exact_b(
        frozen_fixture.frozen,
        frozen_fixture.outer_evidence,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    assert isinstance(drifted, AdaptiveUtilityExactBFallback)
    assert drifted.reason == INVALID_UTILITY_EVIDENCE_REASON


def test_route_rejects_generation_menu_identity_drift(
    frozen_fixture: SimpleNamespace,
) -> None:
    with pytest.raises(ProtocolError, match="candidate target differ"):
        route_frozen_predicted_utility_or_exact_b(
            frozen_fixture.frozen,
            frozen_fixture.target_evidence,
            generation_lock=frozen_fixture.lock,
            candidate_menu=build_candidate_menu(frozen_fixture.lock, "3"),
        )


def test_full_prelabel_router_is_exact_nine_h_canonical_and_policy_complete(
    frozen_fixture: SimpleNamespace,
) -> None:
    first = frozen_fixture.full_router
    second = freeze_full_prelabel_router(
        frozen_fixture.models,
        generation_lock=frozen_fixture.lock,
        partition=frozen_fixture.partition,
        dirichlet_config=frozen_fixture.dirichlet_config,
    )

    assert first == second
    assert tuple(model.outer_target for model in first.models) == CENTERS
    assert len({model.model_sha256 for model in first.models}) == len(CENTERS)
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert FrozenPrelabelRouter.from_canonical_bytes(
        first.to_canonical_bytes()
    ) == first
    payload = first.to_payload()
    assert payload["schema_version"] == FULL_ROUTER_SCHEMA
    assert payload["artifact_role"] == FULL_ROUTER_ROLE
    assert payload["frozen_before_test_label_access"] is True
    assert payload["decision_policy"]["fixed_joint_acceptance_probability"] == (
        FIXED_ACCEPTANCE_PROBABILITY
    )
    assert payload["partition_identity"]["labels_consumed"] is False
    assert payload["claim_status"]["execution_authorized"] is False
    assert len(first.decision_policy_sha256) == 64
    for model, menu in zip(first.models, frozen_fixture.menus, strict=True):
        assert model.candidate_sources == legal_routing_sources(model.outer_target)
        assert model.candidate_menu_hash == menu.menu_hash


def test_full_router_roundtrip_and_replay_reject_policy_or_inventory_drift(
    frozen_fixture: SimpleNamespace,
) -> None:
    receipt = replay_full_prelabel_router(
        frozen_fixture.full_router,
        frozen_fixture.models,
        generation_lock=frozen_fixture.lock,
        partition=frozen_fixture.partition,
        dirichlet_config=frozen_fixture.dirichlet_config,
    )
    assert len(receipt.receipt_sha256) == 64
    assert tuple(target for target, _ in receipt.model_sha256_by_target) == CENTERS

    with pytest.raises(ProtocolError, match="exact CENTERS order"):
        freeze_full_prelabel_router(
            tuple(reversed(frozen_fixture.models)),
            generation_lock=frozen_fixture.lock,
            partition=frozen_fixture.partition,
            dirichlet_config=frozen_fixture.dirichlet_config,
        )

    payload = frozen_fixture.full_router.to_payload()
    policy = dict(payload["decision_policy"])
    policy["support_minimum_bacc_gain"] = 0.1
    payload["decision_policy"] = policy
    payload["decision_policy_sha256"] = canonical_hash(
        {
            "schema_version": "sceptre_complete_decision_policy_v1",
            **policy,
        }
    )
    with pytest.raises(ProtocolError, match="decision thresholds drifted"):
        FrozenPrelabelRouter.from_payload(payload)

    pretty = json.dumps(
        frozen_fixture.full_router.to_payload(),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    with pytest.raises(ProtocolError, match="not canonical"):
        FrozenPrelabelRouter.from_canonical_bytes(pretty)


def test_full_router_binds_typed_per_h_g_proposals_to_phase_receipts(
    frozen_fixture: SimpleNamespace,
) -> None:
    proposals = []
    for model, menu in zip(
        frozen_fixture.models,
        frozen_fixture.menus,
        strict=True,
    ):
        evidence = build_target_prediction_evidence(
            frozen_fixture.raw,
            target_center=model.outer_target,
            raw_source_receipt_hash=RAW_SOURCE_RECEIPT,
        )
        decision = route_frozen_predicted_utility_or_exact_b(
            model,
            evidence,
            generation_lock=frozen_fixture.lock,
            candidate_menu=menu,
        )
        proposal = frozen_fixture.full_router.bind_g_proposal(decision)
        proposals.append(proposal)
        fold_receipt = proposal.to_fold_receipt(3)
        assert fold_receipt.phase == "G_LABEL_FREE"
        assert fold_receipt.target_center == model.outer_target
        assert fold_receipt.partition_hash == frozen_fixture.partition.partition_hash
        assert fold_receipt.payload_hash == proposal.proposal_sha256
        assert proposal.full_router_sha256 == (
            frozen_fixture.full_router.full_router_sha256
        )
        assert proposal.decision_policy_sha256 == (
            frozen_fixture.full_router.decision_policy_sha256
        )
        assert proposal.candidate_menu_hash == menu.menu_hash
        assert proposal.to_payload()["labels_consumed"] is False

    assert tuple(proposal.target_center for proposal in proposals) == CENTERS

    tied_model = _zero_coefficient_freeze(frozen_fixture.frozen)
    tied_decision = route_frozen_predicted_utility_or_exact_b(
        tied_model,
        frozen_fixture.target_evidence,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    with pytest.raises(ProtocolError, match="decision lineage drifted"):
        frozen_fixture.full_router.bind_g_proposal(tied_decision)

    canonical_tie = route_frozen_predicted_utility_or_exact_b(
        frozen_fixture.frozen,
        None,
        generation_lock=frozen_fixture.lock,
        candidate_menu=frozen_fixture.menu,
    )
    fallback = frozen_fixture.full_router.bind_g_proposal(canonical_tie)
    assert fallback.fallback_to_exact_b is True
    assert fallback.proposed_route == EXACT_B_CANDIDATE
