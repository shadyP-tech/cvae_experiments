from __future__ import annotations

from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.calibration_gate import (
    apply_calibration_gate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.evidence_builder import (
    build_target_prediction_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.model_freeze import (
    AdaptiveUtilityRoute,
    FrozenGProposal,
    route_frozen_predicted_utility_or_exact_b,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
    ConfusionCounts,
    FamilyOutcome,
    pool_confusions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.phase_order import (
    SceptrePhaseManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.route_policy import (
    FrozenRoutePolicy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    FreshProcessValidation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.support_tournament import (
    select_support_family,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.uncertainty import (
    SEED_CELL_GRID,
    build_role_prediction_surface,
    paired_dirichlet_route_decision,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from test_sceptre_model_freeze import (
    RAW_SOURCE_RECEIPT,
    frozen_fixture,
)


def _small_identities() -> tuple[CaseIdentity, ...]:
    return tuple(
        CaseIdentity(center, f"case-{center}-{case}", f"sample-{center}-{case}")
        for center in CENTERS
        for case in range(5)
    )


@pytest.fixture(scope="module")
def proposals(frozen_fixture: SimpleNamespace) -> tuple[FrozenGProposal, ...]:
    rows: list[FrozenGProposal] = []
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
        assert isinstance(decision, AdaptiveUtilityRoute)
        rows.append(frozen_fixture.full_router.bind_g_proposal(decision))
    return tuple(rows)


def _outcome(
    candidate: str,
    confusion: ConfusionCounts,
    *,
    fold,
    role: str,
    model,
    partition_hash: str,
    brier: float = 0.2,
    log_loss: float = 0.4,
) -> FamilyOutcome:
    return FamilyOutcome(
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        role=role,
        candidate_center=candidate,
        partition_hash=partition_hash,
        case_set_hash=fold.case_set_hash(role),
        candidate_menu_hash=model.candidate_menu_hash,
        prediction_receipt_hash=canonical_hash(
            {
                "prediction": candidate,
                "role": role,
                "fold": fold.fold_hash,
            }
        ),
        confusion=confusion,
        brier_sum=brier * confusion.row_count,
        log_loss_sum=log_loss * confusion.row_count,
        case_count=len(
            {
                "SELECTION": fold.selection_case_ids,
                "CALIBRATION": fold.calibration_case_ids,
            }[role]
        ),
        exact_b_control_receipt_hash=(
            model.exact_b_control_receipt_hash
            if candidate == EXACT_B_CANDIDATE
            else None
        ),
    )


def _support_decision(
    frozen_fixture: SimpleNamespace,
    proposal: FrozenGProposal,
    fold_ordinal: int,
    *,
    accept_proposal: bool,
    make_other_candidate_best: bool = False,
):
    partition = frozen_fixture.partition
    router = frozen_fixture.full_router
    fold = partition.fold(proposal.target_center, fold_ordinal)
    model = router.model_for_target(proposal.target_center)
    baseline_counts = ConfusionCounts(40, 10, 10, 40)
    exact_b = _outcome(
        EXACT_B_CANDIDATE,
        baseline_counts,
        fold=fold,
        role="SELECTION",
        model=model,
        partition_hash=partition.partition_hash,
    )
    other_best = next(
        source
        for source in legal_routing_sources(proposal.target_center)
        if source != proposal.g_proposed_candidate
    )
    rows = tuple(
        _outcome(
            candidate,
            (
                ConfusionCounts(45, 5, 8, 42)
                if (
                    (accept_proposal and candidate == proposal.g_proposed_candidate)
                    or (make_other_candidate_best and candidate == other_best)
                )
                else baseline_counts
            ),
            fold=fold,
            role="SELECTION",
            model=model,
            partition_hash=partition.partition_hash,
        )
        for candidate in legal_routing_sources(proposal.target_center)
    )
    return select_support_family(
        rows,
        target_center=proposal.target_center,
        fold=fold,
        partition_hash=partition.partition_hash,
        exact_b=exact_b,
        g_proposal=proposal,
        frozen_router=router,
    )


def _calibration_uncertainty(
    frozen_fixture: SimpleNamespace,
    support,
    capability,
):
    fold = frozen_fixture.partition.fold(
        support.target_center,
        support.fold_ordinal,
    )
    observations: list[str] = []
    cases: list[str] = []
    labels: list[int] = []
    for case_id in fold.calibration_case_ids:
        for label in (0, 1):
            observations.append(f"{case_id}::row-{label}")
            cases.append(case_id)
            labels.append(label)
    probability_map = {}
    for action in (*legal_routing_sources(support.target_center), EXACT_B_CANDIDATE):
        probability_map[action] = {
            seed_cell: tuple(
                (
                    (0.9 if label == 1 else 0.1)
                    if action == support.selected_candidate
                    else (0.6 if action == EXACT_B_CANDIDATE else 0.5)
                )
                for label in labels
            )
            for seed_cell in SEED_CELL_GRID
        }
    surface = build_role_prediction_surface(
        target_center=support.target_center,
        fold=fold,
        partition_hash=frozen_fixture.partition.partition_hash,
        role="CALIBRATION",
        observation_ids=observations,
        case_ids=cases,
        labels=labels,
        probabilities_by_action_and_seed=probability_map,
        candidate_menu_hash=support.candidate_menu_hash,
        exact_b_control_receipt_hash=support.exact_b_control_receipt_hash,
        prediction_bundle_sha256=canonical_hash(
            {"calibration_predictions": support.decision_hash}
        ),
        phase_capability=capability,
    )
    return paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate=support.selected_candidate,
        support_selected_candidate=support.selected_candidate,
        config=frozen_fixture.full_router.dirichlet_config,
    )


def _accepted_calibration(
    frozen_fixture: SimpleNamespace,
    support,
    capability,
):
    uncertainty = _calibration_uncertainty(
        frozen_fixture,
        support,
        capability,
    )
    fold = frozen_fixture.partition.fold(
        support.target_center,
        support.fold_ordinal,
    )
    model = frozen_fixture.full_router.model_for_target(support.target_center)
    exact_b = _outcome(
        EXACT_B_CANDIDATE,
        ConfusionCounts(40, 10, 10, 40),
        fold=fold,
        role="CALIBRATION",
        model=model,
        partition_hash=frozen_fixture.partition.partition_hash,
    )
    candidate = _outcome(
        support.selected_candidate,
        ConfusionCounts(45, 5, 8, 42),
        fold=fold,
        role="CALIBRATION",
        model=model,
        partition_hash=frozen_fixture.partition.partition_hash,
        brier=0.1,
        log_loss=0.3,
    )
    calibration = apply_calibration_gate(
        support,
        uncertainty=uncertainty,
        candidate=candidate,
        exact_b=exact_b,
        frozen_router=frozen_fixture.full_router,
    )
    return uncertainty, calibration


def test_three_role_partition_has_45_keys_and_exactly_once_evaluation() -> None:
    partition = build_three_role_partition(
        _small_identities(),
        expected_total_case_count=45,
    )

    assert len(partition.folds) == 45
    for center in CENTERS:
        folds = tuple(fold for fold in partition.folds if fold.target_center == center)
        evaluated = [case for fold in folds for case in fold.evaluation_case_ids]
        assert len(evaluated) == len(set(evaluated)) == 5
        for fold in folds:
            assert set(fold.selection_case_ids).isdisjoint(fold.calibration_case_ids)
            assert set(fold.selection_case_ids).isdisjoint(fold.evaluation_case_ids)
            assert set(fold.calibration_case_ids).isdisjoint(fold.evaluation_case_ids)

    with pytest.raises(ProtocolError, match="duplicated"):
        build_three_role_partition(
            (*_small_identities(), _small_identities()[0]),
            expected_total_case_count=45,
        )
    identities = list(_small_identities())
    identities[-1] = replace(identities[-1], sample_id=identities[0].sample_id)
    with pytest.raises(ProtocolError, match="multiple case rows"):
        build_three_role_partition(identities, expected_total_case_count=45)


def test_support_can_only_validate_g_or_abstain_and_rejects_bundle_substitution(
    frozen_fixture: SimpleNamespace,
    proposals: tuple[FrozenGProposal, ...],
) -> None:
    proposal = proposals[0]
    fallback = _support_decision(
        frozen_fixture,
        proposal,
        0,
        accept_proposal=False,
        make_other_candidate_best=True,
    )
    assert fallback.fallback_required is True
    assert fallback.selected_candidate is None
    assert fallback.winner_set == (proposal.g_proposed_candidate,)

    accepted = _support_decision(
        frozen_fixture,
        proposal,
        0,
        accept_proposal=True,
        make_other_candidate_best=False,
    )
    assert accepted.selected_candidate == proposal.g_proposed_candidate

    forged = replace(
        proposal,
        full_router_sha256=canonical_hash({"wrong_router": True}),
        proposal_sha256="",
    )
    with pytest.raises(ProtocolError, match="G/router/model/control lineage"):
        _support_decision(
            frozen_fixture,
            forged,
            0,
            accept_proposal=True,
        )


def test_complete_45_fold_phase_lifecycle_is_one_way_and_terminally_sealed(
    frozen_fixture: SimpleNamespace,
    proposals: tuple[FrozenGProposal, ...],
) -> None:
    manager = SceptrePhaseManager(
        frozen_fixture.partition,
        frozen_fixture.full_router,
    )
    with pytest.raises(ProtocolError, match="selection capability"):
        manager.issue_selection_capability(CENTERS[0], 0)

    manager.record_label_free_g_decision(proposals[0], 0)
    changed_target_proposal = replace(
        proposals[0],
        evidence_sha256=canonical_hash({"changed_target_evidence": True}),
        proposal_sha256="",
    )
    with pytest.raises(ProtocolError, match="target-global G proposal"):
        manager.record_label_free_g_decision(changed_target_proposal, 1)
    for proposal in proposals:
        first_fold = 1 if proposal is proposals[0] else 0
        for fold_ordinal in range(first_fold, 5):
            manager.record_label_free_g_decision(proposal, fold_ordinal)
    manager.seal_all_g_decisions()

    supports = {}
    routed_key = (proposals[0].target_center, 0)
    first_capability = manager.issue_selection_capability(*routed_key)
    with pytest.raises(TypeError, match="cannot cross process"):
        pickle.dumps(first_capability)
    wrong_fold_support = _support_decision(
        frozen_fixture,
        proposals[0],
        1,
        accept_proposal=True,
    )
    with pytest.raises(ProtocolError, match="fold lineage"):
        manager.record_selection_decision(first_capability, wrong_fold_support)
    first_support = _support_decision(
        frozen_fixture,
        proposals[0],
        0,
        accept_proposal=True,
    )
    manager.record_selection_decision(first_capability, first_support)
    supports[routed_key] = first_support

    with pytest.raises(ProtocolError, match="all 45"):
        manager.seal_all_selection_decisions()
    with pytest.raises(ProtocolError, match="calibration capability"):
        manager.issue_calibration_capability(*routed_key)

    proposal_by_target = {row.target_center: row for row in proposals}
    for center in CENTERS:
        for fold_ordinal in range(5):
            key = (center, fold_ordinal)
            if key == routed_key:
                continue
            capability = manager.issue_selection_capability(*key)
            support = _support_decision(
                frozen_fixture,
                proposal_by_target[center],
                fold_ordinal,
                accept_proposal=False,
            )
            manager.record_selection_decision(capability, support)
            supports[key] = support
    manager.seal_all_selection_decisions()

    for center in CENTERS:
        for fold_ordinal in range(5):
            key = (center, fold_ordinal)
            capability = manager.issue_calibration_capability(*key)
            support = supports[key]
            if key == routed_key:
                uncertainty, calibration = _accepted_calibration(
                    frozen_fixture,
                    support,
                    capability,
                )
                manager.record_calibration_uncertainty(capability, uncertainty)
                forged_calibration = replace(
                    calibration,
                    uncertainty_decision_hash=canonical_hash(
                        {"forged_uncertainty": True}
                    ),
                    decision_hash="",
                )
                with pytest.raises(ProtocolError, match="fold lineage"):
                    manager.record_calibration_decision(
                        capability,
                        forged_calibration,
                    )
            else:
                calibration = apply_calibration_gate(
                    support,
                    uncertainty=None,
                    candidate=None,
                    exact_b=None,
                    frozen_router=frozen_fixture.full_router,
                )
            manager.record_calibration_decision(capability, calibration)

    policy = manager.seal_complete_policy()
    frozen_policy = manager.export_frozen_route_policy()
    assert frozen_policy.route_for(*routed_key) == proposals[0].g_proposed_candidate
    assert frozen_policy.route_for(proposals[0].target_center, 1) == EXACT_B_CANDIDATE
    replayed_policy = FrozenRoutePolicy.from_canonical_bytes(
        frozen_policy.to_canonical_bytes()
    )
    assert replayed_policy == frozen_policy
    adversarial_rows = list(frozen_policy.route_rows)
    target, fold, g_hash, g_candidate, _route, decision_hash = adversarial_rows[0]
    different_expert = next(
        candidate
        for candidate in legal_routing_sources(target)
        if candidate != g_candidate
    )
    adversarial_rows[0] = (
        target,
        fold,
        g_hash,
        g_candidate,
        different_expert,
        decision_hash,
    )
    with pytest.raises(ProtocolError, match="outside G-or-exact-B"):
        replace(
            frozen_policy,
            route_rows=tuple(adversarial_rows),
            policy_artifact_hash="",
        )
    source_hash = canonical_hash({"source": "sealed"})
    reconstruction = canonical_hash(
        {
            "router": frozen_fixture.full_router.full_router_sha256,
            "policy": policy.seal_hash,
        }
    )
    attestation = DurablePreterminalAttestation(
        policy_seal_hash=policy.seal_hash,
        validations=(
            FreshProcessValidation(
                101,
                policy.seal_hash,
                source_hash,
                reconstruction,
            ),
            FreshProcessValidation(
                202,
                policy.seal_hash,
                source_hash,
                reconstruction,
            ),
        ),
    )
    terminal = manager.begin_terminal_evaluation(attestation)
    assert terminal.policy_seal_hash == policy.seal_hash
    assert terminal.route_policy_hash == frozen_policy.policy_artifact_hash
    assert terminal.router_bundle_hash == frozen_fixture.full_router.router_bundle_hash
    with pytest.raises(ProtocolError, match="terminal evaluation"):
        manager.begin_terminal_evaluation(attestation)


def test_bacc_is_recomputed_from_pooled_additive_counts() -> None:
    pooled = pool_confusions(
        (ConfusionCounts(9, 1, 3, 7), ConfusionCounts(1, 9, 1, 9))
    )
    assert pooled == ConfusionCounts(10, 10, 4, 16)
    assert pooled.bacc == pytest.approx(0.65)
