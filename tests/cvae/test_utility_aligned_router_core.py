from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup import (
    build_topup_geometry,
    target_topup_geometry,
)
from midogpp_thesis.cvae.routing.utility_aligned import (
    BASE_ACTION_ID,
    GLOBAL_ACTION_ID,
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    MIN_SUPPORT_BOOTSTRAP_REPLICATES,
    ROUTED_ACTION_ID,
    SEED_PAIR_COUNT,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    CandidateFeatureRow,
    ExactTailUtilityRow,
    build_case_bootstrap_plan,
    build_distributional_feature_surface,
    build_pairwise_preferences,
    build_utility_aligned_action,
    build_utility_aligned_policy,
    fit_utility_aligned_models,
    nested_cardinality_transfer_evaluation,
    permute_interaction_features,
    validate_exact_tail_utility_rows,
)
from midogpp_thesis.cvae.routing.utility_aligned import (
    contracts as contract_facade,
    metrics as metrics_module,
    policy_contracts,
    result_contracts,
    row_contracts,
    serialization as serialization_module,
    surface_contracts,
)
from midogpp_thesis.cvae.routing.utility_aligned.action_adapter import (
    build_utility_aligned_action as adapter_build_utility_aligned_action,
)
from midogpp_thesis.cvae.routing.utility_aligned import policy as policy_module


OUTER_TARGET = "H"
SOURCES = tuple(f"d{index}" for index in range(8))
SEEDS = (17, 42, 101)
EXPECTED_SCIENTIFIC_HASHES = {
    "action": "6c33a19693e1e0ecb64cd822a2df6eb12be78821cccfe60d2502104472c9eebe",
    "bootstrap_plan": "330951b77d4788f2b6f69b52983119a3bdcf95829f69e837077c2347532c00c5",
    "feature_surface": "2c82f67d7246d180edbd903816df3829ffacf8ea6dd0155a7dd841406fe64742",
    "first_row": "96cbdd158f409cc8515e5186bbe87e267836b5500ec9f323807195e4ff32808e",
    "global_crossfit": "37efac044805e5d42329e3226ce84541ce95ffdf33f41fd9ab06866cc8213bfe",
    "global_metrics": "ced994638dc470a391d912c91f57528f0e0dbf6024a7c5476b669411f064e1ce",
    "interaction_crossfit": "d0c89c6690f2d92ad38b3c43a55c4ee8b36bd20611856d447bb1ddcde9219ff4",
    "interaction_metrics": "20f934c9817dd2bdef30025f9ba641e8ac520ef1dd70d8a6b69c9fb7ab262a73",
    "model": "0eb51249de81dc8f09e7b1a764cbda36e4bcf83f02652e0d6a96292086ec74e3",
    "policy": "dee94c539c195e9570baa18bac6c6f4c7ac5b52316bdb5672ed7973824957718",
    "target_surface": "fb1134c6822cf6fb5ec5693f93e9a0d97d4e9e2da54966cbc10f99d4999e11cc",
    "transfer": "1ff4bb311e18cb00e59a2db6f293a4494156715b8ee31aa388ec104f9bc8f269",
    "utility_surface": "9bb48259a723a88eea69389c36e71f7f1754403af3e05f654e720d8f88e00ae8",
}


def _similarity(query_index: int, source_index: int) -> float:
    preferred = (query_index + 1) % len(SOURCES)
    distance = min(
        (source_index - preferred) % len(SOURCES),
        (preferred - source_index) % len(SOURCES),
    )
    return 1.0 - float(distance) / 4.0


def _feature_kwargs(
    *,
    role: str,
    query: str,
    source: str,
    training_seed: int,
    generation_seed: int,
    support_hash: str,
    support_case_count: int,
    similarity: float,
) -> dict[str, object]:
    seed_cell = SEEDS.index(training_seed) * 3 + SEEDS.index(generation_seed)
    seed_shift = float(seed_cell) * 1.0e-4
    reconstruction = 1.1 - 0.35 * similarity + seed_shift
    kl = 0.35 - 0.10 * similarity + seed_shift
    return {
        "role": role,
        "outer_target_id": OUTER_TARGET,
        "query_id": query,
        "candidate_source": source,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "candidate_source_count": (
            INNER_CANDIDATE_COUNT if role == INNER_ROLE else TARGET_CANDIDATE_COUNT
        ),
        "support_partition_hash": support_hash,
        "support_case_count": support_case_count,
        "reconstruction_mean": reconstruction,
        "reconstruction_std": 0.05 + 0.01 * (1.0 - similarity),
        "reconstruction_q25": reconstruction - 0.06,
        "reconstruction_q50": reconstruction,
        "reconstruction_q75": reconstruction + 0.06,
        "kl_mean": kl,
        "kl_std": 0.02 + 0.005 * (1.0 - similarity),
        "kl_q25": kl - 0.025,
        "kl_q50": kl,
        "kl_q75": kl + 0.025,
        "replica_disagreement": 0.01 + 0.002 * (1.0 - similarity),
        "distribution_mmd": 0.2 + (1.0 - similarity),
        "metadata_similarity": similarity,
    }


def _source_inner_rows() -> tuple[tuple[ExactTailUtilityRow, ...], tuple[CandidateFeatureRow, ...]]:
    utility_rows: list[ExactTailUtilityRow] = []
    feature_rows: list[CandidateFeatureRow] = []
    for query_index, query in enumerate(SOURCES):
        candidates = tuple(source for source in SOURCES if source != query)
        for source in candidates:
            source_index = SOURCES.index(source)
            similarity = _similarity(query_index, source_index)
            for training_seed in SEEDS:
                for generation_seed in SEEDS:
                    seed_cell = SEEDS.index(training_seed) * 3 + SEEDS.index(
                        generation_seed
                    )
                    base_bacc = 0.70 + float(seed_cell) * 1.0e-4
                    delta = 0.015 + 0.07 * similarity + float(seed_cell) * 1.0e-5
                    base_hash = (
                        f"base-{OUTER_TARGET}-{query}-{training_seed}-{generation_seed}"
                    )
                    tail_hash = (
                        base_hash
                        if source == candidates[0]
                        else f"tail-{query}-{source}-{training_seed}-{generation_seed}"
                    )
                    utility_rows.append(
                        ExactTailUtilityRow(
                            outer_target_id=OUTER_TARGET,
                            query_id=query,
                            candidate_source=source,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            candidate_source_count=INNER_CANDIDATE_COUNT,
                            support_partition_hash=f"support-{OUTER_TARGET}-{query}",
                            evaluation_partition_hash=f"evaluation-{OUTER_TARGET}-{query}",
                            prediction_seal_hash=f"seal-{OUTER_TARGET}",
                            base_prediction_hash=base_hash,
                            tail_prediction_hash=tail_hash,
                            base_bacc=base_bacc,
                            tail_bacc=base_bacc + delta,
                            support_eval_disjoint=True,
                            predictions_sealed_before_labels=True,
                            source_expert_frozen=True,
                        )
                    )
                    feature_rows.append(
                        CandidateFeatureRow(
                            **_feature_kwargs(
                                role=INNER_ROLE,
                                query=query,
                                source=source,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                support_hash=f"support-{OUTER_TARGET}-{query}",
                                support_case_count=10,
                                similarity=similarity,
                            )
                        )
                    )
    return tuple(utility_rows), tuple(feature_rows)


def _target_feature_rows(
    *,
    support_case_count: int = 10,
    bootstrap_index: int | None = None,
    support_partition_hash: str = "support-H-target-point",
) -> tuple[CandidateFeatureRow, ...]:
    rows: list[CandidateFeatureRow] = []
    for source_index, source in enumerate(SOURCES):
        base_similarity = float(source_index) / float(len(SOURCES) - 1)
        if bootstrap_index is None:
            similarity = base_similarity
            support_hash = support_partition_hash
        else:
            similarity = min(
                1.0,
                max(
                    0.0,
                    base_similarity
                    + 0.02 * math.sin(float(bootstrap_index + source_index)),
                ),
            )
            support_hash = support_partition_hash
        for training_seed in SEEDS:
            for generation_seed in SEEDS:
                rows.append(
                    CandidateFeatureRow(
                        **_feature_kwargs(
                            role=TARGET_ROLE,
                            query=OUTER_TARGET,
                            source=source,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            support_hash=support_hash,
                            support_case_count=support_case_count,
                            similarity=similarity,
                        )
                    )
                )
    return tuple(rows)


@pytest.fixture(scope="module")
def fitted_bundle():
    utility_rows, feature_rows = _source_inner_rows()
    utility_surface = validate_exact_tail_utility_rows(utility_rows)
    feature_surface = build_distributional_feature_surface(feature_rows)
    models = fit_utility_aligned_models(
        feature_surface,
        utility_surface,
        alphas=(0.1,),
    )
    transfer = nested_cardinality_transfer_evaluation(
        models,
        feature_surface,
        utility_surface,
    )
    return utility_surface, feature_surface, models, transfer


def test_contract_facade_and_package_api_resolve_to_split_modules() -> None:
    facade_to_owner = {
        contract_facade.ExactTailUtilityRow: row_contracts.ExactTailUtilityRow,
        contract_facade.CaseBootstrapReplicate: row_contracts.CaseBootstrapReplicate,
        contract_facade.CaseBootstrapPlan: row_contracts.CaseBootstrapPlan,
        contract_facade.CandidateFeatureRow: surface_contracts.CandidateFeatureRow,
        contract_facade.ExactTailUtilitySurface: surface_contracts.ExactTailUtilitySurface,
        contract_facade.FeatureSurface: surface_contracts.FeatureSurface,
        contract_facade.PairwisePreference: surface_contracts.PairwisePreference,
        contract_facade.FoldAudit: result_contracts.FoldAudit,
        contract_facade.CrossfitResult: result_contracts.CrossfitResult,
        contract_facade.RankingMetrics: result_contracts.RankingMetrics,
        contract_facade.CardinalityTransferResult: (
            result_contracts.CardinalityTransferResult
        ),
        contract_facade.UtilityAlignedModels: result_contracts.UtilityAlignedModels,
        contract_facade.UtilityAlignedPolicy: policy_contracts.UtilityAlignedPolicy,
    }
    assert all(facade is owner for facade, owner in facade_to_owner.items())
    assert build_utility_aligned_action is adapter_build_utility_aligned_action
    assert "build_utility_aligned_action" not in policy_module.__all__
    assert not hasattr(policy_module, "build_utility_aligned_action")
    assert "immutable_array" not in contract_facade.__all__
    assert not hasattr(contract_facade, "immutable_array")
    assert metrics_module.__all__ == ()
    assert serialization_module.__all__ == ()


def test_scientific_hashes_are_byte_identical_after_module_split(
    fitted_bundle,
) -> None:
    utility_surface, feature_surface, models, transfer = fitted_bundle
    plan = build_case_bootstrap_plan(
        target_id=OUTER_TARGET,
        support_case_ids=tuple(f"case-{index}" for index in range(10)),
    )
    target_surface = build_distributional_feature_surface(
        _target_feature_rows(support_partition_hash=plan.support_partition_hash)
    )
    bootstraps = tuple(
        build_distributional_feature_surface(
            _target_feature_rows(
                bootstrap_index=replicate.replicate_index,
                support_partition_hash=replicate.support_partition_hash,
            ),
            case_bootstrap_replicate=replicate,
        )
        for replicate in plan.replicates
    )
    policy = build_utility_aligned_policy(
        models,
        target_surface,
        transfer,
        confidence_multiplier=0.0,
        support_bootstrap_features=bootstraps,
        case_bootstrap_plan=plan,
    )
    action = build_utility_aligned_action(
        policy,
        geometry=target_topup_geometry(SOURCES),
    )
    assert action is not None

    assert {
        "action": action.action_hash,
        "bootstrap_plan": plan.plan_hash,
        "feature_surface": feature_surface.surface_hash,
        "first_row": utility_surface.rows[0].row_hash,
        "global_crossfit": models.global_crossfit.crossfit_hash,
        "global_metrics": transfer.global_metrics.metrics_hash,
        "interaction_crossfit": models.interaction_crossfit.crossfit_hash,
        "interaction_metrics": transfer.interaction_metrics.metrics_hash,
        "model": models.model_hash,
        "policy": policy.policy_hash,
        "target_surface": target_surface.surface_hash,
        "transfer": transfer.result_hash,
        "utility_surface": utility_surface.surface_hash,
    } == EXPECTED_SCIENTIFIC_HASHES


def test_exact_tail_surface_is_complete_paired_and_deterministic() -> None:
    utility_rows, _feature_rows = _source_inner_rows()
    first = validate_exact_tail_utility_rows(utility_rows)
    second = validate_exact_tail_utility_rows(tuple(reversed(utility_rows)))

    assert len(first.rows) == 8 * 7 * 9 == 504
    assert first.surface_hash == second.surface_hash
    assert first.row_keys == second.row_keys
    assert first.rows[0].replicate_id.startswith("training_")
    # Distinct actions may legitimately make the same binary predictions.
    assert any(
        row.base_prediction_hash == row.tail_prediction_hash for row in first.rows
    )

    with pytest.raises(ProtocolError, match="nine paired seed cells"):
        validate_exact_tail_utility_rows(utility_rows[:-1])
    with pytest.raises(ProtocolError, match="q != H"):
        replace(utility_rows[0], query_id=OUTER_TARGET)


def test_pairwise_preferences_preserve_every_seed_cell() -> None:
    utility_rows, _feature_rows = _source_inner_rows()
    preferences = build_pairwise_preferences(utility_rows)

    assert len(preferences) == 8 * 9 * 21 == 1512
    assert all(item.left_source < item.right_source for item in preferences)
    assert len({item.preference_hash for item in preferences}) == len(preferences)


def test_distributional_features_are_label_free_cardinality_fenced_and_hashed() -> None:
    _utility_rows, feature_rows = _source_inner_rows()
    first = build_distributional_feature_surface(feature_rows)
    second = build_distributional_feature_surface(tuple(reversed(feature_rows)))

    assert first.role == INNER_ROLE
    assert first.candidate_source_count == 7
    assert first.global_values.shape == (504, 7)
    assert first.interaction_values.shape[0] == 504
    assert first.surface_hash == second.surface_hash
    assert np.array_equal(first.interaction_values, second.interaction_values)
    assert not first.interaction_values.flags.writeable
    assert not any("query" in name or "target" in name for name in first.global_feature_names)

    kwargs = _feature_kwargs(
        role=TARGET_ROLE,
        query=OUTER_TARGET,
        source=SOURCES[0],
        training_seed=17,
        generation_seed=17,
        support_hash="support",
        support_case_count=10,
        similarity=0.5,
    )
    with pytest.raises(TypeError):
        CandidateFeatureRow(**kwargs, labels=(0, 1))  # type: ignore[call-arg]

    target_surface = build_distributional_feature_surface(_target_feature_rows())
    assert target_surface.role == TARGET_ROLE
    assert target_surface.candidate_source_count == 8
    assert target_surface.interaction_values.shape[0] == 8 * 9
    with pytest.raises(ProtocolError, match="one role"):
        build_distributional_feature_surface(
            (*feature_rows, *_target_feature_rows())
        )


def test_cyclic_permutation_control_is_deterministic_and_label_free() -> None:
    _utility_rows, feature_rows = _source_inner_rows()
    surface = build_distributional_feature_surface(feature_rows)
    first = permute_interaction_features(surface, permutation_seed=901)
    second = permute_interaction_features(surface, permutation_seed=901)

    assert first.surface_hash == second.surface_hash
    assert first.permutation_seed == 901
    assert np.array_equal(first.global_values, surface.global_values)
    assert not np.array_equal(first.interaction_values, surface.interaction_values)
    with pytest.raises(ProtocolError, match="unpermuted"):
        permute_interaction_features(first, permutation_seed=902)


def test_case_bootstrap_plan_seals_whole_case_indices() -> None:
    case_ids = tuple(f"case-{index}" for index in range(10))
    first = build_case_bootstrap_plan(
        target_id=OUTER_TARGET,
        support_case_ids=tuple(reversed(case_ids)),
        bootstrap_seed=1234,
    )
    second = build_case_bootstrap_plan(
        target_id=OUTER_TARGET,
        support_case_ids=case_ids,
        bootstrap_seed=1234,
    )

    assert first.plan_hash == second.plan_hash
    assert first.support_case_ids == tuple(sorted(case_ids))
    assert len(first.replicates) == MIN_SUPPORT_BOOTSTRAP_REPLICATES
    assert len({item.replicate_hash for item in first.replicates}) == len(
        first.replicates
    )
    assert all(len(item.sampled_indices) == 10 for item in first.replicates)
    with pytest.raises(ProtocolError, match="eight unique"):
        build_case_bootstrap_plan(
            target_id=OUTER_TARGET,
            support_case_ids=tuple(f"case-{index}" for index in range(7)),
        )
    with pytest.raises(ProtocolError, match="sampled-case partition"):
        build_distributional_feature_surface(
            _target_feature_rows(),
            case_bootstrap_replicate=first.replicates[0],
        )


def test_models_use_strict_nested_six_to_seven_transfer_and_domain_bootstrap(
    fitted_bundle,
) -> None:
    _utility_surface, _feature_surface, models, transfer = fitted_bundle

    for crossfit in (models.global_crossfit, models.interaction_crossfit):
        assert len(crossfit.folds) == 8
        for fold in crossfit.folds:
            assert fold.training_candidate_count_per_query == 6
            assert fold.observation_count == 7 * 6 * 9 == 378
            assert len(fold.heldout_row_indices) == 7 * 9 == 63
            assert fold.heldout_query_id not in fold.training_query_ids
            assert fold.heldout_query_id not in fold.training_source_ids

    assert transfer.training_candidate_count == 6
    assert transfer.evaluation_candidate_count == 7
    assert transfer.deployment_candidate_count == 8
    assert transfer.interaction_metrics.top1_lower_bound > 1.0 / 7.0
    assert transfer.interaction_metrics.spearman_lower_bound > 0.0
    assert transfer.interaction_metrics.selected_utility_lower_bound > 0.0
    assert transfer.normalized_gap_reduction_lower_bound > 0.0
    assert transfer.interaction_metrics.normalized_oracle_gap_upper_bound < 0.46
    assert transfer.global_gate_passed
    assert transfer.eligibility_passed
    assert transfer.claim_role.endswith("eligibility_only_not_7_to_8_evidence")


def test_policy_requires_unique_support_bootstraps_and_abstains_exactly(
    fitted_bundle,
) -> None:
    _utility_surface, _feature_surface, models, transfer = fitted_bundle
    plan = build_case_bootstrap_plan(
        target_id=OUTER_TARGET,
        support_case_ids=tuple(f"case-{index}" for index in range(10)),
    )
    target_surface = build_distributional_feature_surface(
        _target_feature_rows(support_partition_hash=plan.support_partition_hash)
    )
    bootstraps = tuple(
        build_distributional_feature_surface(
            _target_feature_rows(
                bootstrap_index=replicate.replicate_index,
                support_partition_hash=replicate.support_partition_hash,
            ),
            case_bootstrap_replicate=replicate,
        )
        for replicate in plan.replicates
    )

    missing_uncertainty = build_utility_aligned_policy(
        models,
        target_surface,
        transfer,
        confidence_multiplier=0.0,
    )
    assert missing_uncertainty.action_id == BASE_ACTION_ID
    assert missing_uncertainty.selected_source is None
    assert missing_uncertainty.fallback_reason == (
        "support_case_bootstrap_uncertainty_missing_exact_base"
    )

    with pytest.raises(ProtocolError, match="unique hashes"):
        build_utility_aligned_policy(
            models,
            target_surface,
            transfer,
            confidence_multiplier=0.0,
            support_bootstrap_features=(bootstraps[0],) * MIN_SUPPORT_BOOTSTRAP_REPLICATES,
            case_bootstrap_plan=plan,
        )

    forged_hash_only = tuple(
        replace(target_surface, surface_hash=f"forged-bootstrap-{index}")
        for index in range(MIN_SUPPORT_BOOTSTRAP_REPLICATES)
    )
    with pytest.raises(ProtocolError, match="typed case replicates"):
        build_utility_aligned_policy(
            models,
            target_surface,
            transfer,
            confidence_multiplier=0.0,
            support_bootstrap_features=forged_hash_only,
            case_bootstrap_plan=plan,
        )

    routed = build_utility_aligned_policy(
        models,
        target_surface,
        transfer,
        confidence_multiplier=0.0,
        support_bootstrap_features=bootstraps,
        case_bootstrap_plan=plan,
    )
    assert routed.action_id == ROUTED_ACTION_ID
    assert routed.selected_source is not None
    assert routed.support_bootstrap_replicates == MIN_SUPPORT_BOOTSTRAP_REPLICATES
    assert routed.support_bootstrap_standard_deviation > 0.0
    assert not routed.seed_selection_performed

    zero_coefficient_model = replace(
        models.interaction_model,
        intercept=0.5,
        coefficients=np.zeros_like(models.interaction_model.coefficients),
        coefficient_covariance=np.zeros_like(
            models.interaction_model.coefficient_covariance
        ),
        residual_variance=0.04,
    )
    correlated_models = replace(models, interaction_model=zero_coefficient_model)
    correlated = build_utility_aligned_policy(
        correlated_models,
        target_surface,
        transfer,
        confidence_multiplier=0.0,
        support_bootstrap_features=bootstraps,
        case_bootstrap_plan=plan,
    )
    assert correlated.replicate_standard_deviation == pytest.approx(0.0)
    assert correlated.support_bootstrap_standard_deviation == pytest.approx(0.0)
    # Out-of-query residual uncertainty is correlated across technical seeds
    # and therefore enters once, rather than being divided by nine.
    assert correlated.standard_error == pytest.approx(0.2)

    uncertain = build_utility_aligned_policy(
        models,
        target_surface,
        transfer,
        confidence_multiplier=1.0e6,
        support_bootstrap_features=bootstraps,
        case_bootstrap_plan=plan,
    )
    assert uncertain.action_id == BASE_ACTION_ID
    assert uncertain.selected_source is None
    assert uncertain.used_exact_base_fallback

    low_support = build_distributional_feature_surface(
        _target_feature_rows(support_case_count=2)
    )
    insufficient = build_utility_aligned_policy(
        models,
        low_support,
        transfer,
        confidence_multiplier=0.0,
    )
    assert insufficient.action_id == BASE_ACTION_ID
    assert insufficient.fallback_reason == (
        "insufficient_independent_support_cases_exact_base"
    )

    geometry = target_topup_geometry(SOURCES)
    assert build_utility_aligned_action(missing_uncertainty, geometry=geometry) is None
    action = build_utility_aligned_action(routed, geometry=geometry)
    assert action is not None
    assert action.topup_counts[routed.selected_source] == 128
    noncanonical = build_topup_geometry(
        SOURCES,
        base_per_source=16,
        topup_total_per_class=16,
    )
    with pytest.raises(ProtocolError, match="canonical 8x128"):
        build_utility_aligned_action(routed, geometry=noncanonical)


def test_global_control_has_an_independent_gate(fitted_bundle) -> None:
    _utility_surface, _feature_surface, models, transfer = fitted_bundle
    target_surface = build_distributional_feature_surface(_target_feature_rows())
    failed_interaction = replace(
        transfer,
        eligibility_passed=False,
        eligibility_reason="synthetic_failed_interaction_eligibility",
    )

    global_policy = build_utility_aligned_policy(
        models,
        target_surface,
        failed_interaction,
        global_only=True,
        confidence_multiplier=0.0,
    )
    assert global_policy.action_id == GLOBAL_ACTION_ID
    assert global_policy.selected_source is not None
    assert global_policy.global_only

    failed_global = replace(
        failed_interaction,
        global_gate_passed=False,
        global_gate_reason="synthetic_failed_global_gate",
    )
    fallback = build_utility_aligned_policy(
        models,
        target_surface,
        failed_global,
        global_only=True,
        confidence_multiplier=0.0,
    )
    assert fallback.action_id == BASE_ACTION_ID
    assert fallback.fallback_reason == "global_source_quality_gate_failed_exact_base"
