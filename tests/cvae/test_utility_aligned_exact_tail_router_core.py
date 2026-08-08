from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.actions import (
    build_inner_exact_tail_action_library,
    build_inner_exact_tail_actions,
    build_target_actions,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_INNER_UTILITY_ROW_COUNT,
    EXPECTED_TARGET_FEATURE_ROW_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GENERATION_SEEDS,
    GLOBAL_DELTA_ACTION_ID,
    INPUT_ARTIFACT_IDS,
    PERMUTATION_ACTION_ID,
    PERMUTATION_SEED,
    R2_ACTION_ID,
    ROUTING_STATUS,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    candidate_sources,
    h_x_e_action_id,
    inner_candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.features import (
    build_heldout_feature_surfaces,
    build_stage90_feature_surface_set,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.modeling import (
    fit_stage90_heldout_models,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.r2_policy import (
    ROUTER_DIAGNOSTIC_IDS,
    build_r2_diagnostic_plan,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.utility_aligned import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
    CandidateFeatureRow,
    ExactTailUtilityRow,
    permute_interaction_features,
    validate_exact_tail_utility_rows,
)


OUTER = "0"
SEED_PAIRS = tuple(
    (training, generation)
    for training in TRAINING_SEEDS
    for generation in GENERATION_SEEDS
)


def _feature_row(
    *,
    role: str,
    outer: str,
    query: str,
    source: str,
    training_seed: int,
    generation_seed: int,
    support_hash: str,
) -> CandidateFeatureRow:
    source_index = CENTERS.index(source)
    query_index = CENTERS.index(query)
    similarity = 1.0 - abs(source_index - query_index) / float(len(CENTERS))
    seed_index = SEED_PAIRS.index((training_seed, generation_seed))
    rec = 0.8 + 0.02 * source_index + 0.001 * query_index + seed_index * 1.0e-5
    kl = 0.2 + 0.01 * source_index + seed_index * 1.0e-5
    return CandidateFeatureRow(
        role=role,
        outer_target_id=outer,
        query_id=query,
        candidate_source=source,
        training_seed=training_seed,
        generation_seed=generation_seed,
        candidate_source_count=(
            INNER_CANDIDATE_COUNT if role == INNER_ROLE else TARGET_CANDIDATE_COUNT
        ),
        support_partition_hash=support_hash,
        support_case_count=FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        reconstruction_mean=rec,
        reconstruction_std=0.05,
        reconstruction_q25=rec - 0.04,
        reconstruction_q50=rec,
        reconstruction_q75=rec + 0.04,
        kl_mean=kl,
        kl_std=0.02,
        kl_q25=kl - 0.015,
        kl_q50=kl,
        kl_q75=kl + 0.015,
        replica_disagreement=0.01 + seed_index * 1.0e-5,
        distribution_mmd=0.3 + 0.02 * abs(source_index - query_index),
        metadata_similarity=similarity,
    )


def _one_target_rows() -> tuple[
    tuple[ExactTailUtilityRow, ...],
    tuple[CandidateFeatureRow, ...],
    tuple[CandidateFeatureRow, ...],
]:
    utility: list[ExactTailUtilityRow] = []
    inner_features: list[CandidateFeatureRow] = []
    target_features: list[CandidateFeatureRow] = []
    for query in candidate_sources(OUTER):
        support_hash = f"support-{OUTER}-{query}"
        for source in inner_candidate_sources(OUTER, query):
            source_index = CENTERS.index(source)
            query_index = CENTERS.index(query)
            for training_seed, generation_seed in SEED_PAIRS:
                seed_index = SEED_PAIRS.index((training_seed, generation_seed))
                base = 0.62 + seed_index * 1.0e-5
                delta = (
                    0.005
                    + 0.003 * source_index
                    - 0.001 * abs(source_index - query_index)
                    + seed_index * 1.0e-6
                )
                base_hash = (
                    f"base-{OUTER}-{query}-{training_seed}-{generation_seed}"
                )
                utility.append(
                    ExactTailUtilityRow(
                        outer_target_id=OUTER,
                        query_id=query,
                        candidate_source=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        candidate_source_count=INNER_CANDIDATE_COUNT,
                        support_partition_hash=support_hash,
                        evaluation_partition_hash=f"evaluation-{OUTER}-{query}",
                        prediction_seal_hash=f"seal-{OUTER}",
                        base_prediction_hash=base_hash,
                        tail_prediction_hash=(
                            f"tail-{OUTER}-{query}-{source}-{training_seed}-{generation_seed}"
                        ),
                        base_bacc=base,
                        tail_bacc=base + delta,
                        support_eval_disjoint=True,
                        predictions_sealed_before_labels=True,
                        source_expert_frozen=True,
                    )
                )
                inner_features.append(
                    _feature_row(
                        role=INNER_ROLE,
                        outer=OUTER,
                        query=query,
                        source=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        support_hash=support_hash,
                    )
                )
    for source in candidate_sources(OUTER):
        for training_seed, generation_seed in SEED_PAIRS:
            target_features.append(
                _feature_row(
                    role=TARGET_ROLE,
                    outer=OUTER,
                    query=OUTER,
                    source=source,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    support_hash=f"support-{OUTER}-target",
                )
            )
    return tuple(utility), tuple(inner_features), tuple(target_features)


@pytest.fixture(scope="module")
def one_target_bundle():
    utility_rows, inner_rows, target_rows = _one_target_rows()
    utility = validate_exact_tail_utility_rows(utility_rows)
    features = build_heldout_feature_surfaces(
        inner_rows,
        target_rows,
        outer_target_id=OUTER,
    )
    models = fit_stage90_heldout_models(features, utility, alphas=(1.0,))
    plan = build_r2_diagnostic_plan(models, features)
    return utility, features, models, plan


def test_frozen_consumed_stage90_geometry_and_input_order() -> None:
    assert CENTERS == ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    assert EXPECTED_INNER_UTILITY_ROW_COUNT == 4_536
    assert EXPECTED_TARGET_FEATURE_ROW_COUNT == 648
    assert INPUT_ARTIFACT_IDS == (
        "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
        "midogpp_output_uniform_b_v2_generation_lock_v1",
        "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1",
        "midogpp_stage90_utility_aligned_exact_tail_router_validation_cache_v1",
        "midogpp_stage90_utility_aligned_exact_tail_router_validation_manifest_v1",
        "midogpp_routing_metadata_profiles_v1",
    )
    assert candidate_sources("0") == ("1", "2", "3", "5", "6", "7", "8", "9")
    assert inner_candidate_sources("0", "1") == ("2", "3", "5", "6", "7", "8", "9")


def test_feature_api_is_label_free_and_full_set_rejects_one_H_only() -> None:
    prohibited = {"label", "labels", "utility", "bacc", "prediction", "oracle"}
    field_names = {item.name for item in fields(CandidateFeatureRow)}
    signature_names = set(inspect.signature(build_stage90_feature_surface_set).parameters)
    assert not field_names.intersection(prohibited)
    assert not signature_names.intersection(prohibited)
    _utility, inner, target = _one_target_rows()
    with pytest.raises(ProtocolError):
        build_stage90_feature_surface_set(inner, target)


def test_typed_rows_fail_closed_on_H_q_e_leakage() -> None:
    with pytest.raises(ProtocolError):
        _feature_row(
            role=INNER_ROLE,
            outer="0",
            query="1",
            source="0",
            training_seed=17,
            generation_seed=17,
            support_hash="support-0-1",
        )
    with pytest.raises(ProtocolError):
        ExactTailUtilityRow(
            outer_target_id="0",
            query_id="1",
            candidate_source="1",
            training_seed=17,
            generation_seed=17,
            candidate_source_count=7,
            support_partition_hash="support",
            evaluation_partition_hash="evaluation",
            prediction_seal_hash="seal",
            base_prediction_hash="base",
            tail_prediction_hash="tail",
            base_bacc=0.6,
            tail_bacc=0.61,
            support_eval_disjoint=True,
            predictions_sealed_before_labels=True,
            source_expert_frozen=True,
        )


def test_feature_and_model_cardinalities_preserve_strict_exclusion(one_target_bundle) -> None:
    utility, features, models, _plan = one_target_bundle
    assert len(utility.rows) == 8 * 7 * 9
    assert len(features.inner.rows) == 8 * 7 * 9
    assert len(features.target.rows) == 8 * 9
    assert set(models.training_query_ids) == set(candidate_sources(OUTER))
    assert set(models.training_source_ids) == set(candidate_sources(OUTER))
    for bundle in (models.global_and_interaction, models.permuted_interaction):
        assert bundle.outer_target_id == OUTER
        for crossfit in (bundle.global_crossfit, bundle.interaction_crossfit):
            assert len(crossfit.folds) == 8
            for fold in crossfit.folds:
                assert OUTER not in fold.training_query_ids
                assert OUTER not in fold.training_source_ids
                assert fold.heldout_query_id not in fold.training_query_ids
                assert fold.heldout_query_id not in fold.training_source_ids
                assert fold.training_candidate_count_per_query == 6


def test_r2_plan_averages_all_nine_seeds_and_never_authorizes_policy(one_target_bundle) -> None:
    _utility, features, models, first = one_target_bundle
    second = build_r2_diagnostic_plan(models, features)
    assert first.plan_hash == second.plan_hash
    assert first.routing_status == ROUTING_STATUS
    assert first.support_case_count == 2
    assert tuple(first.proposed_source_by_router) == ROUTER_DIAGNOSTIC_IDS
    for router in ROUTER_DIAGNOSTIC_IDS:
        for source in candidate_sources(OUTER):
            assert len(first.seed_predictions_by_router_source[router][source]) == 9
    assert first.policy_authorized is False
    assert first.fallback_authorized is False
    assert first.promotion_authorized is False
    assert first.deployment_authorized is False
    assert first.target_support_labels_used is False
    assert first.target_evaluation_embeddings_used is False
    assert first.development_crossfit_labels_previously_opened is True
    assert first.outer_H_development_rows_excluded_from_plan_H is True
    assert first.predictions_frozen_before_terminal_target_scoring is True
    assert first.outer_H_development_label_rows_used_for_plan_H is False
    assert first.terminal_target_labels_used_for_plan is False
    assert "predictions_frozen_before_target_evaluation_labels" not in first.to_payload()
    assert first.seed_selection_performed is False
    with pytest.raises(ProtocolError):
        replace(first, policy_authorized=True)


def test_permutation_is_fixed_and_hash_deterministic(one_target_bundle) -> None:
    _utility, features, models, plan = one_target_bundle
    first = permute_interaction_features(
        features.target,
        permutation_seed=PERMUTATION_SEED,
    )
    second = permute_interaction_features(
        features.target,
        permutation_seed=PERMUTATION_SEED,
    )
    assert first.surface_hash == second.surface_hash
    assert models.permuted_interaction.permutation_seed == PERMUTATION_SEED
    assert plan.plan_hash == build_r2_diagnostic_plan(models, features).plan_hash


def test_inner_actions_execute_exact_7x144_plus_126_geometry() -> None:
    first = build_inner_exact_tail_actions("0", "1")
    second = build_inner_exact_tail_actions("0", "1")
    assert tuple(action.action_hash for action in first) == tuple(
        action.action_hash for action in second
    )
    assert len(first) == 8
    base, tail = first[0], first[1]
    assert base.to_payload()["action_geometry_label_free"] is True
    assert base.to_payload()["crossfit_development_utility_labels_used_for_route"] is False
    assert "labels_used_to_build" not in base.to_payload()
    assert base.action_id == BASE_ACTION_ID
    assert base.base_per_source_per_class == 144
    assert base.topup_total_per_class == 0
    assert base.final_total_per_class == 1_008
    assert tail.topup_total_per_class == 126
    assert tail.final_total_per_class == 1_134
    assert tail.required_source_capacity_per_class == 270
    assert tail.topup_counts_by_source[tail.selected_source] == 126
    library_a = build_inner_exact_tail_action_library()
    library_b = build_inner_exact_tail_action_library()
    assert library_a.action_library_hash == library_b.action_library_hash


def test_target_menu_executes_B_U_and_single_source_128_tails(one_target_bundle) -> None:
    _utility, _features, _models, plan = one_target_bundle
    actions = build_target_actions(plan)
    by_id = {action.action_id: action for action in actions}
    assert len(actions) == 13
    assert by_id[BASE_ACTION_ID].final_total_per_class == 1_024
    uniform = by_id[UNIFORM_ACTION_ID]
    assert uniform.final_total_per_class == 1_152
    assert set(uniform.topup_counts_by_source.values()) == {16}
    for identifier in (
        GLOBAL_DELTA_ACTION_ID,
        R2_ACTION_ID,
        PERMUTATION_ACTION_ID,
    ):
        action = by_id[identifier]
        assert action.final_total_per_class == 1_152
        assert action.topup_counts_by_source[action.selected_source] == 128
        assert action.required_source_capacity_per_class == 256
        assert action.to_payload()["crossfit_development_utility_labels_used_for_route"] is True
        assert action.to_payload()["outer_H_development_rows_used_for_route"] is False
        assert action.to_payload()["target_support_labels_used_for_route"] is False
        assert action.to_payload()["terminal_target_labels_used_for_route"] is False
        assert action.policy_authorized is False
        assert action.promotion_authorized is False
    for source in candidate_sources(OUTER):
        action = by_id[h_x_e_action_id(source)]
        assert action.selected_source == source
        assert action.topup_counts_by_source[source] == 128
        assert action.to_payload()["crossfit_development_utility_labels_used_for_route"] is False
