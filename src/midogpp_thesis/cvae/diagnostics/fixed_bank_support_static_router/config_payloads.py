"""Canonical scientific config sections for the support-static S4 diagnostic."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    A1_EFFECTIVE_ROWS_PER_CLASS,
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    ACTION_COUNT_PER_TARGET,
    BASE_ROWS_PER_SOURCE_CLASS,
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    HARD_THRESHOLD,
    INPUT_ARTIFACT_IDS,
    METHOD_IDS,
    NULL_DERANGEMENT_ALGORITHM,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    PRE_EVALUATION_METHOD_IDS,
    PUBLICATION_STATUS,
    ROUTING_STATUS,
    SCRATCH_ROOT,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    STAGE_ID,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_DECISION,
    TERMINAL_ORACLE_IDS,
    T_INTERVAL_CONFIDENCE_LEVEL,
    T_INTERVAL_DEGREES_OF_FREEDOM,
    TIE_TOLERANCE,
    TRAINING_SEEDS,
    UNIFORM_ROWS_PER_SOURCE_CLASS,
    WORKSTATION_PROFILE,
)


CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


def canonical_protocol_payload() -> dict[str, object]:
    """Return the exact consumed-test and fold-role protocol."""

    return {
        "schema_version": "midogpp_fixed_bank_support_static_router_s4_protocol_v1",
        "stage": STAGE_ID,
        "dataset_family": DATASET_FAMILY,
        "evaluation_split": EVALUATION_SPLIT,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "consumed_test_data": True,
        "fresh_evidence": False,
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "exact_seed_pair_count": 9,
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(
            EXPECTED_CASE_COUNTS_BY_CENTER
        ),
        "mixed_class_case_count": EXPECTED_MIXED_CLASS_CASE_COUNT,
        "negative_only_case_count": EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
        "positive_only_case_count": EXPECTED_POSITIVE_ONLY_CASE_COUNT,
        "single_class_cases_retained": True,
        "per_case_bacc_stored_or_used": False,
        "partition_unit": "whole_case_within_target_center",
        "partition_seed": OOF_FOLD_SEED,
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "fold_count": OOF_FOLD_COUNT,
        "support_scope": "other_four_same_H_whole_case_folds",
        "evaluation_scope": "one_held_same_H_whole_case_fold",
        "support_evaluation_whole_case_disjoint": True,
        "each_case_evaluated_exactly_once": True,
        "cross_role_case_reuse_only_in_other_folds": True,
        "center_fold_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "candidate_pool_excludes_target_H": True,
        "strict_outer_H_exclusion": True,
        "all_action_probabilities_globally_sealed_before_any_label_access": True,
        "each_fold_decision_invariant_to_held_evaluation_labels": True,
        "role_scoped_label_capabilities_enforced": True,
        "each_H_f_decision_and_seal_precedes_opening_same_H_f_evaluation_role_labels": True,
        "original_six_inputs_only": True,
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "label_free_cache_lineage": "stage70_derived_feature_cache_alias_only",
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_prediction_surfaces_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    """Return the immutable B/U/eight-A1 action library."""

    return {
        "schema_version": "fixed_bank_support_static_router_s4_action_library_v1",
        "action_ids": "B_U_and_eight_A1_source_actions",
        "baseline_action_id": "B",
        "uniform_control_action_id": "U",
        "source_action_id_format": "A1::source={source_center}",
        "decision_candidate_set": "B_plus_eight_frozen_A1_source_actions",
        "U_is_internal_control_not_selection_candidate": True,
        "candidate_source_count_per_target": 8,
        "physical_action_count_per_target": ACTION_COUNT_PER_TARGET,
        "target_task_count": TARGET_TASK_COUNT,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "baseline_rows_per_source_class": BASE_ROWS_PER_SOURCE_CLASS,
        "uniform_rows_per_source_class": UNIFORM_ROWS_PER_SOURCE_CLASS,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "A1_selected_rows_per_class": A1_SELECTED_ROWS_PER_CLASS,
        "A1_other_rows_per_class": A1_OTHER_ROWS_PER_CLASS,
        "A1_selected_row_weight": A1_SELECTED_ROW_WEIGHT,
        "A1_other_row_weight": A1_OTHER_ROW_WEIGHT,
        "A1_selected_row_weight_fraction": "23/16",
        "A1_other_row_weight_fraction": "7/8",
        "A1_effective_rows_per_class": A1_EFFECTIVE_ROWS_PER_CLASS,
        "action_strength_sweep_used": False,
        "class_conditional_action_variant_used": False,
        "source_pair_action_used": False,
        "geometry_selection_used": False,
        "target_expert_used": False,
        "probabilities_averaged_exact_nine_before_routing_or_scoring": True,
        "hard_probability_threshold": HARD_THRESHOLD,
        "previous_probability_surfaces_used": False,
    }


def canonical_support_router_payload() -> dict[str, object]:
    """Return the frozen support-only static selection rule and controls."""

    return {
        "schema_version": "fixed_bank_support_static_router_s4_v1",
        "family": "support_only_static_A1_source_router_v1",
        "method_id": "S4",
        "decision_unit": "one_static_action_per_target_center_and_fold",
        "support_case_fold_count": 4,
        "evaluation_case_fold_count": 1,
        "support_candidate_action_ids": "eight_A1_source_actions_vs_B",
        "support_selection_objective": "pooled_exact_bacc_gain_vs_B",
        "support_sufficient_statistics": [
            "n_positive",
            "true_positive",
            "n_negative",
            "true_negative",
        ],
        "all_eight_A1_candidates_scored": True,
        "selection_rule": (
            "highest_strictly_positive_A1_gain_then_numeric_source_tie_else_B"
        ),
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": "B_then_numeric_source_center",
        "nonpositive_gain_fallback_action": "B",
        "single_class_support_fallback_action": "B",
        "single_class_support_falls_back_to_B": True,
        "support_labels_update_shared_model": False,
        "support_labels_select_features": False,
        "support_labels_select_hyperparameters": False,
        "support_labels_select_thresholds": False,
        "support_labels_select_action_geometry_or_strength": False,
        "case_features_used": False,
        "donor_model_used": False,
        "target_local_calibration_used": False,
        "shared_model_fit_used": False,
        "hyperparameter_search_used": False,
        "G_static_definition": (
            "equal_center_mean_exact_gain_over_q_not_in_H_or_e"
        ),
        "G_static_donor_query_scope": "q_not_in_H_or_e",
        "G_static_candidate_gain_aggregation": "equal_center_mean",
        "G_static_selection_rule": (
            "highest_strictly_positive_gain_then_numeric_source_tie_else_B"
        ),
        "G_static_uses_same_H_support_labels": False,
        "G_static_uses_held_evaluation_labels": False,
        "G_static_candidate_set": "B_plus_eight_frozen_A1_source_actions",
        "G_static_tie_break": "B_then_numeric_source_center",
        "each_H_f_decision_sealed_before_same_H_f_evaluation_capability": True,
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "method_ids": list(METHOD_IDS),
        "pre_evaluation_method_ids": list(PRE_EVALUATION_METHOD_IDS),
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "B_role": "fixed_equal_union_baseline_and_exact_fallback",
        "U_role": "internal_uniform_action_control_not_selection_candidate",
        "G_static_role": (
            "equal_center_q_not_in_H_or_e_static_source_control_or_B"
        ),
        "S4_role": "other_four_same_H_fold_support_static_source_or_B",
        "O_static_role": "terminal_best_static_A1_or_B",
        "O_case_role": "terminal_best_case_action_over_B_and_all_A1",
        "terminal_oracles_are_pre_evaluation_methods": False,
        "terminal_oracles_can_train_select_calibrate_or_seal_a_method": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    """Return descriptive-only evaluation and null summaries."""

    return {
        "primary_endpoint": (
            "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        ),
        "center_utility": "pooled_exact_bacc_from_aggregated_confusion_sums",
        "descriptive_contrasts": [
            "S4-B",
            "S4-U",
            "S4-G_static",
            "O_static-S4",
            "O_case-O_static",
        ],
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_cells_are_not_independent_units": True,
        "descriptive_interval_family": "two_sided_t8_over_nine_centers",
        "descriptive_interval_confidence_level": T_INTERVAL_CONFIDENCE_LEVEL,
        "descriptive_interval_degrees_of_freedom": (
            T_INTERVAL_DEGREES_OF_FREEDOM
        ),
        "permutation_null_count": PERMUTATION_COUNT,
        "permutation_seed": PERMUTATION_SEED,
        "permutation_algorithm_id": NULL_DERANGEMENT_ALGORITHM,
        "permutation_unit": (
            "complete_candidate_A1_sufficient_statistic_contribution_"
            "block_within_support_case"
        ),
        "permutation_case_order_key": (
            "sha256_seed_fold_id_case_id_action_id"
        ),
        "permutation_shift_family": (
            "counter_splitmix64_nonzero_cyclic_shift_1_to_7"
        ),
        "permutation_preserves_class_denominators_tp_tn_and_candidate_multiset": True,
        "permutation_keeps_B_fixed": True,
        "permutation_changes_labels": False,
        "null_selection_plan_row_count": (
            PERMUTATION_COUNT * EXPECTED_CENTER_FOLD_COUNT
        ),
        "null_selection_plan_sealed_before_corresponding_evaluation_capability": True,
        "permutation_output": (
            "descriptive_exceedance_count_and_fraction_only"
        ),
        "confirmatory_p_value_computed": False,
        "confirmatory_gate_defined": False,
        "pass_fail_status_emitted": False,
        "routing_identification_claimed": False,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted_or_used": False,
        "results_are_terminal_consumed_test_diagnostics": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_s4_runtime_v1",
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "source_generation_worker_count": 2,
        "persistent_source_workers": True,
        "gpu_source_phase_precedes_cpu_phase": True,
        "cuda_visible_devices_cleared_before_cpu_phase": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "maximum_total_cpu_threads": 12,
        "multiprocessing_start_method": "spawn",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "target_task_count": TARGET_TASK_COUNT,
        "physical_actions_per_target_task": ACTION_COUNT_PER_TARGET,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "maximum_total_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": (
            "deterministic_restart_from_admission_with_nonrepairing_hash_validation"
        ),
        "clean_scratch_only_after_closed_world_validation_pass": True,
        "two_fresh_process_validation_required": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "routing_status": ROUTING_STATUS,
        "claim_role": CLAIM_ROLE,
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "terminal_stage90_diagnostic": True,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "support_labels_used": True,
        "support_labels_used_for_frozen_static_action_only": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "action_geometry_update_authorized": False,
        "policy_update_authorized": False,
        "model_update_authorized": False,
        "expert_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_prediction_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
        "confirmatory_p_value_or_gate_used": False,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + (
    "CLASSIFIER",
)
