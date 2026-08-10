"""Canonical scientific payloads for the terminal signed-error gate."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
)
from .constants import (
    FEATURE_NAMES,
    INTERCEPT_GRID,
    LAMBDA_GRID,
    MARGIN_BANDWIDTH_LOGIT,
    MAX_ABSOLUTE_CORRECTION_LOGIT,
    METHOD_IDS,
    PERMUTATION_NAMESPACE,
    PROBABILITY_EPSILON,
    RIDGE_ALPHA_GRID,
    STANDARDIZATION_SCALE_FLOOR,
    UNCERTAINTY_Z,
)
from .experiment_contracts import (
    CENTERS,
    CLAIM_ROLE,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
    EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PUBLICATION_STATUS,
    STAGE_ID,
    TRAINING_SEEDS,
)
from .protocol import canonical_consumed_test_protocol


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
    """Bind the package-level protocol plus the concrete workspace topology."""

    protocol = canonical_consumed_test_protocol().to_payload()
    return {
        **protocol,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "exact_seed_pair_count": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(EXPECTED_CASE_COUNTS_BY_CENTER),
        "mixed_class_case_count": EXPECTED_MIXED_CLASS_CASE_COUNT,
        "negative_only_case_count": EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
        "positive_only_case_count": EXPECTED_POSITIVE_ONLY_CASE_COUNT,
        "single_class_cases_retained": True,
        "per_case_bacc_stored_or_used": False,
        "target_geometry": "direct_H_with_B_and_eight_Hxe_actions",
        "baseline_action_id": "B",
        "candidate_action_family": "Hxe",
        "candidate_pool_excludes_target_H": True,
        "candidate_source_count_per_target": (
            EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
        ),
        "action_count_per_target": EXPECTED_ACTION_COUNT_PER_TARGET,
        "target_action_identity_count": EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "oof_fold_count": OOF_FOLD_COUNT,
        "partition_seed": OOF_FOLD_SEED,
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "partition_unit": "whole_case_within_target_center",
        "each_case_evaluated_exactly_once": True,
        "support_scope": "other_four_same_H_whole_case_folds",
        "evaluation_scope": "one_held_same_H_whole_case_fold",
        "heldout_fold_absent_from_its_support_and_decision_fit": True,
        "cross_role_case_reuse_only_in_other_folds": True,
        "center_fold_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "strict_outer_H_exclusion": True,
        "strict_nested_query_q_exclusion": True,
        "target_labels_used_for_shared_model": False,
        "same_H_support_parameter_scope": (
            "B_cal_intercept_and_common_residual_lambda_only"
        ),
        "terminal_endpoint": (
            "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        ),
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "original_six_inputs_only": True,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "label_free_cache_lineage": "stage70_derived_feature_cache_alias_only",
        "previous_stage90_outputs_used": False,
        "hierarchical_output_or_amendment_used": False,
    }


def canonical_probability_surface_payload() -> dict[str, object]:
    return {
        "family": "baseline_and_direct_Hxe_exact_nine_probability_surface_v1",
        "probability_source": "new_run_recomputation_from_fixed_bank_and_cache",
        "previous_stage90_probability_arrays_used": False,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "probabilities_averaged_before_feature_or_threshold_use": True,
        "probability_threshold": 0.5,
        "logit_clip_epsilon": PROBABILITY_EPSILON,
        "global_probability_surface_sealed_before_any_label_access": True,
        "target_expert_used": False,
        "source_expert_or_classifier_updated": False,
    }


def canonical_features_payload() -> dict[str, object]:
    return {
        "family": "signed_baseline_anchored_aggregate_candidate_residual_features_v1",
        "feature_unit": "sample_H_case_sample",
        "label_free": True,
        "feature_names": list(FEATURE_NAMES),
        "feature_count_including_intercept": len(FEATURE_NAMES),
        "candidate_residual_definition": (
            "logit_clip_p_Hxe_minus_logit_clip_p_B"
        ),
        "baseline_predicted_class_branch_used": False,
        "outer_feature_context_excludes": ["H"],
        "nested_feature_context_excludes": ["H", "q"],
        "candidate_identity_one_hot_or_learned_factor_used": False,
        "permutation_namespace": PERMUTATION_NAMESPACE,
        "permutation_unit": (
            "complete_sample_feature_block_within_target_center_context"
        ),
        "permutation_is_nonzero_cyclic_derangement": True,
        "permutation_applied_before_donor_fit": True,
        "permutation_applied_before_target_inference": True,
        "permutation_refits_same_capacity_model": True,
        "permutation_changes_labels_or_gradient_targets": False,
    }


def canonical_model_payload() -> dict[str, object]:
    return {
        "family": "strict_outer_H_nested_q_signed_gradient_ridge_v1",
        "response": (
            "strict_oof_class_balanced_negative_log_loss_logit_gradient"
        ),
        "fit_unit": "sample",
        "ridge_objective": "unweighted_mse_on_rescaled_gradient_target",
        "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
        "alpha_selection": "nested_query_LOCO",
        "alpha_selection_direction": "minimize_validation_mse",
        "alpha_tie_break": "larger_alpha",
        "outer_target_H_absent_from_fit_standardization_and_alpha_selection": True,
        "heldout_query_q_absent_from_nested_fit_and_standardization": True,
        "standardization_fit_scope": "legal_donor_rows_only",
        "standardization_scale_floor": STANDARDIZATION_SCALE_FLOOR,
        "intercept_standardized": False,
        "global_control_model_id": "G",
        "global_control_predictors": ["intercept"],
        "residual_model_id": "R",
        "residual_model_predictors": list(FEATURE_NAMES),
        "permuted_control_model_id": "P",
        "permuted_control_predictors": list(FEATURE_NAMES),
        "permuted_control_is_separately_fit": True,
        "maximum_absolute_correction_logit": MAX_ABSOLUTE_CORRECTION_LOGIT,
        "uncertainty_z": UNCERTAINTY_Z,
        "R_raw_and_R_safe_separately_sealed": True,
        "target_labels_update_shared_model": False,
    }


def canonical_target_support_payload() -> dict[str, object]:
    return {
        "family": "support_only_intercept_then_common_signed_residual_scale_v1",
        "fit_unit": "one_target_H_whole_case_OOF_fold",
        "support_scope": "same_H_nonheldout_whole_cases_only",
        "evaluation_fold_absent": True,
        "B_cal_intercept_grid": list(INTERCEPT_GRID),
        "B_cal_selection_objective": "fixed_class_balanced_log_loss",
        "B_cal_tie_break": "minimum_absolute_intercept_then_numeric",
        "residual_lambda_grid": list(LAMBDA_GRID),
        "residual_lambda_selection_model": "R_safe",
        "residual_lambda_selection_objective": "fixed_class_balanced_log_loss",
        "residual_lambda_tie_break": "smaller_lambda",
        "selected_intercept_shared_across_B_cal_G_R_raw_R_safe_P": True,
        "selected_lambda_shared_across_G_R_raw_R_safe_P": True,
        "full_lambda_path_threshold_crossings_and_fallback_persisted": True,
        "margin_gate": "exp_negative_squared_calibrated_baseline_logit_margin",
        "margin_bandwidth_logit": MARGIN_BANDWIDTH_LOGIT,
        "exact_pooled_bacc_used_for_grid_selection": False,
        "exact_pooled_bacc_used_for_safety_gate": True,
        "safety_gate_contrast": "selected_R_safe_minus_B_cal_on_support",
        "safety_gate_uncertainty_unit": "paired_whole_case_cluster",
        "positive_gate_rule": "LCB_strictly_greater_than_zero",
        "failed_gate_lambda": 0.0,
        "lambda_zero_is_exact_B_cal_fallback": True,
        "support_labels_update_shared_model_features_or_alpha": False,
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "diagnostic_method_ids": list(METHOD_IDS),
        "mandatory_control_ids": ["B", "B_cal", "G", "P"],
        "B_role": "uncalibrated_fixed_equal_union_baseline",
        "B_cal_role": "support_only_intercept_calibrated_baseline",
        "G_role": "strictly_case_independent_intercept_only_signed_control",
        "R_raw_role": "ungated_signed_sample_correction_diagnostic",
        "R_safe_role": "nested_model_uncertainty_gated_signed_sample_correction",
        "P_role": "same_capacity_permuted_feature_alignment_control",
        "B_cal_required_to_separate_threshold_calibration": True,
        "G_required_to_test_case_conditioning": True,
        "P_required_to_test_feature_alignment": True,
        "all_six_method_predictions_sealed_before_evaluation_labels": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": (
            "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        ),
        "primary_contrasts": ["R_safe-B_cal", "R_safe-G", "R_safe-P"],
        "secondary_contrasts": ["R_raw-R_safe", "B_cal-B"],
        "method_ids": list(METHOD_IDS),
        "center_utility": "pooled_exact_bacc_from_aggregated_confusion_sums",
        "case_sufficient_statistic_fields": [
            "n_positive",
            "true_positive",
            "n_negative",
            "true_negative",
        ],
        "per_case_bacc_stored_or_used": False,
        "single_class_cases_retained": True,
        "pooled_scope_requires_both_binary_classes": True,
        "primary_aggregation": "equal_weight_per_target_center",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_repeats_are_not_independent_units": True,
        "whole_case_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "whole_case_cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "whole_case_cluster_bootstrap_scope": (
            "resample_cases_within_each_center_then_equal_center_aggregate"
        ),
        "center_level_confidence_interval": "paired_t_interval_over_nine_centers",
        "confidence_level": 0.95,
        "screening_success_requires_positive_lower_bounds_for": [
            "R_safe-B_cal",
            "R_safe-G",
            "R_safe-P",
        ],
        "evaluation_labels_open_after_all_prediction_seals_only": True,
        "results_are_terminal_consumed_test_diagnostics": True,
        "result_may_authorize_policy_action_promotion_or_later_experiment": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "probability_materialization_device": "cpu",
        "probability_materialization_workers": 4,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "model_workers": 4,
        "model_threads_per_worker": 3,
        "bootstrap_workers": 4,
        "bootstrap_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz_shared_runtime",
        "context_feature_format": "bounded_process_local_float64_target_contexts",
        "scientific_reductions_dtype": "float64",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "target_task_count": EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
        "target_action_identity_count": EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "maximum_total_classifier_fit_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "scratch_preference": [
            "/data/local/fixed_bank_signed_error_gate_v1",
            "artifact_parent",
        ],
        "resume_policy": (
            "hash_validated_source_prediction_task_resume_plus_"
            "deterministic_phase_replay"
        ),
        "context_features_rebuilt_and_hash_revalidated_per_target": True,
        "maximum_concurrent_target_context_builds": 4,
        "cross_target_context_cache_forbidden": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": "DO_NOT_PROMOTE",
        "claim_role": CLAIM_ROLE,
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "terminal_stage90_diagnostic": True,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "support_labels_used": True,
        "support_labels_local_intercept_and_common_lambda_only": True,
        "other_center_labels_used_for_strict_outer_H_nested_q_model_fit": True,
        "evaluation_labels_opened_only_after_all_prediction_seals": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "previous_stage90_outputs_used": False,
        "previous_prediction_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
        "hierarchical_output_or_amendment_used": False,
    }


__all__ = (
    "CLASSIFIER",
    "canonical_claim_boundary_payload",
    "canonical_controls_payload",
    "canonical_evaluation_payload",
    "canonical_features_payload",
    "canonical_model_payload",
    "canonical_probability_surface_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "canonical_target_support_payload",
)
