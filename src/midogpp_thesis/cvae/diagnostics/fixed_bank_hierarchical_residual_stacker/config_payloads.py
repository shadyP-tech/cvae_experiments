"""Canonical scientific payloads for the hierarchical residual stacker."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    CENTERS,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_DONOR_CASE_ACTION_COUNT,
    EXPECTED_DONOR_CLASS_RESPONSE_COUNT,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_OUTER_CANDIDATE_MODEL_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
    EXPECTED_TARGET_CASE_ACTION_FEATURE_COUNT,
    EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PUBLICATION_STATUS,
    SEED_PAIR_COUNT,
    STAGE_ID,
    TRAINING_SEEDS,
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

LOGIT_CLIP_EPSILON = 1.0e-4
PROBABILITY_THRESHOLD = 0.5
SMOOTH_RESPONSE_TEMPERATURE = 0.05
PRIMARY_INTERACTION_RANK = 1
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0)
SUPPORT_INTERCEPT_GRID = (-0.1, -0.05, 0.0, 0.05, 0.1)
SUPPORT_LAMBDA_GRID = (0.0, 0.05, 0.1, 0.2, 0.25)
MAXIMUM_LAMBDA = 0.25
MIXTURE_TEMPERATURE = 0.01
MAX_SOURCES_PER_CLASS = 2
VARIANCE_FLOOR = 1.0e-6
CONFIDENCE_MULTIPLIER = 1.96
TIE_TOLERANCE = 1.0e-12
FEATURE_PERMUTATION_SEED = 90_912_027
CLUSTER_BOOTSTRAP_SEED = 90_912_028
CLUSTER_BOOTSTRAP_REPLICATES = 10_000

LOCAL_RESIDUAL_FEATURE_NAMES = (
    "residual_logit_mean",
    "residual_logit_abs_mean",
    "residual_logit_std",
    "hard_disagreement_rate",
)
MODEL_FEATURE_NAMES = (
    "intercept",
    *LOCAL_RESIDUAL_FEATURE_NAMES,
    "global_source_control",
    "global_source_control_x_residual_logit_mean",
    "global_source_control_x_residual_logit_abs_mean",
    "global_source_control_x_residual_logit_std",
    "global_source_control_x_hard_disagreement_rate",
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_fixed_bank_hierarchical_residual_stacker_v1",
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "consumed_test_data": True,
        "fresh_evidence": False,
        "centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "exact_seed_pair_count": SEED_PAIR_COUNT,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(EXPECTED_CASE_COUNTS_BY_CENTER),
        "mixed_class_case_count": EXPECTED_MIXED_CLASS_CASE_COUNT,
        "negative_only_case_count": EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
        "positive_only_case_count": EXPECTED_POSITIVE_ONLY_CASE_COUNT,
        "single_class_cases_retained": True,
        "per_case_bacc_stored_or_used": False,
        "primary_utility": "pooled_exact_bacc",
        "uncertainty_unit": "paired_whole_case_cluster",
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
        "target_case_action_feature_count": EXPECTED_TARGET_CASE_ACTION_FEATURE_COUNT,
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "oof_fold_count": OOF_FOLD_COUNT,
        "partition_seed": OOF_FOLD_SEED,
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "partition_unit": "whole_case_within_target_center",
        "each_case_evaluated_exactly_once": True,
        "support_scope": "other_four_same_H_whole_case_folds",
        "evaluation_scope": "one_held_same_H_whole_case_fold",
        "minimum_legal_support_case_count": 8,
        "heldout_fold_absent_from_its_support_and_local_calibration": True,
        "cross_role_case_reuse_only_in_other_folds": True,
        "center_fold_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "global_target_probability_seal_before_any_label_access": True,
        "outer_H_absent_from_shared_effect_fit_and_alpha_selection": True,
        "strict_inner_H_q_e_exclusion": True,
        "support_labels_used": True,
        "support_labels_scope": "same_H_nonheldout_whole_cases_only",
        "support_labels_may_update_shared_effect_model": False,
        "support_labels_may_select_rank_features_or_ridge_alpha": False,
        "support_labels_may_select_B_cal_intercept_and_common_lambda_only": True,
        "all_models_sealed_before_same_H_support_label_access": True,
        "all_observed_and_control_actions_sealed_before_evaluation_labels": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "label_free_cache_lineage": "stage70_derived_feature_cache_alias_only",
        "previous_stage90_outputs_used": False,
        "previous_prediction_surface_used": False,
        "previous_scratch_or_checkpoint_used": False,
        "recomputed_from_original_six_inputs": True,
        "metadata_artifact_used": False,
    }


def canonical_probability_surface_payload() -> dict[str, object]:
    return {
        "family": "baseline_and_direct_Hxe_exact_nine_probability_surface_v1",
        "probability_source": "new_run_recomputation_from_fixed_bank_and_cache",
        "previous_stage90_probability_arrays_used": False,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "probabilities_averaged_before_feature_or_threshold_use": True,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "logit_clip_epsilon": LOGIT_CLIP_EPSILON,
        "residual_logit_definition": (
            "logit_clip_p_Hxe_minus_logit_clip_p_B"
        ),
        "residual_logit_clip_bounds": [
            LOGIT_CLIP_EPSILON,
            1.0 - LOGIT_CLIP_EPSILON,
        ],
        "global_source_control_name": "global_source_control",
        "global_source_control_input": "sealed_probabilities_only",
        "global_source_control_metadata_or_label_input": False,
        "global_source_control_definition": (
            "equal_legal_query_mean_of_equal_case_mean_of_row_mean_absolute_"
            "residual_logit"
        ),
        "global_source_control_outer_mask": "query_q_not_in_H_or_e",
        "global_source_control_nested_mask": "query_t_not_in_H_or_e_or_q",
        "training_row_source_s_control_mask": "query_u_not_in_H_or_e_or_s",
        "nested_training_row_source_s_control_mask": (
            "query_u_not_in_H_or_e_or_q_or_s"
        ),
        "global_source_control_equal_query_weight": True,
        "global_source_control_equal_case_weight_within_query": True,
        "candidate_source_identity_one_hot_or_learned_factor_used": False,
        "global_probability_surface_sealed_before_donor_labels": True,
        "current_run_surface_reused_across_all_model_and_control_fits": True,
    }


def canonical_features_payload() -> dict[str, object]:
    return {
        "family": "whole_case_baseline_anchored_residual_logit_features_v1",
        "feature_unit": "whole_case_H_c_e",
        "label_free": True,
        "local_residual_feature_names": list(LOCAL_RESIDUAL_FEATURE_NAMES),
        "local_residual_feature_count": len(LOCAL_RESIDUAL_FEATURE_NAMES),
        "residual_logit_std_ddof": 0,
        "hard_disagreement_definition": (
            "mean_indicator_Hxe_ge_0p5_differs_from_B_ge_0p5"
        ),
        "model_feature_names": list(MODEL_FEATURE_NAMES),
        "model_feature_count_including_intercept": len(MODEL_FEATURE_NAMES),
        "interaction_features": (
            "global_source_control_times_each_four_local_residual_features"
        ),
        "standardization_fit_scope": "legal_donor_training_case_rows_only",
        "standardization_weighting": "equal_case",
        "intercept_standardized": False,
        "zero_variance_feature_policy": "standardized_value_zero",
        "target_or_held_query_rows_used_for_standardization": False,
        "target_case_action_feature_count": EXPECTED_TARGET_CASE_ACTION_FEATURE_COUNT,
        "permutation_control_seed": FEATURE_PERMUTATION_SEED,
        "permutation_unit": (
            "complete_four_local_feature_block_within_query_candidate_case_scope"
        ),
        "permutation_family": (
            "sha256_ordered_nonzero_cyclic_whole_case_shift_v1"
        ),
        "permutation_applied_before_donor_fit": True,
        "permutation_applied_before_target_inference": True,
        "permutation_refits_same_capacity_model": True,
        "permutation_preserves_global_source_control": True,
        "permutation_preserves_residual_vectors": True,
        "permutation_preserves_responses_and_labels": True,
        "permutation_changes_only_case_to_local_feature_assignment": True,
        "permutation_phi_donor_source_must_be_legal_under_same_H_e_q_mask": True,
        "permutation_may_reassign_phi_from_forbidden_original_source": False,
    }


def canonical_hierarchical_model_payload() -> dict[str, object]:
    return {
        "family": "strict_H_q_e_crossfit_class_conditional_ridge_stacker_v1",
        "primary_interaction_rank": PRIMARY_INTERACTION_RANK,
        "rank_semantics": "one_probability_derived_global_source_descriptor",
        "rank_two_challenger_included": False,
        "learned_candidate_identity_factor_used": False,
        "response_unit": "whole_case_query_q_candidate_e_class_y",
        "positive_class_response": (
            "case_mean_sigmoid_(p_Hxe_minus_0p5)_over_T_minus_same_for_B"
        ),
        "negative_class_response": (
            "case_mean_sigmoid_(0p5_minus_p_Hxe)_over_T_minus_same_for_B"
        ),
        "smooth_response_temperature": SMOOTH_RESPONSE_TEMPERATURE,
        "missing_case_class_response_policy": "omit_that_class_response_only",
        "separate_positive_and_negative_models": True,
        "donor_case_action_count": EXPECTED_DONOR_CASE_ACTION_COUNT,
        "donor_class_response_count": EXPECTED_DONOR_CLASS_RESPONSE_COUNT,
        "outer_candidate_model_count": EXPECTED_OUTER_CANDIDATE_MODEL_COUNT,
        "outer_deployed_pair": "target_H_candidate_e",
        "final_training_row_mask": (
            "query_t_and_candidate_s_both_disjoint_from_H_and_e_and_t_not_equal_s"
        ),
        "nested_validation_row": "held_query_q_candidate_e_case_response",
        "nested_training_row_mask": (
            "query_t_and_candidate_s_both_disjoint_from_H_e_q_and_t_not_equal_s"
        ),
        "held_query_q_absent_from_fit_and_standardization": True,
        "deployed_candidate_e_absent_from_fit_and_standardization": True,
        "outer_target_H_absent_from_fit_standardization_and_alpha_selection": True,
        "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
        "alpha_selection": "nested_query_LOCO",
        "alpha_selection_objective": (
            "nested_legal_query_class_count_weighted_squared_error_on_smooth_"
            "class_effect_responses"
        ),
        "alpha_selection_direction": "minimize",
        "alpha_tie_break": "larger_alpha_then_lexicographic",
        "one_alpha_per_outer_H_candidate_e_and_model_family_shared_across_classes": True,
        "target_support_alpha_or_rank_tuning": False,
        "global_control_model_id": "G",
        "global_control_predictors": ["intercept", "global_source_control"],
        "case_conditional_model_id": "R",
        "case_conditional_predictors": list(MODEL_FEATURE_NAMES),
        "permuted_control_model_id": "P",
        "permuted_control_predictors": list(MODEL_FEATURE_NAMES),
        "permuted_control_is_separately_fit": True,
        "permuted_control_reuses_R_coefficients": False,
        "hyperparameters_fixed_before_any_label_access": True,
    }


def canonical_target_support_payload() -> dict[str, object]:
    return {
        "family": "support_only_B_cal_then_common_residual_shrinkage_v1",
        "fit_unit": "one_target_H_whole_case_OOF_fold",
        "support_scope": "same_H_nonheldout_whole_cases_only",
        "evaluation_fold_absent": True,
        "minimum_support_case_count": 8,
        "B_cal_intercept_grid": list(SUPPORT_INTERCEPT_GRID),
        "B_cal_selection_objective": "fixed_class_balanced_log_loss",
        "B_cal_tie_break": "minimum_absolute_intercept_then_numeric",
        "B_cal_probability": "expit_logit_clip_p_B_plus_b_H_fold",
        "residual_lambda_grid": list(SUPPORT_LAMBDA_GRID),
        "residual_lambda_maximum": MAXIMUM_LAMBDA,
        "residual_lambda_selection_model": "R",
        "residual_lambda_selection_objective": "fixed_class_balanced_log_loss",
        "residual_lambda_tie_break": "smaller_lambda",
        "selected_B_cal_intercept_shared_across_B_cal_G_R_P": True,
        "selected_lambda_shared_across_G_R_P": True,
        "exact_pooled_bacc_used_for_grid_selection": False,
        "exact_pooled_bacc_used_for_safety_gate": True,
        "safety_gate_contrast": "selected_R_minus_B_cal_on_support",
        "safety_gate_uncertainty_unit": "paired_whole_case_cluster",
        "case_influence_definition": (
            "0p5_times_[n_c_pos_over_N_pos_times_(case_pos_accuracy_diff_minus_"
            "pooled_pos_accuracy_diff)_plus_n_c_neg_over_N_neg_times_(case_neg_"
            "accuracy_diff_minus_pooled_neg_accuracy_diff)]"
        ),
        "absent_case_class_term_policy": "omit_only_the_absent_class_term",
        "cluster_variance_estimator": (
            "max_m_over_m_minus_1_sum_psi_squared_and_variance_floor"
        ),
        "variance_floor": VARIANCE_FLOOR,
        "confidence_multiplier": CONFIDENCE_MULTIPLIER,
        "lower_confidence_bound": "D_minus_1p96_times_sqrt_V",
        "positive_gate_rule": "LCB_strictly_greater_than_zero",
        "failed_gate_lambda": 0.0,
        "lambda_zero_is_exact_B_cal_fallback": True,
        "shared_effect_models_frozen_before_support_labels": True,
        "support_labels_update_shared_model_features_rank_or_alpha": False,
        "all_support_products_and_actions_sealed_before_evaluation_labels": True,
    }


def canonical_stacker_payload() -> dict[str, object]:
    return {
        "family": "baseline_anchored_sparse_class_conditional_residual_stacker_v1",
        "diagnostic_method_ids": ["B", "B_cal", "G", "R", "P"],
        "baseline_method_id": "B",
        "calibrated_baseline_method_id": "B_cal",
        "global_stack_method_id": "G",
        "case_conditional_stack_method_id": "R",
        "permuted_stack_method_id": "P",
        "alpha_scope": "whole_case_and_predicted_class_direction",
        "score_definition": "class_specific_ridge_predicted_smooth_effect",
        "source_admission_rule": "strictly_positive_predicted_effect",
        "maximum_sources_per_class": MAX_SOURCES_PER_CLASS,
        "top_source_tie_break": "larger_score_then_lexicographic_source",
        "source_weighting": "softmax_over_admitted_top_sources",
        "mixture_temperature": MIXTURE_TEMPERATURE,
        "no_positive_source_effect": "zero_residual_exact_B_cal_fallback",
        "class_gate": "soft_B_cal_probability",
        "soft_class_gate_rationale": (
            "avoid_hard_pseudo_class_sign_reversal_near_the_decision_threshold"
        ),
        "negative_branch_weight": "one_minus_p_B_cal",
        "positive_branch_weight": "p_B_cal",
        "composed_residual": (
            "(1-p_B_cal)*sum_e_alpha0_r_e_plus_p_B_cal*sum_e_alpha1_r_e"
        ),
        "final_probability": (
            "expit_logit_clip_p_B_plus_b_H_fold_plus_lambda_H_fold_times_"
            "composed_residual"
        ),
        "lambda_bounds": [0.0, MAXIMUM_LAMBDA],
        "baseline_anchor_preserved": True,
        "lambda_zero_returns_exact_B_cal": True,
        "global_stack_case_independent": True,
        "case_conditional_stack_uses_local_features": True,
        "permuted_stack_uses_separately_fit_permuted_local_features": True,
        "candidate_pool_excludes_target_H": True,
        "source_expert_or_classifier_updated": False,
        "evaluation_probabilities_or_labels_may_affect_action": False,
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "mandatory_control_ids": ["B", "B_cal", "G", "P"],
        "B_role": "uncalibrated_fixed_equal_union_baseline",
        "B_cal_role": "support_only_intercept_calibrated_baseline",
        "G_role": "case_independent_probability_descriptor_global_stack",
        "P_role": "same_capacity_case_feature_permutation_stack",
        "P_permutation_seed": FEATURE_PERMUTATION_SEED,
        "P_fit_separately_after_permutation": True,
        "P_permutation_applied_to_donor_and_target_local_features": True,
        "P_preserves_global_source_control": True,
        "P_preserves_labels_responses_and_residual_vectors": True,
        "B_cal_required_to_separate_threshold_calibration_from_routing": True,
        "G_required_to_test_case_conditioning": True,
        "P_required_to_test_case_feature_alignment": True,
        "matched_B_cal_intercept_and_lambda_across_G_R_P": True,
        "all_control_actions_sealed_before_evaluation_labels": True,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "center_pooled_exact_bacc_over_whole_case_oof_predictions",
        "primary_contrasts": ["R-B_cal", "R-G", "R-P"],
        "secondary_contrasts": ["R-B", "B_cal-B", "G-B_cal", "P-B_cal"],
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
        "whole_case_cluster_bootstrap_replicates": CLUSTER_BOOTSTRAP_REPLICATES,
        "whole_case_cluster_bootstrap_seed": CLUSTER_BOOTSTRAP_SEED,
        "whole_case_cluster_bootstrap_scope": (
            "resample_cases_within_each_center_then_equal_center_aggregate"
        ),
        "whole_case_cluster_bootstrap_is_conditional_on_observed_centers": True,
        "center_level_confidence_interval": "paired_t_interval_over_nine_centers",
        "confidence_level": 0.95,
        "screening_success_requires_positive_lower_bounds_for": [
            "R-B_cal",
            "R-G",
            "R-P",
        ],
        "metrics": [
            "pooled_exact_bacc",
            "primary_contrasts",
            "secondary_contrasts",
            "nonzero_lambda_coverage",
            "calibration_only_gain",
        ],
        "smooth_metrics_role": "fit_selection_and_descriptive_only",
        "smooth_metrics_may_replace_exact_terminal_endpoint": False,
        "evaluation_labels_open_after_all_action_seals_only": True,
        "results_are_terminal_consumed_test_diagnostics": True,
        "result_may_authorize_policy_action_or_later_experiment": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "model_workers": 4,
        "model_threads_per_worker": 3,
        "bootstrap_workers": 4,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "float32_memmap_plus_hash_index",
        "residual_surface_format": "streamed_float32_memmap",
        "scientific_reductions_dtype": "float64",
        "model_backend": "vectorized_numpy_float64",
        "case_feature_chunk_rows": 2048,
        "bootstrap_chunk_replicates": 256,
        "maximum_parent_resident_memory_bytes": 51_539_607_552,
        "duplicate_full_probability_or_residual_tensor_forbidden": True,
        "current_run_sealed_arrays_reused_across_model_fits": True,
        "previous_stage90_sealed_arrays_used": False,
        "phase_order": (
            "two_gpu_probability_materialization_then_four_by_three_CPU_"
            "feature_model_support_and_bootstrap_phases"
        ),
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
        "outer_candidate_model_count": EXPECTED_OUTER_CANDIDATE_MODEL_COUNT,
        "final_ridge_fit_count": EXPECTED_OUTER_CANDIDATE_MODEL_COUNT * 3 * 2,
        "maximum_nested_ridge_fit_count": (
            EXPECTED_OUTER_CANDIDATE_MODEL_COUNT
            * (len(CENTERS) - 2)
            * len(RIDGE_ALPHA_GRID)
            * 3
            * 2
        ),
        "scratch_preference": [
            "/data/local/fixed_bank_hierarchical_residual_stacker_v1",
            "artifact_parent",
        ],
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
        "prior_experiment_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": "DO_NOT_PROMOTE",
        "consumed_test_data": True,
        "ledger_amendment_required_and_hash_chained": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "claim_role": (
            "known_fixed_bank_label_aware_case_oof_stacking_mechanism_diagnostic"
        ),
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "support_labels_used": True,
        "support_labels_local_calibration_only": True,
        "other_center_labels_used_for_strict_loco_shared_effect_fit": True,
        "evaluation_labels_opened_only_after_all_action_seals": True,
        "diagnostic_candidate_action_probabilities_built": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "screening_gate_may_authorize_policy": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "metadata_artifact_used": False,
        "previous_stage90_outputs_used": False,
        "previous_prediction_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
    }


__all__ = (
    "CLASSIFIER",
    "CLUSTER_BOOTSTRAP_REPLICATES",
    "CLUSTER_BOOTSTRAP_SEED",
    "CONFIDENCE_MULTIPLIER",
    "FEATURE_PERMUTATION_SEED",
    "LOCAL_RESIDUAL_FEATURE_NAMES",
    "LOGIT_CLIP_EPSILON",
    "MAXIMUM_LAMBDA",
    "MAX_SOURCES_PER_CLASS",
    "MIXTURE_TEMPERATURE",
    "MODEL_FEATURE_NAMES",
    "PRIMARY_INTERACTION_RANK",
    "PROBABILITY_THRESHOLD",
    "RIDGE_ALPHA_GRID",
    "SMOOTH_RESPONSE_TEMPERATURE",
    "SUPPORT_INTERCEPT_GRID",
    "SUPPORT_LAMBDA_GRID",
    "TIE_TOLERANCE",
    "VARIANCE_FLOOR",
    "canonical_claim_boundary_payload",
    "canonical_controls_payload",
    "canonical_evaluation_payload",
    "canonical_features_payload",
    "canonical_hierarchical_model_payload",
    "canonical_probability_surface_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
    "canonical_stacker_payload",
    "canonical_target_support_payload",
)
