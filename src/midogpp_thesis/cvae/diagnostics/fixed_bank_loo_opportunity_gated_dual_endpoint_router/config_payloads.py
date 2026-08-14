"""Canonical executable payloads for the dual-endpoint router diagnostic."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    A1_EFFECTIVE_ROWS_PER_CLASS,
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    ACTION_COUNT_PER_TARGET,
    ATTRIBUTION_CONTROL_IDS,
    BASE_ROWS_PER_SOURCE_CLASS,
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    DIRECTION_IDS,
    EVALUATION_SPLIT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    FEATURE_IDS,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    METHOD_IDS,
    PRE_TERMINAL_METHOD_IDS,
    PRIMARY_METHOD_IDS,
    PUBLICATION_STATUS,
    ROUTING_STATUS,
    SCRATCH_ROOT,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    STAGE_ID,
    TARGET_ACTION_IDENTITY_COUNT,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_DECISION,
    TERMINAL_ORACLE_IDS,
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
    """Return the consumed-test label firewall and exact LOO split contract."""

    return {
        "schema_version": (
            "midogpp_fixed_bank_loo_opportunity_gated_"
            "dual_endpoint_router_protocol_v1"
        ),
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
        "held_unit": "one_whole_case_or_group_c_within_target_center_H",
        "held_unit_count": EXPECTED_TOTAL_CASE_COUNT,
        "arbitrary_folds_used": False,
        "support_scope": "same_H_all_whole_cases_except_held_c",
        "terminal_evaluation_scope": "held_whole_case_c_only",
        "support_evaluation_whole_case_disjoint": True,
        "each_case_evaluated_exactly_once": True,
        "candidate_pool_excludes_target_H": True,
        "strict_outer_H_exclusion": True,
        "donor_query_scope": "q_not_in_H_or_e",
        "donor_prior_excludes_H_and_e": True,
        "support_labels_update_source_experts": False,
        "support_labels_update_shared_models": False,
        "all_72_donor_grants_complete_before_route_support": True,
        "all_physical_probabilities_globally_sealed_before_any_label_access": (
            True
        ),
        "label_free_held_case_features_sealed_before_support_labels": True,
        "role_scoped_label_capabilities_enforced": True,
        "route_scoped_support_grants_are_H_minus_c_only": True,
        "route_labels_never_enter_own_fit_scaler_state_or_decision": True,
        "route_local_model_state_never_shared": True,
        "all_218_predictions_and_decisions_sealed_before_terminal_labels": True,
        "all_218_identification_robust_and_portfolio_decisions_sealed_before_"
        "terminal_labels": True,
        "all_aggregate_method_seals_complete_before_terminal_labels": True,
        "terminal_labels_never_train_tune_calibrate_rank_or_select": True,
        "weights_selected_on_same_evaluation_surface": True,
        "weights_are_fixed_before_runtime": True,
        "original_six_inputs_only": True,
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_prediction_surfaces_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    """Return the immutable common B/U/eight-A1 probability contract."""

    return {
        "schema_version": (
            "fixed_bank_loo_opportunity_gated_dual_endpoint_"
            "action_library_v1"
        ),
        "action_ids": "B_U_and_eight_A1_source_actions",
        "baseline_action_id": "B",
        "uniform_control_action_id": "U",
        "source_action_id_format": "A1::source={source_center}",
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
        "probabilities_averaged_exact_nine_before_routing": True,
        "common_row_order_for_both_endpoints": True,
        "common_probability_surface_for_both_endpoints": True,
        "hard_probability_threshold": 0.5,
        "hard_threshold_equal_maps_to_positive": True,
        "previous_probability_surfaces_used": False,
    }


def canonical_identification_endpoint_payload() -> dict[str, object]:
    """Return the frozen opportunity gate and normalized ranking endpoint."""

    return {
        "schema_version": "fixed_bank_ogde_identification_endpoint_v1",
        "method_id": "I_OPPORTUNITY_GATED",
        "direction_ids": list(DIRECTION_IDS),
        "branch_definition": "baseline_B_hard_prediction_before_candidate_flip",
        "held_case_feature_ids": list(FEATURE_IDS),
        "held_case_features_are_label_free": True,
        "support_response": (
            "candidate_directional_flip_correctness_successes_over_trials"
        ),
        "fit_unit": "one_ephemeral_model_per_H_c_e_direction",
        "fit_scope": "same_H_whole_cases_except_c_only",
        "feature_scaler_scope": "same_H_whole_cases_except_c_only",
        "support_denominator_source": "H_minus_c_labels_only",
        "model_family": "ridge_binomial_logistic_newton_irls_v1",
        "feature_standardization": "H_minus_c_unweighted_mean_population_sd",
        "ridge_alpha": 1.0,
        "intercept_penalized": False,
        "max_iterations": 50,
        "convergence_tolerance": 1.0e-12,
        "eta_clip": [-30.0, 30.0],
        "probability_clip": [1.0e-12, 0.999999999999],
        "initialization": "all_zero_coefficients",
        "imputation_used": False,
        "feature_selection_used": False,
        "threshold_tuning_used": False,
        "hyperparameter_search_used": False,
        "nonconvergence_policy": "candidate_invalid_and_OFF_if_none_eligible",
        "zero_trial_policy": "candidate_invalid_and_OFF_if_none_eligible",
        "support_calibrated_output_name": "expected_BACC_proxy",
        "output_is_NELBO": False,
        "output_is_held_case_utility": False,
        "zero_to_one_case_proxy": (
            "m_times_pi_over_2Npos_minus_m_times_1minuspi_over_2Nneg"
        ),
        "one_to_zero_case_proxy": (
            "m_times_pi_over_2Nneg_minus_m_times_1minuspi_over_2Npos"
        ),
        "eligibility_requires_positive_held_flip_count": True,
        "eligibility_requires_strict_positive_case_proxy": True,
        "invalid_fit_is_ineligible": True,
        "case_scale": "mean_absolute_over_exact_eight_candidates",
        "donor_scale": "mean_absolute_over_exact_eight_candidates",
        "normalization_epsilon_used": False,
        "zero_case_scale_policy": "all_candidates_ineligible_and_OFF",
        "zero_donor_scale_policy": "donor_component_exactly_zero",
        "nonfinite_policy": "entire_route_fails_closed_to_OFF",
        "G_definition": (
            "equal_center_mean_directional_gain_over_query_q_not_in_H_or_e"
        ),
        "G_query_scope": "q_not_in_H_or_e",
        "score": (
            "four_fifths_normalized_case_proxy_plus_"
            "one_fifth_normalized_donor_G"
        ),
        "case_proxy_weight_fraction": "4/5",
        "case_proxy_weight_numerator": 4,
        "case_proxy_weight_denominator": 5,
        "donor_prior_weight_fraction": "1/5",
        "donor_prior_weight_numerator": 1,
        "donor_prior_weight_denominator": 5,
        "winner_must_be_strictly_positive": True,
        "candidate_pool": "eligible_non_target_sources_plus_OFF",
        "off_action_id": "OFF",
        "off_score": 0.0,
        "off_probability_source": "B",
        "selection_order": "OFF_then_numeric_source",
        "final_tie_tolerance": 1.0e-12,
        "composition": (
            "B_probability_with_selected_A1_probability_on_matching_B_hard_branch"
        ),
        "sole_endpoint_threshold": 0.5,
    }


def canonical_robust_endpoint_payload() -> dict[str, object]:
    """Return the exact nine-arm robust probability endpoint."""

    arm_grid = [
        {
            "arm_id": f"K{k}::w={numerator}/{denominator}",
            "K": k,
            "w_numerator": numerator,
            "w_denominator": denominator,
        }
        for k in (4, 5, 6)
        for numerator, denominator in ((1, 2), (3, 5), (7, 10))
    ]
    return {
        "schema_version": "fixed_bank_ogde_robust_endpoint_v1",
        "method_id": "R_NINE_ARM_ROBUST",
        "direction_ids": list(DIRECTION_IDS),
        "branch_definition": "baseline_B_hard_prediction_before_candidate_flip",
        "support_score": (
            "pooled_additive_confusion_count_directional_exact_bacc_gain_vs_B"
        ),
        "support_scope": "same_H_whole_cases_except_held_c",
        "per_case_bacc_used_for_scoring_or_selection": False,
        "G_definition": (
            "equal_center_mean_directional_gain_over_query_q_not_in_H_or_e"
        ),
        "G_query_scope": "q_not_in_H_or_e",
        "G_equal_center_aggregation": True,
        "K_grid": [4, 5, 6],
        "w_grid": [0.5, 0.6, 0.7],
        "w_rational_grid": ["1/2", "3/5", "7/10"],
        "arm_grid": arm_grid,
        "arm_count": 9,
        "all_arm_identities_retained_when_selected_endpoints_duplicate": True,
        "source_rank_rule": "descending_G_then_numeric_source",
        "top_K_scope": "eight_legal_non_target_sources_ranked_by_G",
        "endpoint_score": "w_times_support_S_plus_one_minus_w_times_G",
        "endpoint_score_arithmetic": "exact_rational_until_final_tie_check",
        "off_action_id": "OFF",
        "off_score": 0,
        "off_probability_source": "B",
        "selection_candidate_order": "OFF_then_numeric_source",
        "final_tie_tolerance": 1.0e-12,
        "endpoint_composition": (
            "mean_of_nine_selected_endpoint_probabilities_per_B_hard_branch"
        ),
        "off_endpoint_contributes_B_probability": True,
        "probabilities_averaged_before_endpoint_threshold": True,
        "sole_endpoint_threshold": 0.5,
        "hidden_arm_selection_used": False,
        "hyperparameter_search_used": False,
    }


def canonical_portfolio_payload() -> dict[str, object]:
    """Return the fixed prediction-level probability portfolio."""

    return {
        "schema_version": "fixed_bank_ogde_probability_portfolio_v1",
        "method_id": "OGDE_PORTFOLIO",
        "identification_method_id": "I_OPPORTUNITY_GATED",
        "robust_method_id": "R_NINE_ARM_ROBUST",
        "identification_weight_fraction": "3/5",
        "identification_weight_numerator": 3,
        "identification_weight_denominator": 5,
        "robust_weight_fraction": "2/5",
        "robust_weight_numerator": 2,
        "robust_weight_denominator": 5,
        "composition": (
            "three_fifths_I_probability_plus_two_fifths_R_probability"
        ),
        "composition_level": "common_row_aligned_prediction_probability",
        "weights_selected_on_same_evaluation_surface": True,
        "runtime_weight_selection_used": False,
        "sole_final_threshold": 0.5,
        "final_threshold_equal_maps_to_positive": True,
        "CVAE_or_generative_mixture": False,
        "NELBO_or_compatibility_estimate": False,
        "held_case_utility_used_before_terminal_scoring": False,
    }


def canonical_controls_payload() -> dict[str, object]:
    """Return attribution controls; all are descriptive and non-selective."""

    return {
        "method_ids": list(METHOD_IDS),
        "primary_method_ids": list(PRIMARY_METHOD_IDS),
        "pre_terminal_method_ids": list(PRE_TERMINAL_METHOD_IDS),
        "attribution_control_ids": list(ATTRIBUTION_CONTROL_IDS),
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "B_role": "fixed_equal_union_baseline_and_OFF_probability",
        "U_role": "fixed_uniform_A1_control",
        "I_OPPORTUNITY_GATED_role": (
            "strict_opportunity_and_positive_proxy_identification_endpoint"
        ),
        "R_NINE_ARM_ROBUST_role": "independently_recomputed_robust_endpoint",
        "OGDE_PORTFOLIO_role": "fixed_three_fifths_I_two_fifths_R_portfolio",
        "CALIBRATION_ONLY_B_R_role": (
            "three_fifths_B_two_fifths_R_attribution_control"
        ),
        "I_FEATURE_BLOCK_PERMUTED_role": (
            "candidate_feature_block_permutation_refit_reselect_control"
        ),
        "OGDE_FEATURE_BLOCK_PERMUTED_role": (
            "portfolio_using_permuted_identification_endpoint"
        ),
        "I_GATE_ONLY_role": "opportunity_and_positive_proxy_gate_with_OFF_or_B",
        "I_GATE_ONLY_active_probability": (
            "float64_mean_of_all_opportunity_positive_eligible_A1_"
            "probabilities_on_the_matching_branch"
        ),
        "I_GATE_ONLY_preserves_canonical_I_OFF_active_mask": True,
        "I_SOURCE_ONLY_role": (
            "canonical_normalized_source_rank_with_strict_positive_"
            "proxy_OFF_gate_removed"
        ),
        "I_SOURCE_ONLY_candidate_scope": (
            "finite_flip_count_positive_model_valid_opportunities"
        ),
        "I_SOURCE_ONLY_OFF_policy": "OFF_only_when_no_candidate_is_eligible",
        "G_DIRECTIONAL_MATCHED_role": "matched_donor_prior_only_control",
        "O_DIRECTIONAL_STATIC_role": "terminal_directional_static_oracle",
        "O_CASE_DIRECTIONAL_role": "terminal_case_directional_oracle",
        "feature_block_permutation_seed": 20_260_814,
        "feature_block_permutation_algorithm": (
            "splitmix64_route_direction_candidate_block_permutation_v1"
        ),
        "feature_block_permutation_unit": (
            "whole_candidate_feature_vector_within_H_c_direction"
        ),
        "feature_block_exchangeability_claimed": False,
        "permutation_p_value_computed": False,
        "full_pipeline_delete_one_center_recomputation": True,
        "delete_center_recomputes_G_normalization_decisions_R_and_portfolio": True,
        "controls_can_select_model_features_hyperparameters_weights_or_threshold": (
            False
        ),
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "center_pooled_exact_bacc_from_int64_confusion_sums",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_cells_are_not_independent_units": True,
        "primary_descriptive_contrasts": [
            "OGDE_PORTFOLIO-B",
            "OGDE_PORTFOLIO-U",
            "OGDE_PORTFOLIO-R_NINE_ARM_ROBUST",
            "OGDE_PORTFOLIO-CALIBRATION_ONLY_B_R",
        ],
        "probability_metrics": [
            "brier_score",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "threshold_crossing_attribution",
        ],
        "identification_metrics": [
            "off_precision",
            "off_recall",
            "off_balanced_accuracy",
            "overall_exact_action_agreement",
            "active_source_top1_agreement",
            "macro_spearman",
            "normalized_oracle_gap",
        ],
        "observed_vs_B_is_descriptive_only": True,
        "delete_center_results_are_descriptive_only": True,
        "incremental_vs_R_is_inconclusive": True,
        "source_identification_is_established": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "nominal_t_interval_is_a_success_gate": False,
        "delete_center_interval_is_a_success_gate": False,
        "attribution_controls_are_success_gates": False,
        "confusion_count_dtype": "int64",
        "scientific_reduction_dtype": "float64",
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "per_case_bacc_persisted_or_used": False,
        "results_are_terminal_consumed_test_diagnostics": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": (
            "fixed_bank_loo_opportunity_gated_dual_endpoint_runtime_v1"
        ),
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "persistent_generation_worker_count": 2,
        "multiprocessing_start_method": "spawn",
        "gpu_generation_phase_precedes_cpu_phase": True,
        "cuda_visible_devices_cleared_before_cpu_phase": True,
        "parent_cuda_context_forbidden": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "classifier_workers": 4,
        "route_model_workers": 4,
        "classifier_threads_per_worker": 3,
        "launch_blas_threads": 1,
        "maximum_total_cpu_threads": 12,
        "cpu_phase_blas_thread_environment": {
            "OMP_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "3",
            "NUMEXPR_NUM_THREADS": "3",
        },
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "target_task_count": TARGET_TASK_COUNT,
        "target_action_identity_count": TARGET_ACTION_IDENTITY_COUNT,
        "physical_actions_per_target_task": ACTION_COUNT_PER_TARGET,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "maximum_total_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "owned_task_checkpoint_replay_allowed": False,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "resume_policy": (
            "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
        ),
        "successful_phase_checkpoint_cleanup_after_validated_global_seal": True,
        "two_fresh_process_validation_required": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "routing_status": ROUTING_STATUS,
        "claim_role": CLAIM_ROLE,
        "bounded_interpretation": (
            "posthoc_abstention_calibrated_directional_probability_"
            "portfolio_sensitivity_only"
        ),
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "weights_selected_on_same_evaluation_surface": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "terminal_stage90_diagnostic": True,
        "observed_vs_B_is_descriptive_only": True,
        "delete_center_results_are_descriptive_only": True,
        "incremental_vs_R_is_inconclusive": True,
        "source_identification_is_established": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "downstream_utility_claimed": False,
        "predicted_held_case_exact_bacc_claimed": False,
        "generative_composition_claimed": False,
        "NELBO_compatibility_claimed": False,
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
