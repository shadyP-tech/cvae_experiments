"""Canonical executable config for the held-case correctness router."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    A1_EFFECTIVE_ROWS_PER_CLASS,
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    BASE_ROWS_PER_SOURCE_CLASS,
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    PUBLICATION_STATUS,
    SCRATCH_ROOT,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    STAGE_ID,
    TERMINAL_DECISION,
    TRAINING_SEEDS,
    UNIFORM_ROWS_PER_SOURCE_CLASS,
    WORKSTATION_PROFILE,
)


FEATURE_IDS = (
    "directional_flip_rate",
    "baseline_abs_margin_on_directional_flips",
    "candidate_abs_margin_on_directional_flips",
    "directional_probability_shift_on_flips",
    "seed_directional_flip_robustness",
    "candidate_seed_disagreement_on_directional_flips",
)
METHOD_IDS = (
    "B",
    "U",
    "CDCA_LOO",
    "G_directional_matched",
    "CDCA_case_proxy_only",
    "O_directional_static",
    "O_case_directional",
)
PRE_TERMINAL_METHOD_IDS = METHOD_IDS[:5]
TERMINAL_ORACLE_IDS = METHOD_IDS[5:]
DESCRIPTIVE_METHOD_IDS = ("CDCA_feature_block_permutation_descriptive",)


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
    return {
        "schema_version": (
            "midogpp_fixed_bank_case_directional_correctness_"
            "abstention_router_protocol_v1"
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
        "donor_query_scope": "q_not_in_H_or_e",
        "donor_prior_excludes_H_and_e": True,
        "all_72_donor_grants_complete_before_route_support": True,
        "physical_probabilities_globally_sealed_before_any_label_access": True,
        "label_free_held_case_features_sealed_before_support_labels": True,
        "route_scoped_support_grants_are_H_minus_c_only": True,
        "route_labels_never_enter_own_fit_scaler_state_or_decision": True,
        "route_local_model_state_never_shared": True,
        "all_218_predictions_and_decisions_sealed_before_terminal_labels": True,
        "terminal_labels_never_train_tune_rank_select_or_calibrate": True,
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
    return {
        "schema_version": (
            "fixed_bank_case_directional_correctness_abstention_"
            "action_library_v1"
        ),
        "action_ids": "B_U_and_eight_A1_source_actions",
        "baseline_action_id": "B",
        "uniform_control_action_id": "U",
        "source_action_id_format": "A1::source={source_center}",
        "candidate_source_count_per_target": 8,
        "physical_action_count_per_target": 10,
        "target_task_count": 81,
        "target_probability_cell_count": 810,
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
        "target_expert_used": False,
        "probabilities_averaged_exact_nine_before_feature_construction": True,
        "hard_probability_threshold": 0.5,
        "hard_threshold_equal_maps_to_positive": True,
        "previous_probability_surfaces_used": False,
    }


def canonical_case_correctness_router_payload() -> dict[str, object]:
    return {
        "schema_version": (
            "fixed_bank_case_directional_correctness_abstention_router_v1"
        ),
        "method_id": "CDCA_LOO",
        "direction_ids": ["zero_to_one", "one_to_zero"],
        "branch_definition": "baseline_B_hard_prediction_before_candidate_flip",
        "held_case_feature_ids": list(FEATURE_IDS),
        "held_case_features_are_label_free": True,
        "signed_feature_definition": "candidate_probability_minus_B_probability",
        "support_response": (
            "candidate_directional_flip_correctness_successes_over_trials"
        ),
        "one_model_per_H_c_e_direction": True,
        "fit_scope": "same_H_whole_cases_except_c_only",
        "feature_scaler_scope": "same_H_whole_cases_except_c_only",
        "imputation_used": False,
        "feature_selection_used": False,
        "threshold_tuning_used": False,
        "hyperparameter_search_used": False,
        "model_family": "ridge_binomial_logistic_newton_irls_v1",
        "feature_standardization": "H_minus_c_unweighted_mean_population_sd",
        "ridge_alpha": 1.0,
        "intercept_penalized": False,
        "max_iterations": 50,
        "convergence_tolerance": 1.0e-12,
        "eta_clip": [-30.0, 30.0],
        "probability_clip": [1.0e-12, 0.999999999999],
        "initialization": "all_zero_coefficients",
        "nonconvergence_policy": "candidate_invalid_case_proxy_zero",
        "zero_trial_policy": "candidate_invalid_case_proxy_zero",
        "support_denominator_source": "H_minus_c_labels_only",
        "zero_to_one_case_proxy": (
            "m_times_pi_over_2Npos_minus_m_times_1minuspi_over_2Nneg"
        ),
        "one_to_zero_case_proxy": (
            "m_times_pi_over_2Nneg_minus_m_times_1minuspi_over_2Npos"
        ),
        "donor_G_definition": (
            "equal_center_mean_directional_gain_over_q_not_in_H_or_e"
        ),
        "primary_score": "one_half_case_proxy_plus_one_half_G",
        "case_proxy_weight_fraction": "1/2",
        "donor_prior_weight_fraction": "1/2",
        "candidate_pool": "all_eight_non_target_sources_plus_OFF",
        "off_action_id": "OFF",
        "off_score": 0.0,
        "off_probability_source": "B",
        "selection_order": "OFF_then_numeric_source",
        "final_tie_tolerance": 1.0e-12,
        "composition": (
            "B_probability_with_selected_A1_probability_on_matching_B_hard_branch"
        ),
        "sole_final_threshold": 0.5,
        "predicted_held_case_exact_bacc_claimed": False,
        "output_name": "support_calibrated_directional_utility_proxy",
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "method_ids": list(METHOD_IDS),
        "pre_terminal_method_ids": list(PRE_TERMINAL_METHOD_IDS),
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "descriptive_method_ids": list(DESCRIPTIVE_METHOD_IDS),
        "B_role": "fixed_equal_union_baseline_and_OFF_probability",
        "U_role": "fixed_uniform_control",
        "CDCA_LOO_role": "primary_H_minus_c_case_correctness_abstention_router",
        "G_directional_matched_role": "donor_prior_only_directional_router",
        "CDCA_case_proxy_only_role": "H_minus_c_response_model_without_G",
        "O_directional_static_role": "terminal_directional_static_oracle",
        "O_case_directional_role": "terminal_case_directional_oracle",
        "held_feature_candidate_block_permutation_seed": 20_260_814,
        "held_feature_candidate_block_permutation_role": "descriptive_only",
        "held_feature_candidate_block_permutation_algorithm": (
            "splitmix64_route_direction_candidate_block_permutation_v1"
        ),
        "held_feature_candidate_block_exchangeability_claimed": False,
        "permutation_p_value_computed": False,
        "controls_can_select_model_features_hyperparameters_or_threshold": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "center_pooled_exact_bacc_from_int64_confusion_sums",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_cells_are_not_independent_units": True,
        "primary_descriptive_contrasts": [
            "CDCA_LOO-B",
            "CDCA_LOO-U",
            "CDCA_LOO-G_directional_matched",
            "CDCA_LOO-CDCA_case_proxy_only",
        ],
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "nominal_interval_is_a_success_gate": False,
        "feature_permutation_is_a_success_gate": False,
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
            "fixed_bank_case_directional_correctness_abstention_runtime_v1"
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
        "phase_disjoint_gpu_and_cpu_pools": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        "route_model_workers": 4,
        "classifier_workers": 4,
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
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "target_task_count": 81,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
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
        "routing_status": TERMINAL_DECISION,
        "claim_role": CLAIM_ROLE,
        "bounded_interpretation": (
            "posthoc_held_case_correctness_abstention_sensitivity_only"
        ),
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "terminal_stage90_diagnostic": True,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "downstream_utility_claimed": False,
        "predicted_held_case_exact_bacc_claimed": False,
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
    "DESCRIPTIVE_METHOD_IDS",
    "FEATURE_IDS",
    "METHOD_IDS",
    "PRE_TERMINAL_METHOD_IDS",
    "TERMINAL_ORACLE_IDS",
)
