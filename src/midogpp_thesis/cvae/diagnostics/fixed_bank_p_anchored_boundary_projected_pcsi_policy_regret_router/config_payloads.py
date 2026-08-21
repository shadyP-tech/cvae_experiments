"""Canonical executable config sections for PCSI-PARC."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .constants import (
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    B_ROWS_PER_SOURCE_CLASS,
    BLOCKED_CONTROL_METHOD_ID,
    CENTERS,
    CLAIM_ROLE,
    COMPOSED_POLICY_IDS,
    CPU_WORKERS,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    LEGACY_CONTROL_METHOD_ID,
    LOG_LOSS_CLIP_EPSILON,
    PRIMARY_METHOD_ID,
    PROJECTED_NO_PARC_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    PROJECTION_ONE_TO_ZERO_VALUE,
    PROJECTION_ZERO_TO_ONE_VALUE,
    PUBLICATION_STATUS,
    SCRATCH_ROOT,
    SIGN_PRESERVING_SHRINKAGE,
    TARGET_POSTERIOR_C,
    TARGET_POSTERIOR_MAX_ITER,
    TARGET_POSTERIOR_RANDOM_STATE,
    TARGET_POSTERIOR_SOLVER,
    TERMINAL_DECISION,
    TRANSPORT_SCALE_FLOOR,
    U_ROWS_PER_SOURCE_CLASS,
    UNPROJECTED_PARC_METHOD_ID,
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


def canonical_action_library_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_parc_action_library_v1",
        "centers": list(CENTERS),
        "actions_per_target": 10,
        "action_ids": "B_U_and_eight_A1_source_actions",
        "B_rows_per_source_class": B_ROWS_PER_SOURCE_CLASS,
        "U_rows_per_source_class": U_ROWS_PER_SOURCE_CLASS,
        "A1_selected_rows_per_class": A1_SELECTED_ROWS_PER_CLASS,
        "A1_other_rows_per_class": A1_OTHER_ROWS_PER_CLASS,
        "A1_selected_row_weight": A1_SELECTED_ROW_WEIGHT,
        "A1_other_row_weight": A1_OTHER_ROW_WEIGHT,
        "target_expert_excluded": True,
        "probabilities_averaged_exact_nine_before_routing": True,
        "physical_probability_storage_dtype": "float32",
        "previous_probability_surface_used": False,
    }


def canonical_policy_menu_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_parc_policy_menu_v1",
        "policy_ids": list(COMPOSED_POLICY_IDS),
        "primary_policy_id": PRIMARY_METHOD_ID,
        "projected_no_policy_regret_control_id": PROJECTED_NO_PARC_METHOD_ID,
        "raw_full_action_PARC_control_id": UNPROJECTED_PARC_METHOD_ID,
        "fresh_legacy_dual_veto_control_id": LEGACY_CONTROL_METHOD_ID,
        "blocked_fingerprint_control_id": BLOCKED_CONTROL_METHOD_ID,
        "protected_fallback": "P_PROTECTED",
        "endpoint_nomination_used": False,
        "actions_selected_from_terminal_labels": False,
        "projection": {
            "geometry_id": PROJECTION_GEOMETRY_ID,
            "zero_to_one_value": PROJECTION_ZERO_TO_ONE_VALUE,
            "one_to_zero_value": PROJECTION_ONE_TO_ZERO_VALUE,
            "off_crossing_P_bytes_preserved": True,
            "equivalence_identity": "complete_little_endian_binary32_output_vector",
            "provenance_tie_order": [
                "B",
                "I_OPPORTUNITY_GATED",
                "R_NINE_ARM_ROBUST",
            ],
        },
        "target_posterior": {
            "family": "sklearn_logistic_regression",
            "C": TARGET_POSTERIOR_C,
            "class_weight": "balanced",
            "solver": TARGET_POSTERIOR_SOLVER,
            "max_iter": TARGET_POSTERIOR_MAX_ITER,
            "random_state": TARGET_POSTERIOR_RANDOM_STATE,
            "natural_prevalence_correction": True,
            "fit_scope": "route_local_H_minus_c",
            "shared_across_routes": False,
            "final_classifier": False,
        },
        "projected_selection_gate": (
            "crossing_gt_0_and_target_influence_gt_1e_minus_15_and_"
            "predicted_Brier_le_0_and_predicted_log_loss_le_0"
        ),
        "projected_selection_score": "target_influence",
        "utility_model": {
            "family": "joint_three_response_weighted_ridge",
            "alpha": 1.0,
            "direction_intercept_count": 2,
            "direction_intercepts_penalized": False,
            "standardized_slope_count": 12,
            "response_ids": [
                "bacc_contribution_delta",
                "brier_contribution_delta",
                "log_loss_contribution_delta",
            ],
            "weighting": (
                "equal_donor_then_equal_case_then_equal_surviving_equivalence_class"
            ),
            "solve_dtype": "float64",
            "singular_fallback": "pinv_rcond_1e_minus_12",
        },
        "policy_regret": {
            "pseudo_target_fit_scope": "H_and_J_double_excluded",
            "gain_coordinates": ["BACC", "negative_Brier", "negative_log_loss"],
            "regret_correction": (
                "componentwise_worst_observed_donor_clamped_at_zero"
            ),
            "projected_replay_count": EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY,
            "raw_full_action_replay_count": EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY,
            "finite_sample_coverage_claimed": False,
        },
        "transport": {
            "semantics": "support_conditioned_endpoint_reconstructed_P_B_I_R",
            "endpoint_support_scope": "endpoint_target_T_minus_held_case_c",
            "actual_source_prior_scope": "q_not_in_endpoint_target_T_or_source_e",
            "donor_source_prior_scope": (
                "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
            ),
            "source_prior_labels_used_upstream": True,
            "route_local_support_labels_used_upstream": True,
            "held_case_evaluation_capability_used_directly": False,
            "pseudo_evaluation_capability_used_directly": False,
            "terminal_evaluation_capability_used_directly": False,
            "label_free_claim": False,
            "uses_pre_equivalence_endpoint_crossing_rates": True,
            "screens_sealed_before_pseudo_evaluation_capability_open": True,
            "screens_sealed_before_terminal_evaluation_capability_open": True,
            "identity_level_route_noninterference_required": True,
            "identity_level_route_noninterference_proven": False,
            "authorization_valid": False,
            "protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
            "feature_count": 12,
            "scale": "median_MAD_1_4826",
            "scale_floor": TRANSPORT_SCALE_FLOOR,
            "minimum_reference_center_count": 3,
            "required_targets": "H_and_all_eight_donors",
            "equality_passes": True,
        },
        "primary_gate": (
            "all_transport_pass_and_every_corrected_gain_coordinate_gt_0"
        ),
        "one_equivalence_class_per_direction": True,
        "raw_full_action_control_geometry": "RAW_FULL_ACTION_PARC",
        "fresh_legacy_sign_preserving_shrinkage": SIGN_PRESERVING_SHRINKAGE,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_metric": "equal_center_BACC",
        "secondary_metrics": [
            "sample_pooled_BACC",
            "global_Brier",
            "equal_center_Brier",
            "global_log_loss",
            "equal_center_log_loss",
        ],
        "log_loss_clip_epsilon": LOG_LOSS_CLIP_EPSILON,
        "center_contrasts_against": "P_PROTECTED",
        "primary_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "proper_loss_safety_rule": (
            "report_mean_center_Brier_and_log_loss_deltas_vs_P_without_"
            "postseal_route_changes"
        ),
        "proper_loss_safety_may_change_same_surface_routes": False,
        "policy_regret_information_diagnostics": [
            "target_influence_vs_realized_projected_action_BACC_midrank_Spearman",
            "predicted_whole_policy_gain_vs_realized_whole_policy_gain_midrank_Spearman",
            "transport_distance_by_center",
            "projected_equivalence_class_multiplicity",
            "selected_policy_helpful_vs_harmful_counts",
            "case_harm_rate",
        ],
        "oracle_diagnostics": [
            "endpoint_top1_agreement_case_weighted_and_equal_center",
            "endpoint_rank_case_weighted_and_equal_center",
            "normalized_endpoint_oracle_gap_case_weighted_and_equal_center",
        ],
        "selection_control": "exact_2_power_9_center_sign_flip_max_over_fixed_method_menu",
        "route_pipeline_refit_inside_null_replicate": False,
        "information_gate_is_terminal_only": True,
        "information_gate_may_change_same_surface_routes": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "raw_labels_persisted": False,
        "diagnostic_rows_are_not_independent_inference_units": True,
        "blocked_fingerprint_unit": "complete_feature_row_cyclic_shift_within_case",
        "calibration_validated": False,
        "finite_sample_coverage_claimed": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_parc_workstation_runtime_v1",
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
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "endpoint_workers": CPU_WORKERS,
        "classifier_workers": CPU_WORKERS,
        "posterior_utility_replay_workers": CPU_WORKERS,
        "route_model_workers": CPU_WORKERS,
        "classifier_threads_per_worker": 3,
        "endpoint_threads_per_worker": 3,
        "target_posterior_threads_per_worker": 1,
        "utility_threads_per_worker": 1,
        "policy_replay_threads_per_worker": 1,
        "launch_blas_threads": 1,
        "maximum_total_cpu_threads": 12,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "target_task_count": 81,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "outer_route_count": EXPECTED_OUTER_PLAN_COUNT,
        "double_exclusion_pair_count": EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
        "expected_outer_endpoint_model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
        "expected_target_posterior_model_fit_count": (
            EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        ),
        "expected_utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
        "expected_projected_PARC_utility_model_fit_count": (
            EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY
        ),
        "expected_raw_full_action_PARC_utility_model_fit_count": (
            EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY
        ),
        "expected_fresh_legacy_utility_model_fit_count": (
            EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT
        ),
        "expected_policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT,
        "expected_projected_policy_replay_count": (
            EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
        ),
        "expected_raw_full_action_policy_replay_count": (
            EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
        ),
        "prior_rebinding_additional_endpoint_model_fit_count": 0,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only",
        "owned_task_checkpoint_replay_allowed": False,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "two_fresh_process_validation_required": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_role": CLAIM_ROLE,
        "bounded_interpretation": (
            "target_support_conditioned_boundary_projected_whole_policy_regret_"
            "sensitivity_on_consumed_MIDOGpp_test_only"
        ),
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "projection_selection_transport_and_policy_regret_rule_predeclared_for_this_run": True,
        "held_case_evaluation_capability_used_before_route_seal": False,
        "target_support_labels_used": True,
        "transport_semantics": "support_conditioned_endpoint_reconstructed_P_B_I_R",
        "transport_source_prior_labels_used_upstream": True,
        "transport_route_local_support_labels_used_upstream": True,
        "transport_held_case_evaluation_capability_used_directly": False,
        "transport_pseudo_evaluation_capability_used_directly": False,
        "transport_terminal_evaluation_capability_used_directly": False,
        "transport_label_free_claim": False,
        "transport_identity_level_route_noninterference_required": True,
        "transport_identity_level_route_noninterference_proven": False,
        "transport_authorization_valid": False,
        "transport_protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
        "execution_authorized": False,
        "unlabeled_target_deployment_claimed": False,
        "fresh_evidence": False,
        "terminal_stage90_diagnostic": True,
        "thesis_specific_integration_novelty_only": True,
        "generic_ensemble_or_calibration_method_novelty_claimed": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "finite_sample_conformal_coverage_claimed": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
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
        "previous_probability_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + ("CLASSIFIER",)
