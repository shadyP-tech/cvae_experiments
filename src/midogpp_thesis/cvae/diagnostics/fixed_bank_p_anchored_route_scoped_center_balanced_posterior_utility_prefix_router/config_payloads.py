"""Canonical executable config sections for terminal-only CBPUPR v1."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .constants import (
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    B_ROWS_PER_SOURCE_CLASS,
    BLOCKED_CONTROL_METHOD_ID,
    CANDIDATE_ONLY_METHOD_ID,
    CENTERS,
    CLAIM_ROLE,
    COMPOSED_POLICY_IDS,
    CPU_WORKERS,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_TOTAL_POSTERIOR_MODEL_FIT_COUNT,
    GENERATED_CACHE_FORMAT,
    GENERATION_WORKERS_PER_DEVICE,
    LOG_LOSS_CLIP_EPSILON,
    MIN_SUPPORTED_DONOR_CENTER_COUNT,
    OBSERVED_MAX_CONTROL_METHOD_ID,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    PORTFOLIO_METHOD_ID,
    POSTERIOR_FITS_PER_ROUTE_AND_CONTROL,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    RUN_RECOVERY_POLICY,
    SCRATCH_ROOT,
    SOURCE_JOB_COUNT,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    SOURCE_STREAM_COUNT,
    SOURCE_WORKERS_PER_DEVICE,
    TARGET_POSTERIOR_C,
    TARGET_POSTERIOR_MAX_ITER,
    TARGET_POSTERIOR_RANDOM_STATE,
    TARGET_POSTERIOR_SOLVER,
    TARGET_POSTERIOR_TOLERANCE,
    TARGET_POSTERIOR_PROBABILITY_CLIP,
    TARGET_ACTION_IDENTITY_COUNT,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TARGET_UNIQUE_CLASSIFIER_FIT_COUNT,
    TERMINAL_DECISION,
    U_ROWS_PER_SOURCE_CLASS,
    UTILITY_RESPONSE_IDS,
    UTILITY_ZERO_TOLERANCE,
    TRANSPORT_MAD_SCALE,
    TRANSPORT_SCALE_FLOOR,
    TRANSPORT_MIN_REFERENCE_CENTER_COUNT,
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
        "schema_version": "fixed_bank_cbpupr_action_library_v1",
        "centers": list(CENTERS),
        "actions_per_target": PHYSICAL_ACTION_COUNT_PER_TARGET,
        "action_ids": "B_U_and_eight_A1_source_actions",
        "B_rows_per_source_class": B_ROWS_PER_SOURCE_CLASS,
        "U_rows_per_source_class": U_ROWS_PER_SOURCE_CLASS,
        "A1_selected_rows_per_class": A1_SELECTED_ROWS_PER_CLASS,
        "A1_other_rows_per_class": A1_OTHER_ROWS_PER_CLASS,
        "A1_selected_row_weight": A1_SELECTED_ROW_WEIGHT,
        "A1_other_row_weight": A1_OTHER_ROW_WEIGHT,
        "target_expert_excluded": True,
        "probabilities_averaged_exact_nine_before_routing": True,
        "previous_probability_surface_used": False,
    }


def canonical_policy_menu_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cbpupr_policy_menu_v1",
        "protected_fallback": PORTFOLIO_METHOD_ID,
        "exact_P_fallback_required": True,
        "exact_P_fallback_storage_contract": "byte_for_byte_float32_identity",
        "policy_ids": list(COMPOSED_POLICY_IDS),
        "primary_policy_id": PRIMARY_METHOD_ID,
        "candidate_only_control_id": CANDIDATE_ONLY_METHOD_ID,
        "observed_max_prefix_control_id": OBSERVED_MAX_CONTROL_METHOD_ID,
        "cyclic_fingerprint_prefix_control_id": BLOCKED_CONTROL_METHOD_ID,
        "candidate_endpoint_ids": [
            "B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST"
        ],
        "candidate_action_rule": (
            "proper_safe_then_maximum_expected_BACC_then_B_I_R_then_action_hash"
        ),
        "expected_utility_coordinates": list(UTILITY_RESPONSE_IDS),
        "expected_utility_model": "analytic_posterior_expected_utility",
        "posterior_augmented_center_denominators": True,
        "donor_response_regression_used": False,
        "target_posterior": {
            "family": "sklearn_logistic_regression",
            "C": TARGET_POSTERIOR_C,
            "class_weight": "balanced",
            "solver": TARGET_POSTERIOR_SOLVER,
            "max_iter": TARGET_POSTERIOR_MAX_ITER,
            "random_state": TARGET_POSTERIOR_RANDOM_STATE,
            "tolerance": TARGET_POSTERIOR_TOLERANCE,
            "probability_clip": TARGET_POSTERIOR_PROBABILITY_CLIP,
            "natural_prevalence_correction": True,
            "fit_scope": "route_local_H_minus_c",
            "fits_per_route_and_control": POSTERIOR_FITS_PER_ROUTE_AND_CONTROL,
            "inner_crossfit_or_OOF_reliability_used": False,
            "pseudo_H_J_d_reuses_sealed_J_minus_d_posterior": True,
            "outer_H_support_rows_or_labels_enter_fit_or_normalization": False,
            "outer_H_frozen_label_free_expert_fingerprint_covariates_present": True,
            "posterior_is_outer_H_covariate_invariant": False,
            "outer_H_specific_refit_performed": False,
            "shared_across_distinct_target_case_routes": False,
            "referenced_by_outer_H_pseudo_wrappers": True,
            "full_fitted_model_DTO_persisted": True,
            "held_case_eta_replayed_from_fitted_DTO_during_validation": True,
            "optimizer_refit_during_bundle_validation": False,
            "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
        },
        "calibration_unit": "donor_center",
        "calibration": "center_balanced_median_conditional_overprediction_bias",
        "minimum_supported_donor_center_count": MIN_SUPPORTED_DONOR_CENTER_COUNT,
        "pseudo_calibration": "leave_pseudo_donor_J_out",
        "prefix_unit": "complete_case_policy",
        "prefix_order": (
            "descending_corrected_expected_BACC_then_case_id_then_policy_hash"
        ),
        "prefix_feasibility": (
            "aggregate_BACC_positive_and_aggregate_favorable_Brier_and_log_"
            "loss_nonnegative"
        ),
        "prefix_selection": (
            "maximum_corrected_aggregate_BACC_then_smaller_K_then_prefix_hash"
        ),
        "utility_eligibility_zero_tolerance": UTILITY_ZERO_TOLERANCE,
        "prefix_feasibility_zero_tolerance": UTILITY_ZERO_TOLERANCE,
        "finite_sample_coverage_claimed": False,
        "confidence_bound_claimed": False,
        "numeric_transport_is_authorization_gate": False,
        "structural_transport_lineage_is_authorization_gate": True,
        "zero_MAD_numeric_transport_division_forbidden": True,
        "numeric_transport_MAD_scale": TRANSPORT_MAD_SCALE,
        "numeric_transport_zero_scale_threshold": TRANSPORT_SCALE_FLOOR,
        "numeric_transport_minimum_reference_centers": (
            TRANSPORT_MIN_REFERENCE_CENTER_COUNT
        ),
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
        "center_contrasts_against": PORTFOLIO_METHOD_ID,
        "primary_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "fixed_method_menu": [PORTFOLIO_METHOD_ID, *COMPOSED_POLICY_IDS],
        "selection_control": (
            "exact_2_power_9_center_sign_flip_max_over_fixed_method_menu"
        ),
        "route_pipeline_refit_inside_null_replicate": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "raw_labels_persisted": False,
        "diagnostic_rows_are_not_independent_inference_units": True,
        "nonzero_route_count_is_not_success": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cbpupr_workstation_runtime_v1",
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": SOURCE_WORKERS_PER_DEVICE,
        "generation_workers_per_device": GENERATION_WORKERS_PER_DEVICE,
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
        "generated_cache_format": GENERATED_CACHE_FORMAT,
        "probability_surface_format": "sealed_compressed_float32_npz",
        "dense_array_manifest_format": "json_hash_offsets_only",
        "numeric_transport_MAD_scale": TRANSPORT_MAD_SCALE,
        "numeric_transport_zero_scale_threshold": TRANSPORT_SCALE_FLOOR,
        "numeric_transport_minimum_reference_centers": (
            TRANSPORT_MIN_REFERENCE_CENTER_COUNT
        ),
        "worker_DTOs_are_plain_pickle_safe_values": True,
        "classifier_workers": CPU_WORKERS,
        "route_model_workers": CPU_WORKERS,
        "classifier_threads_per_worker": 3,
        "target_posterior_threads_per_worker": 1,
        "launch_blas_threads": 1,
        "maximum_total_cpu_threads": 12,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": SOURCE_JOB_COUNT,
        "source_stream_count": SOURCE_STREAM_COUNT,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "target_task_count": TARGET_TASK_COUNT,
        "target_action_identity_count": TARGET_ACTION_IDENTITY_COUNT,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": TARGET_UNIQUE_CLASSIFIER_FIT_COUNT,
        "maximum_total_classifier_fit_count": TARGET_UNIQUE_CLASSIFIER_FIT_COUNT,
        "outer_route_count": EXPECTED_OUTER_PLAN_COUNT,
        "ordered_H_J_pair_count": EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
        "expected_outer_endpoint_model_fit_count": (
            EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        ),
        "expected_target_posterior_model_fit_count": (
            EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        ),
        "pseudo_route_count": EXPECTED_PSEUDO_ROUTE_COUNT,
        "expected_pseudo_posterior_model_fit_count": (
            EXPECTED_PSEUDO_POSTERIOR_MODEL_FIT_COUNT
        ),
        "expected_total_posterior_model_fit_count": (
            EXPECTED_TOTAL_POSTERIOR_MODEL_FIT_COUNT
        ),
        "donor_response_model_fit_count": 0,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": RUN_RECOVERY_POLICY,
        "owned_task_checkpoint_replay_allowed": False,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "two_fresh_process_validation_required": True,
        "validation_endpoint_optimizer_refit_count": 0,
        "validation_posterior_optimizer_refit_count": 0,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_role": CLAIM_ROLE,
        "bounded_interpretation": (
            "center_balanced_posterior_expected_utility_prefix_sensitivity_"
            "on_consumed_MIDOGpp_test_only"
        ),
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "prior_consumed_test_findings_informed_method_design": True,
        "target_support_labels_used": True,
        "target_evaluation_labels_used_before_route_seal": False,
        "fresh_evidence": False,
        "terminal_stage90_diagnostic": True,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "expert_selection_claimed": False,
        "deployment_claimed": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_probability_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
        "all_fitted_DTO_outputs_replayed_during_validation": True,
        "optimizer_refit_during_bundle_validation": False,
        "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + (
    "CLASSIFIER",
)
