"""Canonical, path-independent P-DCAPS configuration sections."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from .identity import (
    ACTION_FAMILIES,
    ACTION_ONLY_METHOD_ID,
    ACTION_STRATA,
    CYCLIC_METHOD_ID,
    LEGACY_METHOD_ID,
    METHOD_MENU,
    METRICS,
    POLICY_ONLY_METHOD_ID,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    P_METHOD_ID,
    RIDGE_ALPHA,
    TERMINAL_DECISION,
    TIE_TOLERANCE,
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
    """Return the independently recomputed fixed-bank action contract."""

    return {
        "schema_version": "pdcaps_action_library_v1",
        "centers": list(CENTERS),
        "excluded_center": "4",
        "physical_actions_per_target": 10,
        "physical_action_ids": "B_U_and_eight_A1_source_actions",
        "candidate_action_families": list(ACTION_FAMILIES),
        "candidate_action_strata": [list(row) for row in ACTION_STRATA],
        "B_rows_per_source_class": 128,
        "U_rows_per_source_class": 144,
        "A1_selected_rows_per_class": 256,
        "A1_other_rows_per_class": 128,
        "A1_selected_row_weight": 1.4375,
        "A1_other_row_weight": 0.875,
        "target_expert_excluded": True,
        "probabilities_averaged_exact_nine_before_routing": True,
        "physical_probability_storage_dtype": "float32",
        "all_crossing_actions_retained_until_action_surface_seal": True,
        "previous_probability_surface_used": False,
        "previous_stage90_output_used": False,
    }


def canonical_policy_menu_payload() -> dict[str, object]:
    """Return the fixed action and complete-prefix selection contract."""

    return {
        "schema_version": "pdcaps_policy_menu_v1",
        "method_ids": list(METHOD_MENU),
        "protected_fallback": P_METHOD_ID,
        "primary_method_id": PRIMARY_METHOD_ID,
        "exact_P_fallback_required": True,
        "exact_P_fallback_storage_contract": "byte_for_byte_float32_identity",
        "action_response_model": {
            "family": "equal_center_weighted_ridge",
            "response_coordinates": list(METRICS),
            "alpha": RIDGE_ALPHA,
            "hyperparameter_selection_used": False,
            "standardization": "fit_rows_only",
            "solve_dtype": "float64",
            "features": [
                "predicted_favorable_metric",
                "crossing_fraction",
                "six_action_family_direction_indicators",
                "predicted_metric_x_action_stratum_interactions",
            ],
            "weighting": "equal_center_then_route_then_action_cell",
            "target_fit_scope": "all_J_not_equal_H",
            "pseudo_fit_scope": "all_K_not_in_H_or_J",
            "held_case_role": "SCORED_RESPONSE_ONLY_AFTER_SURFACE_SEAL",
        },
        "stratum_reliability_gate": {
            "minimum_represented_donor_centers": 6,
            "bacc_spearman_strictly_positive": True,
            "equal_center_mean_realized_bacc_strictly_positive": True,
            "strict_majority_positive_donor_center_bacc_means": True,
            "equal_center_mean_realized_brier_nonnegative": True,
            "equal_center_mean_realized_log_nonnegative": True,
            "class_domain_support_required": True,
            "nonfinite_or_unsupported_action": "P_PROTECTED",
            "applied_before_within_case_argmax": True,
        },
        "action_selection": {
            "feasibility": (
                "calibrated_BACC_positive_and_calibrated_favorable_Brier_and_"
                "log_nonnegative"
            ),
            "ranking": (
                "maximum_calibrated_BACC_then_B_I_R_then_zero_to_one_"
                "then_one_to_zero_then_action_hash"
            ),
            "tie_tolerance": TIE_TOLERANCE,
            "tie_or_failure_fallback": P_METHOD_ID,
        },
        "policy_surface_model": {
            "family": "equal_center_weighted_ridge",
            "response_coordinates": list(METRICS),
            "alpha": RIDGE_ALPHA,
            "hyperparameter_selection_used": False,
            "standardization": "fit_rows_only",
            "solve_dtype": "float64",
            "features": [
                "predicted_favorable_metric",
                "normalized_prefix_depth",
                "maximum_positive_candidate_share",
                "six_action_stratum_proportions",
            ],
            "weighting": "equal_center_then_complete_policy_cell",
            "target_fit_scope": "all_J_not_equal_H",
            "pseudo_fit_scope": "all_K_not_in_H_or_J",
            "all_complete_prefix_cells_scored": True,
        },
        "lower_envelope": {
            "formula": (
                "policy_prediction_minus_equal_center_mean_positive_OOF_"
                "overprediction_minus_maximum_delete_donor_change"
            ),
            "out_of_fold_only": True,
            "finite_sample_coverage_claimed": False,
            "confidence_bound_claimed": False,
        },
        "prefix_selection": {
            "feasibility": (
                "lower_envelope_BACC_positive_and_lower_envelope_favorable_"
                "Brier_and_log_nonnegative"
            ),
            "ranking": "maximum_BACC_then_smaller_K_then_prefix_hash",
            "tie_tolerance": TIE_TOLERANCE,
            "empty_or_failed_surface_fallback": P_METHOD_ID,
        },
        "legacy_control_recomputed_same_run": True,
        "legacy_control": {
            "method_id": LEGACY_METHOD_ID,
            "implementation_scope": "same_run_package_local_only",
            "response_scope": "donor_pseudo_responses_only",
            "pooling": "equal_center",
            "selection_grid": "normalized_target_prefix_depth_grid",
            "donor_prefix_match": (
                "nearest_normalized_depth_then_smaller_k_then_prefix_hash"
            ),
            "feasibility": (
                "pooled_BACC_strictly_positive_and_pooled_favorable_Brier_"
                "and_log_nonnegative"
            ),
            "ranking": (
                "maximum_pooled_BACC_then_smaller_target_k_then_target_"
                "prefix_hash"
            ),
            "target_labels_used": False,
            "exact_P_permitted_at_k0": True,
        },
        "method_controls": {
            "schema_version": "pdcaps_fixed_method_controls_v1",
            "fixed_menu": {
                P_METHOD_ID: {
                    "source_layer": "protected_P_probability_surface",
                    "selected_actions": "none_exact_P_byte_identity",
                    "admission_H": "not_applied",
                    "expected_posterior_control_id": "IDENTITY",
                    "same_physical_surface_as_identity_required": True,
                    "role": "protected_reference",
                },
                PRIMARY_METHOD_ID: {
                    "source_layer": (
                        "identity_action_selection_then_selected_policy_prefix"
                    ),
                    "selected_actions": (
                        "selected_policy_prefix_only_when_Admission_H_passes"
                    ),
                    "admission_H": "required_pseudo_only_same_run",
                    "expected_posterior_control_id": "IDENTITY",
                    "same_physical_surface_as_identity_required": True,
                    "role": "primary_terminal_diagnostic",
                },
                ACTION_ONLY_METHOD_ID: {
                    "source_layer": "identity_target_action_selections",
                    "selected_actions": (
                        "all_nonfallback_target_action_selections"
                    ),
                    "admission_H": "bypassed_terminal_ablation",
                    "expected_posterior_control_id": "IDENTITY",
                    "same_physical_surface_as_identity_required": True,
                    "role": "terminal_action_layer_ablation",
                },
                POLICY_ONLY_METHOD_ID: {
                    "source_layer": "identity_selected_target_policy_prefix",
                    "selected_actions": "selected_policy_prefix",
                    "admission_H": "bypassed_terminal_ablation",
                    "expected_posterior_control_id": "IDENTITY",
                    "same_physical_surface_as_identity_required": True,
                    "role": "terminal_admission_layer_ablation",
                },
                LEGACY_METHOD_ID: {
                    "source_layer": (
                        "typed_same_run_legacy_center_pooled_target_decision"
                    ),
                    "selected_actions": "sealed_legacy_target_prefix",
                    "admission_H": "not_applied",
                    "expected_posterior_control_id": "IDENTITY",
                    "same_physical_surface_as_identity_required": True,
                    "role": "terminal_legacy_control",
                },
                CYCLIC_METHOD_ID: {
                    "source_layer": (
                        "distinct_cyclic_posterior_action_and_policy_result"
                    ),
                    "selected_actions": (
                        "cyclic_selected_policy_prefix_only_when_cyclic_"
                        "Admission_H_passes"
                    ),
                    "admission_H": "required_cyclic_pseudo_only_same_run",
                    "expected_posterior_control_id": (
                        "WITHIN_CASE_CYCLIC_SHIFT"
                    ),
                    "same_physical_surface_as_identity_required": True,
                    "distinct_action_surface_seal_from_identity_required": True,
                    "role": "terminal_cyclic_poison_control",
                },
            },
            "caller_selected_action_hashes_permitted": False,
            "typed_source_and_seal_binding_required": True,
            "terminal_diagnostic_only": True,
            "routing_authorized": False,
            "promotion_allowed": False,
            "target_labels_used": False,
        },
        "cyclic_poison_control_predeclared": True,
        "terminal_labels_may_change_same_surface_routes": False,
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
        "log_loss_clip_epsilon": 1.0e-12,
        "center_contrasts_against": P_METHOD_ID,
        "primary_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "fixed_method_menu": list(METHOD_MENU),
        "selection_control": "exact_2_power_9_center_sign_flip_max_over_fixed_menu",
        "route_pipeline_refit_inside_null_replicate": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "terminal_diagnostics": [
            "action_expected_vs_realized_midrank_spearman_by_stratum",
            "policy_expected_vs_realized_midrank_spearman",
            "joint_safe_routed_policy_rate",
            "normalized_endpoint_oracle_gap",
            "case_harm_rate",
            "center_action_frequencies",
        ],
        "preterminal_admission_uses_pseudo_donor_responses_only": True,
        "target_terminal_labels_may_change_routes": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "raw_labels_persisted": False,
        "diagnostic_rows_are_not_independent_inference_units": True,
        "nonzero_route_count_is_not_success": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    """Return the frozen workstation topology and resource ceilings."""

    return {
        "schema_version": "pdcaps_workstation_runtime_v1",
        "execution_authorized": False,
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "persistent_generation_worker_count": 2,
        "gpu_generation_phase_precedes_cpu_phase": True,
        "cuda_visible_devices_cleared_before_cpu_phase": True,
        "parent_cuda_context_forbidden": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "multiprocessing_start_method": "spawn",
        "outer_task_unit": "one_complete_outer_center_H",
        "outer_task_count": 9,
        "outer_worker_count": 4,
        "outer_process_workers": 4,
        "nested_process_pools_forbidden": True,
        "worker_DTOs_are_plain_pickle_safe_values": True,
        "worker_DTOs_forbid_mappingproxy_estimators_handles_and_closures": True,
        "worker_results_are_manifest_hashes_and_compact_offsets_only": True,
        "classifier_threads_per_worker": 3,
        "classifier_workers": 4,
        "calibration_threads_per_worker": 1,
        "launch_blas_threads": 1,
        "maximum_total_cpu_threads": 12,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107374182400,
        "minimum_artifact_disk_free_bytes": 17179869184,
        "minimum_scratch_disk_free_bytes": 34359738368,
        "minimum_gpu_free_mib_per_device": 18000,
        "tf32_enabled": False,
        "amp_enabled": False,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "generated_cache_format": "float32_npy_memmap",
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "target_task_count": 81,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "dense_integer_indexed_surfaces": True,
        "maximum_dense_surface_bytes": 8589934592,
        "outer_chunks_written_atomically": True,
        "serial_process_hash_equivalence_required": True,
        "physical_probability_cell_count": 810,
        "held_case_route_count": 218,
        "ordered_H_J_pair_count": 72,
        "pseudo_case_route_count": 1744,
        "attempted_action_cell_count": 10464,
        "expected_primary_crossing_action_replay_count": 2680,
        "maximum_crossing_action_replay_count": 10464,
        "expected_existing_candidate_prefix_cell_count": 690,
        "maximum_pseudo_prefix_cell_count": 381936,
        "action_response_numerical_ridge_fit_count": 999,
        "action_response_serialized_model_count": 1755,
        "unordered_reliability_exclusion_pair_fit_reuse": True,
        "policy_surface_numerical_ridge_fit_count": 999,
        "policy_surface_serialized_model_count": 1755,
        "total_numerical_ridge_fit_count": 1998,
        "scratch_preference": [
            "/data/local/fixed_bank_p_anchored_route_scoped_donor_crossfit_"
            "action_policy_surface_router_v1",
            "artifact_parent",
        ],
        "resume_policy": "no_cross_run_recovery_atomic_outer_H_chunks_only",
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "previous_stage90_scratch_reuse_forbidden": True,
        "preterminal_fresh_process_validation_count": 2,
        "final_fresh_process_validation_count": 2,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "schema_version": "pdcaps_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_role": (
            "posthoc_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
            "action_policy_surface_router_diagnostic"
        ),
        "bounded_interpretation": (
            "donor_crossfit_action_and_policy_surface_sensitivity_on_"
            "consumed_MIDOGpp_test_only"
        ),
        "execution_authorized": False,
        "authorization_basis": "implementation_and_non_authorizing_registration_only",
        "consumed_test_data": True,
        "fresh_evidence": False,
        "method_development_is_posthoc": True,
        "prior_consumed_test_findings_informed_method_design": True,
        "prior_consumed_test_bytes_used_as_scientific_inputs": False,
        "target_support_labels_used": True,
        "target_support_labels_are_non_deployable_consumed_test_support": True,
        "held_case_evaluation_capability_used_before_route_seal": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_probability_surface_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
        "source_experts_frozen": True,
        "generation_lock_frozen": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "generic_calibration_method_novelty_claimed": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "finite_sample_coverage_claimed": False,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "model_update_authorized": False,
        "expert_update_authorized": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + (
    "CLASSIFIER",
)
