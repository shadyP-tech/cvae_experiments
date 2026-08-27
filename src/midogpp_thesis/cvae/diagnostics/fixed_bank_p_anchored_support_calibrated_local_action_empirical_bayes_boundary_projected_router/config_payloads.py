"""Frozen scientific, workstation, and claim payloads for SCALE-BP v1."""

from __future__ import annotations

from .controls import METHOD_IDS, REQUIRED_CONTROL_METHOD_IDS
from .experiment_contracts import CANONICAL_SCRATCH_ROOT
from .identity import (
    ACTION_FAMILIES,
    CLAIM_SCOPE,
    DIRECTIONS,
    METRICS,
    METHOD_MENU,
    MAXIMUM_HARMFUL_SELECTED_POLICY_COUNT,
    MAXIMUM_NORMALIZED_ORACLE_GAP,
    MINIMUM_OPPORTUNITY_CASES,
    MINIMUM_REPRESENTED_CENTERS,
    MINIMUM_WITHIN_CASE_SPEARMAN,
    PUBLICATION_STATUS,
    RIDGE_ALPHA,
    SUPPORT_FOLD_COUNT,
    TERMINAL_DECISION,
    TIE_TOLERANCE,
    WORKSPACE_STATUS,
)


def action_geometry_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_action_geometry_v1",
        "anchor": "P",
        "families": list(ACTION_FAMILIES),
        "directions": list(DIRECTIONS),
        "crossing_threshold": 0.5,
        "primary_projection": "NEAREST_FLOAT32_ON_REQUIRED_SIDE_OF_HALF",
        "off_crossing_probabilities": "BYTE_EXACT_P",
        "no_crossing_case_result": "BYTE_EXACT_P",
        "full_endpoint_role": "SENSITIVITY_ONLY",
        "physical_probability_cell_count": 810,
        "endpoint_derivation_rules": {
            "B": "EXACT_NINE_MEAN_B",
            "I": "DIRECTIONAL_ROW_EXTREME_OF_EIGHT_TARGET_EXCLUDED_A1_MEANS",
            "R": "ROW_MEDIAN_OF_U_AND_EIGHT_TARGET_EXCLUDED_A1_MEANS",
        },
        "endpoint_surface_receipt_required": True,
        "endpoint_receipt_construction": "REDERIVE_FROM_EXACT_90_PHYSICAL_CELLS",
        "arbitrary_endpoint_probability_constructor_allowed": False,
        "physical_surface_construction": "FACTORY_ISSUED_FROM_READ_ONLY_MEMMAP_SLICE",
        "physical_bank_receipt_required": True,
        "physical_bank_receipt_cell_count": 810,
        "physical_cell_to_file_offset_slice_and_row_identity_mapping_required": True,
        "physical_bank_receipt_hash_propagated_to_endpoint_evidence": True,
        "physical_surface_memmap_reference_slice_and_row_index_hash_required": True,
        "stored_dtype": "float32",
        "scientific_reduction_dtype": "float64",
    }


def support_folds_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_support_folds_v1",
        "support_definition": "H_MINUS_C",
        "fold_count": SUPPORT_FOLD_COUNT,
        "assignment": "DETERMINISTIC_WHOLE_CASE_HASH_BALANCED",
        "patient_slide_case_disjoint": True,
        "route_identity_inventory_source": "EXACT_MANIFEST_SAMPLE_KEYS",
        "support_and_evaluation_sample_key_hashes_required": True,
        "route_scope_witness_required": True,
        "scored_fold_excluded_from_own_fit": True,
        "held_case_c_excluded_from_all_folds": True,
        "minimum_nonempty_training_folds": 2,
        "fold_failure_result": "BYTE_EXACT_P",
    }


def influence_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_sample_influence_v1",
        "metrics": list(METRICS),
        "unit": "CROSSING_SAMPLE_THEN_CASE_ACTION_AGGREGATE",
        "bacc_denominator_source": "SUPPORT_ONLY_WITH_PREDECLARED_HELD_MASS_ESTIMATE",
        "true_whole_H_denominators_forbidden": True,
        "descriptors": [
            "p_threshold_margin",
            "action_threshold_margin",
            "endpoint_displacement",
            "posterior_eta",
            "posterior_uncertainty",
            "entropy",
            "seed_sd",
            "positive_vote_fraction",
            "crossing_family",
            "crossing_direction",
            "case_size",
            "support_denominator_geometry",
            "bank_effective_sample_size",
        ],
        "nonfinite_result": "BYTE_EXACT_P",
    }


def donor_prior_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_donor_prior_v1",
        "fit_centers": "J_NOT_EQUAL_H",
        "final_training_centers": "EXACT_ALL_CENTERS_EXCEPT_H",
        "pseudo_training_centers": "EXACT_ALL_CENTERS_EXCEPT_H_AND_J",
        "canonical_218_case_inventory_required": True,
        "pseudo_target_and_candidate_exclusions_required": True,
        "ridge_alpha": RIDGE_ALPHA,
        "equal_center_weighting": True,
        "shared_hyperparameter_source": "DONOR_ONLY_FIXED",
        "target_support_updates_global_coefficients": False,
        "target_support_updates_scalers": False,
    }


def local_residual_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_local_residual_v1",
        "fit_scope": "ROUTE_LOCAL_H_C_ONLY",
        "crossfit_folds": SUPPORT_FOLD_COUNT,
        "residual_target": "REALIZED_MINUS_DONOR_PREDICTED_SAMPLE_INFLUENCE",
        "own_fold_label_influence": False,
        "held_case_label_influence": False,
        "may_update_global_state": False,
        "insufficient_support_result": "BYTE_EXACT_P",
    }


def empirical_bayes_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_empirical_bayes_v1",
        "formula": "DONOR_PRIOR_PLUS_WEIGHT_TIMES_LOCAL_OOF_RESIDUAL",
        "between_center_variance_source": "DONOR_ONLY",
        "local_standard_error_source": "ROUTE_LOCAL_OOF_SUPPORT_ONLY",
        "weight_interval": [0.0, 1.0],
        "support_labels_tune_hyperparameters": False,
        "degenerate_variance_result": "DONOR_ONLY_OR_BYTE_EXACT_P",
    }


def uncertainty_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_uncertainty_v1",
        "envelope": "SELECTION_AWARE_MAX_OOF_RESIDUAL",
        "computed_before_argmax": True,
        "bacc_lower_bound_must_be_positive": True,
        "brier_and_log_must_be_nonworsening": True,
        "delete_center_sensitivity_required": True,
        "descriptive_only": True,
        "confidence_bound_claimed": False,
        "conformal_claimed": False,
    }


def selection_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_selection_v1",
        "method_menu": list(METHOD_MENU),
        "selection_unit": "DIRECT_CASE_ACTION",
        "learned_prefix_layer": False,
        "at_most_one_family_per_direction": True,
        "joint_composition_requires_all_metric_safety": True,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_winner": "P_PROTECTED",
        "any_gate_failure_result": "BYTE_EXACT_P",
    }


def admission_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_pseudo_case_admission_v1",
        "unit": "ALL_CANONICAL_PSEUDO_CASES_AND_ALL_ACTIONS",
        "sealed_expected_replay_inventory_required": True,
        "omitted_pseudo_contexts_forbidden": True,
        "full_algorithm_replay_required": True,
        "caller_supplied_metric_or_oracle_rows_allowed": False,
        "factory_sealed_case_replay_results_required": True,
        "factory_sealed_outer_evidence_bundle_required": True,
        "factory_sealed_all_outer_evidence_bundle_required": True,
        "all_outer_H_replay_required": True,
        "expected_outer_H_count": 9,
        "expected_H_J_d_context_count": 1744,
        "every_outer_H_must_pass_before_final_routing": True,
        "terminal_denominator_source": (
            "EXACT_FACTORY_SEALED_PSEUDO_CENTER_LABEL_POPULATION"
        ),
        "caller_supplied_terminal_denominators_allowed": False,
        "same_pseudo_center_label_population_across_all_H_and_d_required": True,
        "primary_and_support_permutation_invoke_exact_case_route_engine": True,
        "oracle_derivation": "ENUMERATE_P_SAFE_SINGLES_AND_DISJOINT_PAIRS",
        "input_action_policy_and_oracle_roots_required": True,
        "exclude_outer_H_pseudo_J_and_held_d": True,
        "equal_center_weighting": True,
        "no_opportunity_center_utility": 0.0,
        "primary_diagnostics": [
            "top1_action_oracle_agreement",
            "within_opportunity_spearman",
            "normalized_oracle_gap",
            "center_stability",
        ],
        "positive_equal_center_bacc_required": True,
        "brier_and_log_nonworsening_required": True,
        "legacy_noninferiority_required": True,
        "harm_budget_required": True,
        "controls_must_be_inactive": True,
        "minimum_opportunity_cases": MINIMUM_OPPORTUNITY_CASES,
        "minimum_represented_centers": MINIMUM_REPRESENTED_CENTERS,
        "minimum_within_case_spearman": MINIMUM_WITHIN_CASE_SPEARMAN,
        "maximum_normalized_oracle_gap": MAXIMUM_NORMALIZED_ORACLE_GAP,
        "maximum_harmful_selected_policy_count": (
            MAXIMUM_HARMFUL_SELECTED_POLICY_COUNT
        ),
        "admission_thresholds_caller_overridable": False,
        "failed_learnability_result": "ABORT_BEFORE_TERMINAL_LABELS",
    }


def controls_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_controls_v1",
        "required_methods": list(METHOD_IDS),
        "required_control_methods": list(REQUIRED_CONTROL_METHOD_IDS),
        "support_permutation_block": "WHOLE_CASE_WITHIN_ROUTE_LOCAL_SUPPORT",
        "control_identity_hash_isolated": True,
        "controls_may_not_authorize_primary": True,
    }


def workstation_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_planned_workstation_runtime_v1",
        "workspace_status": WORKSPACE_STATUS,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "direct_runner_rejects_before_mutation": True,
        "workspace_runner_rejects_before_mutation": True,
        "output_root_creation_allowed": False,
        "scratch_root_creation_allowed": False,
        "lock_creation_allowed": False,
        "cross_run_recovery_allowed": False,
        "source_manifest_required": True,
        "source_manifest_validated_during_config_load": True,
        "source_manifest_checked_before_any_gpu_or_label_access": True,
        "generation_device_ids": ["cuda:0", "cuda:1"],
        "persistent_generation_worker_count": 2,
        "physical_cells_materialized_once": 810,
        "physical_store": "FLOAT32_READ_ONLY_MEMMAP",
        "physical_bank_receipt": "FACTORY_ISSUED_EXACT_810_CELL_MAPPING",
        "physical_bank_root_and_member_symlinks_forbidden": True,
        "physical_bank_out_of_root_paths_forbidden": True,
        "physical_bank_missing_duplicate_or_overlapping_cells_forbidden": True,
        "outer_tasks_require_one_shared_physical_bank_receipt": True,
        "physical_surface_factory_requires_validated_memmap_slice": True,
        "memmap_semantic_role_byte_extent_row_index_and_hash_required": True,
        "memmap_loader_mode": "READ_ONLY_VALIDATED_SLICE",
        "cpu_phase_cuda_visible_devices": "",
        "outer_worker_start_method": "spawn",
        "outer_cpu_worker_count": 4,
        "outer_worker_task_unit": "ONE_COMPLETE_OUTER_H",
        "final_route_inventory_receipt_required": True,
        "final_route_inventory_center_count": 9,
        "final_route_inventory_case_count": 218,
        "partial_or_reordered_task_and_result_universe_allowed": False,
        "support_folds_inside_outer_worker": "SEQUENTIAL",
        "method_replay_inside_outer_worker": "SEQUENTIAL_SHARED_PRIMARY_KERNELS",
        "support_permutation_materialization": "IN_MEMORY_AGGREGATE_RECORD_ROTATION",
        "terminal_label_request_repr_and_pickle_forbidden": True,
        "durable_replay_output": "AGGREGATE_HASHED_EVIDENCE_ONLY",
        "blas_threads_per_outer_worker": 1,
        "nested_process_pools_forbidden": True,
        "worker_payload": "PRIMITIVE_FROZEN_DTOS_HASHES_AND_OFFSETS_ONLY",
        "mappingproxy_across_process_boundary_forbidden": True,
        "estimator_across_process_boundary_forbidden": True,
        "open_handle_or_memmap_object_across_process_boundary_forbidden": True,
        "atomic_outer_center_chunks": True,
        "scratch_root": CANONICAL_SCRATCH_ROOT,
    }


def claim_boundary_payload() -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v1_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "bounded_interpretation": (
            "planned_support_calibrated_local_action_diagnostic_on_consumed_"
            "MIDOGpp_test_only"
        ),
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "target_terminal_labels_may_open": False,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "confidence_bound_claimed": False,
        "finite_sample_coverage_claimed": False,
        "promotion_eligible": False,
        "deployment_claimed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def scientific_payloads() -> dict[str, dict[str, object]]:
    """Return all path-independent method sections in canonical order."""

    return {
        "action_geometry": action_geometry_payload(),
        "support_folds": support_folds_payload(),
        "influence": influence_payload(),
        "donor_prior": donor_prior_payload(),
        "local_residual": local_residual_payload(),
        "empirical_bayes": empirical_bayes_payload(),
        "uncertainty": uncertainty_payload(),
        "selection": selection_payload(),
        "admission": admission_payload(),
        "controls": controls_payload(),
    }


__all__ = (
    "action_geometry_payload",
    "admission_payload",
    "claim_boundary_payload",
    "controls_payload",
    "donor_prior_payload",
    "empirical_bayes_payload",
    "influence_payload",
    "local_residual_payload",
    "scientific_payloads",
    "selection_payload",
    "support_folds_payload",
    "uncertainty_payload",
    "workstation_payload",
)
