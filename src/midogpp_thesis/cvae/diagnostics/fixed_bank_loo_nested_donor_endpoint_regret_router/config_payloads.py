"""Canonical executable config sections for the nested-regret diagnostic."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .constants import (
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    B_ROWS_PER_SOURCE_CLASS,
    CENTERS,
    CLAIM_ROLE,
    CPU_WORKERS,
    EXPECTED_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_ORDERED_VOTER_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_UNORDERED_PAIR_COUNT,
    LOG_LOSS_CLIP_EPSILON,
    PUBLICATION_STATUS,
    SCRATCH_ROOT,
    TERMINAL_DECISION,
    U_ROWS_PER_SOURCE_CLASS,
    WORKSTATION_PROFILE,
)
from .controls import predeclared_policy_menu
from .protocol import frozen_protocol_payload


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
        "schema_version": "fixed_bank_nested_regret_action_library_v1",
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
        "previous_probability_surface_used": False,
    }


def canonical_policy_menu_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_nested_regret_policy_menu_v1",
        "policies": [row.to_payload() for row in predeclared_policy_menu()],
        "primary_policy_id": "NDR_MODEL_BASED",
        "center_block_feasibility_method_id": "NDR_CENTER_BLOCK_FEASIBILITY",
        "protected_fallback": "P_PROTECTED",
        "policy_selected_from_terminal_utility": False,
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
        "switch_attribution": "sample_threshold_crossings_helpful_or_harmful",
        "oracle_diagnostics": [
            "endpoint_top1_agreement_case_weighted_and_equal_center",
            "endpoint_rank_case_weighted_and_equal_center",
            "normalized_endpoint_oracle_gap_case_weighted_and_equal_center",
            "candidate_regret_spearman",
        ],
        "selection_control": (
            "exact_2_power_9_center_sign_flip_max_over_presealed_"
            "fixed_decision_policy_menu"
        ),
        "policy_identity_reselected_inside_every_null_replicate": True,
        "route_pipeline_refit_inside_null_replicate": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "raw_labels_persisted": False,
        "case_rows_are_not_independent_inference_units": True,
        "calibration_validated": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_nested_regret_workstation_runtime_v1",
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
        "classifier_workers": CPU_WORKERS,
        "route_model_workers": CPU_WORKERS,
        "classifier_threads_per_worker": 3,
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
        "unordered_pair_state_count": EXPECTED_UNORDERED_PAIR_COUNT,
        "ordered_voter_count": EXPECTED_ORDERED_VOTER_COUNT,
        "expected_endpoint_model_fit_count": EXPECTED_ENDPOINT_MODEL_FIT_COUNT,
        "unordered_pair_state_reused_for_both_ordered_voters": True,
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
        "bounded_interpretation": "center_balanced_outer_center_excluded_nested_donor_regret_sensitivity_only",
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "support_multiplier_selected_on_same_evaluation_surface": True,
        "route_decision_label_blind": False,
        "protected_fallback_label_blind": False,
        "fresh_evidence": False,
        "terminal_stage90_diagnostic": True,
        "thesis_specific_integration_novelty_only": True,
        "generic_conformal_method_novelty_claimed": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "downstream_utility_claimed": False,
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
        "previous_prediction_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + (
    "CLASSIFIER",
)
