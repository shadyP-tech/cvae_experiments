"""Canonical executable configuration sections for PCSI-RACR."""

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
    EXPECTED_FINAL_CASE_PREDICTION_COUNT,
    EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT,
    EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT,
    EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT,
    EXPECTED_TRANSPORT_SCREEN_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    LOG_LOSS_CLIP_EPSILON,
    PRIMARY_METHOD_ID,
    PROJECTION_ONE_TO_ZERO_VALUE,
    PROJECTION_ZERO_TO_ONE_VALUE,
    PROJECTED_NO_ENVELOPE_METHOD_ID,
    PUBLICATION_STATUS,
    RAW_OBSERVED_MAX_METHOD_ID,
    SCRATCH_ROOT,
    TERMINAL_DECISION,
    TRANSPORT_SCALE_FLOOR,
    U_ROWS_PER_SOURCE_CLASS,
    WORKSTATION_PROFILE,
)
from .transport import TRANSPORT_PROTOCOL_CONTRACT


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
        "schema_version": "fixed_bank_pcsi_racr_action_library_v1",
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
        "schema_version": "fixed_bank_pcsi_racr_policy_menu_v1",
        "policy_ids": [
            "P_PROTECTED",
            PRIMARY_METHOD_ID,
            RAW_OBSERVED_MAX_METHOD_ID,
            PROJECTED_NO_ENVELOPE_METHOD_ID,
        ],
        "primary_policy_id": PRIMARY_METHOD_ID,
        "raw_whole_pipeline_sensitivity_id": RAW_OBSERVED_MAX_METHOD_ID,
        "projected_no_envelope_sensitivity_id": (
            PROJECTED_NO_ENVELOPE_METHOD_ID
        ),
        "legacy_dual_veto_removed": True,
        "blocked_fingerprint_is_not_a_final_surface": True,
        "protected_fallback": "P_PROTECTED",
        "decision_enum": ["CHANGE", "ABSTAIN_TO_P"],
        "projection": {
            "zero_to_one_value": PROJECTION_ZERO_TO_ONE_VALUE,
            "one_to_zero_value": PROJECTION_ONE_TO_ZERO_VALUE,
            "off_crossing_P_bytes_preserved": True,
            "binary32_exact": True,
        },
        "candidate_gate": (
            "crossing_gt_0_and_target_influence_gt_1e_minus_15_and_"
            "predicted_raw_Brier_le_0_and_predicted_raw_log_loss_le_0"
        ),
        "candidate_tie_order": "exact_max_then_B_I_R_then_action_hash",
        "projected_and_raw_reselect_candidates": True,
        "projected_q0_reuses_projected_candidate_and_transport": True,
        "utility_model": {
            "family": "training_residual_shifted_leave_donor_ridge_ensemble",
            "alpha": 1.0,
            "equal_donor_then_equal_case_then_equal_surviving_class": True,
            "unbiasedness_claimed": False,
            "projected_fit_count": EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY,
            "raw_fit_count": EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY,
        },
        "observed_envelope": {
            "id": "OBSERVED_DONOR_CASE_ENVELOPE",
            "formula": "max_zero_max_all_cases_then_max_all_donor_centers",
            "all_cases_required": True,
            "strict_three_coordinate_gate": True,
            "equality_abstains": True,
            "conformal": False,
            "finite_sample_coverage": False,
            "tail_probability_claimed": False,
        },
        "upper_median_is_unscored_annotation": True,
        "descriptor_match_is_unscored_annotation": True,
        "transport": dict(TRANSPORT_PROTOCOL_CONTRACT),
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_racr_evaluation_v1",
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
        "estimand": "cross_fitted_case_policy_mosaic",
        "deployable_center_policy_claimed": False,
        "case_contributions_add_to_center_metric_difference": True,
        "descriptive_only": True,
        "success_gate_defined": False,
        "nominal_coverage_claimed": False,
        "nominal_significance_claimed": False,
        "raw_labels_persisted": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_racr_workstation_runtime_v1",
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
        "endpoint_threads_per_worker": 3,
        "posterior_utility_replay_workers": CPU_WORKERS,
        "route_model_workers": CPU_WORKERS,
        "classifier_threads_per_worker": 3,
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
        "double_exclusion_pair_count": 72,
        "expected_outer_endpoint_model_fit_count": (
            EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        ),
        "expected_target_posterior_model_fit_count": (
            EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        ),
        "expected_utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
        "expected_projected_utility_model_fit_count": (
            EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY
        ),
        "expected_raw_utility_model_fit_count": (
            EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY
        ),
        "expected_legacy_utility_model_fit_count": 0,
        "expected_role_bound_transport_descriptor_count": (
            EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT
        ),
        "expected_numeric_transport_leaf_count": (
            EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT
        ),
        "expected_transport_reference_summary_count": (
            EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT
        ),
        "expected_transport_screen_count": EXPECTED_TRANSPORT_SCREEN_COUNT,
        "expected_policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT,
        "expected_primary_geometry_decision_count": (
            EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT
        ),
        "expected_projected_no_envelope_decision_count": (
            EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT
        ),
        "expected_final_case_prediction_count": (
            EXPECTED_FINAL_CASE_PREDICTION_COUNT
        ),
        "replay_shard_count": 144,
        "canonical_task_order": (
            "geometry_outer_center_donor_center_case_id_role"
        ),
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": (
            "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
        ),
        "owned_task_checkpoint_replay_allowed": False,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "two_fresh_process_validation_required": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_racr_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_role": CLAIM_ROLE,
        "claim_boundary": "NON_GUARANTEE_CONSUMED_TEST_ONLY",
        "consumed_test_data": True,
        "whole_test_dataset_reused": True,
        "fresh_evidence": False,
        "execution_authorized": True,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "reconstruction_or_fidelity_evidence_claimed": False,
        "generated_embedding_mutated": False,
        "source_expert_updated": False,
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
    }


__all__ = tuple(
    name for name in globals() if name.startswith("canonical_")
) + ("CLASSIFIER",)
