"""Frozen scientific and execution protocol for the terminal-only CBPUPR v3 repair."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    BLOCKED_CONTROL_METHOD_ID,
    CANONICAL_PHYSICAL_ROW_ORDER,
    CANONICAL_POSTERIOR_ROW_ORDER,
    CANDIDATE_ONLY_METHOD_ID,
    CENTERS,
    CLAIM_ROLE,
    CLAIM_SCOPE,
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    EXECUTION_REVISION,
    EXECUTION_SCHEMA_REVISION,
    EXPERIMENT_ID,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    FIXED_CONTROL_MENU,
    FAILED_V2_EXPERIMENT_ID,
    FAILED_V2_OUTPUT_ARTIFACT_ID,
    FAILED_V2_SCRATCH_ROOT,
    MIN_SUPPORTED_DONOR_CENTER_COUNT,
    OBSERVED_MAX_CONTROL_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    REPAIR_BASE_COMMIT,
    REPAIR_CODE_IDENTITY,
    TERMINAL_DECISION,
    TARGET_POSTERIOR_TOLERANCE,
    TARGET_POSTERIOR_PROBABILITY_CLIP,
    UTILITY_ZERO_TOLERANCE,
    TRANSPORT_MAD_SCALE,
    TRANSPORT_SCALE_FLOOR,
    TRANSPORT_MIN_REFERENCE_CENTER_COUNT,
    UTILITY_RESPONSE_IDS,
    V1_FAILURE_EXCEPTION,
    V1_FAILURE_PHASE,
    V2_FAILURE_EXCEPTION,
    V2_FAILURE_PHASE,
)
from .hashing import canonical_hash
from .source_seal import source_seal_identity


PROTOCOL_SCHEMA_VERSION = "fixed_bank_cbpupr_protocol_v3"


def frozen_protocol_payload() -> dict[str, object]:
    """Return a new JSON-native copy of the immutable protocol contract."""

    source_identity = source_seal_identity()
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "execution_schema_revision": EXECUTION_SCHEMA_REVISION,
        "repair_code_identity": REPAIR_CODE_IDENTITY,
        "repair_base_commit": REPAIR_BASE_COMMIT,
        "repair_source_manifest_member": source_identity[
            "repair_source_manifest_member"
        ],
        "repair_source_manifest_sha256": source_identity[
            "repair_source_manifest_sha256"
        ],
        "repair_source_tree_sha256": source_identity[
            "repair_source_tree_sha256"
        ],
        "repair_source_member_count": source_identity[
            "repair_source_member_count"
        ],
        "repair_source_manifest_checked_pre_gpu": True,
        "repair_source_identity_persisted_in_protocol_manifest": True,
        "mechanical_repair_only": True,
        "scientific_protocol_unchanged_from_v1": True,
        "scientific_protocol_unchanged_from_v2": True,
        "scientific_method_changed_from_v1": False,
        "scientific_method_changed_from_v2": False,
        "canonical_row_order_repair_only": False,
        "canonical_row_order_repair_verified": True,
        "canonical_row_order_repair_from_v1_retained": True,
        "global_surface_lineage_repair_only": True,
        "global_surface_lineage_repair_verified": True,
        "outer_endpoint_job_carries_global_physical_surface_hash": True,
        "outer_endpoint_job_carries_center_surface_hash": True,
        "global_plan_hash_compared_only_to_global_job_hash": True,
        "center_surface_hash_compared_only_to_prepared_center_hash": True,
        "quarantined_v1_experiment_id": QUARANTINED_V1_EXPERIMENT_ID,
        "quarantined_v1_output_artifact_id": QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
        "v1_output_quarantined": True,
        "v1_failure_preterminal": False,
        "v1_failure_phase": V1_FAILURE_PHASE,
        "v1_failure_exception": V1_FAILURE_EXCEPTION,
        "v1_target_terminal_capability_had_opened": True,
        "v1_terminal_outputs_had_persisted": True,
        "v1_final_validation_passed": False,
        "quarantined_v1_output_used": False,
        "quarantined_v1_scratch_or_checkpoint_used": False,
        "quarantined_v1_terminal_outputs_used": False,
        "prior_v1_label_capability_history_used": False,
        "prior_v1_amendment_used": False,
        "failed_v2_experiment_id": FAILED_V2_EXPERIMENT_ID,
        "failed_v2_output_artifact_id": FAILED_V2_OUTPUT_ARTIFACT_ID,
        "failed_v2_scratch_root": FAILED_V2_SCRATCH_ROOT,
        "v2_failure_preterminal": True,
        "v2_failure_phase": V2_FAILURE_PHASE,
        "v2_failure_exception": V2_FAILURE_EXCEPTION,
        "v2_target_terminal_access_intent_persisted": False,
        "v2_target_terminal_capability_had_opened": False,
        "v2_terminal_outputs_had_persisted": False,
        "v2_final_validation_passed": False,
        "failed_v2_output_used": False,
        "failed_v2_scratch_or_checkpoint_used": False,
        "failed_v2_preterminal_outputs_used": False,
        "prior_v2_label_capability_history_used": False,
        "prior_v2_amendment_used": False,
        "prior_v2_execution_authorization_reused": False,
        "dataset_family": "MIDOG++",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "bounded_interpretation": (
            "center_balanced_posterior_expected_utility_prefix_sensitivity_"
            "on_consumed_MIDOGpp_test_only"
        ),
        "method_development_is_posthoc": True,
        "prior_consumed_test_findings_informed_method_design": True,
        "prior_consumed_test_bytes_used_as_scientific_inputs": False,
        "preexisting_v2_semantic_artifacts_used": False,
        "fresh_v3_workspace_aliases_required": True,
        "physical_probability_surface_recomputed_from_original_inputs": True,
        "previous_prediction_surfaces_used": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
        "stage50_stage60_or_stage70_result_used": False,
        "centers": list(CENTERS),
        "held_unit": "one_whole_case_or_group_c_within_target_center_H",
        "held_unit_count": EXPECTED_OUTER_PLAN_COUNT,
        "outer_route_count": EXPECTED_OUTER_PLAN_COUNT,
        "ordered_H_J_pair_count": EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
        "outer_support_scope": "H_minus_c",
        "pseudo_route_scope": (
            "J_minus_d_posterior_reuse_with_outer_H_role_exclusion_not_"
            "covariate_exclusion"
        ),
        "donor_calibration_scope": "centers_not_in_H_or_J",
        "target_support_labels_used": True,
        "target_support_labels_are_non_deployable_consumed_test_support": True,
        "target_support_labels_may_update_source_experts_or_shared_models": False,
        "outer_case_labels_enter_own_route": False,
        "pseudo_case_labels_enter_own_candidate": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "source_experts_frozen": True,
        "generation_lock_frozen": True,
        "target_evaluation_labels_used_before_route_seal": False,
        "terminal_labels_from_this_run_used_to_define_policy": False,
        "all_target_and_pseudo_candidates_sealed_before_pseudo_evaluation": True,
        "all_replays_and_calibrations_sealed_before_target_decisions": True,
        "all_target_decisions_and_aggregate_predictions_sealed_before_terminal_labels": True,
        "canonical_physical_row_order": CANONICAL_PHYSICAL_ROW_ORDER,
        "posterior_prediction_and_model_row_order": CANONICAL_POSTERIOR_ROW_ORDER,
        "raw_source_store_permutations_canonicalized_before_worker_transfer": True,
        "preterminal_closed_world_validation_required": True,
        "preterminal_parent_validation_required": True,
        "preterminal_fresh_process_validation_count": 2,
        "preterminal_validation_attested_before_terminal_labels": True,
        "terminal_label_loader_forbidden_before_preterminal_attestation": True,
        "final_fresh_process_validation_count": 2,
        "phase_chain": [
            "PhysicalSeal",
            "OuterAndDoubleExclusionPlanSeal",
            "LegalSupportCapabilityGrants",
            "TargetAndPseudoCandidateSeal",
            "PreEvaluationSeal",
            "PseudoEvaluationGrant",
            "PseudoReplayAndCalibrationSeal",
            "DecisionAndAggregateSeal",
            "PreterminalClosedWorldSeal",
            "PreterminalParentValidation",
            "PreterminalFreshProcessValidationX2",
            "PreterminalValidationAttestation",
            "TargetTerminalLabelGrant",
        ],
        "endpoint_methods": list(ENDPOINT_METHOD_IDS),
        "protected_fallback_method_id": PORTFOLIO_METHOD_ID,
        "exact_P_fallback_required": True,
        "exact_P_fallback_storage_contract": "byte_for_byte_float32_identity",
        "primary_method_id": PRIMARY_METHOD_ID,
        "fixed_control_menu": list(FIXED_CONTROL_MENU),
        "composed_policy_ids": list(COMPOSED_POLICY_IDS),
        "candidate_only_control_id": CANDIDATE_ONLY_METHOD_ID,
        "observed_max_control_id": OBSERVED_MAX_CONTROL_METHOD_ID,
        "cyclic_fingerprint_control_id": BLOCKED_CONTROL_METHOD_ID,
        "posterior_expected_utility_coordinates": list(UTILITY_RESPONSE_IDS),
        "posterior_expected_utility_uses_posterior_augmented_center_denominators": True,
        "target_posterior_fit_once_per_target_case_control": True,
        "target_posterior_shared_across_distinct_target_case_routes": False,
        "target_posterior_referenced_by_outer_H_pseudo_wrappers": True,
        "pseudo_outer_H_support_rows_or_labels_enter_posterior_fit_or_normalization": False,
        "pseudo_outer_H_frozen_label_free_expert_fingerprint_covariates_present": True,
        "pseudo_posterior_is_outer_H_covariate_invariant": False,
        "pseudo_outer_H_excluded_from_actionable_endpoint_source_selection": True,
        "pseudo_outer_H_and_J_excluded_from_donor_calibration": True,
        "target_posterior_tolerance": TARGET_POSTERIOR_TOLERANCE,
        "target_posterior_probability_clip": TARGET_POSTERIOR_PROBABILITY_CLIP,
        "utility_eligibility_zero_tolerance": UTILITY_ZERO_TOLERANCE,
        "prefix_feasibility_zero_tolerance": UTILITY_ZERO_TOLERANCE,
        "candidate_action_rule": (
            "proper_safe_then_maximum_expected_BACC_then_B_I_R_then_action_hash"
        ),
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
        "numeric_transport_is_authorization_gate": False,
        "structural_transport_lineage_is_authorization_gate": True,
        "zero_MAD_numeric_transport_division_forbidden": True,
        "numeric_transport_MAD_scale": TRANSPORT_MAD_SCALE,
        "numeric_transport_zero_scale_threshold": TRANSPORT_SCALE_FLOOR,
        "numeric_transport_minimum_reference_centers": (
            TRANSPORT_MIN_REFERENCE_CENTER_COUNT
        ),
        "finite_sample_conformal_coverage_claimed": False,
        "confidence_bound_claimed": False,
        "calibrated_uncertainty_claimed": False,
        "nominal_significance_claimed": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "expert_selection_claimed": False,
        "deployment_claimed": False,
        "promotion_eligible": False,
        "may_authorize_routing": False,
        "may_authorize_policy_update": False,
        "may_authorize_promotion": False,
        "may_authorize_deployment": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "generic_consumer_authorized": False,
        "raw_labels_may_be_persisted": False,
        "raw_sample_or_image_paths_may_be_persisted": False,
        "scratch_reuse_forbidden": True,
        "cross_run_recovery_forbidden": True,
        "two_fresh_process_validation_required": True,
        "full_fitted_endpoint_state_DTOs_persisted": True,
        "full_fitted_posterior_model_DTOs_persisted": True,
        "all_fitted_DTO_outputs_replayed_during_validation": True,
        "optimizer_refits_during_bundle_validation": False,
        "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
    }


FROZEN_PROTOCOL_HASH = canonical_hash(frozen_protocol_payload())
FROZEN_PROTOCOL: Mapping[str, object] = MappingProxyType(frozen_protocol_payload())


def validate_frozen_protocol(payload: Mapping[str, object]) -> None:
    """Fail closed if any executable protocol field drifts."""

    if dict(payload) != frozen_protocol_payload():
        raise ProtocolError("CBPUPR frozen protocol drifted.")
    if canonical_hash(payload) != FROZEN_PROTOCOL_HASH:
        raise ProtocolError("CBPUPR protocol hash drifted.")


__all__ = (
    "FROZEN_PROTOCOL",
    "FROZEN_PROTOCOL_HASH",
    "PROTOCOL_SCHEMA_VERSION",
    "frozen_protocol_payload",
    "validate_frozen_protocol",
)
