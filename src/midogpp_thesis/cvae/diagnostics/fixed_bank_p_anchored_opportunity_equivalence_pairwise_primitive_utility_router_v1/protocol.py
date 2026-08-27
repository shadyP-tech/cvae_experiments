"""Frozen leakage firewall for the planned OE-PPUR v1 diagnostic."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from .contracts import claim_boundary_payload, direct_input_policy_payload
from .hashing import canonical_hash
from .identity import (
    ACTION_IDS,
    CENTERS,
    EXCLUDED_CENTERS,
    EXPERIMENT_ID,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    METRICS,
    PRIMITIVES,
)
from .manifest_contract import canonical_terminal_manifest_contract_payload


def _protocol_body() -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v1_terminal_protocol_v3",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "held_case_route_count": EXPECTED_CASE_COUNT,
        "eligible_center_ids": list(CENTERS),
        "excluded_center_ids": list(EXCLUDED_CENTERS),
        "held_unit": "whole_case_patient_slide_group",
        "scope_roles": {
            "H": "FINAL_TARGET_CENTER",
            "J": "PSEUDO_TARGET_CENTER",
            "K": "HYPERPARAMETER_VALIDATION_ONLY_CENTER",
            "L": "RESIDUAL_CALIBRATION_ONLY_CENTER",
            "d": "WHOLE_HELD_PSEUDO_TARGET_CASE",
        },
        "FoldScope_role": "PSEUDO_AND_NESTED_H_J_K_L_d_CONTEXT_ONLY",
        "FinalOuterScope_role": "FINAL_H_ONLY_AFTER_SOURCE_CHOICES_FROZEN",
        "H_J_K_L_pairwise_distinct": True,
        "H_J_K_L_excluded_from_estimator_fit": True,
        "nested_scopes_share_one_outer_H": True,
        "K_may_only_score_hyperparameters": True,
        "nested_K_rotation_centers": "EXACT_C_MINUS_H",
        "nested_K_rotation_complete_once_per_source_center": True,
        "L_may_only_calibrate_residuals": True,
        "d_identity": "EXPLICIT_CENTER_AND_WHOLE_CASE_TUPLE_WITH_AUDIT_HASH",
        "d_excluded_from_nested_posterior_ranker_validation_and_residual_calibration": True,
        "d_recovered_in_legal_final_source_refit": True,
        "final_nested_K_L_choices_source_only_and_hash_frozen": True,
        "final_estimator_fit_centers": "EXACT_C_MINUS_H",
        "final_H_excluded_from_every_learner_normalizer_calibrator_and_candidate_pool": True,
        "target_H_labels_closed_preterminal": True,
        "target_support_labels_for_final_routing": False,
        "terminal_labels_transient_nonserializable": True,
        "source_only_row_posterior": True,
        "target_identity_feature_forbidden": True,
        "candidate_action_ids": list(ACTION_IDS),
        "candidate_pool_receipt_type": "CandidatePoolReceipt",
        "candidate_pool_receipt_required_at_pairwise_fit_and_selection": True,
        "candidate_pool_inventory": "EXACT_ONE_FROZEN_EXPERT_PER_C_MINUS_H_CENTER",
        "protected_fallback": "P_PROTECTED_BYTE_EXACT",
        "structural_noops_excluded_from_fit_and_ranking": True,
        "identical_projected_surfaces_collapsed": True,
        "opportunity_receipt_type": "OpportunityCaseReceipt",
        "typed_opportunity_receipt_required_at_pairwise_fit_and_selection": True,
        "opportunity_receipt_embeds_typed_canonical_opportunity_set": True,
        "opportunity_candidate_action_inventory": "EXACT_FROZEN_CANDIDATE_ACTION_IDS",
        "opportunity_receipt_is_label_free_and_predecision": True,
        "primitive_targets": list(PRIMITIVES),
        "primitive_computation_precedes_metric_normalization": True,
        "primitive_expected_label_probability_type": "RowPosteriorPrediction",
        "typed_row_posterior_prediction_required_for_primitive_and_denominator": True,
        "primitive_action_id_exact_match_to_opportunity_member": True,
        "primitive_protected_baseline_probability_hash_exact_match_to_opportunity": True,
        "primitive_candidate_probability_hash_exact_match_to_opportunity_member": True,
        "normalization_policy": "ACTION_INVARIANT_EXPECTED_CLASS_TOTALS_PER_CASE_SCOPE",
        "primitive_denominator_row_count_exact_match": True,
        "primitive_denominator_scope_id_exact_match": True,
        "primitive_denominator_row_manifest_hash_exact_match": True,
        "primitive_denominator_posterior_model_hash_exact_match": True,
        "primitive_denominator_posterior_receipt_hash_exact_match": True,
        "pairwise_fit_response_type": "NormalizedUtility",
        "pairwise_fit_response_metric": "EXPECTED_BACC_GAIN_ONLY",
        "pairwise_fit_exact_matches_utility_action_and_probability_surface_to_opportunity": True,
        "brier_and_log_roles": "SELECTION_SAFETY_AND_TERMINAL_DIAGNOSTIC_ONLY",
        "brier_or_log_may_enter_pairwise_ranking_response": False,
        "reported_metrics": list(METRICS),
        "ranking_target": "ANTISYMMETRIC_PAIRWISE_EXPECTED_BACC_GAIN_CONTRAST",
        "absolute_action_value_ranking_forbidden": True,
        "action_specific_family_direction_interactions_required": True,
        "nested_complete_K_rotation_hyperparameter_selection_required": True,
        "rotating_L_center_case_OOF_residual_calibration_required": True,
        "residual_calibration_one_sided_alpha": 0.2,
        "residual_calibration_minimum_distinct_L_centers": 4,
        "infeasible_residual_quantile_result": "UNCERTAINTY_UNAVAILABLE_AND_EXACT_P",
        "global_pooled_uncertainty_fallback_allowed": False,
        "uncertainty_action_specific": True,
        "uncalibrated_descriptive_bound_may_not_authorize_selection": True,
        "prelabel_selection_decision_type": "SelectionDecision",
        "prelabel_selection_decision_hash_frozen_before_terminal_label_open": True,
        "selection_ledger_entry_opportunity_receipt_type": "OpportunityCaseReceipt",
        "selection_ledger_entry_requires_typed_opportunity_receipt": True,
        "selection_ledger_entry_exact_matches_opportunity_receipt_center_and_case": True,
        "selection_ledger_entry_exact_matches_decision_opportunity_receipt_hash": True,
        "prelabel_selection_decision_ledger_type": "SelectionDecisionLedger",
        "prelabel_selection_decision_ledger_case_count": EXPECTED_CASE_COUNT,
        "prelabel_selection_decision_ledger_requires_exact_dataset_case_manifest_hash": True,
        "canonical_terminal_manifest_contract": (
            canonical_terminal_manifest_contract_payload()
        ),
        "canonical_terminal_manifest_receipt_required_by_selection_ledger": True,
        "canonical_terminal_manifest_receipt_hash_bound_in_preterminal_phase": True,
        "terminal_label_gate_requires_canonical_terminal_manifest_receipt": True,
        "preterminal_input_lineage_type": "PreterminalInputLineage",
        "preterminal_raw_hash_constructor_allowed": False,
        "promoted_bank_validation_receipt_required": True,
        "frozen_generation_lock_receipt_required": True,
        "short_semantic_lock_ids_distinct_from_full_sha256_file_pins": True,
        "candidate_probability_surface_receipt_type": (
            "CandidateProbabilitySurfaceReceipt"
        ),
        "candidate_probability_surface_receipt_scope": (
            "GPU_TO_CPU_TRANSPORT_BYTES_AND_CANONICAL_ROWS_ONLY"
        ),
        "cpu_outer_probability_input_hash_must_be_gpu_output_hash": True,
        "parsed_probability_matrix_science_receipt_implemented": False,
        "outer_selection_lineage_type": "OuterSelectionLineage",
        "outer_selection_lineage_inventory": "EXACT_ONE_PER_ELIGIBLE_H",
        "each_case_decision_must_match_its_H_specific_source_pool_model_and_calibration": True,
        "preterminal_phase_uses_canonical_per_H_lineage_surface_hashes": True,
        "preterminal_phase_receipt_type": "PreterminalPhaseReceipt",
        "preterminal_phase_direct_construction_allowed": False,
        (
            "preterminal_phase_binds_config_protocol_source_candidate_model_"
            "calibration_opportunity_hashes"
        ): True,
        "terminal_label_capability_requires_exact_complete_decision_ledger_hash": True,
        "terminal_label_capability_openable_in_current_contract": False,
        "terminal_admission_joins_exact_selection_decision_lineage": True,
        "selection_exact_matches_utility_action_and_probability_surface_to_opportunity": True,
        "terminal_admission_aggregation": "CASE_THEN_EQUAL_CENTER",
        "zero_active_action_cases_in_admission_coverage_denominator": True,
        "micro_pooled_terminal_admission_forbidden": True,
        "exact_P_on_any_failed_gate": True,
        "route_policy_proxy_role": "DOWNSTREAM_ACTION_SELECTION_DIAGNOSTIC_PROXY",
        "route_policy_proxy_is_true_utility": False,
        "route_policy_proxy_is_cvae_compatibility": False,
        "route_policy_proxy_is_nelbo_compatibility": False,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "output_or_scratch_resolution_allowed": False,
        "cross_run_recovery_allowed": False,
        "predecessor_output_artifact_scratch_lease_or_report_recovery_allowed": False,
        "combined_diagnostic_adapter_and_neutral_core_source_seal_required": True,
        "combined_source_seal_recomputed_and_exact_matched_on_config_load": True,
        "source_fence_checked_before_execution_rejection": True,
        "publication_and_claim_boundary": claim_boundary_payload(),
        "direct_input_policy": direct_input_policy_payload(),
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def frozen_protocol_payload() -> dict[str, object]:
    body = _protocol_body()
    return {**body, "protocol_hash": canonical_hash(body)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if dict(payload) != frozen_protocol_payload():
        raise ProtocolError("OE-PPUR protocol contract drifted.")


__all__ = ("ProtocolError", "frozen_protocol_payload", "validate_protocol_payload")
