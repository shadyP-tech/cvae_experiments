"""Claim-boundary reports for the consumed-test case-aware diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .experiment_contracts import EXPERIMENT_ID


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    test_cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_proxy_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "split": "test",
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": test_cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "fixed_support_case_count_per_center": 8,
        "support_aggregation_primary": "equal_case",
        "row_weighted_aggregation_role": "control_only",
        "primary_response": "exact_nine_probability_ensemble_bacc_delta",
        "smooth_response": "exact_nine_probability_ensemble_soft_bacc_delta",
        "smooth_response_role": "separately_fit_postseal_diagnostic_only",
        "response_count_per_response": 504,
        "test_labels_construct_postseal_response_rows": True,
        "label_derived_responses_feed_strict_crossfit_diagnostic_models": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "strict_all_role_H_q_e_exclusion": True,
        "support_labels_used": False,
        "development_predictions_sealed_before_labels": True,
        "test_split_previously_consumed": True,
        "repurposed_for_method_development": True,
        "fresh_evidence": False,
        "target_actions_built": False,
        "prior_stage90_output_consumed": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
    }
    return {**unhashed, "protocol_manifest_hash": stable_hash(unhashed)}


def prelabel_phase_payload(
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    development_prediction_seal_hash: str,
    feature_lock_hash: str,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_proxy_prelabel_phase_v1",
        "status": "SEALED_BEFORE_TEST_LABEL_ACCESS",
        "config_contract_hash": config_contract_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "case_aware_feature_lock_hash": feature_lock_hash,
        "candidate_feature_row_count": 504,
        "support_probabilities_only_for_features": True,
        "evaluation_probabilities_used_as_features": False,
        "test_labels_opened": False,
        "support_labels_opened": False,
        "target_actions_built": False,
    }
    return {**unhashed, "phase_hash": stable_hash(unhashed)}


def leakage_report_payload(
    *,
    support_partition_lock_hash: str,
    development_prediction_seal_hash: str,
    feature_lock_hash: str,
    crossfit_fold_lock_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_case_aware_proxy_leakage_report_v1",
        "status": "PASS",
        "support_partition_lock_hash": support_partition_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "case_aware_feature_lock_hash": feature_lock_hash,
        "crossfit_fold_lock_hash": crossfit_fold_lock_hash,
        "crossfit_response_names": ["exact_bacc_delta", "smooth_bacc_delta"],
        "exact_response_is_primary": True,
        "smooth_response_is_diagnostic_only": True,
        "test_labels_construct_postseal_response_rows": True,
        "label_derived_responses_feed_strict_crossfit_diagnostic_models": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "whole_case_support_evaluation_disjoint": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "test_labels_opened_after_global_prediction_seal": True,
        "strict_all_role_H_q_e_exclusion": True,
        "outer_centers_are_inference_units": True,
        "seed_patch_and_case_rows_are_not_inference_units": True,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "target_actions_built": False,
        "previous_stage90_outputs_used": False,
        "stage60_or_stage70_prediction_outputs_used": False,
        "diagnostic_only": True,
    }


def publication_decision_payload(audit_result: Mapping[str, object]) -> dict[str, object]:
    if (
        "primary_proxy_information_gate_passed" not in audit_result
        or "audit_result_hash" not in audit_result
    ):
        raise ProtocolError("Case-aware audit result is incomplete for publication.")
    informative = audit_result["primary_proxy_information_gate_passed"]
    if type(informative) is not bool:
        raise ProtocolError("Case-aware primary proxy gate must be boolean.")
    return {
        "schema_version": "midogpp_stage90_case_aware_proxy_publication_decision_v1",
        "decision": (
            "EXPLORATORY_CASE_AWARE_SIGNAL_REQUIRES_FRESH_CONFIRMATION"
            if informative
            else "NO_RELIABLE_CASE_AWARE_PROXY_SIGNAL_USE_EXACT_B"
        ),
        "audit_result_hash": str(audit_result.get("audit_result_hash", "")),
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "test_split_previously_consumed": True,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "fresh_confirmation_claimed": False,
        "promotion_eligible": False,
        "policy_update_authorized": False,
        "action_selection_authorized": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_deployable_selection": False,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    if status not in {"RUNNING", "FAILED", "COMPLETE"}:
        raise ValueError("Case-aware audit run state is invalid.")
    return {
        "schema_version": "midogpp_stage90_case_aware_proxy_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "test_split_previously_consumed": True,
        "promotion_eligible": False,
    }


__all__ = (
    "leakage_report_payload",
    "prelabel_phase_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
