"""Claim-boundary and execution reports for the proxy-information audit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from ....common.hashing import stable_hash
from .contracts import EXPERIMENT_ID


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_proxy_information_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "response_count": 504,
        "technical_seed_row_count": 4536,
        "technical_seed_rows_are_independent_units": False,
        "strict_H_q_e_exclusion": True,
        "support_labels_used": False,
        "development_predictions_sealed_before_labels": True,
        "target_actions_built": False,
        "target_labels_opened": False,
        "prior_stage90_output_consumed": False,
        "fresh_evidence": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
    }
    return {**unhashed, "protocol_manifest_hash": stable_hash(unhashed)}


def prelabel_phase_payload(
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    development_prediction_seal_hash: str,
    proxy_feature_lock_hash: str,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_proxy_information_prelabel_phase_v1",
        "status": "SEALED_BEFORE_DEVELOPMENT_LABEL_ACCESS",
        "config_contract_hash": config_contract_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "proxy_feature_lock_hash": proxy_feature_lock_hash,
        "proxy_feature_row_count": 504,
        "support_probabilities_only_for_proxy_features": True,
        "evaluation_probabilities_used_as_features": False,
        "development_labels_opened": False,
        "support_labels_opened": False,
        "target_labels_opened": False,
        "actions_or_oracles_materialized": False,
    }
    return {**unhashed, "phase_hash": stable_hash(unhashed)}


def leakage_report_payload(
    *, support_partition_lock_hash: str, development_prediction_seal_hash: str,
    proxy_feature_lock_hash: str, crossfit_fold_lock_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_proxy_information_leakage_report_v1",
        "status": "PASS",
        "support_partition_lock_hash": support_partition_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "proxy_feature_lock_hash": proxy_feature_lock_hash,
        "crossfit_fold_lock_hash": crossfit_fold_lock_hash,
        "whole_case_support_evaluation_disjoint": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "development_labels_opened_after_global_prediction_seal": True,
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction": True,
        "seed_or_patch_rows_used_as_independent_observations": False,
        "target_actions_built": False,
        "target_labels_opened": False,
        "previous_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "diagnostic_only": True,
    }


def publication_decision_payload(audit_result: Mapping[str, object]) -> dict[str, object]:
    informative = bool(audit_result.get("proxy_information_gate_passed", False))
    return {
        "schema_version": "midogpp_stage90_proxy_information_publication_decision_v1",
        "decision": (
            "EXPLORATORY_PROXY_SIGNAL_DETECTED_REQUIRES_FRESH_CONFIRMATION"
            if informative
            else "NO_RELIABLE_PROXY_SIGNAL_USE_EXACT_B"
        ),
        "audit_result_hash": str(audit_result.get("audit_result_hash", "")),
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
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
        raise ValueError("Proxy-audit run state is invalid.")
    return {
        "schema_version": "midogpp_stage90_proxy_information_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "promotion_eligible": False,
    }


__all__ = (
    "leakage_report_payload",
    "prelabel_phase_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
