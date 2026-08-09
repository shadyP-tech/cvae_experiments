"""Protocol, leakage, publication, and runner-state reports."""

from __future__ import annotations

from typing import Mapping

from .experiment_contracts import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_label_aware_case_oof_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "protocol": dict(getattr(config, "protocol")),
        "global_prior": dict(getattr(config, "global_prior")),
        "posterior": dict(getattr(config, "posterior")),
        "decision": dict(getattr(config, "decision")),
        "evaluation": dict(getattr(config, "evaluation")),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "support_labels_used": True,
        "evaluation_labels_inaccessible_until_all_decisions_sealed": True,
        "fresh_evidence": False,
        "terminal_consumed_test_diagnostic_only": True,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    prior_count: int,
    decision_count: int,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_label_aware_case_oof_leakage_report_v1",
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "loco_prior_count": prior_count,
        "fold_decision_count": decision_count,
        "probabilities_globally_sealed_before_any_label": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
        "G_H_sealed_before_H_support_access": True,
        "whole_case_support_evaluation_overlap_count": 0,
        "evaluation_labels_opened_after_all_decisions": capability_report.get("evaluation_labels_opened") is True,
        "target_expert_used": False,
        "source_expert_updated": False,
        "shared_model_updated_with_target_labels": False,
        "smooth_metric_may_affect_decision": False,
        "prior_stage90_output_consumed": False,
        "generic_consumer_authorized": False,
    }


def publication_decision_payload(evaluation: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_label_aware_case_oof_publication_decision_v1",
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "claim_role": "label_aware_known_bank_case_oof_ceiling",
        "fresh_evidence": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "scientific_result_hash": evaluation.get("scientific_result_hash"),
    }


def run_state_payload(status: str, phase: str, *, error: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_label_aware_case_oof_run_state_v1",
        "status": status,
        "phase": phase,
        "terminal_consumed_test_diagnostic_only": True,
    }
    if error is not None:
        payload["error"] = error
    return payload


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
