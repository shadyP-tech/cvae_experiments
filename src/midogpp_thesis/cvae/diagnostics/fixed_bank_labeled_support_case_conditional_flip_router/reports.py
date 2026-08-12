"""Claim-bound report payloads for the terminal diagnostic."""

from __future__ import annotations

from typing import Mapping

from .constants import METHOD_IDS
from .experiment_contracts import CLAIM_ROLE, PUBLICATION_STATUS


def protocol_manifest_payload(
    config: object,
    *,
    protocol: object,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_labeled_support_flip_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol": getattr(protocol, "to_payload")(),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "action_library": dict(getattr(config, "action_library")),
        "flip_features": dict(getattr(config, "flip_features")),
        "routing": dict(getattr(config, "routing")),
        "controls": dict(getattr(config, "controls")),
        "evaluation": dict(getattr(config, "evaluation")),
        "runtime": dict(getattr(config, "runtime")),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "probabilities_recomputed_from_original_six_inputs": True,
        "three_role_whole_case_partitions": True,
        "each_fold_plan_invariant_to_its_held_evaluation_labels": True,
        "fresh_evidence": False,
        "terminal_consumed_test_diagnostic_only": True,
        "prior_stage90_output_prediction_or_scratch_consumed": False,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    feature_seal_hash: str,
    capability_report: Mapping[str, object],
    donor_model_seal_count: int,
    fold_decision_seal_count: int,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_labeled_support_flip_leakage_report_v1",
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "label_free_feature_seal_hash": feature_seal_hash,
        "H_specific_donor_model_seal_count": donor_model_seal_count,
        "fold_decision_seal_count": fold_decision_seal_count,
        "all_probabilities_and_features_sealed_before_label_capabilities": True,
        "heldout_H_labels_used_in_H_specific_donor_model": False,
        "donor_model_shared_across_target_H": False,
        "selection_calibration_evaluation_case_overlap_count_per_fold": 0,
        "every_fold_decision_invariant_to_its_held_evaluation_labels": True,
        "terminal_scoring_after_all_45_fold_decision_seals": capability_report.get("terminal_scoring_opened") is True,
        "evaluation_labels_used_for_decisions": False,
        "terminal_oracles_used_for_decisions": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "shared_model_updated_with_target_labels": False,
        "prior_stage90_output_prediction_or_scratch_consumed": False,
        "raw_labels_persisted": False,
        "per_case_bacc_used": False,
    }


def publication_decision_payload(sealed_result_hash: str) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_labeled_support_flip_publication_v1",
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": PUBLICATION_STATUS,
        "claim_role": CLAIM_ROLE,
        "sealed_result_hash": sealed_result_hash,
        "method_ids": list(METHOD_IDS),
        "primary_heuristic_router": "F_S",
        "heuristic_prediction_bound_descriptive_only": True,
        "calibrated_case_confidence_or_safety_claimed": False,
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "policy_model_expert_or_deployment_update_authorized": False,
        "may_feed_stage50_stage60_stage70_or_another_experiment": False,
    }


def run_state_payload(
    status: str,
    phase: str,
    *,
    error: str | None = None,
    error_class: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "fixed_bank_labeled_support_flip_run_state_v1",
        "status": status,
        "phase": phase,
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
    }
    if error is not None and error_class is None:
        raise ValueError("FAILED run states require an explicit error_class")
    if error is not None:
        payload["error"] = error
        payload["error_class"] = error_class
    elif error_class is not None:
        raise ValueError("error_class requires an error message")
    return payload


__all__ = (
    "leakage_report_payload", "protocol_manifest_payload", "publication_decision_payload", "run_state_payload",
)
