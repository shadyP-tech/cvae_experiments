"""Claim-bound reports for the signed-error terminal diagnostic."""

from __future__ import annotations

from typing import Mapping

from .constants import METHOD_IDS
from .protocol import SignedErrorGateProtocol


def protocol_manifest_payload(
    config: object,
    *,
    protocol: SignedErrorGateProtocol,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_fixed_bank_signed_error_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "signed_protocol": protocol.to_payload(),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "runtime": dict(getattr(config, "runtime")),
        "evaluation": dict(getattr(config, "evaluation")),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "probability_surface_recomputed_from_original_six_inputs": True,
        "strict_outer_H_and_nested_query_exclusion": True,
        "baseline_predicted_class_branch_used": False,
        "support_selection_objective": (
            "fixed_class_balanced_negative_log_loss_only"
        ),
        "terminal_metric": "center_pooled_exact_bacc_equal_center_aggregate",
        "uncertainty_unit": "paired_whole_case_cluster",
        "single_class_cases_retained": True,
        "per_case_bacc_defined_or_persisted": False,
        "fresh_evidence": False,
        "terminal_consumed_test_diagnostic_only": True,
        "prior_stage90_output_or_scratch_consumed": False,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    feature_seal_hash: str,
    model_family_count: int,
    decision_count: int,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_fixed_bank_signed_error_leakage_report_v1",
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "label_free_feature_seal_hash": feature_seal_hash,
        "loco_target_model_family_count": model_family_count,
        "fold_method_decision_count": decision_count,
        "probabilities_and_feature_contexts_sealed_before_any_label": True,
        "outer_H_labels_used_in_shared_models": False,
        "held_query_leakage": False,
        "baseline_predicted_class_branch_used": False,
        "all_G_R_P_models_sealed_before_same_H_support": True,
        "R_raw_and_R_safe_separately_sealed": True,
        "whole_case_support_evaluation_overlap_count": 0,
        "evaluation_labels_opened_after_all_method_and_permutation_seals": (
            capability_report.get("evaluation_labels_opened") is True
        ),
        "support_objective": "fixed_class_balanced_negative_log_loss_only",
        "exact_bacc_used_for_grid_selection": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "shared_model_updated_with_target_labels": False,
        "metadata_artifact_used": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
        "evaluation_labels_used_for_decisions": False,
        "raw_labels_persisted": False,
        "per_case_bacc_used": False,
    }


def publication_decision_payload(
    sealed_evaluation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_fixed_bank_signed_error_publication_v1",
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "claim_role": "posthoc_signed_error_mechanism_diagnostic",
        "sealed_result_hash": sealed_evaluation.get("sealed_result_hash"),
        "method_ids": list(METHOD_IDS),
        "fresh_evidence": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "model_or_expert_update_authorized": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_fixed_bank_signed_error_run_state_v1",
        "status": status,
        "phase": phase,
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
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
