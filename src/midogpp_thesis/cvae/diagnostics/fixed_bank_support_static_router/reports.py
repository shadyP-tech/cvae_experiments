"""Claim-safe report payloads for the terminal consumed-test S4 diagnostic."""

from __future__ import annotations

from typing import Mapping

from .experiment_contracts import METHOD_IDS, TERMINAL_DECISION


def protocol_manifest_payload(
    config: object,
    *,
    protocol: object,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol": getattr(protocol, "to_payload")(),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "action_library": dict(getattr(config, "action_library")),
        "support_router": dict(getattr(config, "support_router")),
        "controls": dict(getattr(config, "controls")),
        "evaluation": dict(getattr(config, "evaluation")),
        "runtime": dict(getattr(config, "runtime")),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "probabilities_recomputed_from_original_six_inputs": True,
        "whole_case_five_fold_support_evaluation": True,
        "each_route_decision_invariant_to_own_evaluation_labels": True,
        "fresh_evidence": False,
        "terminal_consumed_test_diagnostic_only": True,
        "prior_stage90_output_prediction_or_scratch_consumed": False,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    probability_surface_hash: str,
    capability_report: Mapping[str, object],
    global_static_seal_hash: str,
    decision_seal_hash: str,
    null_seal_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_leakage_report_v1",
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "probability_surface_hash": probability_surface_hash,
        "global_static_selection_seal_hash": global_static_seal_hash,
        "all_route_decision_seal_hash": decision_seal_hash,
        "action_identity_null_seal_hash": null_seal_hash,
        "all_810_probabilities_sealed_before_label_capabilities": True,
        "fold_plan_count": capability_report.get("fold_plan_count"),
        "route_decision_seal_count": capability_report.get(
            "route_decision_seal_count"
        ),
        "null_route_plan_seal_count": capability_report.get(
            "null_selection_seal_count"
        ),
        "pre_evaluation_aggregate_decision_seal_count": capability_report.get(
            "pre_evaluation_aggregate_decision_seal_count"
        ),
        "pre_evaluation_aggregate_null_plan_seal_count": capability_report.get(
            "pre_evaluation_aggregate_null_plan_seal_count"
        ),
        "all_route_and_null_aggregate_seals_recorded_before_evaluation_labels": (
            capability_report.get(
                "all_route_and_null_aggregate_seals_recorded_before_evaluation_labels"
            )
            is True
        ),
        "every_route_decision_excludes_own_evaluation_labels": capability_report.get(
            "every_route_decision_sealed_before_own_evaluation_labels"
        )
        is True,
        "each_null_route_plan_sealed_before_own_evaluation_labels": capability_report.get(
            "every_null_selection_sealed_before_own_evaluation_labels"
        )
        is True,
        "other_fold_labels_may_be_used_for_support": True,
        "impossible_global_no_label_rule_imposed": False,
        "evaluation_labels_used_for_own_route_decision": False,
        "terminal_oracles_used_for_decisions": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "shared_model_updated_with_target_labels": False,
        "donor_model_used": False,
        "target_local_calibration_used": False,
        "previous_stage90_output_prediction_or_scratch_consumed": False,
        "raw_labels_persisted": False,
        "per_case_bacc_used": False,
    }


def publication_decision_payload(sealed_result_hash: str) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_support_static_router_publication_v1",
        "decision": TERMINAL_DECISION,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "claim_role": "posthoc_known_fixed_bank_support_static_router_s4_diagnostic",
        "sealed_result_hash": sealed_result_hash,
        "method_ids": list(METHOD_IDS),
        "primary_diagnostic_method": "S4",
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "exchangeability_claimed": False,
        "confirmatory_p_value": False,
        "null_summary_in_pass_gate": False,
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
        "schema_version": "fixed_bank_support_static_router_run_state_v1",
        "status": status,
        "phase": phase,
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_supported": False,
        "deterministic_restart_from_admission_requires_hash_validation": True,
        "terminal_checkpoint_recovery_supported": False,
        "terminal_checkpoint_is_atomicity_boundary_only": True,
    }
    if error is not None and error_class is None:
        raise ValueError("FAILED S4 run states require an error_class.")
    if error is not None:
        payload["error"] = error
        payload["error_class"] = error_class
    elif error_class is not None:
        raise ValueError("S4 error_class requires an error message.")
    return payload


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
