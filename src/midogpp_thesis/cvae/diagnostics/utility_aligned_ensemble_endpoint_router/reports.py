"""Canonical claim-boundary and phase reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .contracts import CLAIM_SCOPE, DATASET_FAMILY, EXPERIMENT_ID, PUBLICATION_STATUS, ROUTING_STATUS, STAGE_ID


def protocol_manifest_payload(
    config: object, *, input_artifact_hashes: Mapping[str, str],
    validation_cache_binding_hash: str, firewall: Mapping[str, object]
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID, "dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID, "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "validation_cache_binding_hash": validation_cache_binding_hash,
        "pre_gpu_firewall": dict(firewall), "consumed_validation_data": True,
        "fresh_evidence": False, "terminal_diagnostic": True,
        "routing_status": ROUTING_STATUS, "support_case_count": 2,
        "primary_development_response_count": 504,
        "descriptive_seed_row_count": 4536,
        "target_action_identity_count": 1053,
        "target_unique_classifier_fit_count": 810,
        "development_predictions_sealed_before_labels": True,
        "target_probe_sealed_before_plan": True,
        "target_predictions_sealed_before_terminal_labels": True,
        "technical_seed_rows_may_feed_model": False,
        "may_feed_stage60": False, "may_feed_stage70": False,
        "may_feed_deployable_selection": False,
    }
    return {**unhashed, "protocol_manifest_hash": stable_hash(unhashed)}


def phase_completion_payload(
    phase: str, *, config_contract_hash: str, bindings: Mapping[str, object],
    counts: Mapping[str, int], development_labels_opened: bool,
    terminal_target_labels_opened: bool,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_phase_completion_v1",
        "phase": phase, "status": "COMPLETE", "config_contract_hash": config_contract_hash,
        "bindings": dict(bindings), "counts": {str(key): int(value) for key, value in counts.items()},
        "development_labels_opened": development_labels_opened,
        "terminal_target_labels_opened": terminal_target_labels_opened,
        "diagnostic_only": True,
    }
    return {**unhashed, "phase_hash": stable_hash(unhashed)}


def leakage_report_payload(**bindings: object) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_leakage_report_v1",
        "status": "PASS", **bindings,
        "whole_case_support_evaluation_disjoint": True,
        "strict_H_q_e_exclusion": True,
        "support_labels_used": False,
        "development_labels_opened_after_global_development_seal": True,
        "target_probe_sealed_before_plan": True,
        "target_evaluation_used_to_build_plan": False,
        "target_labels_opened_after_global_target_seal": True,
        "previous_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "policy_or_action_update_after_target_labels": False,
        "diagnostic_only": True,
    }


def scoring_summary_payload(
    endpoint_rows: Sequence[object], inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[object]
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_scoring_summary_v1",
        "target_endpoint_row_count": len(endpoint_rows),
        "contrast_inference_row_count": len(inference_rows),
        "oracle_target_count": len(oracle_rows),
        "primary_contrasts": {
            str(row["contrast_id"]): {
                "mean_paired_bacc_delta": float(row["mean_paired_bacc_delta"]),
                "ci95_lower": float(row["ci95_lower"]),
                "ci95_upper": float(row["ci95_upper"]),
            }
            for row in inference_rows if row.get("contrast_role") == "primary_predeclared"
        },
        "inference_unit": "target_center", "center_count": 9,
        "technical_seed_cells_are_independent_units": False,
        "consumed_data_diagnostic_only": True,
    }


def publication_decision_payload(scoring_summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_publication_decision_v1",
        "decision": "DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": PUBLICATION_STATUS, "routing_status": ROUTING_STATUS,
        "reason": "consumed_validation_and_two_support_cases_below_fresh_policy_minimum",
        "scoring_summary_hash": stable_hash(dict(scoring_summary)),
        "routing_quality_claimed": False, "target_specific_router_success_claimed": False,
        "downstream_utility_claimed": False, "fresh_confirmation_claimed": False,
        "promotion_eligible": False, "policy_update_authorized": False,
        "fallback_authorized": False, "may_feed_stage60": False,
        "may_feed_stage70": False, "may_feed_deployable_selection": False,
    }


def runtime_summary_payload(
    preflight: Mapping[str, object], *, counts: Mapping[str, int],
    source_cache_staging: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_runtime_summary_v1",
        "workstation_preflight": dict(preflight), "counts": dict(counts),
        "source_cache_staging": dict(source_cache_staging),
        "generation_devices": ["cuda:0", "cuda:1"],
        "classifier_workers": 4, "classifier_threads_per_worker": 3,
        "phase_disjoint_gpu_and_cpu_pools": True, "hash_validated_resume": True,
        "target_unique_classifier_fit_count": 810, "tf32_enabled": False, "amp_enabled": False,
    }


def run_state_payload(status: str, phase: str, *, error: str | None = None) -> dict[str, object]:
    if status not in {"RUNNING", "FAILED", "COMPLETE"}:
        raise ValueError("Ensemble-endpoint run state is invalid.")
    return {
        "schema_version": "midogpp_stage90_ensemble_endpoint_run_state_v1",
        "status": status, "phase": phase, "error": error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True, "promotion_eligible": False,
    }


__all__ = (
    "leakage_report_payload", "phase_completion_payload", "protocol_manifest_payload",
    "publication_decision_payload", "run_state_payload", "runtime_summary_payload",
    "scoring_summary_payload",
)
