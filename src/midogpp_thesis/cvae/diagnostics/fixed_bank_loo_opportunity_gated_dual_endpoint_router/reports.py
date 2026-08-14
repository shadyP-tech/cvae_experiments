"""Small, reconstructive report and seal payload builders."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .constants import PUBLICATION_STATUS, TERMINAL_DECISION
from .experiment_contracts import CLAIM_ROLE, CLAIM_SCOPE, STAGE_ID
from .hashing import canonical_hash, json_native


def seal_payload(
    schema_version: str,
    *,
    bindings: Mapping[str, object],
    **facts: object,
) -> dict[str, object]:
    unhashed = {
        "schema_version": schema_version,
        "bindings": json_native(dict(bindings)),
        **json_native(facts),
    }
    return {**unhashed, "seal_hash": canonical_hash(unhashed)}


def protocol_manifest_payload(
    config: object,
    *,
    protocol: object,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    protocol_hash = str(
        getattr(protocol, "protocol_hash", getattr(protocol, "contract_hash", ""))
    )
    unhashed = {
        "schema_version": "fixed_bank_dual_endpoint_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol_hash,
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "input_artifact_hashes": dict(input_artifact_hashes),
        "cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "input_artifact_count": len(input_artifact_hashes),
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "target_expert_used": False,
        "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed": False,
        "publication_status": PUBLICATION_STATUS,
    }
    return {**unhashed, "protocol_manifest_hash": canonical_hash(unhashed)}


def run_state_payload(
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_dual_endpoint_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "error_class": error_class,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }


def leakage_report_payload(
    *,
    physical_prelabel_seal_hash: str,
    feature_seal_hash: str,
    aggregate_plan_decision_seal_hash: str,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_dual_endpoint_leakage_report_v1",
        "status": "PASS",
        "physical_prelabel_seal_hash": physical_prelabel_seal_hash,
        "label_free_feature_seal_hash": feature_seal_hash,
        "aggregate_plan_decision_seal_hash": aggregate_plan_decision_seal_hash,
        "capability_report": dict(capability_report),
        "all_218_decisions_and_endpoint_probabilities_sealed_before_terminal_labels": True,
        "donor_scope": "q_not_in_H_or_e",
        "support_scope": "H_minus_c_complete_case_block",
        "route_state_shared_across_routes": False,
        "target_expert_used": False,
        "raw_labels_persisted": False,
        "sample_or_image_paths_persisted": False,
        "fresh_evidence": False,
    }


def publication_decision_payload(
    terminal_seal_hash: str, *, diagnostic_summary: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_dual_endpoint_publication_decision_v1",
        "status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "terminal_evaluation_seal_hash": terminal_seal_hash,
        "diagnostic_summary": dict(diagnostic_summary),
        "significance_claim_authorized": False,
        "routing_success_claim_authorized": False,
        "active_expert_identification_claim_authorized": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "seal_payload",
)
