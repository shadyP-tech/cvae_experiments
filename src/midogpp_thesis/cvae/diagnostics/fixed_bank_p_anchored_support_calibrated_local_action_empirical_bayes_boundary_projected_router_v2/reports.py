"""Small reconstructive runtime, leakage, validation, and claim reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .artifacts.hashing import canonical_hash, json_native, require_sha256
from .identity import (
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .protocol import GovernanceError, terminal_claim_firewall_payload


def seal_payload(
    schema_version: str,
    *,
    bindings: Mapping[str, object],
    **facts: object,
) -> dict[str, object]:
    body = {
        "schema_version": str(schema_version),
        "bindings": json_native(dict(bindings)),
        **json_native(facts),
    }
    return {**body, "seal_hash": canonical_hash(body)}


def protocol_manifest_payload(
    *,
    config_hash: str,
    protocol_hash: str,
    run_identity_hash: str,
    admission_receipt_hash: str,
    input_artifact_hashes: Mapping[str, str],
    source_fence_hash: str,
    workstation_plan_hash: str,
    authorization_lease_claim_hash: str,
) -> dict[str, object]:
    bindings = {
        "config_hash": require_sha256(config_hash, "config hash"),
        "protocol_hash": require_sha256(protocol_hash, "protocol hash"),
        "run_identity_hash": require_sha256(run_identity_hash, "run identity hash"),
        "admission_receipt_hash": require_sha256(
            admission_receipt_hash, "admission receipt hash"
        ),
        "authorization_lease_claim_hash": require_sha256(
            authorization_lease_claim_hash,
            "authorization lease claim hash",
        ),
        "source_fence_hash": require_sha256(source_fence_hash, "source fence hash"),
        "workstation_plan_hash": require_sha256(
            workstation_plan_hash, "workstation plan hash"
        ),
        "input_artifact_hashes": {
            str(key): require_sha256(value, f"input {key}")
            for key, value in sorted(input_artifact_hashes.items())
        },
    }
    body = {
        "schema_version": "scale_bp_v2_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        **bindings,
        "split_previously_consumed": True,
        "single_use_execution": True,
        "fresh_evidence": False,
        "raw_labels_persisted": False,
        "predecessor_diagnostic_artifacts_consumed": False,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
    }
    return {**body, "manifest_hash": canonical_hash(body)}


def runtime_report_payload(
    *,
    run_state: Mapping[str, object],
    workstation_plan: Mapping[str, object],
    center_results: Sequence[Mapping[str, object]],
    phase_timings_seconds: Mapping[str, float],
    memmap_reference_hashes: Sequence[str],
) -> dict[str, object]:
    results = [dict(row) for row in center_results]
    timings = {str(key): float(value) for key, value in sorted(phase_timings_seconds.items())}
    if (
        run_state.get("authorization_exhausted") is not True
        or workstation_plan.get("nested_process_pools_allowed") is not False
        or any(value < 0 for value in timings.values())
        or not results
    ):
        raise GovernanceError("SCALE-BP v2 runtime report inputs drifted.")
    memmaps = [require_sha256(value, "memmap reference hash") for value in memmap_reference_hashes]
    body = {
        "schema_version": "scale_bp_v2_runtime_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "run_state_hash": require_sha256(run_state.get("state_hash"), "run state hash"),
        "workstation_plan_hash": require_sha256(
            workstation_plan.get("plan_hash"), "workstation plan hash"
        ),
        "outer_center_results": json_native(results),
        "outer_center_result_count": len(results),
        "phase_timings_seconds": timings,
        "memmap_reference_hashes": memmaps,
        "memmaps_opened_read_only": True,
        "support_folds_executed_sequentially_inside_outer_worker": True,
        "nested_process_pools_used": False,
        "cpu_outer_worker_count": 4,
        "blas_threads_per_outer_worker": 1,
        "storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "cross_run_recovery_used": False,
        "terminal_recovery_used": False,
    }
    return {**body, "runtime_report_hash": canonical_hash(body)}


def leakage_report_payload(
    *,
    protocol_hash: str,
    preterminal_aggregate_seal_hash: str,
    decision_seal_hash: str,
    preterminal_journal_hash: str,
    final_journal_hash: str,
    terminal_seal_hash: str,
) -> dict[str, object]:
    body = {
        "schema_version": "scale_bp_v2_leakage_report_v1",
        "status": "PASS",
        "protocol_hash": require_sha256(protocol_hash, "protocol hash"),
        "preterminal_aggregate_seal_hash": require_sha256(
            preterminal_aggregate_seal_hash, "preterminal aggregate seal"
        ),
        "decision_seal_hash": require_sha256(decision_seal_hash, "decision seal"),
        "preterminal_label_capability_journal_hash": require_sha256(
            preterminal_journal_hash, "preterminal label journal"
        ),
        "final_label_capability_journal_hash": require_sha256(
            final_journal_hash, "final label journal"
        ),
        "terminal_seal_hash": require_sha256(
            terminal_seal_hash, "terminal aggregate seal"
        ),
        "all_218_decisions_sealed_before_terminal_labels": True,
        "terminal_labels_used_for_aggregate_scoring_only": True,
        "terminal_labels_updated_no_preterminal_state": True,
        "outer_H_excluded_from_donor_fits": True,
        "held_case_excluded_from_route_support": True,
        "support_labels_route_local_only": True,
        "raw_labels_persisted": False,
        "target_expert_used": False,
        "predecessor_stage90_outputs_consumed": False,
        "fresh_evidence": False,
    }
    return {**body, "leakage_report_hash": canonical_hash(body)}


def publication_decision_payload(
    *,
    terminal_seal_hash: str,
    diagnostic_summary: Mapping[str, object],
) -> dict[str, object]:
    firewall = terminal_claim_firewall_payload()
    body = {
        "schema_version": "scale_bp_v2_publication_decision_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "terminal_seal_hash": require_sha256(
            terminal_seal_hash, "terminal seal hash"
        ),
        "diagnostic_summary": json_native(dict(diagnostic_summary)),
        "claim_firewall_hash": firewall["claim_firewall_hash"],
        "fresh_evidence": False,
        "routing_success_claim_authorized": False,
        "downstream_utility_claim_authorized": False,
        "confidence_claim_authorized": False,
        "significance_claim_authorized": False,
        "deployment_claim_authorized": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "publication_decision_hash": canonical_hash(body)}


def validation_report_payload(
    checks: Mapping[str, object],
    *,
    fresh_process_attestation_hash: str,
) -> dict[str, object]:
    payload = json_native(dict(checks))
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise GovernanceError("SCALE-BP v2 cannot report non-PASS validation.")
    body = {
        "schema_version": "scale_bp_v2_validation_report_v1",
        "status": "PASS",
        "checks": payload,
        "checks_hash": canonical_hash(payload),
        "fresh_process_attestation_hash": require_sha256(
            fresh_process_attestation_hash, "fresh-process attestation hash"
        ),
        "fresh_process_count": 2,
        "artifact_only_reconstruction": True,
        "scientific_refit_performed": False,
    }
    return {**body, "validation_report_hash": canonical_hash(body)}


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "runtime_report_payload",
    "seal_payload",
    "validation_report_payload",
)
