"""Aggregate-only scientific, leakage, and runtime reports for P-DCAPS v2."""

from __future__ import annotations

from typing import Mapping

from ....protocol import ProtocolError
from ..identity import PUBLICATION_STATUS, TERMINAL_DECISION
from .identity import canonical_hash, require_sha256
from .terminal.contracts import TerminalEvaluationResult


FINAL_REPORT_SCHEMAS = {
    "reports/diagnostic_summary.json": "pdcaps_v2_diagnostic_summary_v1",
    "reports/label_capability_report.json": (
        "pdcaps_v2_label_capability_report_v1"
    ),
    "reports/leakage_report.json": "pdcaps_v2_leakage_report_v1",
    "reports/publication_decision.json": "pdcaps_v2_publication_decision_v1",
    "reports/runtime_summary.json": "pdcaps_v2_runtime_summary_v1",
}
FINAL_REPORT_MEMBERS = tuple(FINAL_REPORT_SCHEMAS)
_FINAL_REPORT_KEYS = {
    "reports/diagnostic_summary.json": {
        "schema_version",
        "terminal_result_hash",
        "method_rows",
        "selection_control",
        "router_diagnostics",
        "publication_status",
        "terminal_decision",
        "nonzero_route_count_is_not_success",
        "routing_success_claimed",
        "fresh_evidence",
        "promotion_allowed",
        "report_hash",
    },
    "reports/label_capability_report.json": {
        "schema_version",
        "lifecycle_audit",
        "target_labels_opened_only_after_durable_preterminal_attestation",
        "raw_labels_persisted",
        "sample_paths_persisted",
        "report_hash",
    },
    "reports/leakage_report.json": {
        "schema_version",
        "status",
        "input_binding",
        "pre_gpu_firewall",
        "lifecycle_hash",
        "source_snapshot_manifest_sha256",
        "source_snapshot_tree_sha256",
        "exact_six_inputs",
        "outer_center_excluded_from_every_scientific_fit",
        "pseudo_center_excluded_from_own_prediction",
        "held_case_labels_used_only_for_scored_response",
        "target_labels_can_change_preterminal_decisions",
        "v1_or_prior_v2_state_used",
        "raw_labels_persisted",
        "fresh_evidence",
        "report_hash",
    },
    "reports/publication_decision.json": {
        "schema_version",
        "terminal_result_hash",
        "publication_status",
        "terminal_decision",
        "bounded_interpretation",
        "fresh_evidence",
        "routing_success_claimed",
        "downstream_utility_claimed",
        "promotion_eligible",
        "may_feed_another_experiment",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_recipe_selection",
        "report_hash",
    },
    "reports/runtime_summary.json": {
        "schema_version",
        "workstation_preflight_status",
        "workstation_preflight_schema",
        "physical_surface_hash",
        "route_runtime_hash",
        "pseudo_runtime_hash",
        "outer_execution_mode",
        "outer_worker_count",
        "outer_runtime_hash",
        "outer_science_hash",
        "outer_chunk_manifest_hash",
        "outer_chunks_written_atomically",
        "outer_chunks_verified_after_write",
        "outer_chunks_are_scratch_only",
        "gpu_then_cpu_phase_order",
        "cpu_phase_cuda_hidden",
        "outer_blas_threads_per_worker",
        "nested_process_pools",
        "worker_DTOs_plain_pickle_safe",
        "serial_process_hash_equivalence_source_test_gated",
        "full_run_serial_replay_performed",
        "scratch_recovery_used",
        "report_hash",
    },
}


def diagnostic_summary_payload(
    result: TerminalEvaluationResult,
) -> dict[str, object]:
    base = {
        "schema_version": "pdcaps_v2_diagnostic_summary_v1",
        "terminal_result_hash": result.result_hash,
        "method_rows": [dict(row) for row in result.method_rows],
        "selection_control": dict(result.selection_control),
        "router_diagnostics": dict(result.router_diagnostics),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "nonzero_route_count_is_not_success": True,
        "routing_success_claimed": False,
        "fresh_evidence": False,
        "promotion_allowed": False,
    }
    return {**base, "report_hash": canonical_hash(base)}


def label_capability_report_payload(
    lifecycle_audit: Mapping[str, object],
) -> dict[str, object]:
    base = {
        "schema_version": "pdcaps_v2_label_capability_report_v1",
        "lifecycle_audit": dict(lifecycle_audit),
        "target_labels_opened_only_after_durable_preterminal_attestation": True,
        "raw_labels_persisted": False,
        "sample_paths_persisted": False,
    }
    return {**base, "report_hash": canonical_hash(base)}


def leakage_report_payload(
    *,
    input_binding: Mapping[str, object],
    pre_gpu_firewall: Mapping[str, object],
    lifecycle_audit: Mapping[str, object],
    source_snapshot: Mapping[str, object],
) -> dict[str, object]:
    source_manifest = require_sha256(
        source_snapshot.get(
            "manifest_sha256",
            source_snapshot.get("source_snapshot_manifest_sha256"),
        ),
        "v2 leakage source manifest",
    )
    source_tree = require_sha256(
        source_snapshot.get(
            "tree_sha256",
            source_snapshot.get("source_snapshot_tree_sha256"),
        ),
        "v2 leakage source tree",
    )
    base = {
        "schema_version": "pdcaps_v2_leakage_report_v1",
        "status": "PASS",
        "input_binding": dict(input_binding),
        "pre_gpu_firewall": dict(pre_gpu_firewall),
        "lifecycle_hash": lifecycle_audit.get("lifecycle_hash"),
        "source_snapshot_manifest_sha256": source_manifest,
        "source_snapshot_tree_sha256": source_tree,
        "exact_six_inputs": True,
        "outer_center_excluded_from_every_scientific_fit": True,
        "pseudo_center_excluded_from_own_prediction": True,
        "held_case_labels_used_only_for_scored_response": True,
        "target_labels_can_change_preterminal_decisions": False,
        "v1_or_prior_v2_state_used": False,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
    }
    return {**base, "report_hash": canonical_hash(base)}


def publication_decision_payload(
    result: TerminalEvaluationResult,
) -> dict[str, object]:
    base = {
        "schema_version": "pdcaps_v2_publication_decision_v1",
        "terminal_result_hash": result.result_hash,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "bounded_interpretation": (
            "donor_crossfit_action_and_policy_surface_sensitivity_on_"
            "consumed_MIDOGpp_test_only"
        ),
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
    }
    return {**base, "report_hash": canonical_hash(base)}


def runtime_summary_payload(
    *,
    preflight: Mapping[str, object],
    physical_surface_hash: str,
    route_runtime_hash: str,
    pseudo_runtime_hash: str,
    outer_execution: Mapping[str, object],
    outer_chunk_manifest: Mapping[str, object],
) -> dict[str, object]:
    base = {
        "schema_version": "pdcaps_v2_runtime_summary_v1",
        "workstation_preflight_status": preflight.get("status"),
        "workstation_preflight_schema": preflight.get("schema_version"),
        "physical_surface_hash": physical_surface_hash,
        "route_runtime_hash": route_runtime_hash,
        "pseudo_runtime_hash": pseudo_runtime_hash,
        "outer_execution_mode": outer_execution.get("execution_mode"),
        "outer_worker_count": outer_execution.get("worker_count"),
        "outer_runtime_hash": outer_execution.get("runtime_hash"),
        "outer_science_hash": outer_execution.get("science_hash"),
        "outer_chunk_manifest_hash": outer_chunk_manifest.get("manifest_hash"),
        "outer_chunks_written_atomically": outer_chunk_manifest.get(
            "written_atomically"
        ),
        "outer_chunks_verified_after_write": outer_chunk_manifest.get(
            "verified_after_write"
        ),
        "outer_chunks_are_scratch_only": True,
        "gpu_then_cpu_phase_order": True,
        "cpu_phase_cuda_hidden": True,
        "outer_blas_threads_per_worker": 1,
        "nested_process_pools": False,
        "worker_DTOs_plain_pickle_safe": True,
        "serial_process_hash_equivalence_source_test_gated": True,
        "full_run_serial_replay_performed": False,
        "scratch_recovery_used": False,
    }
    return {**base, "report_hash": canonical_hash(base)}


def validation_report_payload(
    attestation: Mapping[str, object],
) -> dict[str, object]:
    base = {
        "schema_version": "pdcaps_v2_validation_report_v1",
        "status": "PASS",
        "final_attestation_hash": attestation.get("attestation_hash"),
        "fresh_python_process_count": attestation.get(
            "fresh_python_process_count"
        ),
        "validator_process_ids": list(
            attestation.get("validator_process_ids", ())
        ),
        "validator_result_hashes": list(
            attestation.get("validator_result_hashes", ())
        ),
        "reconstructed_checks": dict(
            attestation.get("reconstructed_checks", {})
        ),
        "semantic_reconstruction_without_refit": True,
        "formal_claim_authorized": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
    }
    return {**base, "report_hash": canonical_hash(base)}


def validate_final_report_payloads(
    reports: Mapping[str, Mapping[str, object]],
    *,
    terminal_result_hash: str,
    terminal_result_payload: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Validate the complete terminal-report set before or after persistence."""

    result_hash = require_sha256(
        terminal_result_hash, "v2 terminal report result"
    )
    if set(reports) != set(FINAL_REPORT_MEMBERS):
        raise ProtocolError("P-DCAPS v2 final report inventory drifted.")
    checked: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for member in FINAL_REPORT_MEMBERS:
        report = dict(reports[member])
        base = {
            key: value for key, value in report.items() if key != "report_hash"
        }
        if (
            set(report) != _FINAL_REPORT_KEYS[member]
            or report.get("schema_version") != FINAL_REPORT_SCHEMAS[member]
            or report.get("report_hash") != canonical_hash(base)
        ):
            raise ProtocolError("P-DCAPS v2 final report hash drifted.")
        checked[member] = report
        hashes[member] = str(report["report_hash"])

    diagnostic = checked["reports/diagnostic_summary.json"]
    capability = checked["reports/label_capability_report.json"]
    leakage = checked["reports/leakage_report.json"]
    publication = checked["reports/publication_decision.json"]
    runtime = checked["reports/runtime_summary.json"]
    lifecycle = capability.get("lifecycle_audit")
    terminal = (
        None if terminal_result_payload is None else dict(terminal_result_payload)
    )
    if (
        diagnostic.get("terminal_result_hash") != result_hash
        or diagnostic.get("publication_status") != PUBLICATION_STATUS
        or diagnostic.get("terminal_decision") != TERMINAL_DECISION
        or diagnostic.get("routing_success_claimed") is not False
        or diagnostic.get("fresh_evidence") is not False
        or diagnostic.get("promotion_allowed") is not False
        or diagnostic.get("nonzero_route_count_is_not_success") is not True
        or (
            terminal is not None
            and (
                terminal.get("result_hash") != result_hash
                or diagnostic.get("method_rows") != terminal.get("method_rows")
                or diagnostic.get("selection_control")
                != terminal.get("selection_control")
                or diagnostic.get("router_diagnostics")
                != terminal.get("router_diagnostics")
            )
        )
        or capability.get(
            "target_labels_opened_only_after_durable_preterminal_attestation"
        )
        is not True
        or capability.get("raw_labels_persisted") is not False
        or capability.get("sample_paths_persisted") is not False
        or not isinstance(lifecycle, dict)
        or lifecycle.get("phase") != "TERMINAL"
        or not lifecycle.get("durable_preterminal_attestation_hash")
        or lifecycle.get("target_labels_can_change_preterminal_decisions")
        is not False
        or lifecycle.get("publication_status") != PUBLICATION_STATUS
        or lifecycle.get("terminal_decision") != TERMINAL_DECISION
        or lifecycle.get("raw_labels_persisted") is not False
        or leakage.get("status") != "PASS"
        or leakage.get("exact_six_inputs") is not True
        or leakage.get("target_labels_can_change_preterminal_decisions")
        is not False
        or leakage.get("v1_or_prior_v2_state_used") is not False
        or leakage.get("raw_labels_persisted") is not False
        or leakage.get("fresh_evidence") is not False
        or publication.get("terminal_result_hash") != result_hash
        or publication.get("publication_status") != PUBLICATION_STATUS
        or publication.get("terminal_decision") != TERMINAL_DECISION
        or any(
            publication.get(key) is not False
            for key in (
                "fresh_evidence",
                "routing_success_claimed",
                "downstream_utility_claimed",
                "promotion_eligible",
                "may_feed_another_experiment",
                "may_feed_stage50",
                "may_feed_stage60",
                "may_feed_stage70",
                "may_feed_recipe_selection",
            )
        )
        or runtime.get("workstation_preflight_status") != "PASS"
        or runtime.get("outer_execution_mode") != "spawn"
        or runtime.get("outer_worker_count") != 4
        or runtime.get("outer_chunks_written_atomically") is not True
        or runtime.get("outer_chunks_verified_after_write") is not True
        or runtime.get("cpu_phase_cuda_hidden") is not True
        or runtime.get("outer_blas_threads_per_worker") != 1
        or runtime.get("nested_process_pools") is not False
        or runtime.get("scratch_recovery_used") is not False
    ):
        raise ProtocolError("P-DCAPS v2 terminal-only report contract drifted.")
    return hashes


__all__ = (
    "FINAL_REPORT_MEMBERS",
    "FINAL_REPORT_SCHEMAS",
    "diagnostic_summary_payload",
    "label_capability_report_payload",
    "leakage_report_payload",
    "publication_decision_payload",
    "runtime_summary_payload",
    "validate_final_report_payloads",
    "validation_report_payload",
)
