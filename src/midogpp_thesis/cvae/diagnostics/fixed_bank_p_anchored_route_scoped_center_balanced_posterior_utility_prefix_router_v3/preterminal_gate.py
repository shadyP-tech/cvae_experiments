"""Durable contracts for the label-closed preterminal validation gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_io import persist_json
from .bundle import PRETERMINAL_SCIENTIFIC_MEMBERS
from .hashing import canonical_hash


FRESH_WORKER_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def preterminal_validation_checks_payload(
    *,
    config_contract_hash: str,
    protocol_contract_hash: str,
    content_index_hash: str,
    outer_route_count: int,
    target_posterior_model_fit_count: int,
    pseudo_posterior_reference_count: int,
    preterminal_hash: str,
) -> dict[str, object]:
    payload = {
        "schema_version": "fixed_bank_cbpupr_preterminal_validation_checks_v1",
        "status": "PASS",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "content_index_hash": content_index_hash,
        "preterminal_scientific_member_count": len(
            PRETERMINAL_SCIENTIFIC_MEMBERS
        ),
        "outer_route_count": outer_route_count,
        "target_posterior_model_fit_count": target_posterior_model_fit_count,
        "pseudo_posterior_model_fit_count": 0,
        "pseudo_posterior_reference_count": pseudo_posterior_reference_count,
        "preterminal_hash": preterminal_hash,
        "terminal_opened": False,
        "terminal_product_count": 0,
        "terminal_only_consumed_test": True,
        "formal_claim_authorized": False,
    }
    return {**payload, "validation_checks_hash": canonical_hash(payload)}


def persist_preterminal_capability_report(
    root: Path, capability_report: Mapping[str, object]
) -> None:
    events = capability_report.get("events")
    if (
        capability_report.get("schema_version")
        != "fixed_bank_cbpupr_label_access_audit_v1"
        or capability_report.get("aggregate_seal_complete") is not True
        or capability_report.get("terminal_opened") is not False
        or capability_report.get("raw_labels_persisted") is not False
        or not isinstance(events, list)
        or any(
            isinstance(row, Mapping)
            and row.get("role") == "target_terminal_after_aggregate_seal"
            for row in events
        )
    ):
        raise ProtocolError("CBPUPR preterminal capability audit drifted.")
    persist_json(
        root / "reports/preterminal_label_capability_report.json",
        capability_report,
    )


def preterminal_validation_report_payload(
    checks: Mapping[str, object], attestation: Mapping[str, object]
) -> dict[str, object]:
    payload = {
        "schema_version": "fixed_bank_cbpupr_preterminal_validation_report_v1",
        "status": "PASS",
        "checks": dict(checks),
        "fresh_process_attestation_hash": attestation["attestation_hash"],
        "terminal_opened": False,
        "formal_claim_authorized": False,
    }
    return {**payload, "validation_report_hash": canonical_hash(payload)}


def persist_preterminal_validation_report(
    root: Path, payload: Mapping[str, object]
) -> None:
    persist_json(root / "reports/preterminal_validation_report.json", payload)


def persist_preterminal_validation_seal(
    root: Path,
    *,
    checks: Mapping[str, object],
    attestation: Mapping[str, object],
    report: Mapping[str, object],
) -> dict[str, object]:
    expected_report = preterminal_validation_report_payload(checks, attestation)
    if dict(report) != expected_report:
        raise ProtocolError("CBPUPR preterminal validation report lineage drifted.")
    payload = {
        "schema_version": "fixed_bank_cbpupr_preterminal_validation_seal_v1",
        "status": "PASS",
        "preterminal_content_index_hash": checks["content_index_hash"],
        "preterminal_validation_checks_hash": checks["validation_checks_hash"],
        "preterminal_hash": checks["preterminal_hash"],
        "fresh_process_attestation_hash": attestation["attestation_hash"],
        "validation_report_hash": report["validation_report_hash"],
        "terminal_opened": False,
        "formal_claim_authorized": False,
    }
    sealed = {**payload, "validation_seal_hash": canonical_hash(payload)}
    persist_json(root / "manifests/preterminal_validation_seal.json", sealed)
    return sealed


def validate_preterminal_gate_artifacts(
    root: Path, *, expected_checks: Mapping[str, object]
) -> dict[str, object]:
    expected = dict(expected_checks)
    check_unhashed = {
        key: value
        for key, value in expected.items()
        if key != "validation_checks_hash"
    }
    if (
        set(expected)
        != {
            "schema_version",
            "status",
            "config_contract_hash",
            "protocol_contract_hash",
            "content_index_hash",
            "preterminal_scientific_member_count",
            "outer_route_count",
            "target_posterior_model_fit_count",
            "pseudo_posterior_model_fit_count",
            "pseudo_posterior_reference_count",
            "preterminal_hash",
            "terminal_opened",
            "terminal_product_count",
            "terminal_only_consumed_test",
            "formal_claim_authorized",
            "validation_checks_hash",
        }
        or expected.get("schema_version")
        != "fixed_bank_cbpupr_preterminal_validation_checks_v1"
        or expected.get("status") != "PASS"
        or expected.get("terminal_opened") is not False
        or expected.get("terminal_product_count") != 0
        or expected.get("formal_claim_authorized") is not False
        or expected.get("validation_checks_hash")
        != canonical_hash(check_unhashed)
    ):
        raise ProtocolError("CBPUPR preterminal validation checks drifted.")
    expected_hash = canonical_hash(expected)
    attestation = read_json(
        root / "reports/preterminal_fresh_process_attestation.json"
    )
    children = attestation.get("child_process_results")
    child_ids = attestation.get("child_process_ids")
    parent_id = attestation.get("parent_process_id")
    if (
        not isinstance(children, list)
        or len(children) != 2
        or not isinstance(child_ids, list)
        or len(child_ids) != 2
        or not all(type(value) is int and value > 0 for value in child_ids)
        or len(set(child_ids)) != 2
        or type(parent_id) is not int
        or parent_id <= 0
        or parent_id in child_ids
    ):
        raise ProtocolError(
            "CBPUPR preterminal fresh-process identity topology drifted."
        )
    for ordinal, (child, child_id) in enumerate(
        zip(children, child_ids, strict=True), start=1
    ):
        worker_result = {
            "process_id": child_id,
            "validation_phase": "preterminal",
            "checks": expected,
        }
        if child != {
            "ordinal": ordinal,
            "process_id": child_id,
            "exit_code": 0,
            "result_hash": canonical_hash(worker_result),
        }:
            raise ProtocolError(
                "CBPUPR preterminal fresh-process result reconstruction drifted."
            )
    unhashed = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_hash"
    }
    expected_unhashed = {
        "schema_version": (
            "fixed_bank_cbpupr_preterminal_fresh_process_attestation_v1"
        ),
        "status": "PASS",
        "validation_phase": "preterminal",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "cuda_visible_devices": "",
        "worker_thread_environment": FRESH_WORKER_THREAD_ENVIRONMENT,
        "parent_process_id": parent_id,
        "child_process_ids": child_ids,
        "child_process_results": children,
        "reconstructed_checks_exactly_equal": True,
        "reconstructed_checks_hash": expected_hash,
        "validator_entrypoint": "validate_preterminal_bundle",
        "terminal_opened": False,
    }
    if unhashed != expected_unhashed or attestation.get(
        "attestation_hash"
    ) != canonical_hash(unhashed):
        raise ProtocolError("CBPUPR preterminal fresh-process attestation drifted.")

    report = read_json(root / "reports/preterminal_validation_report.json")
    expected_report = preterminal_validation_report_payload(expected, attestation)
    if report != expected_report:
        raise ProtocolError("CBPUPR preterminal validation report drifted.")
    seal = read_json(root / "manifests/preterminal_validation_seal.json")
    expected_seal_unhashed = {
        "schema_version": "fixed_bank_cbpupr_preterminal_validation_seal_v1",
        "status": "PASS",
        "preterminal_content_index_hash": expected["content_index_hash"],
        "preterminal_validation_checks_hash": expected[
            "validation_checks_hash"
        ],
        "preterminal_hash": expected["preterminal_hash"],
        "fresh_process_attestation_hash": attestation["attestation_hash"],
        "validation_report_hash": report["validation_report_hash"],
        "terminal_opened": False,
        "formal_claim_authorized": False,
    }
    expected_seal = {
        **expected_seal_unhashed,
        "validation_seal_hash": canonical_hash(expected_seal_unhashed),
    }
    if seal != expected_seal:
        raise ProtocolError("CBPUPR preterminal validation seal drifted.")
    return seal


__all__ = (
    "FRESH_WORKER_THREAD_ENVIRONMENT",
    "persist_preterminal_capability_report",
    "persist_preterminal_validation_report",
    "persist_preterminal_validation_seal",
    "preterminal_validation_checks_payload",
    "preterminal_validation_report_payload",
    "validate_preterminal_gate_artifacts",
)
