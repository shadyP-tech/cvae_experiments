"""Durable aggregate-only final-member persistence for OE-PPUR v3.

This module is deliberately downstream of the preterminal decision seal and
the one-shot terminal evaluator.  It has no API accepting labels, prediction
vectors, case identifiers, or per-case oracle diagnostics.  Its only terminal
input is :class:`AggregateOnlyTerminalReceipt`.

The public output façade calls the private persistence composer only after
preterminal persistence and fresh-process validation have completed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import errno
import hashlib
import json
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .config import RouterV3Config, validate_authorization_ready_config
from .execution.inputs import SevenInputContractReceipt, validate_seven_input_contract
from .hashing import canonical_hash, canonical_json_bytes
from .identity import (
    EXPECTED_CASE_COUNT,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .protocol import claim_boundary_payload
from .source_seal import SourceSealReceipt, validate_source_seal
from .source_supervision import SourceTrainingSurface
from .execution.preterminal_artifact import (
    FinalAggregateAttestationReceipt,
    _reconstruct_final_aggregate_attestation,
)
from .terminal.contracts import (
    AggregateOnlyTerminalReceipt,
    ArtifactOnlyPreterminalAttestationReceipt,
    GuardedPreterminalBoundary,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
)


FINAL_BINDING_MEMBER = "provenance/final_aggregate_binding.json"
TERMINAL_RESULT_MEMBER = "tables/terminal_result.json"
DIAGNOSTIC_SUMMARY_MEMBER = "reports/diagnostic_summary.json"
LEAKAGE_REPORT_MEMBER = "reports/leakage_report.json"
PUBLICATION_DECISION_MEMBER = "reports/publication_decision.json"
RUNTIME_SUMMARY_MEMBER = "reports/runtime_summary.json"
CLAIM_BOUNDARY_MEMBER = "reports/claim_boundary.json"
FINAL_ATTESTATION_MEMBER = "reports/final_fresh_process_attestation.json"
CONTENT_INDEX_MEMBER = "manifests/content_index.json"
VALIDATION_REPORT_MEMBER = "reports/validation_report.json"
VALIDATION_INDEX_MEMBER = "manifests/validation_index.json"
COMPLETE_ARTIFACT_INDEX_MEMBER = "manifests/complete_artifact_index.json"
TERMINAL_METRICS_MEMBER = "reports/terminal_metrics.json"

FINAL_PAYLOAD_MEMBERS = (
    FINAL_BINDING_MEMBER,
    TERMINAL_RESULT_MEMBER,
    DIAGNOSTIC_SUMMARY_MEMBER,
    LEAKAGE_REPORT_MEMBER,
    PUBLICATION_DECISION_MEMBER,
    RUNTIME_SUMMARY_MEMBER,
    CLAIM_BOUNDARY_MEMBER,
)
FINAL_INDEXED_MEMBERS = (*FINAL_PAYLOAD_MEMBERS, FINAL_ATTESTATION_MEMBER)

# Exact catalog-required file inventory for the complete v3 output.  The run
# lock is an internal same-run member and therefore tracked separately below.
COMPLETE_CATALOG_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "provenance/execution_admission.json",
    "provenance/authorization_consumption_lease.json",
    FINAL_BINDING_MEMBER,
    "physical/source_streams/arrays/frozen_source_streams.npy",
    "physical/source_streams/manifests/frozen_source_stream_index.json",
    "physical/source_streams/manifests/frozen_source_stream_lock.json",
    "physical/predictions/arrays/fixed_bank_a1_action_probabilities.npz",
    "physical/predictions/manifests/fixed_bank_a1_prediction_index.json",
    "physical/predictions/manifests/fixed_bank_a1_prediction_seal.json",
    "arrays/preterminal_probability_matrix.npy",
    "manifests/preterminal_result.json",
    "reports/launch_receipts.json",
    "reports/preterminal_fresh_process_attestation.json",
    TERMINAL_METRICS_MEMBER,
    TERMINAL_RESULT_MEMBER,
    DIAGNOSTIC_SUMMARY_MEMBER,
    LEAKAGE_REPORT_MEMBER,
    PUBLICATION_DECISION_MEMBER,
    RUNTIME_SUMMARY_MEMBER,
    CLAIM_BOUNDARY_MEMBER,
    CONTENT_INDEX_MEMBER,
    FINAL_ATTESTATION_MEMBER,
    VALIDATION_REPORT_MEMBER,
    VALIDATION_INDEX_MEMBER,
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    "reports/run_state.json",
)
COMPLETE_INTERNAL_MEMBERS = (".run.lock",)

_FORBIDDEN_SUBSTRINGS = (
    "raw_label",
    "row_label",
    "case_id",
    "row_id",
    "sample_id",
    "prediction_vector",
    "oracle_action",
    "rank_diagnostic",
    "per_case",
    "per_row",
)


def _sha256_file(path: Path) -> str:
    raw, _metadata = _read_regular_bytes_nofollow(path)
    return hashlib.sha256(raw).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    raw, _metadata = _read_regular_bytes_nofollow(path)

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 final output member is unreadable.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("OE-PPUR v3 final output member is not a JSON object.")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    _ensure_safe_parent(path)
    normalized = json.loads(canonical_json_bytes(dict(payload)).decode("ascii"))
    raw = (
        json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short OE-PPUR v3 final output write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno == errno.EEXIST or path.exists() or path.is_symlink():
            raise ProtocolError(
                "OE-PPUR v3 final output refuses overwrite or recovery."
            ) from exc
        raise ProtocolError("OE-PPUR v3 final output exclusive write failed.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_directory(path.parent)
    if _read_json_object(path) != normalized:
        raise ProtocolError("OE-PPUR v3 final output read-back drifted.")


def _read_regular_bytes_nofollow(path: Path) -> tuple[bytes, os.stat_result]:
    candidate = Path(os.path.abspath(path))
    _reject_symlink_chain(candidate)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 final output member is absent or unsafe.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError(
                "OE-PPUR v3 final output member is not a unique regular file."
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if _stat_payload(before) != _stat_payload(after) or len(raw) != before.st_size:
        raise ProtocolError("OE-PPUR v3 final output member changed while read.")
    return raw, before


def _ensure_safe_parent(path: Path) -> None:
    candidate = Path(os.path.abspath(path))
    if candidate != path or path == Path(path.anchor):
        raise ProtocolError("OE-PPUR v3 final output path is unsafe.")
    missing: list[Path] = []
    current = path.parent
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        current = current.parent
    _reject_symlink_chain(current)
    if not current.is_dir():
        raise ProtocolError("OE-PPUR v3 final output parent is unsafe.")
    for directory in reversed(missing):
        try:
            directory.mkdir(exist_ok=False)
        except OSError as exc:
            raise ProtocolError("OE-PPUR v3 final output parent creation failed.") from exc
        _fsync_directory(directory.parent)
    _reject_symlink_chain(path.parent)
    if not path.parent.is_dir():
        raise ProtocolError("OE-PPUR v3 final output parent is unsafe.")


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 final output path contains a symlink.")
        if current == current.parent:
            return
        current = current.parent


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 final output directory is unsafe.") from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _contains_forbidden_payload(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            # TerminalReceipt constrains this dictionary to its fixed aggregate
            # metric vocabulary.  Metric names may contain "oracle" or "rank",
            # but its scalar values are not per-case diagnostics.
            if lowered == "aggregate_metrics" and isinstance(child, Mapping):
                if any(_contains_forbidden_payload(item) for item in child.values()):
                    return True
                continue
            if (
                any(token in lowered for token in _FORBIDDEN_SUBSTRINGS)
                and child is not False
            ):
                return True
            if _contains_forbidden_payload(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_forbidden_payload(child) for child in value)
    return False


def _assert_aggregate_only(value: Mapping[str, object]) -> None:
    if _contains_forbidden_payload(value):
        raise ProtocolError("OE-PPUR v3 final output would persist forbidden terminal data.")
    encoded = canonical_json_bytes(dict(value)).decode("ascii").lower()
    # Uniform-B v2 is legitimate upstream provenance.  OE-PPUR v2 is not.
    if "opportunity_equivalence_pairwise_primitive_utility_router_v2" in encoded:
        raise ProtocolError("OE-PPUR v3 final output reused predecessor v2 state.")
    if (
        '"cross_run_recovery_allowed":true' in encoded
        or '"recovery_allowed":true' in encoded
    ):
        raise ProtocolError("OE-PPUR v3 final output enables recovery.")


def _binding_payload(
    *,
    config: RouterV3Config,
    inputs: SevenInputContractReceipt,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
    boundary: GuardedPreterminalBoundary,
    attestations: tuple[ArtifactOnlyPreterminalAttestationReceipt, ...],
    terminal: AggregateOnlyTerminalReceipt,
    final_attestation: FinalAggregateAttestationReceipt,
) -> dict[str, object]:
    if (
        config.execution_authorized is not True
        or inputs.execution_authorized is not True
        or config.seven_input_contract_hash != inputs.receipt_hash
        or config.protocol_hash != str(config.to_payload()["protocol"]["protocol_hash"])
        or boundary.seven_input_contract_hash != inputs.receipt_hash
        or boundary.source_seal_hash != source_seal.combined_source_sha256
        or source_surface.receipt.target_rows_present
        or source_surface.receipt.target_labels_used
        or source_surface.receipt.receipt_hash
        != config.source_supervision_content_sha256
        or source_surface.receipt.row_order_sha256
        != config.source_supervision_row_order_sha256
        or source_surface.receipt.contract.producer_source_seal_sha256
        != source_seal.combined_source_sha256
        or source_surface.receipt.contract.producer_source_seal_sha256
        != config.source_supervision_producer_seal_sha256
        or source_surface.receipt.compiler_recomputation_receipt_sha256
        != config.source_supervision_recomputation_receipt_sha256
        or boundary.source_training_surface_receipt_hash
        != source_surface.receipt.receipt_hash
        or terminal.boundary_receipt_hash != boundary.receipt_hash
        or terminal.decision_ledger_receipt_hash != boundary.decision_ledger_receipt_hash
        or final_attestation.terminal_receipt_hash != terminal.receipt_hash
        or terminal.evaluated_case_count != EXPECTED_CASE_COUNT
        or terminal.exact_p_fallback_count != boundary.exact_p_fallback_count
    ):
        raise ProtocolError("OE-PPUR v3 final aggregate lineage drifted.")
    if (
        len(attestations) != 2
        or len({row.receipt_hash for row in attestations}) != 2
        or len({row.process_pid for row in attestations}) != 2
        or len({row.artifact_file_sha256 for row in attestations}) != 1
        or len({row.artifact_file_identity_sha256 for row in attestations}) != 1
        or len({row.validator_runtime_sha256 for row in attestations}) != 1
        or any(
            row.sealed_ledger_receipt_hash != boundary.decision_ledger_receipt_hash
            for row in attestations
        )
        or tuple(row.receipt_hash for row in attestations)
        != boundary.preterminal_attestation_hashes
    ):
        raise ProtocolError("OE-PPUR v3 final output lacks its bound two-process attestation.")
    payload = {
        "schema_version": "oe_ppur_v3_final_aggregate_binding_v1",
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": config.contract_hash,
        "protocol_hash": config.protocol_hash,
        "seven_input_contract_hash": inputs.receipt_hash,
        "source_seal_hash": source_seal.combined_source_sha256,
        "source_seal_receipt_hash": source_seal.receipt_hash,
        "source_supervision_contract_hash": source_surface.receipt.contract.contract_hash,
        "source_training_surface_receipt_hash": source_surface.receipt.receipt_hash,
        "source_training_surface_hash": source_surface.surface_hash,
        "preterminal_boundary_receipt_hash": boundary.receipt_hash,
        "preterminal_ledger_receipt_hash": boundary.decision_ledger_receipt_hash,
        "preterminal_attestation_receipt_hashes": [
            row.receipt_hash for row in attestations
        ],
        "terminal_receipt_hash": terminal.receipt_hash,
        "final_attestation_hash": final_attestation.receipt_hash,
        "evaluated_case_count": terminal.evaluated_case_count,
        "exact_p_fallback_count": terminal.exact_p_fallback_count,
        "aggregate_only": True,
        "raw_labels_persisted": False,
        "v2_input_or_state_used": False,
        "cross_run_recovery_allowed": False,
    }
    _assert_aggregate_only(payload)
    return payload


def _runtime_payload(value: Mapping[str, object]) -> dict[str, object]:
    runtime = dict(value)
    if not runtime or _contains_forbidden_payload(runtime):
        raise ProtocolError("OE-PPUR v3 runtime summary is not aggregate-only.")
    payload = {
        "schema_version": "oe_ppur_v3_runtime_summary_v1",
        "runtime": runtime,
        "cross_run_recovery_allowed": False,
        "raw_labels_persisted": False,
    }
    _assert_aggregate_only(payload)
    return payload


def _reports(
    *,
    binding: Mapping[str, object],
    terminal: AggregateOnlyTerminalReceipt,
    runtime: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    metrics = {key: value for key, value in terminal.aggregate_metrics}
    common = {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "raw_labels_persisted": False,
        "v2_input_or_state_used": False,
        "cross_run_recovery_allowed": False,
    }
    reports = {
        TERMINAL_RESULT_MEMBER: terminal.to_payload(),
        DIAGNOSTIC_SUMMARY_MEMBER: {
            "schema_version": "oe_ppur_v3_aggregate_diagnostic_summary_v1",
            "terminal_receipt_hash": terminal.receipt_hash,
            "evaluated_case_count": terminal.evaluated_case_count,
            "routed_case_count": terminal.routed_case_count,
            "exact_p_fallback_count": terminal.exact_p_fallback_count,
            "aggregate_metrics": metrics,
            **common,
        },
        LEAKAGE_REPORT_MEMBER: {
            "schema_version": "oe_ppur_v3_final_leakage_report_v1",
            "status": "PASS",
            "source_supervision_target_rows_present": False,
            "source_supervision_target_labels_used": False,
            "preterminal_decisions_sealed_before_terminal": True,
            "terminal_result_aggregate_only": True,
            "raw_labels_persisted": False,
            "v2_input_or_state_used": False,
            "cross_run_recovery_allowed": False,
            "binding_hash": canonical_hash(binding),
        },
        PUBLICATION_DECISION_MEMBER: {
            "schema_version": "oe_ppur_v3_publication_decision_v1",
            "decision": TERMINAL_DECISION,
            "publication_status": PUBLICATION_STATUS,
            "promotion_allowed": False,
            "fresh_routing_claim_allowed": False,
            "downstream_utility_claim_allowed": False,
            "nelbo_compatibility_claim_allowed": False,
            "raw_labels_persisted": False,
        },
        RUNTIME_SUMMARY_MEMBER: _runtime_payload(runtime),
        CLAIM_BOUNDARY_MEMBER: {
            "schema_version": "oe_ppur_v3_final_claim_boundary_v1",
            "claim_boundary": claim_boundary_payload(execution_authorized=True),
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "terminal_result_aggregate_only": True,
            "raw_labels_persisted": False,
            "v2_input_or_state_used": False,
            "cross_run_recovery_allowed": False,
        },
    }
    for payload in reports.values():
        _assert_aggregate_only(payload)
    return reports


def _persist_final_aggregate_members(
    root: str | Path,
    *,
    config: RouterV3Config,
    seven_input_contract: SevenInputContractReceipt,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
    preterminal_boundary: GuardedPreterminalBoundary,
    preterminal_attestations: Sequence[ArtifactOnlyPreterminalAttestationReceipt],
    terminal_receipt: AggregateOnlyTerminalReceipt,
    final_attestation: FinalAggregateAttestationReceipt,
    runtime_summary: Mapping[str, object],
) -> Path:
    """Write final aggregate-only members after the two-process final gate.

    The destination is an already-created, preterminal artifact root.  Every
    member written here is exclusive; this code intentionally offers no repair
    or cross-run recovery path.
    """

    destination = Path(root)
    if (
        not destination.is_absolute()
        or not destination.is_dir()
        or destination.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v3 final output root is absent or unsafe.")
    config = validate_authorization_ready_config(config)
    inputs = validate_seven_input_contract(
        seven_input_contract, execution_authorized=True
    )
    seal = validate_source_seal(source_seal)
    if type(source_surface) is not SourceTrainingSurface:
        raise ProtocolError("OE-PPUR v3 final output source surface is untyped.")
    if type(preterminal_boundary) is not GuardedPreterminalBoundary:
        raise ProtocolError("OE-PPUR v3 final output boundary is untyped.")
    if type(terminal_receipt) is not AggregateOnlyTerminalReceipt:
        raise ProtocolError("OE-PPUR v3 final output terminal receipt is untyped.")
    if type(final_attestation) is not FinalAggregateAttestationReceipt:
        raise ProtocolError("OE-PPUR v3 final output attestation is untyped.")
    attestations = tuple(preterminal_attestations)
    if any(
        type(row) is not ArtifactOnlyPreterminalAttestationReceipt
        for row in attestations
    ):
        raise ProtocolError("OE-PPUR v3 final output preterminal attestations are untyped.")

    binding = _binding_payload(
        config=config,
        inputs=inputs,
        source_seal=seal,
        source_surface=source_surface,
        boundary=preterminal_boundary,
        attestations=attestations,
        terminal=terminal_receipt,
        final_attestation=final_attestation,
    )
    persisted_attestation = _read_json_object(destination / FINAL_ATTESTATION_MEMBER)
    reconstructed_attestation = _reconstruct_final_aggregate_attestation(
        persisted_attestation
    )
    persisted_terminal_payload = _read_json_object(
        destination / TERMINAL_METRICS_MEMBER
    )
    persisted_terminal = _reconstruct_persisted_aggregate_only_terminal_receipt(
        persisted_terminal_payload
    )
    terminal_raw, terminal_stat = _read_regular_bytes_nofollow(
        destination / TERMINAL_METRICS_MEMBER
    )
    if (
        reconstructed_attestation != final_attestation
        or persisted_terminal.to_payload() != terminal_receipt.to_payload()
        or final_attestation.terminal_receipt_hash != terminal_receipt.receipt_hash
        or final_attestation.terminal_file_sha256
        != hashlib.sha256(terminal_raw).hexdigest()
        or final_attestation.terminal_file_identity_sha256
        != canonical_hash(
            {
                "schema_version": "oe_ppur_v3_terminal_file_identity_v1",
                "stat": _stat_payload(terminal_stat),
            }
        )
    ):
        raise ProtocolError("OE-PPUR v3 final attestation bytes drifted before assembly.")
    _assert_aggregate_only(persisted_attestation)
    payloads = {
        FINAL_BINDING_MEMBER: binding,
        **_reports(binding=binding, terminal=terminal_receipt, runtime=runtime_summary),
    }
    if tuple(payloads) != FINAL_PAYLOAD_MEMBERS:
        raise ProtocolError("OE-PPUR v3 final output member order drifted.")
    for member, payload in payloads.items():
        _write_json_exclusive(destination / member, payload)

    member_hashes = {
        member: _sha256_file(destination / member) for member in FINAL_INDEXED_MEMBERS
    }
    content_body = {
        "schema_version": "oe_ppur_v3_final_aggregate_content_index_v1",
        "binding_hash": canonical_hash(binding),
        "terminal_receipt_hash": terminal_receipt.receipt_hash,
        "final_attestation_hash": final_attestation.receipt_hash,
        "members": member_hashes,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "aggregate_only": True,
        "raw_labels_persisted": False,
        "v2_input_or_state_used": False,
        "cross_run_recovery_allowed": False,
    }
    _assert_aggregate_only(content_body)
    content = {**content_body, "content_index_hash": canonical_hash(content_body)}
    _write_json_exclusive(destination / CONTENT_INDEX_MEMBER, content)

    validation_body = {
        "schema_version": "oe_ppur_v3_final_aggregate_validation_report_v1",
        "status": "PASS",
        "content_index_hash": content["content_index_hash"],
        "final_attestation_hash": final_attestation.receipt_hash,
        "terminal_receipt_hash": terminal_receipt.receipt_hash,
        "preterminal_ledger_receipt_hash": preterminal_boundary.decision_ledger_receipt_hash,
        "aggregate_metrics": {
            key: value for key, value in terminal_receipt.aggregate_metrics
        },
        "exact_p_fallback_count": terminal_receipt.exact_p_fallback_count,
        "raw_labels_persisted": False,
        "per_case_diagnostics_persisted": False,
        "v2_input_or_state_used": False,
        "cross_run_recovery_allowed": False,
    }
    _assert_aggregate_only(validation_body)
    validation = {
        **validation_body,
        "validation_report_hash": canonical_hash(validation_body),
    }
    _write_json_exclusive(destination / VALIDATION_REPORT_MEMBER, validation)

    validation_index_body = {
        "schema_version": "oe_ppur_v3_final_aggregate_validation_index_v1",
        "status": "PASS",
        "content_index_hash": content["content_index_hash"],
        "content_index_file_sha256": _sha256_file(destination / CONTENT_INDEX_MEMBER),
        "validation_report_hash": validation["validation_report_hash"],
        "validation_report_file_sha256": _sha256_file(
            destination / VALIDATION_REPORT_MEMBER
        ),
        "final_attestation_hash": final_attestation.receipt_hash,
        "final_attestation_file_sha256": _sha256_file(
            destination / FINAL_ATTESTATION_MEMBER
        ),
        "fresh_process_count": 2,
        "aggregate_only": True,
        "raw_labels_persisted": False,
        "v2_input_or_state_used": False,
        "cross_run_recovery_allowed": False,
    }
    _assert_aggregate_only(validation_index_body)
    validation_index = {
        **validation_index_body,
        "validation_index_hash": canonical_hash(validation_index_body),
    }
    _write_json_exclusive(destination / VALIDATION_INDEX_MEMBER, validation_index)
    _fsync_directory(destination)
    return destination


__all__: tuple[str, ...] = ()
