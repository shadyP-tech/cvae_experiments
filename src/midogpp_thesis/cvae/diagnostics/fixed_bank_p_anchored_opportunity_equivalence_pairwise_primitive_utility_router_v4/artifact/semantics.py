"""Semantic reopening for a completed, aggregate-only v4 artifact.

This layer intentionally validates durable v4 receipts rather than replaying
the scientific fit.  It reopens the exact preterminal files and aggregate-only
terminal bundle, binds them to the prepared/committed run state, and rejects
raw-label or predecessor operational residue.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

from ....protocol import ProtocolError
from .completion import discover_completion_commit
from .contracts import (
    FINAL_BINDING_MEMBER,
    TERMINAL_METRICS_MEMBER,
    CompletionCommitReceipt,
)
from ..execution.preterminal_artifact import (
    MANIFEST_MEMBER,
    MATRIX_MEMBER,
    PRETERMINAL_ATTESTATION_MEMBER,
    _validate_preterminal_files,
)
from ..hashing import canonical_hash, require_sha256
from ..output_persistence import (
    _assert_aggregate_only,
    _read_json_object,
    _sha256_file,
)
from ..output_validation import (
    FinalAggregateBundleReceipt,
    _issue_final_aggregate_bundle,
    validate_final_aggregate_bundle,
)
from ..terminal.contracts import (
    _reconstruct_persisted_aggregate_only_terminal_receipt,
)


@dataclass(frozen=True, slots=True)
class _PreparedStateBinding:
    complete_payload: dict[str, object]
    complete_file_sha256: str
    state_hash: str
    receipt_hash: str
    final_bundle_receipt_hash: str


@dataclass(frozen=True, slots=True)
class _SemanticReopenResult:
    semantic_validation_hash: str
    source_seal_hash: str
    final_bundle_receipt_hash: str


def _validate_prepared_complete_state_for_build(
    root: Path,
    expected_complete_state: object,
) -> tuple[_PreparedStateBinding, FinalAggregateBundleReceipt]:
    from ..run_state import (
        PreparedCompleteRunState,
        prepare_complete_run_state,
        validate_prepared_complete_run_state,
    )

    if type(expected_complete_state) is not PreparedCompleteRunState:
        raise ProtocolError("OE-PPUR v4 complete seal requires prepared state.")
    prepared = validate_prepared_complete_run_state(expected_complete_state)
    if prepared.artifact_root != root:
        raise ProtocolError("OE-PPUR v4 prepared state root drifted.")
    final_bundle = validate_final_aggregate_bundle(
        root,
        expected_receipt=_issue_final_aggregate_bundle(root),
    )
    rebuilt = prepare_complete_run_state(root, final_bundle=final_bundle)
    if rebuilt != prepared or prepared.final_bundle_receipt_hash != final_bundle.receipt_hash:
        raise ProtocolError("OE-PPUR v4 prepared state/final bundle drifted.")
    payload = _thaw_json(prepared.complete_payload)
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v4 prepared COMPLETE payload is malformed.")
    return (
        _PreparedStateBinding(
            complete_payload=payload,
            complete_file_sha256=hashlib.sha256(
                prepared.canonical_complete_bytes
            ).hexdigest(),
            state_hash=prepared.state_hash,
            receipt_hash=prepared.receipt_hash,
            final_bundle_receipt_hash=prepared.final_bundle_receipt_hash,
        ),
        final_bundle,
    )


def _require_prepared_complete_state_type(value: object) -> None:
    from ..run_state import PreparedCompleteRunState

    if type(value) is not PreparedCompleteRunState:
        raise ProtocolError("OE-PPUR v4 complete seal requires prepared state.")


def _validate_committed_complete_state(
    root: Path,
    *,
    expected_complete_state: object | None,
) -> dict[str, object]:
    from ..lease_claim import (
        AuthorizationLeaseClaim,
        LEASE_DIRECTORY_NAME,
        validate_authorization_lease,
    )
    from ..lease_io import read_json_regular
    from ..run_state import (
        PreparedCompleteRunState,
        read_terminal_run_state,
        validate_terminal_run_state,
    )

    terminal = validate_terminal_run_state(read_terminal_run_state(root))
    if terminal.status != "COMPLETE" or terminal.phase != "COMPLETE":
        raise ProtocolError("OE-PPUR v4 complete artifact state is not COMPLETE.")
    lease_path = root.parent / LEASE_DIRECTORY_NAME
    claim_payload = read_json_regular(
        lease_path / "claim.json",
        role="authorization claim",
    )
    claim = validate_authorization_lease(
        AuthorizationLeaseClaim(
            lease_path,
            claim_payload,
            str(claim_payload.get("claim_hash", "")),
        )
    )
    completion = discover_completion_commit(claim)
    if type(completion) is not CompletionCommitReceipt:
        raise ProtocolError("OE-PPUR v4 COMPLETE lacks its durable commit journal.")
    payload = _read_json_object(root / "reports/run_state.json")
    if payload.get("state_hash") != terminal.state_hash:
        raise ProtocolError("OE-PPUR v4 complete state receipt drifted from bytes.")
    if expected_complete_state is not None:
        if type(expected_complete_state) is not PreparedCompleteRunState:
            raise ProtocolError("OE-PPUR v4 expected prepared state is untyped.")
        expected_payload = _thaw_json(expected_complete_state.complete_payload)
        if (
            expected_complete_state.artifact_root != root
            or expected_complete_state.state_hash != terminal.state_hash
            or expected_payload != payload
        ):
            raise ProtocolError("OE-PPUR v4 committed state differs from preparation.")
    return payload


def _semantic_reopen_complete_artifact(
    root: Path,
    *,
    complete_state_payload: Mapping[str, object],
    final_bundle: FinalAggregateBundleReceipt,
) -> _SemanticReopenResult:
    if type(final_bundle) is not FinalAggregateBundleReceipt:
        raise ProtocolError("OE-PPUR v4 semantic reopen final bundle is untyped.")
    validated_bundle = validate_final_aggregate_bundle(
        root,
        expected_receipt=final_bundle,
    )
    preterminal_payload = _read_json_object(root / MANIFEST_MEMBER)
    preterminal = _validate_preterminal_files(
        root / MANIFEST_MEMBER,
        root / MATRIX_MEMBER,
        expected_ledger_hash=str(preterminal_payload.get("decision_ledger_hash", "")),
        expected_result_hash=str(preterminal_payload.get("result_hash", "")),
    )
    preterminal_attestation = _read_json_object(
        root / PRETERMINAL_ATTESTATION_MEMBER
    )
    attestation_rows = preterminal_attestation.get("attestations")
    guarded_boundary = preterminal_attestation.get("guarded_boundary")
    binding = _read_json_object(root / FINAL_BINDING_MEMBER)
    admission = _read_json_object(root / "provenance/execution_admission.json")
    replay = _read_json_object(root / "preparation/sealed_execution_replay.json")
    terminal_payload = _read_json_object(root / TERMINAL_METRICS_MEMBER)
    _assert_aggregate_only(binding)
    _assert_aggregate_only(admission)
    _assert_aggregate_only(replay)
    _assert_aggregate_only(terminal_payload)
    terminal = _reconstruct_persisted_aggregate_only_terminal_receipt(
        terminal_payload
    )
    source_seal_hash = require_sha256(
        binding.get("source_seal_hash"),
        "complete artifact source seal",
    )
    launch_hash = require_sha256(
        binding.get("execution_launch_authority_sha256"),
        "complete artifact launch authority",
    )
    admission_hash = require_sha256(
        binding.get("seven_input_admission_hash"),
        "complete artifact admission",
    )
    admission_body = {
        key: value for key, value in admission.items() if key != "receipt_hash"
    }
    replay_body = {
        key: value for key, value in replay.items() if key != "receipt_hash"
    }
    run_identity_hash = require_sha256(
        complete_state_payload.get("run_identity_hash"),
        "complete artifact run identity",
    )
    attestation_receipt_hashes: tuple[str, ...] = ()
    if isinstance(attestation_rows, list) and all(
        isinstance(row, dict) for row in attestation_rows
    ):
        attestation_receipt_hashes = tuple(
            str(row.get("receipt_hash")) for row in attestation_rows
        )
        for row in attestation_rows:
            row_body = {
                key: value for key, value in row.items() if key != "receipt_hash"
            }
            if (
                set(row)
                != {
                    "schema_version",
                    "sealed_ledger_receipt_hash",
                    "artifact_file_sha256",
                    "artifact_file_identity_sha256",
                    "validator_runtime_sha256",
                    "process_pid",
                    "artifact_only",
                    "raw_path_present",
                    "raw_labels_present",
                    "receipt_hash",
                }
                or row.get("schema_version")
                != "oe_ppur_v4_artifact_only_preterminal_attestation_v1"
                or row.get("receipt_hash") != canonical_hash(row_body)
                or row.get("sealed_ledger_receipt_hash")
                != preterminal["sealed_ledger_receipt_hash"]
                or row.get("artifact_file_sha256")
                != preterminal["artifact_file_sha256"]
                or row.get("artifact_file_identity_sha256")
                != preterminal["artifact_file_identity_sha256"]
                or row.get("artifact_only") is not True
                or row.get("raw_path_present") is not False
                or row.get("raw_labels_present") is not False
            ):
                raise ProtocolError(
                    "OE-PPUR v4 complete preterminal attestation drifted."
                )
    guarded_body = (
        {
            key: value
            for key, value in guarded_boundary.items()
            if key != "receipt_hash"
        }
        if isinstance(guarded_boundary, dict)
        else {}
    )
    if (
        launch_hash == "0" * 64
        or binding.get("terminal_receipt_hash") != terminal.receipt_hash
        or admission.get("schema_version")
        != "oe_ppur_v4_seven_input_run_admission_v1"
        or admission.get("receipt_hash") != canonical_hash(admission_body)
        or admission.get("receipt_hash") != admission_hash
        or admission.get("execution_launch_authority_sha256") != launch_hash
        or _sha256_file(root / "preparation/execution_launch_authority.json")
        != launch_hash
        or admission.get("final_envelope_sha256")
        != _sha256_file(root / "preparation/final_authorization_envelope.json")
        or replay.get("schema_version") != "oe_ppur_v4_sealed_execution_replay_v1"
        or replay.get("receipt_hash") != canonical_hash(replay_body)
        or replay.get("receipt_hash") != admission.get("sealed_replay_receipt_hash")
        or replay.get("workspace_snapshot_sha256")
        != admission.get("workspace_snapshot_sha256")
        or replay.get("workspace_plan_sha256")
        != admission.get("workspace_plan_sha256")
        or replay.get("final_envelope_sha256")
        != admission.get("final_envelope_sha256")
        or replay.get("target_labels_opened") is not False
        or replay.get("filesystem_mutation_performed") is not False
        or binding.get("preterminal_ledger_receipt_hash")
        != preterminal["sealed_ledger_receipt_hash"]
        or preterminal_attestation.get("schema_version")
        != "oe_ppur_v4_two_fresh_preterminal_attestations_v1"
        or preterminal_attestation.get("fresh_process_count") != 2
        or preterminal_attestation.get("target_labels_opened") is not False
        or not isinstance(attestation_rows, list)
        or len(attestation_rows) != 2
        or len({row.get("receipt_hash") for row in attestation_rows if isinstance(row, dict)})
        != 2
        or len({row.get("process_pid") for row in attestation_rows if isinstance(row, dict)})
        != 2
        or len(set(attestation_receipt_hashes)) != 2
        or len(
            {
                row.get("validator_runtime_sha256")
                for row in attestation_rows
                if isinstance(row, dict)
            }
        )
        != 1
        or not isinstance(guarded_boundary, dict)
        or guarded_boundary.get("schema_version")
        != "oe_ppur_v4_guarded_preterminal_boundary_v1"
        or guarded_boundary.get("receipt_hash") != canonical_hash(guarded_body)
        or tuple(guarded_boundary.get("preterminal_attestation_hashes", ()))
        != attestation_receipt_hashes
        or guarded_boundary.get("case_count") != 218
        or guarded_boundary.get("case_inventory_sha256")
        != preterminal_payload.get("case_inventory_sha256")
        or guarded_boundary.get("exact_p_fallback_count")
        != preterminal_payload.get("exact_p_count")
        or guarded_boundary.get("decision_ledger_receipt_hash")
        != preterminal["sealed_ledger_receipt_hash"]
        or complete_state_payload.get("seven_input_admission_hash")
        != admission_hash
        or complete_state_payload.get("execution_launch_authority_sha256")
        != launch_hash
        or complete_state_payload.get("status") != "COMPLETE"
        or complete_state_payload.get("phase") != "COMPLETE"
        or complete_state_payload.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("OE-PPUR v4 complete semantic lineage drifted.")
    semantic_payload = {
        "schema_version": "oe_ppur_v4_complete_semantic_reopen_v1",
        "artifact_root": root.as_posix(),
        "run_identity_hash": run_identity_hash,
        "seven_input_admission_hash": admission_hash,
        "execution_launch_authority_sha256": launch_hash,
        "source_seal_hash": source_seal_hash,
        "preterminal_result_hash": preterminal["result_hash"],
        "preterminal_ledger_hash": preterminal["sealed_ledger_receipt_hash"],
        "terminal_receipt_hash": terminal.receipt_hash,
        "terminal_metrics_file_sha256": _sha256_file(root / TERMINAL_METRICS_MEMBER),
        "final_bundle_receipt_hash": validated_bundle.receipt_hash,
        "raw_labels_persisted": False,
        "per_case_values_persisted": False,
        "v3_operational_state_used": False,
    }
    return _SemanticReopenResult(
        semantic_validation_hash=canonical_hash(semantic_payload),
        source_seal_hash=source_seal_hash,
        final_bundle_receipt_hash=validated_bundle.receipt_hash,
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ProtocolError("OE-PPUR v4 prepared state is not canonical JSON.")


__all__: tuple[str, ...] = ()
