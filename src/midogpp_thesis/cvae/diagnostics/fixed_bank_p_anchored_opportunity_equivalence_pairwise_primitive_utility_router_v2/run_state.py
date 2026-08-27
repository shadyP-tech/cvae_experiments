"""Monotone, irrecoverable run state for OE-PPUR v2."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from ...protocol import ProtocolError
from .authorization_lease import (
    AuthorizationLeaseClaim,
    validate_authorization_lease,
)
from .execution_admission import SixInputAdmissionReceipt
from .hashing import canonical_hash, canonical_json_bytes, require_sha256
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .run_paths import is_exact_workspace_launch_envelope


RUN_STATE_SCHEMA = "oe_ppur_v2_single_use_run_state_v1"
PHASE_ORDER = (
    "ADMITTED",
    "INPUTS_SEALED",
    "PROBABILITY_MATRIX_SEALED",
    "OUTER_FOLDS_COMPLETE",
    "PRETERMINAL_DECISIONS_SEALED",
    "PRETERMINAL_ATTESTED",
    "TERMINAL_AGGREGATES_SCORED",
    "FINAL_ATTESTED",
    "COMPLETE",
)


def build_run_identity_hash(
    *,
    config_hash: str,
    protocol_hash: str,
    source_contract_hash: str,
    admission_receipt_hash: str,
) -> str:
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v2_run_identity_v1",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "config_hash": require_sha256(config_hash, "config hash"),
            "protocol_hash": require_sha256(protocol_hash, "protocol hash"),
            "source_contract_hash": require_sha256(
                source_contract_hash, "source contract hash"
            ),
            "admission_receipt_hash": require_sha256(
                admission_receipt_hash, "admission receipt hash"
            ),
        }
    )


def create_single_use_run(
    admission: SixInputAdmissionReceipt,
    lease: AuthorizationLeaseClaim,
    *,
    run_identity_hash: str,
) -> dict[str, object]:
    """Create roots only after the external authorization is consumed."""

    if not isinstance(admission, SixInputAdmissionReceipt):
        raise ProtocolError("OE-PPUR v2 run requires typed admission.")
    validated_lease = validate_authorization_lease(
        lease, expected_claim_hash=lease.claim_hash
    )
    root = Path(admission.artifact_root)
    scratch = Path(admission.scratch_root)
    prepared = is_exact_workspace_launch_envelope(root)
    if (root.exists() or root.is_symlink()) and not prepared:
        raise ProtocolError("OE-PPUR v2 output root already exists; recovery forbidden.")
    if scratch.exists() or scratch.is_symlink():
        raise ProtocolError("OE-PPUR v2 scratch root already exists; recovery forbidden.")
    run_hash = require_sha256(run_identity_hash, "run identity hash")
    if (
        validated_lease.payload.get("run_identity_hash") != run_hash
        or validated_lease.payload.get("six_input_admission_hash")
        != admission.receipt_hash
        or validated_lease.payload.get("artifact_root") != str(root)
        or validated_lease.payload.get("scratch_root") != str(scratch)
    ):
        raise ProtocolError("OE-PPUR v2 run/lease binding drifted.")
    if not prepared:
        root.mkdir(parents=True, exist_ok=False)
        for relative in ("manifests", "provenance", "reports", "tables"):
            (root / relative).mkdir()
    _atomic_json(
        root / "provenance/authorization_consumption_lease.json",
        validated_lease.to_payload(),
    )
    try:
        scratch.mkdir(parents=True, exist_ok=False)
    except BaseException as exc:
        state = _initial_state(admission, validated_lease, run_hash)
        failed = _replace_state(
            state,
            status="FAILED_EXHAUSTED",
            phase="ADMITTED",
            evidence_hash=canonical_hash(
                {
                    "schema_version": "oe_ppur_v2_root_creation_failure_v1",
                    "error_class": type(exc).__name__,
                }
            ),
            error_class=type(exc).__name__,
        )
        _atomic_json(root / "reports/run_state.json", failed)
        raise ProtocolError(
            "OE-PPUR v2 scratch creation failed; authorization is exhausted."
        ) from exc
    _atomic_json(root / "provenance/execution_admission.json", admission.to_payload())
    lock_body = {
        "schema_version": "oe_ppur_v2_run_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "run_identity_hash": run_hash,
        "authorization_lease_claim_hash": validated_lease.claim_hash,
        "authorization_exhausted": True,
        "recovery_allowed": False,
    }
    _write_exclusive_json(root / ".run.lock", {**lock_body, "lock_hash": canonical_hash(lock_body)})
    state = _initial_state(admission, validated_lease, run_hash)
    _atomic_json(root / "reports/run_state.json", state)
    return state


def transition_run(
    artifact_root: str | Path,
    next_phase: str,
    *,
    evidence_hash: str,
    expected_phase: str,
) -> dict[str, object]:
    root = Path(artifact_root)
    state = read_run_state(root)
    target = str(next_phase)
    current = str(state["phase"])
    if (
        state.get("status") != "RUNNING"
        or current != expected_phase
        or target not in PHASE_ORDER
        or target in {"ADMITTED", "COMPLETE"}
        or PHASE_ORDER.index(target) != PHASE_ORDER.index(current) + 1
        or (
            target == "TERMINAL_AGGREGATES_SCORED"
            and current != "PRETERMINAL_ATTESTED"
        )
    ):
        raise ProtocolError("OE-PPUR v2 run transition is out of order.")
    updated = _replace_state(
        state,
        status="RUNNING",
        phase=target,
        evidence_hash=require_sha256(evidence_hash, "transition evidence hash"),
        error_class=None,
    )
    _atomic_json(root / "reports/run_state.json", updated)
    return updated


def mark_failed_exhausted(
    artifact_root: str | Path,
    *,
    error_class: str,
    evidence_hash: str,
) -> dict[str, object]:
    root = Path(artifact_root)
    state = read_run_state(root)
    if state.get("status") != "RUNNING":
        raise ProtocolError("OE-PPUR v2 failed state cannot be rewritten.")
    updated = _replace_state(
        state,
        status="FAILED_EXHAUSTED",
        phase=str(state["phase"]),
        evidence_hash=require_sha256(evidence_hash, "failure evidence hash"),
        error_class=_safe_text(error_class),
    )
    _atomic_json(root / "reports/run_state.json", updated)
    return updated


def mark_complete(
    artifact_root: str | Path,
    *,
    final_attestation_hash: str,
) -> dict[str, object]:
    root = Path(artifact_root)
    state = read_run_state(root)
    if state.get("status") != "RUNNING" or state.get("phase") != "FINAL_ATTESTED":
        raise ProtocolError("OE-PPUR v2 completion requires final attestation.")
    updated = _replace_state(
        state,
        status="COMPLETE",
        phase="COMPLETE",
        evidence_hash=require_sha256(
            final_attestation_hash, "final attestation hash"
        ),
        error_class=None,
    )
    _atomic_json(root / "reports/run_state.json", updated)
    return updated


def read_run_state(artifact_root: str | Path) -> dict[str, object]:
    path = Path(artifact_root) / "reports/run_state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v2 run state is unreadable.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("OE-PPUR v2 run state is malformed.")
    history = value.get("transitions")
    unhashed = {key: item for key, item in value.items() if key != "state_hash"}
    if (
        value.get("schema_version") != RUN_STATE_SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or value.get("status") not in {"RUNNING", "FAILED_EXHAUSTED", "COMPLETE"}
        or value.get("phase") not in PHASE_ORDER
        or value.get("authorization_exhausted") is not True
        or value.get("cross_run_recovery_allowed") is not False
        or value.get("terminal_recovery_allowed") is not False
        or value.get("raw_labels_persisted") is not False
        or not isinstance(history, list)
        or value.get("transition_count") != len(history)
        or value.get("state_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("OE-PPUR v2 run state drifted.")
    previous: str | None = None
    for sequence, row in enumerate(history):
        if (
            not isinstance(row, Mapping)
            or row.get("sequence") != sequence
            or row.get("previous_transition_hash") != previous
            or row.get("transition_hash")
            != canonical_hash(
                {key: item for key, item in row.items() if key != "transition_hash"}
            )
        ):
            raise ProtocolError("OE-PPUR v2 transition chain drifted.")
        previous = str(row["transition_hash"])
    return value


def _initial_state(
    admission: SixInputAdmissionReceipt,
    lease: AuthorizationLeaseClaim,
    run_hash: str,
) -> dict[str, object]:
    base = {
        "schema_version": RUN_STATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": run_hash,
        "config_hash": admission.config_contract_hash,
        "protocol_hash": admission.protocol_hash,
        "source_contract_hash": admission.source_contract_hash,
        "six_input_admission_hash": admission.receipt_hash,
        "authorization_lease_claim_hash": lease.claim_hash,
        "status": "RUNNING",
        "phase": "ADMITTED",
        "transition_count": 0,
        "transitions": [],
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "scratch_recovery_allowed": False,
        "raw_labels_persisted": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_class": None,
    }
    return {**base, "state_hash": canonical_hash(base)}


def _replace_state(
    state: Mapping[str, object],
    *,
    status: str,
    phase: str,
    evidence_hash: str,
    error_class: str | None,
) -> dict[str, object]:
    history = list(state["transitions"])
    previous = None if not history else history[-1]["transition_hash"]
    transition_body = {
        "sequence": len(history),
        "from_phase": state["phase"],
        "to_phase": phase,
        "status": status,
        "evidence_hash": evidence_hash,
        "previous_transition_hash": previous,
    }
    history.append(
        {**transition_body, "transition_hash": canonical_hash(transition_body)}
    )
    body = {
        **{key: item for key, item in state.items() if key != "state_hash"},
        "status": status,
        "phase": phase,
        "transition_count": len(history),
        "transitions": history,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_class": error_class,
    }
    return {**body, "state_hash": canonical_hash(body)}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        data = canonical_json_bytes(payload) + b"\n"
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short write while persisting OE-PPUR v2 run lock")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


__all__ = (
    "PHASE_ORDER",
    "build_run_identity_hash",
    "create_single_use_run",
    "mark_complete",
    "mark_failed_exhausted",
    "read_run_state",
    "transition_run",
)
