"""Irrecoverable single-use run state for executable SCALE-BP v2."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
from typing import Iterator, Mapping

from .authorization_lease import (
    AuthorizationLeaseClaim,
    LEASE_DIRECTORY_NAME,
    validate_authorization_lease,
)
from .artifacts.hashing import canonical_hash, json_native, require_sha256
from .artifacts.io import atomic_json, member_path, read_json_object
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .protocol import GovernanceError


RUN_STATE_SCHEMA = "scale_bp_v2_single_use_run_state_v1"
RUN_LOCK_SCHEMA = "scale_bp_v2_single_use_run_lock_v1"
PHASE_ORDER = (
    "ADMITTED",
    "INPUTS_SEALED",
    "PHYSICAL_SURFACE_SEALED",
    "OUTER_CENTERS_COMPLETE",
    "PRETERMINAL_SEALED",
    "PRETERMINAL_ATTESTED",
    "TERMINAL_OPEN",
    "TERMINAL_SCORED",
    "FINAL_INDEX_SEALED",
    "FINAL_ATTESTED",
    "COMPLETE",
)


def create_single_use_run(
    artifact_root: str | Path,
    scratch_root: str | Path,
    *,
    run_identity_hash: str,
    admission_receipt: Mapping[str, object] | object,
    authorization_lease: AuthorizationLeaseClaim,
    config_hash: str,
    protocol_hash: str,
) -> dict[str, object]:
    """Consume the one-shot authorization and create both previously absent roots."""

    artifact = _absolute_root(artifact_root, "artifact")
    scratch = _absolute_root(scratch_root, "scratch")
    if artifact == scratch:
        raise GovernanceError("SCALE-BP v2 artifact and scratch roots must differ.")
    prepared_workspace_root = _is_exact_workspace_launch_envelope(artifact)
    if artifact.exists() or artifact.is_symlink():
        if not prepared_workspace_root:
            reject_existing_run(artifact)
    if scratch.exists() or scratch.is_symlink():
        reject_existing_run(scratch)
    run_hash = require_sha256(run_identity_hash, "run identity hash")
    config = require_sha256(config_hash, "config hash")
    protocol = require_sha256(protocol_hash, "protocol hash")
    receipt = _receipt_payload(
        admission_receipt,
        artifact_root=artifact,
        scratch_root=scratch,
        config_hash=config,
    )
    receipt_hash = require_sha256(receipt.get("receipt_hash"), "admission receipt hash")
    lease = validate_authorization_lease(authorization_lease)
    if (
        lease.payload.get("artifact_root") != str(artifact)
        or lease.payload.get("scratch_root") != str(scratch)
        or lease.payload.get("authorization_lease_path")
        != receipt.get("authorization_lease_path")
        or lease.payload.get("config_contract_hash") != config
        or lease.payload.get("protocol_hash") != protocol
        or lease.payload.get("admission_receipt_hash") != receipt_hash
        or lease.payload.get("run_identity_hash") != run_hash
    ):
        raise GovernanceError("SCALE-BP v2 run/authorization lease binding drifted.")
    if not prepared_workspace_root:
        artifact.mkdir(parents=True, exist_ok=False)
    atomic_json(
        member_path(artifact, "provenance/authorization_consumption_lease.json"),
        lease.to_payload(),
    )
    try:
        scratch.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        _write_initial_failed_state(
            artifact,
            run_identity_hash=run_hash,
            config_hash=config,
            protocol_hash=protocol,
            admission_receipt_hash=receipt_hash,
            authorization_lease_path=str(lease.path),
            authorization_lease_claim_hash=lease.claim_hash,
            error_class=type(exc).__name__,
            error="scratch_root_creation_failed",
        )
        raise GovernanceError(
            "SCALE-BP v2 scratch creation failed; authorization is exhausted."
        ) from exc
    atomic_json(
        member_path(artifact, "provenance/execution_admission.json"), receipt
    )
    lock_body = {
        "schema_version": RUN_LOCK_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": run_hash,
        "admission_receipt_hash": receipt_hash,
        "authorization_lease_path": str(lease.path),
        "authorization_lease_claim_hash": lease.claim_hash,
        "process_id_at_launch": os.getpid(),
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }
    atomic_json(artifact / ".run.lock", {**lock_body, "lock_hash": canonical_hash(lock_body)})
    state = _new_state(
        run_identity_hash=run_hash,
        config_hash=config,
        protocol_hash=protocol,
        admission_receipt_hash=receipt_hash,
        authorization_lease_path=str(lease.path),
        authorization_lease_claim_hash=lease.claim_hash,
    )
    atomic_json(member_path(artifact, "reports/run_state.json"), state)
    return state


def _is_exact_workspace_launch_envelope(root: Path) -> bool:
    """Recheck the mutation-free workspace envelope at authorization consume."""

    if not root.exists() or root.is_symlink() or not root.is_dir():
        return False
    members = tuple(root.rglob("*"))
    if any(member.is_symlink() for member in members):
        return False
    files = tuple(
        sorted(member.relative_to(root).as_posix() for member in members if member.is_file())
    )
    directories = tuple(
        sorted(
            member.relative_to(root).as_posix()
            for member in members
            if member.is_dir()
        )
    )
    return files == (
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
    ) and directories == (
        "manifests",
        "provenance",
        "reports",
        "tables",
    )


def reject_existing_run(
    artifact_root: str | Path, scratch_root: str | Path | None = None
) -> None:
    """Reject every previous or partial attempt; no recovery path exists."""

    artifact = _absolute_root(artifact_root, "artifact")
    roots = (artifact,) if scratch_root is None else (
        artifact,
        _absolute_root(scratch_root, "scratch"),
    )
    existing = [path for path in roots if path.exists() or path.is_symlink()]
    if not existing:
        return
    detail = "existing_or_unsafe_root"
    state_path = artifact / "reports/run_state.json"
    if state_path.is_file() and not state_path.is_symlink():
        state = read_json_object(state_path)
        detail = f"status={state.get('status')},phase={state.get('phase')}"
    raise GovernanceError(
        "SCALE-BP v2 single-use run already exists; recovery/rerun forbidden; "
        f"{detail}."
    )


def transition_run(
    artifact_root: str | Path,
    next_phase: str,
    *,
    evidence_hash: str,
    expected_phase: str | None = None,
) -> dict[str, object]:
    """Advance monotonically while retaining the hash chain in run state."""

    root = _absolute_root(artifact_root, "artifact")
    target = str(next_phase)
    evidence = require_sha256(evidence_hash, "run transition evidence hash")
    if target not in PHASE_ORDER or target in {"ADMITTED", "COMPLETE"}:
        raise GovernanceError("SCALE-BP v2 run transition target drifted.")
    with _state_lock(root):
        state = read_run_state(root)
        current = str(state["phase"])
        if (
            state.get("status") != "RUNNING"
            or (expected_phase is not None and current != str(expected_phase))
            or PHASE_ORDER.index(target) <= PHASE_ORDER.index(current)
            or (
                PHASE_ORDER.index(target) >= PHASE_ORDER.index("TERMINAL_OPEN")
                and PHASE_ORDER.index(current) < PHASE_ORDER.index("PRETERMINAL_ATTESTED")
            )
        ):
            raise GovernanceError("SCALE-BP v2 run transition is out of order.")
        return _replace_state(
            root,
            state,
            status="RUNNING",
            phase=target,
            evidence_hash=evidence,
            error=None,
            error_class=None,
        )


def mark_failed_exhausted(
    artifact_root: str | Path,
    *,
    error: str,
    error_class: str | None = None,
) -> dict[str, object]:
    root = _absolute_root(artifact_root, "artifact")
    with _state_lock(root):
        state = read_run_state(root)
        if state.get("status") != "RUNNING":
            raise GovernanceError("SCALE-BP v2 failed state cannot be rewritten.")
        detail = _safe_error(error)
        evidence = canonical_hash(
            {
                "schema_version": "scale_bp_v2_failure_evidence_v1",
                "phase": state["phase"],
                "error": detail,
                "error_class": None if error_class is None else _safe_error(error_class),
            }
        )
        return _replace_state(
            root,
            state,
            status="FAILED_EXHAUSTED",
            phase=str(state["phase"]),
            evidence_hash=evidence,
            error=detail,
            error_class=None if error_class is None else _safe_error(error_class),
        )


def mark_complete(
    artifact_root: str | Path, *, final_validation_hash: str
) -> dict[str, object]:
    root = _absolute_root(artifact_root, "artifact")
    validation_hash = require_sha256(
        final_validation_hash, "final validation hash"
    )
    with _state_lock(root):
        state = read_run_state(root)
        if state.get("status") != "RUNNING" or state.get("phase") != "FINAL_ATTESTED":
            raise GovernanceError("SCALE-BP v2 completion requires final attestation.")
        return _replace_state(
            root,
            state,
            status="COMPLETE",
            phase="COMPLETE",
            evidence_hash=validation_hash,
            error=None,
            error_class=None,
        )


def read_run_state(artifact_root: str | Path) -> dict[str, object]:
    root = _absolute_root(artifact_root, "artifact")
    state = read_json_object(member_path(root, "reports/run_state.json"))
    history = state.get("transitions")
    unhashed = {key: value for key, value in state.items() if key != "state_hash"}
    if (
        state.get("schema_version") != RUN_STATE_SCHEMA
        or state.get("experiment_id") != EXPERIMENT_ID
        or state.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or state.get("phase") not in PHASE_ORDER
        or state.get("status") not in {"RUNNING", "FAILED_EXHAUSTED", "COMPLETE"}
        or state.get("authorization_consumed") is not True
        or state.get("authorization_exhausted") is not True
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
        or state.get("scratch_recovery_allowed") is not False
        or not isinstance(history, list)
        or state.get("transition_count") != len(history)
        or state.get("state_hash") != canonical_hash(unhashed)
    ):
        raise GovernanceError("SCALE-BP v2 run state drifted.")
    for role in (
        "run_identity_hash",
        "config_hash",
        "protocol_hash",
        "admission_receipt_hash",
        "authorization_lease_claim_hash",
    ):
        require_sha256(state.get(role), role)
    lease_path = state.get("authorization_lease_path")
    if not isinstance(lease_path, str) or not Path(lease_path).is_absolute():
        raise GovernanceError("SCALE-BP v2 run-state lease path drifted.")
    previous: str | None = None
    for sequence, row in enumerate(history):
        if (
            not isinstance(row, Mapping)
            or row.get("sequence") != sequence
            or row.get("previous_transition_hash") != previous
        ):
            raise GovernanceError("SCALE-BP v2 run-state transition chain drifted.")
        transition_hash = row.get("transition_hash")
        body = {key: value for key, value in row.items() if key != "transition_hash"}
        if transition_hash != canonical_hash(body):
            raise GovernanceError("SCALE-BP v2 run-state transition hash drifted.")
        previous = require_sha256(transition_hash, "run transition hash")
    return state


def build_run_identity_hash(
    *,
    config_hash: str,
    protocol_hash: str,
    admission_receipt_hash: str,
) -> str:
    """Derive the one-shot run identity before constructing the label journal."""

    return canonical_hash(
        {
            "schema_version": "scale_bp_v2_run_identity_v1",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "config_hash": require_sha256(config_hash, "config hash"),
            "protocol_hash": require_sha256(protocol_hash, "protocol hash"),
            "admission_receipt_hash": require_sha256(
                admission_receipt_hash, "admission receipt hash"
            ),
            "single_use": True,
        }
    )


def _new_state(
    *,
    run_identity_hash: str,
    config_hash: str,
    protocol_hash: str,
    admission_receipt_hash: str,
    authorization_lease_path: str,
    authorization_lease_claim_hash: str,
) -> dict[str, object]:
    transition = _transition_row(
        sequence=0,
        previous_hash=None,
        status="RUNNING",
        phase="ADMITTED",
        evidence_hash=admission_receipt_hash,
        error=None,
        error_class=None,
    )
    body = {
        "schema_version": RUN_STATE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": run_identity_hash,
        "config_hash": config_hash,
        "protocol_hash": protocol_hash,
        "admission_receipt_hash": admission_receipt_hash,
        "authorization_lease_path": authorization_lease_path,
        "authorization_lease_claim_hash": require_sha256(
            authorization_lease_claim_hash, "authorization lease claim hash"
        ),
        "status": "RUNNING",
        "phase": "ADMITTED",
        "error": None,
        "error_class": None,
        "updated_at_utc": _utc_now(),
        "transition_count": 1,
        "transitions": [transition],
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "scratch_recovery_allowed": False,
    }
    return {**body, "state_hash": canonical_hash(body)}


def _replace_state(
    root: Path,
    state: Mapping[str, object],
    *,
    status: str,
    phase: str,
    evidence_hash: str,
    error: str | None,
    error_class: str | None,
) -> dict[str, object]:
    history = list(state["transitions"])  # type: ignore[arg-type]
    previous = str(history[-1]["transition_hash"])
    history.append(
        _transition_row(
            sequence=len(history),
            previous_hash=previous,
            status=status,
            phase=phase,
            evidence_hash=evidence_hash,
            error=error,
            error_class=error_class,
        )
    )
    body = {
        **{key: value for key, value in state.items() if key != "state_hash"},
        "status": status,
        "phase": phase,
        "error": error,
        "error_class": error_class,
        "updated_at_utc": _utc_now(),
        "transition_count": len(history),
        "transitions": history,
    }
    payload = {**body, "state_hash": canonical_hash(body)}
    atomic_json(
        member_path(root, "reports/run_state.json"), payload, replace=True
    )
    return payload


def _transition_row(
    *,
    sequence: int,
    previous_hash: str | None,
    status: str,
    phase: str,
    evidence_hash: str,
    error: str | None,
    error_class: str | None,
) -> dict[str, object]:
    body = {
        "schema_version": "scale_bp_v2_run_transition_v1",
        "sequence": sequence,
        "previous_transition_hash": previous_hash,
        "status": status,
        "phase": phase,
        "evidence_hash": require_sha256(evidence_hash, "transition evidence hash"),
        "error": error,
        "error_class": error_class,
    }
    return {**body, "transition_hash": canonical_hash(body)}


def _receipt_payload(
    receipt: Mapping[str, object] | object,
    *,
    artifact_root: Path,
    scratch_root: Path,
    config_hash: str,
) -> dict[str, object]:
    if isinstance(receipt, Mapping):
        payload = dict(receipt)
    else:
        to_payload = getattr(receipt, "to_payload", None)
        if not callable(to_payload):
            raise GovernanceError("SCALE-BP v2 admission receipt is malformed.")
        value = to_payload()
        if not isinstance(value, Mapping):
            raise GovernanceError("SCALE-BP v2 admission receipt is malformed.")
        payload = dict(value)
    json_native(payload)
    receipt_hash = payload.get("receipt_hash")
    if (
        payload.get("schema_version")
        != "scale_bp_v2_single_use_execution_admission_v1"
        or payload.get("status") != "ADMITTED_SINGLE_USE"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("config_contract_hash") != config_hash
        or payload.get("artifact_root") != str(artifact_root)
        or payload.get("scratch_root") != str(scratch_root)
        or payload.get("authorization_lease_path")
        != str(artifact_root.parent / LEASE_DIRECTORY_NAME)
        or payload.get("single_use_execution_identity") is not True
        or payload.get("consumed_test_reuse_authorized") is not True
        or payload.get("predecessor_state_used") is not False
        or payload.get("mutation_performed") is not False
        or receipt_hash
        != canonical_hash({key: value for key, value in payload.items() if key != "receipt_hash"})
    ):
        raise GovernanceError("SCALE-BP v2 admission receipt is not a clean PASS.")
    return payload


def _write_initial_failed_state(
    root: Path,
    *,
    run_identity_hash: str,
    config_hash: str,
    protocol_hash: str,
    admission_receipt_hash: str,
    authorization_lease_path: str,
    authorization_lease_claim_hash: str,
    error_class: str,
    error: str,
) -> None:
    state = _new_state(
        run_identity_hash=run_identity_hash,
        config_hash=config_hash,
        protocol_hash=protocol_hash,
        admission_receipt_hash=admission_receipt_hash,
        authorization_lease_path=authorization_lease_path,
        authorization_lease_claim_hash=authorization_lease_claim_hash,
    )
    evidence = canonical_hash({"error": error, "error_class": error_class})
    failed = _replace_state(
        root,
        state,
        status="FAILED_EXHAUSTED",
        phase="ADMITTED",
        evidence_hash=evidence,
        error=error,
        error_class=error_class,
    )
    atomic_json(member_path(root, "reports/run_state.json"), failed, replace=True)


@contextmanager
def _state_lock(root: Path) -> Iterator[None]:
    path = root / ".state.lock"
    if path.is_symlink():
        raise GovernanceError("SCALE-BP v2 state lock is unsafe.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _absolute_root(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise GovernanceError(f"SCALE-BP v2 {role} root must be absolute and nonsymlinked.")
    return path


def _safe_error(value: object) -> str:
    text = " ".join(str(value).split())[:500]
    return text or "unspecified_failure"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = (
    "PHASE_ORDER",
    "RUN_STATE_SCHEMA",
    "build_run_identity_hash",
    "create_single_use_run",
    "mark_complete",
    "mark_failed_exhausted",
    "read_run_state",
    "reject_existing_run",
    "transition_run",
)
