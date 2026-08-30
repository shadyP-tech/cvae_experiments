"""Monotone same-run lifecycle state facade for OE-PPUR v4."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .run_admission import SevenInputRunAdmission
from .run_state_contracts import (
    PHASE_ORDER,
    PreparedCompleteRunState,
    TerminalRunStateReceipt,
    _issue_terminal_run_state_receipt,
    _validate_transition_history,
)
from . import run_state_completion as _completion
from . import run_state_storage as _storage


def build_run_identity_hash(admission: SevenInputRunAdmission) -> str:
    if type(admission) is not SevenInputRunAdmission:
        raise ProtocolError("OE-PPUR v4 run identity requires typed admission.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v4_run_identity_v1",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "config_contract_hash": admission.config_contract_hash,
            "protocol_hash": admission.protocol_hash,
            "seven_input_contract_hash": admission.seven_input_contract_hash,
            "source_seal_hash": admission.source_seal_hash,
            "source_training_surface_receipt_hash": (
                admission.source_training_surface_receipt_hash
            ),
            "seven_input_admission_hash": admission.receipt_hash,
            "workspace_snapshot_sha256": admission.workspace_snapshot_sha256,
            "workspace_plan_sha256": admission.workspace_plan_sha256,
            "final_envelope_sha256": admission.final_envelope_sha256,
            "execution_launch_authority_sha256": (
                admission.execution_launch_authority_sha256
            ),
        }
    )


def create_single_use_run(
    admission: SevenInputRunAdmission,
    lease: AuthorizationLeaseClaim,
    *,
    run_identity_hash: str,
) -> dict[str, object]:
    if type(admission) is not SevenInputRunAdmission:
        raise ProtocolError("OE-PPUR v4 run requires typed admission.")
    validated = validate_authorization_lease(lease)
    run_hash = require_sha256(run_identity_hash, "run identity hash")
    if (
        validated.payload.get("run_identity_hash") != run_hash
        or validated.payload.get("seven_input_admission_hash")
        != admission.receipt_hash
        or validated.payload.get("workspace_snapshot_sha256")
        != admission.workspace_snapshot_sha256
        or validated.payload.get("workspace_plan_sha256")
        != admission.workspace_plan_sha256
        or validated.payload.get("final_envelope_sha256")
        != admission.final_envelope_sha256
        or validated.payload.get("execution_launch_authority_sha256")
        != admission.execution_launch_authority_sha256
    ):
        raise ProtocolError("OE-PPUR v4 run/lease binding drifted.")
    root = admission.artifact_root
    scratch = admission.scratch_root
    if scratch.exists() or scratch.is_symlink():
        raise ProtocolError("OE-PPUR v4 scratch root is not fresh.")
    state = _initial_state(admission, validated, run_hash)
    try:
        scratch.mkdir(parents=True, exist_ok=False)
    except BaseException as exc:
        failed = _replace_state(
            state,
            status="FAILED_EXHAUSTED",
            phase="ADMITTED",
            evidence_hash=canonical_hash(
                {
                    "schema_version": "oe_ppur_v4_scratch_creation_failure_v1",
                    "error_class": type(exc).__name__,
                }
            ),
            error_class=type(exc).__name__,
        )
        write_exclusive_json(root / "reports/run_state.json", failed)
        raise ProtocolError(
            "OE-PPUR v4 scratch creation failed; authorization is exhausted."
        ) from exc
    write_exclusive_json(
        root / "provenance/execution_admission.json", admission.to_payload()
    )
    write_exclusive_json(
        root / "provenance/authorization_consumption_lease.json",
        validated.to_payload(),
    )
    lock_body = {
        "schema_version": "oe_ppur_v4_run_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "run_identity_hash": run_hash,
        "authorization_lease_claim_hash": validated.claim_hash,
        "authorization_exhausted": True,
        "recovery_allowed": False,
    }
    write_exclusive_json(
        root / ".run.lock",
        {**lock_body, "lock_hash": canonical_hash(lock_body)},
    )
    write_exclusive_json(root / "reports/run_state.json", state)
    return state


def transition_run(
    artifact_root: Path,
    next_phase: str,
    *,
    expected_phase: str,
    evidence_hash: str,
) -> dict[str, object]:
    state = read_run_state(artifact_root)
    current = str(state["phase"])
    target = str(next_phase)
    if (
        state.get("status") != "RUNNING"
        or current != expected_phase
        or target not in PHASE_ORDER
        or target in {"ADMITTED", "COMPLETE"}
        or PHASE_ORDER.index(target) != PHASE_ORDER.index(current) + 1
    ):
        raise ProtocolError("OE-PPUR v4 run transition is out of order.")
    updated = _replace_state(
        state,
        status="RUNNING",
        phase=target,
        evidence_hash=require_sha256(evidence_hash, "transition evidence hash"),
        error_class=None,
    )
    atomic_json(Path(artifact_root) / "reports/run_state.json", updated)
    return updated


def mark_failed_exhausted(
    artifact_root: Path,
    *,
    error_class: str,
    evidence_hash: str,
) -> TerminalRunStateReceipt:
    state = read_run_state(artifact_root)
    if state.get("status") != "RUNNING":
        raise ProtocolError("OE-PPUR v4 failed state cannot be rewritten.")
    updated = _replace_state(
        state,
        status="FAILED_EXHAUSTED",
        phase=str(state["phase"]),
        evidence_hash=require_sha256(evidence_hash, "failure evidence hash"),
        error_class=_safe_text(error_class),
    )
    atomic_json(Path(artifact_root) / "reports/run_state.json", updated)
    return _terminal_state_receipt(Path(artifact_root), updated)


def prepare_complete_run_state(
    artifact_root: Path,
    *,
    final_bundle: object,
) -> PreparedCompleteRunState:
    """Prepare immutable COMPLETE bytes without exposing COMPLETE on disk."""

    return _completion.prepare_complete_run_state(
        Path(artifact_root),
        final_bundle=final_bundle,
        read_state=read_run_state,
        replace_state=_replace_state,
    )


def validate_prepared_complete_run_state(
    prepared: PreparedCompleteRunState,
) -> PreparedCompleteRunState:
    """Revalidate typed preparation and the unchanged pending disk state."""

    return _completion.validate_prepared_complete_run_state(
        prepared,
        read_state=read_run_state,
    )


def commit_complete_run_state(
    prepared: PreparedCompleteRunState,
    *,
    completion_commit: object,
) -> TerminalRunStateReceipt:
    """Commit exact prepared COMPLETE bytes only after journal validation."""

    return _completion.commit_complete_run_state(
        prepared,
        completion_commit=completion_commit,
        read_state=read_run_state,
        write_state=atomic_json,
        issue_terminal=_terminal_state_receipt,
    )


def mark_complete(
    artifact_root: Path,
    *,
    final_bundle: object,
) -> TerminalRunStateReceipt:
    """Reject the retired journal-free completion path."""

    del artifact_root, final_bundle
    raise ProtocolError("OE-PPUR v4 completion requires a durable commit journal.")


def validate_terminal_run_state(
    receipt: TerminalRunStateReceipt,
) -> TerminalRunStateReceipt:
    if type(receipt) is not TerminalRunStateReceipt:
        raise ProtocolError("OE-PPUR v4 terminal run state is untyped.")
    observed = _terminal_state_receipt(
        receipt.artifact_root,
        read_run_state(receipt.artifact_root),
    )
    if observed != receipt:
        raise ProtocolError("OE-PPUR v4 terminal run state changed after issuance.")
    return observed


def read_terminal_run_state(artifact_root: Path) -> TerminalRunStateReceipt:
    """Read a COMPLETE or FAILED_EXHAUSTED state into a gated receipt."""

    root = Path(artifact_root)
    return _terminal_state_receipt(root, read_run_state(root))


def read_run_state(artifact_root: Path) -> dict[str, object]:
    path = Path(artifact_root) / "reports/run_state.json"
    payload = _storage.read_json_regular_nofollow(path)
    body = {key: value for key, value in payload.items() if key != "state_hash"}
    history = payload.get("transitions")
    status = payload.get("status")
    phase = payload.get("phase")
    if (
        payload.get("schema_version") != "oe_ppur_v4_single_use_run_state_v1"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or status not in {"RUNNING", "FAILED_EXHAUSTED", "COMPLETE"}
        or phase not in PHASE_ORDER
        or (status == "COMPLETE") != (phase == "COMPLETE")
        or (status == "RUNNING" and phase == "COMPLETE")
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_consumed") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
        or payload.get("scratch_recovery_allowed") is not False
        or payload.get("raw_labels_persisted") is not False
        or payload.get("execution_launch_authority_sha256") == "0" * 64
        or not isinstance(history, list)
        or payload.get("transition_count") != len(history)
        or payload.get("state_hash") != canonical_hash(body)
    ):
        raise ProtocolError("OE-PPUR v4 run state drifted.")
    for role in (
        "run_identity_hash",
        "config_contract_hash",
        "protocol_hash",
        "source_seal_hash",
        "seven_input_admission_hash",
        "workspace_snapshot_sha256",
        "workspace_plan_sha256",
        "final_envelope_sha256",
        "execution_launch_authority_sha256",
        "authorization_lease_claim_hash",
        "state_hash",
    ):
        require_sha256(payload.get(role), role.replace("_", " "))
    _validate_transition_history(
        history,
        final_status=str(status),
        final_phase=str(phase),
    )
    return payload


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Compatibility facade for durable atomic canonical-JSON replacement."""

    _storage.atomic_json(Path(path), payload)


def write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    """Compatibility facade for durable exclusive canonical-JSON creation."""

    _storage.write_exclusive_json(Path(path), payload)


def _initial_state(
    admission: SevenInputRunAdmission,
    lease: AuthorizationLeaseClaim,
    run_identity_hash: str,
) -> dict[str, object]:
    body = {
        "schema_version": "oe_ppur_v4_single_use_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": run_identity_hash,
        "config_contract_hash": admission.config_contract_hash,
        "protocol_hash": admission.protocol_hash,
        "source_seal_hash": admission.source_seal_hash,
        "seven_input_admission_hash": admission.receipt_hash,
        "workspace_snapshot_sha256": admission.workspace_snapshot_sha256,
        "workspace_plan_sha256": admission.workspace_plan_sha256,
        "final_envelope_sha256": admission.final_envelope_sha256,
        "execution_launch_authority_sha256": (
            admission.execution_launch_authority_sha256
        ),
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
    return {**body, "state_hash": canonical_hash(body)}


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
        **{key: value for key, value in state.items() if key != "state_hash"},
        "status": status,
        "phase": phase,
        "transition_count": len(history),
        "transitions": history,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_class": error_class,
    }
    return {**body, "state_hash": canonical_hash(body)}


def _terminal_state_receipt(
    artifact_root: Path,
    state: Mapping[str, object],
) -> TerminalRunStateReceipt:
    history = state.get("transitions")
    if (
        state.get("status") not in {"COMPLETE", "FAILED_EXHAUSTED"}
        or not isinstance(history, list)
        or not history
        or not isinstance(history[-1], Mapping)
    ):
        raise ProtocolError("OE-PPUR v4 run is not in a terminal state.")
    return _issue_terminal_run_state_receipt(
        artifact_root=Path(artifact_root),
        status=str(state["status"]),
        phase=str(state["phase"]),
        state_hash=str(state["state_hash"]),
        run_identity_hash=str(state["run_identity_hash"]),
        authorization_lease_claim_hash=str(
            state["authorization_lease_claim_hash"]
        ),
        evidence_hash=str(history[-1]["evidence_hash"]),
    )


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


__all__ = (
    "PHASE_ORDER",
    "PreparedCompleteRunState",
    "TerminalRunStateReceipt",
    "atomic_json",
    "build_run_identity_hash",
    "commit_complete_run_state",
    "create_single_use_run",
    "mark_complete",
    "mark_failed_exhausted",
    "prepare_complete_run_state",
    "read_terminal_run_state",
    "read_run_state",
    "transition_run",
    "validate_prepared_complete_run_state",
    "validate_terminal_run_state",
    "write_exclusive_json",
)
