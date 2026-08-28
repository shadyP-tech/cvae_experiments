"""Monotone same-run lifecycle state for OE-PPUR v3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType

from ...protocol import ProtocolError
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .hashing import canonical_hash, canonical_json_bytes, require_sha256
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .run_admission import SevenInputRunAdmission


PHASE_ORDER = (
    "ADMITTED",
    "INPUTS_SEALED",
    "PHYSICAL_PROBABILITIES_MATERIALIZED",
    "PRETERMINAL_DECISIONS_SEALED",
    "PRETERMINAL_ATTESTED",
    "TERMINAL_AGGREGATES_SCORED",
    "FINAL_ATTESTED",
    "COMPLETION_PENDING",
    "COMPLETE",
)

_TERMINAL_STATE_TOKEN = object()
_PREPARED_COMPLETE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PreparedCompleteRunState:
    """Factory-gated canonical COMPLETE bytes prepared while disk is pending."""

    artifact_root: Path
    payload: Mapping[str, object]
    state_hash: str
    pending_state_hash: str
    final_bundle_receipt_hash: str
    run_identity_hash: str
    authorization_lease_claim_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _PREPARED_COMPLETE_TOKEN:
            raise ProtocolError(
                "OE-PPUR v3 prepared COMPLETE state bypassed typed validation."
            )
        root = Path(self.artifact_root)
        rendered = _thaw_json(self.payload)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or not isinstance(rendered, dict)
            or rendered.get("status") != "COMPLETE"
            or rendered.get("phase") != "COMPLETE"
            or rendered.get("state_hash") != self.state_hash
            or rendered.get("run_identity_hash") != self.run_identity_hash
            or rendered.get("authorization_lease_claim_hash")
            != self.authorization_lease_claim_hash
        ):
            raise ProtocolError("OE-PPUR v3 prepared COMPLETE state drifted.")
        body = {key: value for key, value in rendered.items() if key != "state_hash"}
        history = rendered.get("transitions")
        if (
            canonical_hash(body) != self.state_hash
            or not isinstance(history, list)
            or not history
            or history[-1].get("from_phase") != "COMPLETION_PENDING"
            or history[-1].get("to_phase") != "COMPLETE"
            or history[-1].get("status") != "COMPLETE"
            or history[-1].get("evidence_hash")
            != self.final_bundle_receipt_hash
        ):
            raise ProtocolError("OE-PPUR v3 prepared COMPLETE payload drifted.")
        _validate_transition_history(
            history,
            final_status="COMPLETE",
            final_phase="COMPLETE",
        )
        object.__setattr__(self, "artifact_root", root)
        object.__setattr__(self, "payload", _freeze_json(rendered))
        for role in (
            "state_hash",
            "pending_state_hash",
            "final_bundle_receipt_hash",
            "run_identity_hash",
            "authorization_lease_claim_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    @property
    def complete_payload(self) -> Mapping[str, object]:
        """Immutable canonical COMPLETE mapping for whole-artifact sealing."""

        return self.payload

    @property
    def canonical_complete_bytes(self) -> bytes:
        """Exact bytes later committed to ``reports/run_state.json``."""

        return canonical_json_bytes(_thaw_json(self.payload)) + b"\n"

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_prepared_complete_run_state_v1",
            "artifact_root": self.artifact_root.as_posix(),
            "state_hash": self.state_hash,
            "pending_state_hash": self.pending_state_hash,
            "final_bundle_receipt_hash": self.final_bundle_receipt_hash,
            "run_identity_hash": self.run_identity_hash,
            "authorization_lease_claim_hash": self.authorization_lease_claim_hash,
        }


@dataclass(frozen=True, slots=True)
class TerminalRunStateReceipt:
    """Factory-gated terminal state re-read from durable run-state bytes."""

    artifact_root: Path
    status: str
    phase: str
    state_hash: str
    run_identity_hash: str
    authorization_lease_claim_hash: str
    evidence_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _TERMINAL_STATE_TOKEN:
            raise ProtocolError("OE-PPUR v3 terminal run state bypassed durable validation.")
        root = Path(self.artifact_root)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or self.status not in {"COMPLETE", "FAILED_EXHAUSTED"}
            or self.phase not in PHASE_ORDER
            or (self.status == "COMPLETE") != (self.phase == "COMPLETE")
        ):
            raise ProtocolError("OE-PPUR v3 terminal run-state receipt drifted.")
        object.__setattr__(self, "artifact_root", root)
        for role in (
            "state_hash",
            "run_identity_hash",
            "authorization_lease_claim_hash",
            "evidence_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_terminal_run_state_receipt_v1",
            "artifact_root": self.artifact_root.as_posix(),
            "status": self.status,
            "phase": self.phase,
            "state_hash": self.state_hash,
            "run_identity_hash": self.run_identity_hash,
            "authorization_lease_claim_hash": self.authorization_lease_claim_hash,
            "evidence_hash": self.evidence_hash,
        }


def build_run_identity_hash(admission: SevenInputRunAdmission) -> str:
    if type(admission) is not SevenInputRunAdmission:
        raise ProtocolError("OE-PPUR v3 run identity requires typed admission.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_run_identity_v1",
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
        }
    )


def create_single_use_run(
    admission: SevenInputRunAdmission,
    lease: AuthorizationLeaseClaim,
    *,
    run_identity_hash: str,
) -> dict[str, object]:
    if type(admission) is not SevenInputRunAdmission:
        raise ProtocolError("OE-PPUR v3 run requires typed admission.")
    validated = validate_authorization_lease(lease)
    run_hash = require_sha256(run_identity_hash, "run identity hash")
    if (
        validated.payload.get("run_identity_hash") != run_hash
        or validated.payload.get("seven_input_admission_hash")
        != admission.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v3 run/lease binding drifted.")
    root = admission.artifact_root
    scratch = admission.scratch_root
    if scratch.exists() or scratch.is_symlink():
        raise ProtocolError("OE-PPUR v3 scratch root is not fresh.")
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
                    "schema_version": "oe_ppur_v3_scratch_creation_failure_v1",
                    "error_class": type(exc).__name__,
                }
            ),
            error_class=type(exc).__name__,
        )
        atomic_json(root / "reports/run_state.json", failed)
        raise ProtocolError(
            "OE-PPUR v3 scratch creation failed; authorization is exhausted."
        ) from exc
    atomic_json(root / "provenance/execution_admission.json", admission.to_payload())
    atomic_json(
        root / "provenance/authorization_consumption_lease.json",
        validated.to_payload(),
    )
    lock_body = {
        "schema_version": "oe_ppur_v3_run_lock_v1",
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
    atomic_json(root / "reports/run_state.json", state)
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
        raise ProtocolError("OE-PPUR v3 run transition is out of order.")
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
        raise ProtocolError("OE-PPUR v3 failed state cannot be rewritten.")
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

    from .output_artifact import (
        FinalAggregateBundleReceipt,
        validate_final_aggregate_bundle,
    )

    if type(final_bundle) is not FinalAggregateBundleReceipt:
        raise ProtocolError("OE-PPUR v3 completion requires a typed final bundle.")
    validated_bundle = validate_final_aggregate_bundle(
        artifact_root,
        expected_receipt=final_bundle,
    )
    state = read_run_state(artifact_root)
    transitions = state.get("transitions")
    if (
        state.get("status") != "RUNNING"
        or state.get("phase") != "COMPLETION_PENDING"
        or not isinstance(transitions, list)
        or not transitions
        or transitions[-1].get("evidence_hash")
        != validated_bundle.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v3 completion preparation requires pending state.")
    prepared_payload = _replace_state(
        state,
        status="COMPLETE",
        phase="COMPLETE",
        evidence_hash=validated_bundle.receipt_hash,
        error_class=None,
    )
    # Preparation is a pure projection of the durable pending state.  Reusing
    # its already-recorded timestamp makes independent rebuilds byte-identical.
    prepared_body = {
        **{
            key: value
            for key, value in prepared_payload.items()
            if key != "state_hash"
        },
        "updated_at_utc": state["updated_at_utc"],
    }
    prepared_payload = {
        **prepared_body,
        "state_hash": canonical_hash(prepared_body),
    }
    return PreparedCompleteRunState(
        artifact_root=Path(artifact_root),
        payload=prepared_payload,
        state_hash=str(prepared_payload["state_hash"]),
        pending_state_hash=str(state["state_hash"]),
        final_bundle_receipt_hash=validated_bundle.receipt_hash,
        run_identity_hash=str(state["run_identity_hash"]),
        authorization_lease_claim_hash=str(
            state["authorization_lease_claim_hash"]
        ),
        _factory_token=_PREPARED_COMPLETE_TOKEN,
    )


def validate_prepared_complete_run_state(
    prepared: PreparedCompleteRunState,
) -> PreparedCompleteRunState:
    """Revalidate typed preparation and the unchanged pending disk state."""

    if type(prepared) is not PreparedCompleteRunState:
        raise ProtocolError("OE-PPUR v3 prepared COMPLETE state is untyped.")
    observed = read_run_state(prepared.artifact_root)
    if (
        observed.get("status") != "RUNNING"
        or observed.get("phase") != "COMPLETION_PENDING"
        or observed.get("state_hash") != prepared.pending_state_hash
        or observed.get("run_identity_hash") != prepared.run_identity_hash
        or observed.get("authorization_lease_claim_hash")
        != prepared.authorization_lease_claim_hash
        or prepared.receipt_hash != canonical_hash(prepared.to_payload())
    ):
        raise ProtocolError("OE-PPUR v3 prepared COMPLETE state changed before commit.")
    return prepared


def commit_complete_run_state(
    prepared: PreparedCompleteRunState,
    *,
    completion_commit: object,
) -> TerminalRunStateReceipt:
    """Commit exact prepared COMPLETE bytes only after journal validation."""

    from .completion_transaction import (
        CompletionCommitReceipt,
        validate_completion_commit,
    )

    validated_prepared = validate_prepared_complete_run_state(prepared)
    if type(completion_commit) is not CompletionCommitReceipt:
        raise ProtocolError("OE-PPUR v3 COMPLETE commit requires typed journal.")
    validated_commit = validate_completion_commit(
        completion_commit,
        expected_prepared_state=validated_prepared,
    )
    if (
        validated_commit.prepared_state_receipt_hash
        != validated_prepared.receipt_hash
        or validated_commit.prepared_state_hash != validated_prepared.state_hash
    ):
        raise ProtocolError("OE-PPUR v3 completion journal/state binding drifted.")
    payload = _thaw_json(validated_prepared.payload)
    if not isinstance(payload, dict):  # pragma: no cover - constructor guarded
        raise ProtocolError("OE-PPUR v3 prepared COMPLETE payload is malformed.")
    atomic_json(
        validated_prepared.artifact_root / "reports/run_state.json",
        payload,
    )
    observed = _terminal_state_receipt(
        validated_prepared.artifact_root,
        read_run_state(validated_prepared.artifact_root),
    )
    if observed.state_hash != validated_prepared.state_hash:
        raise ProtocolError("OE-PPUR v3 committed COMPLETE bytes drifted.")
    return observed


def mark_complete(
    artifact_root: Path,
    *,
    final_bundle: object,
) -> TerminalRunStateReceipt:
    """Reject the retired journal-free completion path."""

    del artifact_root, final_bundle
    raise ProtocolError("OE-PPUR v3 completion requires a durable commit journal.")


def validate_terminal_run_state(
    receipt: TerminalRunStateReceipt,
) -> TerminalRunStateReceipt:
    if type(receipt) is not TerminalRunStateReceipt:
        raise ProtocolError("OE-PPUR v3 terminal run state is untyped.")
    observed = _terminal_state_receipt(
        receipt.artifact_root,
        read_run_state(receipt.artifact_root),
    )
    if observed != receipt:
        raise ProtocolError("OE-PPUR v3 terminal run state changed after issuance.")
    return observed


def read_terminal_run_state(artifact_root: Path) -> TerminalRunStateReceipt:
    """Read a COMPLETE or FAILED_EXHAUSTED state into a gated receipt."""

    root = Path(artifact_root)
    return _terminal_state_receipt(root, read_run_state(root))


def read_run_state(artifact_root: Path) -> dict[str, object]:
    path = Path(artifact_root) / "reports/run_state.json"
    payload = _read_json_regular_nofollow(path)
    body = {key: value for key, value in payload.items() if key != "state_hash"}
    history = payload.get("transitions")
    status = payload.get("status")
    phase = payload.get("phase")
    if (
        payload.get("schema_version") != "oe_ppur_v3_single_use_run_state_v1"
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
        or not isinstance(history, list)
        or payload.get("transition_count") != len(history)
        or payload.get("state_hash") != canonical_hash(body)
    ):
        raise ProtocolError("OE-PPUR v3 run state drifted.")
    for role in (
        "run_identity_hash",
        "config_contract_hash",
        "protocol_hash",
        "source_seal_hash",
        "seven_input_admission_hash",
        "authorization_lease_claim_hash",
        "state_hash",
    ):
        require_sha256(payload.get(role), role.replace("_", " "))
    _validate_transition_history(history, final_status=str(status), final_phase=str(phase))
    return payload


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _ensure_parent_directory(path)
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProtocolError("OE-PPUR v3 atomic state target is unsafe.")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        if _read_json_regular_nofollow(path) != dict(payload):
            raise ProtocolError("OE-PPUR v3 atomic state read-back drifted.")
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    _ensure_parent_directory(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 exclusive state write failed.") from exc
    try:
        raw = canonical_json_bytes(payload) + b"\n"
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short OE-PPUR v3 artifact write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    if _read_json_regular_nofollow(path) != dict(payload):
        raise ProtocolError("OE-PPUR v3 exclusive state read-back drifted.")


def _initial_state(
    admission: SevenInputRunAdmission,
    lease: AuthorizationLeaseClaim,
    run_identity_hash: str,
) -> dict[str, object]:
    body = {
        "schema_version": "oe_ppur_v3_single_use_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": run_identity_hash,
        "config_contract_hash": admission.config_contract_hash,
        "protocol_hash": admission.protocol_hash,
        "source_seal_hash": admission.source_seal_hash,
        "seven_input_admission_hash": admission.receipt_hash,
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
        raise ProtocolError("OE-PPUR v3 run is not in a terminal state.")
    return TerminalRunStateReceipt(
        artifact_root=Path(artifact_root),
        status=str(state["status"]),
        phase=str(state["phase"]),
        state_hash=str(state["state_hash"]),
        run_identity_hash=str(state["run_identity_hash"]),
        authorization_lease_claim_hash=str(
            state["authorization_lease_claim_hash"]
        ),
        evidence_hash=str(history[-1]["evidence_hash"]),
        _factory_token=_TERMINAL_STATE_TOKEN,
    )


def _validate_transition_history(
    history: list[object],
    *,
    final_status: str,
    final_phase: str,
) -> None:
    previous_hash: str | None = None
    current_phase = "ADMITTED"
    current_status = "RUNNING"
    for sequence, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ProtocolError("OE-PPUR v3 transition chain drifted.")
        transition_body = {
            key: value for key, value in row.items() if key != "transition_hash"
        }
        target_phase = row.get("to_phase")
        target_status = row.get("status")
        evidence = row.get("evidence_hash")
        if (
            set(row)
            != {
                "sequence",
                "from_phase",
                "to_phase",
                "status",
                "evidence_hash",
                "previous_transition_hash",
                "transition_hash",
            }
            or row.get("sequence") != sequence
            or row.get("from_phase") != current_phase
            or row.get("previous_transition_hash") != previous_hash
            or row.get("transition_hash") != canonical_hash(transition_body)
            or require_sha256(evidence, "transition evidence hash") != evidence
            or current_status != "RUNNING"
        ):
            raise ProtocolError("OE-PPUR v3 transition chain drifted.")
        if target_status == "RUNNING":
            if (
                target_phase not in PHASE_ORDER
                or target_phase in {"ADMITTED", "COMPLETE"}
                or PHASE_ORDER.index(str(target_phase))
                != PHASE_ORDER.index(current_phase) + 1
            ):
                raise ProtocolError("OE-PPUR v3 transition chain drifted.")
        elif target_status == "FAILED_EXHAUSTED":
            if target_phase != current_phase or sequence != len(history) - 1:
                raise ProtocolError("OE-PPUR v3 transition chain drifted.")
        elif target_status == "COMPLETE":
            if (
                current_phase != "COMPLETION_PENDING"
                or target_phase != "COMPLETE"
                or sequence != len(history) - 1
            ):
                raise ProtocolError("OE-PPUR v3 transition chain drifted.")
        else:
            raise ProtocolError("OE-PPUR v3 transition chain drifted.")
        current_phase = str(target_phase)
        current_status = str(target_status)
        previous_hash = str(row["transition_hash"])
    if current_phase != final_phase or current_status != final_status:
        if not (not history and final_phase == "ADMITTED" and final_status == "RUNNING"):
            raise ProtocolError("OE-PPUR v3 transition chain/state drifted.")


def _read_json_regular_nofollow(path: Path) -> dict[str, object]:
    _reject_symlink_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 run state is unreadable.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ProtocolError("OE-PPUR v3 run state is unsafe.")
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
        raise ProtocolError("OE-PPUR v3 run state changed while read.")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 run state is unreadable.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v3 run state is malformed.")
    return payload


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_parent_directory(path: Path) -> None:
    candidate = Path(os.path.abspath(path))
    if candidate != path or path == Path(path.anchor):
        raise ProtocolError("OE-PPUR v3 state path is unsafe.")
    missing: list[Path] = []
    current = path.parent
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        current = current.parent
    _reject_symlink_chain(current)
    if not current.is_dir():
        raise ProtocolError("OE-PPUR v3 state parent is unsafe.")
    for directory in reversed(missing):
        try:
            directory.mkdir(exist_ok=False)
        except OSError as exc:
            raise ProtocolError("OE-PPUR v3 state parent creation failed.") from exc
        _fsync_directory(directory.parent)
    _reject_symlink_chain(path.parent)


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 state path contains a symlink.")
        if current == current.parent:
            return
        current = current.parent


def _stat_payload(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ProtocolError("OE-PPUR v3 prepared state is not canonical JSON.")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ProtocolError("OE-PPUR v3 prepared state is not canonical JSON.")


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
