"""Immutable receipt contracts for the OE-PPUR v4 run-state machine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

from ...protocol import ProtocolError
from .durable_io import canonical_json_file_bytes
from .hashing import canonical_hash, require_sha256


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
                "OE-PPUR v4 prepared COMPLETE state bypassed typed validation."
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
            raise ProtocolError("OE-PPUR v4 prepared COMPLETE state drifted.")
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
            raise ProtocolError("OE-PPUR v4 prepared COMPLETE payload drifted.")
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

        rendered = _thaw_json(self.payload)
        if not isinstance(rendered, dict):  # pragma: no cover - constructor guarded
            raise ProtocolError("OE-PPUR v4 prepared COMPLETE payload is malformed.")
        return canonical_json_file_bytes(rendered)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_prepared_complete_run_state_v1",
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
            raise ProtocolError(
                "OE-PPUR v4 terminal run state bypassed durable validation."
            )
        root = Path(self.artifact_root)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or self.status not in {"COMPLETE", "FAILED_EXHAUSTED"}
            or self.phase not in PHASE_ORDER
            or (self.status == "COMPLETE") != (self.phase == "COMPLETE")
        ):
            raise ProtocolError("OE-PPUR v4 terminal run-state receipt drifted.")
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
            "schema_version": "oe_ppur_v4_terminal_run_state_receipt_v1",
            "artifact_root": self.artifact_root.as_posix(),
            "status": self.status,
            "phase": self.phase,
            "state_hash": self.state_hash,
            "run_identity_hash": self.run_identity_hash,
            "authorization_lease_claim_hash": self.authorization_lease_claim_hash,
            "evidence_hash": self.evidence_hash,
        }


def _issue_prepared_complete_run_state(
    *,
    artifact_root: Path,
    payload: Mapping[str, object],
    state_hash: str,
    pending_state_hash: str,
    final_bundle_receipt_hash: str,
    run_identity_hash: str,
    authorization_lease_claim_hash: str,
) -> PreparedCompleteRunState:
    return PreparedCompleteRunState(
        artifact_root=artifact_root,
        payload=payload,
        state_hash=state_hash,
        pending_state_hash=pending_state_hash,
        final_bundle_receipt_hash=final_bundle_receipt_hash,
        run_identity_hash=run_identity_hash,
        authorization_lease_claim_hash=authorization_lease_claim_hash,
        _factory_token=_PREPARED_COMPLETE_TOKEN,
    )


def _issue_terminal_run_state_receipt(
    *,
    artifact_root: Path,
    status: str,
    phase: str,
    state_hash: str,
    run_identity_hash: str,
    authorization_lease_claim_hash: str,
    evidence_hash: str,
) -> TerminalRunStateReceipt:
    return TerminalRunStateReceipt(
        artifact_root=artifact_root,
        status=status,
        phase=phase,
        state_hash=state_hash,
        run_identity_hash=run_identity_hash,
        authorization_lease_claim_hash=authorization_lease_claim_hash,
        evidence_hash=evidence_hash,
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
            raise ProtocolError("OE-PPUR v4 transition chain drifted.")
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
            raise ProtocolError("OE-PPUR v4 transition chain drifted.")
        if target_status == "RUNNING":
            if (
                target_phase not in PHASE_ORDER
                or target_phase in {"ADMITTED", "COMPLETE"}
                or PHASE_ORDER.index(str(target_phase))
                != PHASE_ORDER.index(current_phase) + 1
            ):
                raise ProtocolError("OE-PPUR v4 transition chain drifted.")
        elif target_status == "FAILED_EXHAUSTED":
            if target_phase != current_phase or sequence != len(history) - 1:
                raise ProtocolError("OE-PPUR v4 transition chain drifted.")
        elif target_status == "COMPLETE":
            if (
                current_phase != "COMPLETION_PENDING"
                or target_phase != "COMPLETE"
                or sequence != len(history) - 1
            ):
                raise ProtocolError("OE-PPUR v4 transition chain drifted.")
        else:
            raise ProtocolError("OE-PPUR v4 transition chain drifted.")
        current_phase = str(target_phase)
        current_status = str(target_status)
        previous_hash = str(row["transition_hash"])
    if current_phase != final_phase or current_status != final_status:
        if not (
            not history
            and final_phase == "ADMITTED"
            and final_status == "RUNNING"
        ):
            raise ProtocolError("OE-PPUR v4 transition chain/state drifted.")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ProtocolError("OE-PPUR v4 prepared state is not canonical JSON.")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ProtocolError("OE-PPUR v4 prepared state is not canonical JSON.")


__all__ = ()
