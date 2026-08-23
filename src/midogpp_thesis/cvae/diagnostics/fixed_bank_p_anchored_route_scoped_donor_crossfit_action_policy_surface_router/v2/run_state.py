"""Monotonic single-attempt run-state records for P-DCAPS v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json
from ..identity import PUBLICATION_STATUS, TERMINAL_DECISION
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    canonical_hash,
    require_sha256,
)


RUN_STATE_MEMBER = "reports/run_state.json"
_RUN_STATE_KEYS = {
    "schema_version",
    "experiment_id",
    "status",
    "phase",
    "process_id",
    "config_hash",
    "authorization_basis",
    "authorization_scope",
    "single_use_execution_identity",
    "authorization_exhausted",
    "cross_run_recovery_allowed",
    "terminal_recovery_allowed",
    "scratch_recovery_used",
    "v1_state_used",
    "publication_status",
    "terminal_decision",
    "fresh_evidence",
    "may_feed_another_experiment",
    "bound_hashes",
    "error_class",
    "error",
    "state_hash",
}
PHASE_ORDER = (
    "BEGIN",
    "WORKSTATION_PREFLIGHT",
    "FRESH_PHYSICAL_810_CELLS",
    "ROUTE_SURFACES_AND_PSEUDO_RESPONSES",
    "FOUR_SPAWN_OUTER_H_WORKERS",
    "DURABLE_PRETERMINAL_BARRIER",
    "TWO_FRESH_PRETERMINAL_VALIDATORS",
    "TERMINAL_LABELS_AND_DIAGNOSTICS",
    "TWO_FRESH_FINAL_VALIDATORS",
    "COMPLETE",
)


def write_run_state(
    root: Path,
    *,
    config_hash: str,
    status: str,
    phase: str,
    bound_hashes: Mapping[str, str] | None = None,
    error_class: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    """Atomically record the current phase; no state authorizes recovery."""

    digest = require_sha256(config_hash, "v2 run-state config")
    state = str(status)
    phase_id = str(phase)
    if (
        state not in {"RUNNING", "COMPLETE", "FAILED"}
        or phase_id not in PHASE_ORDER
        or (state == "COMPLETE" and phase_id != "COMPLETE")
        or (state == "RUNNING" and phase_id == "COMPLETE")
    ):
        raise ProtocolError("P-DCAPS v2 run-state transition drifted.")
    hashes = {
        str(role): require_sha256(value, f"v2 run-state {role}")
        for role, value in sorted((bound_hashes or {}).items())
    }
    state_path = Path(root) / RUN_STATE_MEMBER
    if state_path.is_symlink():
        raise ProtocolError("P-DCAPS v2 run state is not authenticated.")
    if state_path.exists():
        previous = read_json(state_path)
        _validate_previous_state(
            previous,
            config_hash=digest,
            next_phase=phase_id,
            next_bound_hashes=hashes,
        )
    base = {
        "schema_version": "pdcaps_v2_single_attempt_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "phase": phase_id,
        "process_id": os.getpid(),
        "config_hash": digest,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "single_use_execution_identity": True,
        "authorization_exhausted": state in {"COMPLETE", "FAILED"},
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "scratch_recovery_used": False,
        "v1_state_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
        "bound_hashes": hashes,
        "error_class": None if error_class is None else str(error_class),
        "error": None if error is None else str(error)[:2000],
    }
    payload = {**base, "state_hash": canonical_hash(base)}
    atomic_json(state_path, payload)
    return payload


def _validate_previous_state(
    previous: Mapping[str, object],
    *,
    config_hash: str,
    next_phase: str,
    next_bound_hashes: Mapping[str, str],
) -> None:
    """Authenticate the active attempt before allowing an append-only update."""

    previous_base = {
        key: value for key, value in previous.items() if key != "state_hash"
    }
    previous_status = str(previous.get("status"))
    previous_phase = str(previous.get("phase"))
    previous_hashes = previous.get("bound_hashes")
    if (
        set(previous) != _RUN_STATE_KEYS
        or previous.get("schema_version")
        != "pdcaps_v2_single_attempt_run_state_v1"
        or previous.get("state_hash") != canonical_hash(previous_base)
        or previous.get("experiment_id") != EXPERIMENT_ID
        or previous.get("config_hash") != config_hash
        or previous.get("process_id") != os.getpid()
        or previous.get("authorization_basis") != AUTHORIZATION_BASIS
        or previous.get("authorization_scope") != AUTHORIZATION_SCOPE
        or previous.get("single_use_execution_identity") is not True
        or previous.get("authorization_exhausted") is not False
        or previous.get("cross_run_recovery_allowed") is not False
        or previous.get("terminal_recovery_allowed") is not False
        or previous.get("scratch_recovery_used") is not False
        or previous.get("v1_state_used") is not False
        or previous.get("publication_status") != PUBLICATION_STATUS
        or previous.get("terminal_decision") != TERMINAL_DECISION
        or previous.get("fresh_evidence") is not False
        or previous.get("may_feed_another_experiment") is not False
        or previous.get("error_class") is not None
        or previous.get("error") is not None
        or previous_status != "RUNNING"
        or previous_phase not in PHASE_ORDER
        or PHASE_ORDER.index(next_phase) < PHASE_ORDER.index(previous_phase)
        or not isinstance(previous_hashes, Mapping)
    ):
        raise ProtocolError("P-DCAPS v2 prior run state is not authenticated.")
    authenticated_hashes = {
        str(role): require_sha256(value, f"prior v2 run-state {role}")
        for role, value in previous_hashes.items()
    }
    if (
        tuple(authenticated_hashes) != tuple(sorted(authenticated_hashes))
        or any(
            next_bound_hashes.get(role) != value
            for role, value in authenticated_hashes.items()
        )
    ):
        raise ProtocolError("P-DCAPS v2 run-state bound hashes are not append-only.")


__all__ = ("PHASE_ORDER", "RUN_STATE_MEMBER", "write_run_state")
