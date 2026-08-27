"""Monotonic in-attempt run state; never a recovery capability."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .authorization_lease import AuthorizationLease
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    canonical_hash,
    require_sha256,
)


RUN_STATE_MEMBER = "reports/run_state.json"
PHASE_ORDER = (
    "BEGIN",
    "WORKSTATION_PREFLIGHT",
    "SOURCE_INNER_DEVELOPMENT_FREEZE",
    "FRESH_PHYSICAL_SOURCE_STREAMS",
    "FRESH_PHYSICAL_PREDICTION_SURFACE",
    "ALL_G_DECISIONS_SEALED",
    "ALL_SELECTION_DECISIONS_SEALED",
    "ALL_CALIBRATION_DECISIONS_SEALED",
    "ROUTE_POLICY_SEALED",
    "DURABLE_PRETERMINAL_BARRIER",
    "TWO_FRESH_PRETERMINAL_VALIDATORS",
    "TERMINAL_LABELS_AND_DIAGNOSTICS",
    "TWO_FRESH_FINAL_VALIDATORS",
    "POSTVALIDATION_INDEX_AUTHENTICATED",
    "FINALIZING_AUTHORIZATION",
    "COMPLETE",
)


def write_run_state(
    root: Path,
    *,
    authorization_lease: AuthorizationLease,
    config_hash: str,
    status: str,
    phase: str,
    bound_hashes: Mapping[str, str] | None = None,
    error_class: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    if not isinstance(authorization_lease, AuthorizationLease) or (
        authorization_lease.process_id != os.getpid()
    ):
        raise ProtocolError("SCEPTRE v2 run state must follow its process lease.")
    digest = require_sha256(config_hash, "run-state config")
    if (
        status not in {"RUNNING", "COMPLETE", "FAILED", "FINALIZATION_ERROR"}
        or phase not in PHASE_ORDER
        or (status == "COMPLETE" and phase != "COMPLETE")
        or (status == "RUNNING" and phase == "COMPLETE")
        or (
            status == "RUNNING"
            and authorization_lease.status != "CLAIMED_IN_PROGRESS"
        )
        or (
            status == "COMPLETE"
            and authorization_lease.status != "COMPLETE_EXHAUSTED"
        )
        or (
            status == "FAILED"
            and authorization_lease.status != "FAILED_EXHAUSTED"
        )
        or (
            status == "FINALIZATION_ERROR"
            and (
                phase != "FINALIZING_AUTHORIZATION"
                or authorization_lease.status
                not in {"CLAIMED_IN_PROGRESS", "COMPLETE_EXHAUSTED"}
            )
        )
    ):
        raise ProtocolError("SCEPTRE v2 run-state transition drifted.")
    hashes = {
        str(role): require_sha256(value, f"run-state {role}")
        for role, value in sorted((bound_hashes or {}).items())
    }
    path = Path(root) / RUN_STATE_MEMBER
    if path.is_symlink():
        raise ProtocolError("SCEPTRE v2 run state is unsafe.")
    if path.exists():
        _validate_previous(
            read_json(path),
            authorization_lease=authorization_lease,
            config_hash=digest,
            next_phase=phase,
            next_hashes=hashes,
        )
    base = {
        "schema_version": "sceptre_v2_single_attempt_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "phase": phase,
        "process_id": os.getpid(),
        "authorization_lease_hash": authorization_lease.lease_hash,
        "config_hash": digest,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "single_use_execution_identity": True,
        "authorization_exhausted": status
        in {"COMPLETE", "FAILED", "FINALIZATION_ERROR"},
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "prior_router_state_used": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
        "bound_hashes": hashes,
        "error_class": error_class,
        "error": None if error is None else str(error)[:2000],
    }
    payload = {**base, "state_hash": canonical_hash(base)}
    atomic_json(path, payload)
    return payload


def _validate_previous(
    previous: Mapping[str, object],
    *,
    authorization_lease: AuthorizationLease,
    config_hash: str,
    next_phase: str,
    next_hashes: Mapping[str, str],
) -> None:
    base = {key: value for key, value in previous.items() if key != "state_hash"}
    previous_hashes = previous.get("bound_hashes")
    previous_phase = str(previous.get("phase"))
    if (
        previous.get("state_hash") != canonical_hash(base)
        or previous.get("schema_version") != "sceptre_v2_single_attempt_run_state_v1"
        or previous.get("experiment_id") != EXPERIMENT_ID
        or previous.get("status") != "RUNNING"
        or previous.get("process_id") != os.getpid()
        or previous.get("config_hash") != config_hash
        or previous.get("authorization_exhausted") is not False
        or previous.get("cross_run_recovery_allowed") is not False
        or previous.get("terminal_recovery_allowed") is not False
        or previous_phase not in PHASE_ORDER
        or PHASE_ORDER.index(next_phase) < PHASE_ORDER.index(previous_phase)
        or not isinstance(previous_hashes, Mapping)
    ):
        raise ProtocolError("SCEPTRE v2 prior run state is not authenticated.")
    previous_lease_hash = require_sha256(
        previous.get("authorization_lease_hash"), "prior authorization lease"
    )
    if previous_lease_hash != authorization_lease.lease_hash:
        lease_payload = read_json(authorization_lease.root / "lease.json")
        if lease_payload.get("predecessor_lease_hash") != previous_lease_hash:
            raise ProtocolError("SCEPTRE v2 terminal lease transition drifted.")
    authenticated = {
        str(role): require_sha256(value, f"prior run-state {role}")
        for role, value in previous_hashes.items()
    }
    if any(next_hashes.get(role) != value for role, value in authenticated.items()):
        raise ProtocolError("SCEPTRE v2 run-state hashes are not append-only.")


__all__ = ("PHASE_ORDER", "RUN_STATE_MEMBER", "write_run_state")
