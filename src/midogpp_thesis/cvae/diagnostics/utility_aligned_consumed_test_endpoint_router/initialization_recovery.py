"""Exact same-root retry boundary for the pre-compute cache-schema repair."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .artifact_io import read_json


FAILED_CACHE_IDENTITY_STATE = {
    "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
    "status": "FAILED",
    "phase": "INITIALIZING",
    "promotion_eligible": False,
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_requires_hash_validation": True,
    "error": "ProtocolError: Endpoint-router test-cache identity drifted.",
}
INITIALIZING_RECOVERY_FILES = frozenset(
    {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "reports/run_state.json",
    }
)


def detect_initializing_cache_identity_recovery(root: Path) -> bool:
    """Recognize only the original false cache-identity failure boundary."""

    state_path = root / "reports/run_state.json"
    if not state_path.is_file():
        return False
    state = read_json(state_path)
    if not (
        state.get("status") == "FAILED"
        and state.get("phase") == "INITIALIZING"
        and state.get("error") == FAILED_CACHE_IDENTITY_STATE["error"]
    ):
        return False
    observed = frozenset(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != ".run.lock"
    )
    state_unhashed = {
        key: value for key, value in state.items() if key != "updated_at_utc"
    }
    updated = state.get("updated_at_utc")
    if (
        observed != INITIALIZING_RECOVERY_FILES
        or state_unhashed != FAILED_CACHE_IDENTITY_STATE
        or not isinstance(updated, str)
        or not updated
    ):
        missing = sorted(INITIALIZING_RECOVERY_FILES.difference(observed))
        extras = sorted(observed.difference(INITIALIZING_RECOVERY_FILES))
        raise ProtocolError(
            "Endpoint-router initialization recovery boundary drifted: "
            f"missing={missing}, extras={extras}, "
            f"state_matches={state_unhashed == FAILED_CACHE_IDENTITY_STATE}."
        )
    return True


__all__ = (
    "FAILED_CACHE_IDENTITY_STATE",
    "INITIALIZING_RECOVERY_FILES",
    "detect_initializing_cache_identity_recovery",
)
