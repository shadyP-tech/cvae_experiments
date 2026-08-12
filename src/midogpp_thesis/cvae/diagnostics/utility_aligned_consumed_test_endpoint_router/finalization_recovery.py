"""Exact same-root boundaries for terminal validation and revalidation."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .artifact_io import read_json, relative_files
from .bundle import REQUIRED_FILES


FAILED_FEATURE_PARTITION_BINDING_STATE = {
    "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
    "status": "FAILED",
    "phase": "CLOSED_WORLD_VALIDATION",
    "promotion_eligible": False,
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_requires_hash_validation": True,
    "error": "ProtocolError: Candidate feature partition binding drifted.",
}
FINALIZATION_RECOVERY_FILES = frozenset(REQUIRED_FILES).difference(
    {"reports/validation_report.json"}
)
COMPLETE_ENDPOINT_ROUTER_STATE = {
    "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
    "status": "COMPLETE",
    "phase": "COMPLETE",
    "promotion_eligible": False,
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_requires_hash_validation": True,
    "error": None,
}
COMPLETE_REVALIDATION_FILES = frozenset(REQUIRED_FILES)


def detect_feature_partition_binding_finalization_recovery(root: Path) -> bool:
    """Recognize only the fully persisted failed-validation snapshot."""

    state_path = root / "reports/run_state.json"
    if state_path.is_symlink():
        raise ProtocolError(
            "Endpoint-router finalization recovery state must not be a symlink."
        )
    if not state_path.is_file():
        return False
    state = read_json(state_path)
    if state.get("status") != "FAILED":
        return False
    if (
        state.get("phase") != FAILED_FEATURE_PARTITION_BINDING_STATE["phase"]
        or state.get("error") != FAILED_FEATURE_PARTITION_BINDING_STATE["error"]
    ):
        return False

    observed = set(relative_files(root))
    observed.discard(".run.lock")
    observed_files = frozenset(observed)
    state_unhashed = {
        key: value for key, value in state.items() if key != "updated_at_utc"
    }
    updated = state.get("updated_at_utc")
    if (
        observed_files != FINALIZATION_RECOVERY_FILES
        or state_unhashed != FAILED_FEATURE_PARTITION_BINDING_STATE
        or not isinstance(updated, str)
        or not updated
    ):
        missing = sorted(FINALIZATION_RECOVERY_FILES.difference(observed_files))
        extras = sorted(observed_files.difference(FINALIZATION_RECOVERY_FILES))
        raise ProtocolError(
            "Endpoint-router finalization recovery boundary drifted: "
            f"missing={missing}, extras={extras}, "
            "state_matches="
            f"{state_unhashed == FAILED_FEATURE_PARTITION_BINDING_STATE}."
        )
    return True


def detect_complete_endpoint_router_revalidation(root: Path) -> bool:
    """Recognize only the exact terminal bundle accepted for revalidation."""

    state_path = root / "reports/run_state.json"
    if state_path.is_symlink():
        raise ProtocolError(
            "Endpoint-router complete revalidation state must not be a symlink."
        )
    if not state_path.is_file():
        return False
    state = read_json(state_path)
    if state.get("status") != "COMPLETE":
        return False

    observed = set(relative_files(root))
    observed.discard(".run.lock")
    observed_files = frozenset(observed)
    state_unhashed = {
        key: value for key, value in state.items() if key != "updated_at_utc"
    }
    updated = state.get("updated_at_utc")
    if (
        observed_files != COMPLETE_REVALIDATION_FILES
        or state_unhashed != COMPLETE_ENDPOINT_ROUTER_STATE
        or not isinstance(updated, str)
        or not updated
    ):
        missing = sorted(COMPLETE_REVALIDATION_FILES.difference(observed_files))
        extras = sorted(observed_files.difference(COMPLETE_REVALIDATION_FILES))
        raise ProtocolError(
            "Endpoint-router complete revalidation boundary drifted: "
            f"missing={missing}, extras={extras}, "
            f"state_matches={state_unhashed == COMPLETE_ENDPOINT_ROUTER_STATE}."
        )
    return True


__all__ = (
    "COMPLETE_ENDPOINT_ROUTER_STATE",
    "COMPLETE_REVALIDATION_FILES",
    "FAILED_FEATURE_PARTITION_BINDING_STATE",
    "FINALIZATION_RECOVERY_FILES",
    "detect_complete_endpoint_router_revalidation",
    "detect_feature_partition_binding_finalization_recovery",
)
