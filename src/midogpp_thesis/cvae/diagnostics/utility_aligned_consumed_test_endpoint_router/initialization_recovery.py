"""Exact same-root retry boundaries for pre-label identity repairs."""

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
FAILED_EMBEDDING_IDENTITY_STATE = {
    "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
    "status": "FAILED",
    "phase": "SOURCE_AND_LABEL_FREE_FEATURES",
    "promotion_eligible": False,
    "terminal_consumed_test_diagnostic_only": True,
    "automatic_resume_requires_hash_validation": True,
    "error": "ProtocolError: Consumed-test embedding row identity drifted.",
}
SOURCE_FEATURE_RECOVERY_FILES = frozenset(
    {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "arrays/frozen_source_streams.npy",
        "manifests/action_library.json",
        "manifests/frozen_source_stream_index.json",
        "manifests/frozen_source_stream_lock.json",
        "manifests/protocol_manifest.json",
        "manifests/support_partition_lock.json",
        "reports/run_state.json",
        "reports/workstation_preflight.json",
        "tables/support_partitions.csv",
    }
)


def detect_initializing_cache_identity_recovery(root: Path) -> bool:
    """Recognize only either original false pre-label identity failure."""

    state_path = root / "reports/run_state.json"
    if not state_path.is_file():
        return False
    state = read_json(state_path)
    if state.get("status") != "FAILED":
        return False
    if (
        state.get("phase") == FAILED_CACHE_IDENTITY_STATE["phase"]
        and state.get("error") == FAILED_CACHE_IDENTITY_STATE["error"]
    ):
        expected_state = FAILED_CACHE_IDENTITY_STATE
        expected_files = INITIALIZING_RECOVERY_FILES
        boundary = "initialization"
    elif (
        state.get("phase") == FAILED_EMBEDDING_IDENTITY_STATE["phase"]
        and state.get("error") == FAILED_EMBEDDING_IDENTITY_STATE["error"]
    ):
        expected_state = FAILED_EMBEDDING_IDENTITY_STATE
        expected_files = SOURCE_FEATURE_RECOVERY_FILES
        boundary = "source-feature"
    else:
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
        observed != expected_files
        or state_unhashed != expected_state
        or not isinstance(updated, str)
        or not updated
    ):
        missing = sorted(expected_files.difference(observed))
        extras = sorted(observed.difference(expected_files))
        raise ProtocolError(
            f"Endpoint-router {boundary} recovery boundary drifted: "
            f"missing={missing}, extras={extras}, "
            f"state_matches={state_unhashed == expected_state}."
        )
    return True


__all__ = (
    "FAILED_CACHE_IDENTITY_STATE",
    "FAILED_EMBEDDING_IDENTITY_STATE",
    "INITIALIZING_RECOVERY_FILES",
    "SOURCE_FEATURE_RECOVERY_FILES",
    "detect_initializing_cache_identity_recovery",
)
