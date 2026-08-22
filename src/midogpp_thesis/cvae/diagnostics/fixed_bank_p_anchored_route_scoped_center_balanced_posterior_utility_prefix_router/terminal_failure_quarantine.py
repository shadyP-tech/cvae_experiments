"""Exact, retry-safe quarantine for the failed terminal CBPUPR v1 run."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import stat
from typing import Iterable, Iterator, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    relative_files,
    validate_content_index,
)
from .constants import (
    CENTERS,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    PUBLICATION_STATUS,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    QUARANTINED_V1_SCRATCH_ROOT,
)
from .hashing import canonical_hash


FAILED_TERMINAL_LINEAGE_ERROR = (
    "CBPUPR persisted posterior prediction/model lineage drifted."
)
FAILED_TERMINAL_LINEAGE_FINAL_MEMBERS = frozenset(
    {
        "reports/fresh_process_attestation.json",
        "reports/validation_report.json",
    }
)
FAILED_TERMINAL_LINEAGE_FILES = frozenset(REQUIRED_FILES) - (
    FAILED_TERMINAL_LINEAGE_FINAL_MEMBERS
)
FAILED_TERMINAL_LINEAGE_ARTIFACT_DIRECTORIES = frozenset(
    {"arrays", "manifests", "provenance", "reports", "tables"}
)
FAILED_TERMINAL_LINEAGE_SCRATCH_DIRECTORIES = frozenset(
    {
        "prediction_cache",
        "prediction_cache/checkpoints",
        "source_generation",
        "source_generation/arrays",
        "source_generation/checkpoints",
        "source_generation/manifests",
    }
)
FAILED_TERMINAL_LINEAGE_SCRATCH_FILES = frozenset(
    {
        "source_generation/arrays/frozen_source_streams.npy",
        "source_generation/manifests/frozen_source_stream_index.json",
        "source_generation/manifests/frozen_source_stream_lock.json",
    }
)

_SCRATCH_TO_ARTIFACT_SOURCE_MEMBERS = {
    "source_generation/arrays/frozen_source_streams.npy": (
        "arrays/frozen_source_streams.npy"
    ),
    "source_generation/manifests/frozen_source_stream_index.json": (
        "manifests/frozen_source_stream_index.json"
    ),
    "source_generation/manifests/frozen_source_stream_lock.json": (
        "manifests/frozen_source_stream_lock.json"
    ),
}
_TERMINAL_LINEAGE_QUARANTINE_SUFFIX = re.compile(
    r"\.quarantine-terminal-lineage-([0-9]{8}T[0-9]{6}Z)"
)
_RUN_STATE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "phase",
        "error",
        "error_class",
        "updated_at_utc",
        "cross_run_recovery_allowed",
        "terminal_recovery_allowed",
    }
)
_CAPABILITY_KEYS = frozenset(
    {
        "schema_version",
        "plan_seal_hash",
        "event_count",
        "events",
        "target_candidate_seal_complete",
        "pre_evaluation_seal_complete",
        "pseudo_evaluation_route_count",
        "calibration_seal_complete",
        "decision_count",
        "aggregate_seal_complete",
        "terminal_opened",
        "raw_labels_persisted",
        "audit_hash",
    }
)
_EXPECTED_TERMINAL_LINEAGE_CAPABILITY_EVENT_COUNT = (
    len(CENTERS) * (len(CENTERS) - 1)
    + len(CENTERS) * (len(CENTERS) - 1) * (len(CENTERS) - 2)
    + EXPECTED_OUTER_PLAN_COUNT
    + EXPECTED_PSEUDO_ROUTE_COUNT
    + 1
)


def audit_failed_terminal_lineage_for_quarantine(
    root: Path,
) -> Mapping[str, object]:
    """Certify the one exact terminal-lineage failure and its owned scratch."""

    logical_root = _logical_source_root(root)
    scratch = _terminal_scratch_root()
    if _is_present(logical_root) and not logical_root.is_dir():
        raise ProtocolError("CBPUPR terminal-lineage root is absent or unsafe.")
    if _is_present(scratch) and not scratch.is_dir():
        raise ProtocolError("CBPUPR terminal-lineage scratch is absent or unsafe.")
    with _exclusive_existing_terminal_run_lock(logical_root):
        return _audit_locked_failed_terminal_lineage(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=scratch,
            observed_scratch=scratch,
        )


def quarantine_failed_terminal_lineage(
    root: Path,
    *,
    artifact_destination: Path,
    scratch_destination: Path,
) -> Mapping[str, object]:
    """Preserve the failed terminal artifact and scratch without reusing bytes.

    Scratch moves first. A retry accepts only that exact interrupted state or
    the completed two-move state, so failed v1 bytes can never become resumable
    input.
    """

    logical_root = _logical_source_root(root)
    logical_scratch = _terminal_scratch_root()
    quarantine_root, quarantine_scratch = _terminal_quarantine_destinations(
        logical_root,
        Path(artifact_destination),
        logical_scratch,
        Path(scratch_destination),
    )
    paths = (
        logical_root,
        logical_scratch,
        quarantine_root,
        quarantine_scratch,
    )
    if any(path.is_symlink() for path in paths):
        raise ProtocolError("CBPUPR terminal-lineage quarantine path is a symlink.")

    # (artifact source, scratch source, artifact destination, scratch destination)
    initial = (True, True, False, False)
    scratch_moved = (True, False, False, True)
    completed = (False, False, True, True)
    state = tuple(_is_present(path) for path in paths)
    if state not in {initial, scratch_moved, completed}:
        raise ProtocolError("CBPUPR terminal-lineage quarantine state is unsafe.")

    if state == completed:
        with _exclusive_existing_terminal_run_lock(quarantine_root):
            audit = _audit_locked_failed_terminal_lineage(
                logical_root=logical_root,
                observed_root=quarantine_root,
                logical_scratch=logical_scratch,
                observed_scratch=quarantine_scratch,
            )
        return _terminal_quarantine_receipt(
            source_root=logical_root,
            quarantine_root=quarantine_root,
            source_scratch=logical_scratch,
            quarantine_scratch=quarantine_scratch,
            audit=audit,
        )

    with _exclusive_existing_terminal_run_lock(logical_root):
        locked_state = tuple(_is_present(path) for path in paths)
        if locked_state not in {initial, scratch_moved}:
            raise ProtocolError("CBPUPR terminal-lineage quarantine state changed.")
        observed_scratch = (
            logical_scratch if locked_state == initial else quarantine_scratch
        )
        audit = _audit_locked_failed_terminal_lineage(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=logical_scratch,
            observed_scratch=observed_scratch,
        )
        root_snapshot = _tree_stat_snapshot(
            logical_root, (".run.lock", *sorted(FAILED_TERMINAL_LINEAGE_FILES))
        )
        scratch_snapshot = _tree_stat_snapshot(
            observed_scratch, sorted(FAILED_TERMINAL_LINEAGE_SCRATCH_FILES)
        )

        if locked_state == initial:
            if _is_present(quarantine_scratch):
                raise ProtocolError(
                    "CBPUPR terminal-lineage scratch destination appeared."
                )
            os.rename(logical_scratch, quarantine_scratch)
            _assert_tree_stat_snapshot(quarantine_scratch, scratch_snapshot)
        if _is_present(quarantine_root):
            raise ProtocolError(
                "CBPUPR terminal-lineage artifact destination appeared."
            )
        os.rename(logical_root, quarantine_root)
        # The descriptor keeps the moved root locked through post-move checks.
        if (
            _is_present(logical_root)
            or _is_present(logical_scratch)
            or not quarantine_root.is_dir()
            or not quarantine_scratch.is_dir()
        ):
            raise ProtocolError(
                "CBPUPR terminal-lineage quarantine move was incomplete."
            )
        _assert_tree_stat_snapshot(quarantine_root, root_snapshot)
        _assert_tree_stat_snapshot(quarantine_scratch, scratch_snapshot)
        _assert_failed_terminal_artifact_inventory(quarantine_root)
        _assert_failed_terminal_scratch_inventory(quarantine_scratch)
        # Recompute the full content-index and scratch/source hashes at the
        # destinations while the moved artifact lock remains held.  Stat
        # snapshots cannot detect a same-size in-place mutation in the narrow
        # interval between the pre-move audit and the renames.
        post_move_audit = _audit_locked_failed_terminal_lineage(
            logical_root=logical_root,
            observed_root=quarantine_root,
            logical_scratch=logical_scratch,
            observed_scratch=quarantine_scratch,
        )
        if post_move_audit != audit:
            raise ProtocolError("CBPUPR terminal-lineage moved bytes drifted.")
        audit = post_move_audit

    return _terminal_quarantine_receipt(
        source_root=logical_root,
        quarantine_root=quarantine_root,
        source_scratch=logical_scratch,
        quarantine_scratch=quarantine_scratch,
        audit=audit,
    )


def _audit_locked_failed_terminal_lineage(
    *,
    logical_root: Path,
    observed_root: Path,
    logical_scratch: Path,
    observed_scratch: Path,
) -> Mapping[str, object]:
    _assert_failed_terminal_artifact_inventory(observed_root)
    assert_closed_world(observed_root, allow_incomplete=True)

    state = read_json(observed_root / "reports/run_state.json")
    if (
        set(state) != _RUN_STATE_KEYS
        or state.get("schema_version") != "fixed_bank_cbpupr_run_state_v1"
        or state.get("status") != "FAILED"
        or state.get("phase") != "CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION"
        or state.get("error") != FAILED_TERMINAL_LINEAGE_ERROR
        or state.get("error_class") != "ProtocolError"
        or not isinstance(state.get("updated_at_utc"), str)
        or not str(state.get("updated_at_utc"))
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR terminal-lineage failed state drifted.")

    protocol = read_json(observed_root / "manifests/protocol_manifest.json")
    protocol_unhashed = {
        key: value
        for key, value in protocol.items()
        if key != "protocol_manifest_hash"
    }
    if (
        protocol.get("schema_version")
        != "fixed_bank_cbpupr_protocol_manifest_v1"
        or protocol.get("experiment_id") != QUARANTINED_V1_EXPERIMENT_ID
        or protocol.get("output_artifact_id")
        != QUARANTINED_V1_OUTPUT_ARTIFACT_ID
        or protocol.get("publication_status") != PUBLICATION_STATUS
        or protocol.get("test_split_previously_consumed") is not True
        or protocol.get("fresh_evidence") is not False
        or protocol.get("previous_stage90_output_or_checkpoint_used") is not False
        or protocol.get("protocol_manifest_hash")
        != canonical_hash(protocol_unhashed)
    ):
        raise ProtocolError("CBPUPR terminal-lineage v1 identity drifted.")

    capability = read_json(observed_root / "reports/label_capability_report.json")
    events = capability.get("events")
    outer_plan_seal = read_json(observed_root / "manifests/outer_plan_seal.json")
    if (
        set(capability) != _CAPABILITY_KEYS
        or capability.get("schema_version")
        != "fixed_bank_cbpupr_label_access_audit_v1"
        or not isinstance(events, list)
        or len(events) != _EXPECTED_TERMINAL_LINEAGE_CAPABILITY_EVENT_COUNT
        or capability.get("event_count") != len(events)
        or capability.get("audit_hash") != canonical_hash(events)
        or any(
            not isinstance(event, dict)
            or event.get("raw_labels_persisted") is not False
            for event in events
        )
        or sum(
            isinstance(event, dict)
            and event.get("role") == "target_terminal_after_aggregate_seal"
            for event in events
        )
        != 1
        or capability.get("plan_seal_hash") != outer_plan_seal.get("seal_hash")
        or capability.get("target_candidate_seal_complete") is not True
        or capability.get("pre_evaluation_seal_complete") is not True
        or capability.get("pseudo_evaluation_route_count")
        != EXPECTED_PSEUDO_ROUTE_COUNT
        or capability.get("calibration_seal_complete") is not True
        or capability.get("decision_count") != 4 * EXPECTED_OUTER_PLAN_COUNT
        or capability.get("aggregate_seal_complete") is not True
        or capability.get("terminal_opened") is not True
        or capability.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("CBPUPR terminal-lineage capability state drifted.")

    content = validate_content_index(observed_root)
    content_rows = content.get("members")
    if not isinstance(content_rows, list):
        raise ProtocolError("CBPUPR terminal-lineage content index drifted.")
    by_member = {
        str(row["member"]): {
            "path": str(row["member"]),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in content_rows
        if isinstance(row, dict)
    }
    if set(by_member) != set(CONTENT_INDEX_MEMBERS):
        raise ProtocolError("CBPUPR terminal-lineage content members drifted.")

    observed = frozenset(relative_files(observed_root))
    artifact_members = []
    for relative in (".run.lock", *sorted(observed)):
        if relative in by_member:
            artifact_members.append(by_member[relative])
        else:
            path = observed_root / relative
            artifact_members.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    scratch_members = _audit_failed_terminal_scratch(observed_scratch)
    artifact_by_path = {
        str(member["path"]): member for member in artifact_members
    }
    scratch_by_path = {str(member["path"]): member for member in scratch_members}
    if any(
        {
            "size_bytes": scratch_by_path[scratch_relative]["size_bytes"],
            "sha256": scratch_by_path[scratch_relative]["sha256"],
        }
        != {
            "size_bytes": artifact_by_path[artifact_relative]["size_bytes"],
            "sha256": artifact_by_path[artifact_relative]["sha256"],
        }
        for scratch_relative, artifact_relative in (
            _SCRATCH_TO_ARTIFACT_SOURCE_MEMBERS.items()
        )
    ):
        raise ProtocolError(
            "CBPUPR terminal-lineage scratch/source artifact bytes differ."
        )

    payload: dict[str, object] = {
        "schema_version": "fixed_bank_cbpupr_terminal_lineage_quarantine_audit_v1",
        "status": "PASS",
        "source_root": str(logical_root),
        "source_scratch_root": str(logical_scratch),
        "source_run_status": "FAILED",
        "source_run_phase": "CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION",
        "source_error_class": "ProtocolError",
        "source_error": FAILED_TERMINAL_LINEAGE_ERROR,
        "content_index_hash": content.get("content_index_hash"),
        "terminal_capability_opened": True,
        "persisted_terminal_outputs_present": True,
        "fresh_process_attestation_present": False,
        "final_validation_report_present": False,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "fresh_v2_execution_identity_required": True,
        "fresh_v2_workspace_prepare_required": True,
        "eligible_next_action": (
            "MOVE_SCRATCH_THEN_WHOLE_FAILED_ROOT_TO_QUARANTINE_ONLY"
        ),
        "artifact_members": artifact_members,
        "scratch_directories": sorted(
            FAILED_TERMINAL_LINEAGE_SCRATCH_DIRECTORIES
        ),
        "scratch_members": scratch_members,
    }
    return {**payload, "quarantine_audit_hash": canonical_hash(payload)}


def _audit_failed_terminal_scratch(
    observed_scratch: Path,
) -> list[dict[str, object]]:
    files = _assert_failed_terminal_scratch_inventory(observed_scratch)
    return [
        {
            "path": relative,
            "size_bytes": (observed_scratch / relative).stat().st_size,
            "sha256": sha256_file(observed_scratch / relative),
        }
        for relative in sorted(files)
    ]


def _assert_failed_terminal_artifact_inventory(root: Path) -> None:
    _assert_regular_tree(root, role="terminal-lineage artifact")
    members = tuple(root.rglob("*"))
    directories = frozenset(
        path.relative_to(root).as_posix() for path in members if path.is_dir()
    )
    files = frozenset(
        path.relative_to(root).as_posix() for path in members if path.is_file()
    )
    if (
        directories != FAILED_TERMINAL_LINEAGE_ARTIFACT_DIRECTORIES
        or files != FAILED_TERMINAL_LINEAGE_FILES | {".run.lock"}
    ):
        raise ProtocolError("CBPUPR terminal-lineage artifact inventory drifted.")


def _assert_failed_terminal_scratch_inventory(root: Path) -> frozenset[str]:
    _assert_regular_tree(root, role="terminal-lineage scratch")
    members = tuple(root.rglob("*"))
    directories = frozenset(
        path.relative_to(root).as_posix() for path in members if path.is_dir()
    )
    files = frozenset(
        path.relative_to(root).as_posix() for path in members if path.is_file()
    )
    if (
        directories != FAILED_TERMINAL_LINEAGE_SCRATCH_DIRECTORIES
        or files != FAILED_TERMINAL_LINEAGE_SCRATCH_FILES
    ):
        raise ProtocolError("CBPUPR terminal-lineage scratch inventory drifted.")
    return files


def _terminal_quarantine_receipt(
    *,
    source_root: Path,
    quarantine_root: Path,
    source_scratch: Path,
    quarantine_scratch: Path,
    audit: Mapping[str, object],
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "schema_version": "fixed_bank_cbpupr_terminal_lineage_quarantine_receipt_v1",
        "status": "PASS",
        "source_root": str(source_root),
        "quarantine_root": str(quarantine_root),
        "source_scratch_root": str(source_scratch),
        "quarantine_scratch_root": str(quarantine_scratch),
        "source_root_absent_after_move": True,
        "source_scratch_absent_after_move": True,
        "whole_artifact_move_completed": True,
        "whole_scratch_move_completed": True,
        "move_order": ["scratch", "artifact"],
        "same_parent_atomic_rename_per_root": True,
        "quarantined_bytes_may_feed_rerun": False,
        "quarantined_v1_results_may_be_promoted": False,
        "fresh_v2_execution_identity_required": True,
        "fresh_v2_workspace_prepare_required": True,
        "pre_move_audit": dict(audit),
    }
    return {**payload, "quarantine_receipt_hash": canonical_hash(payload)}


def _logical_source_root(root: Path) -> Path:
    unresolved = Path(root)
    if unresolved.is_symlink():
        raise ProtocolError("CBPUPR terminal-lineage root is a symlink.")
    parent = unresolved.absolute().parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("CBPUPR terminal-lineage root parent is unsafe.")
    return parent.resolve() / unresolved.name


def _terminal_scratch_root() -> Path:
    scratch = Path(QUARANTINED_V1_SCRATCH_ROOT)
    if not scratch.is_absolute() or str(scratch) != QUARANTINED_V1_SCRATCH_ROOT:
        raise ProtocolError("CBPUPR terminal-lineage scratch root drifted.")
    if scratch.is_symlink():
        raise ProtocolError("CBPUPR terminal-lineage scratch root is a symlink.")
    if scratch.parent.is_symlink() or not scratch.parent.is_dir():
        raise ProtocolError("CBPUPR terminal-lineage scratch parent is unsafe.")
    return scratch.parent.resolve() / scratch.name


def _terminal_quarantine_destinations(
    source_root: Path,
    artifact_destination: Path,
    source_scratch: Path,
    scratch_destination: Path,
) -> tuple[Path, Path]:
    requested = (artifact_destination, scratch_destination)
    if any(path.is_symlink() for path in requested):
        raise ProtocolError("CBPUPR terminal-lineage quarantine path is a symlink.")
    if any(path.parent.is_symlink() or not path.parent.is_dir() for path in requested):
        raise ProtocolError("CBPUPR terminal-lineage quarantine parent is unsafe.")
    artifact = artifact_destination.parent.resolve() / artifact_destination.name
    scratch = scratch_destination.parent.resolve() / scratch_destination.name
    if artifact.parent != source_root.parent or scratch.parent != source_scratch.parent:
        raise ProtocolError(
            "CBPUPR terminal-lineage quarantine destination is not same-parent."
        )
    artifact_match = re.fullmatch(
        re.escape(source_root.name) + _TERMINAL_LINEAGE_QUARANTINE_SUFFIX.pattern,
        artifact.name,
    )
    scratch_match = re.fullmatch(
        re.escape(source_scratch.name) + _TERMINAL_LINEAGE_QUARANTINE_SUFFIX.pattern,
        scratch.name,
    )
    if artifact_match is None or scratch_match is None:
        raise ProtocolError("CBPUPR terminal-lineage quarantine name is unsafe.")
    artifact_timestamp = artifact_match.group(1)
    scratch_timestamp = scratch_match.group(1)
    try:
        datetime.strptime(artifact_timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ProtocolError(
            "CBPUPR terminal-lineage quarantine timestamp is invalid."
        ) from exc
    if artifact_timestamp != scratch_timestamp:
        raise ProtocolError(
            "CBPUPR terminal-lineage quarantine timestamps differ."
        )
    return artifact, scratch


def _assert_regular_tree(root: Path, *, role: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError(f"CBPUPR {role} root is absent or unsafe.")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ProtocolError(f"CBPUPR {role} tree contains an unsafe member.")


def _tree_stat_snapshot(
    root: Path, members: Iterable[object]
) -> dict[str, tuple[int, int, int]]:
    snapshot: dict[str, tuple[int, int, int]] = {}
    for relative in members:
        path = root / str(relative)
        value = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise ProtocolError("CBPUPR quarantine snapshot member is unsafe.")
        snapshot[str(relative)] = (value.st_dev, value.st_ino, value.st_size)
    return snapshot


def _assert_tree_stat_snapshot(
    root: Path, snapshot: Mapping[str, tuple[int, int, int]]
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("CBPUPR quarantine destination is absent or unsafe.")
    for relative, expected in snapshot.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("CBPUPR quarantine moved member is unsafe.")
        value = path.stat(follow_symlinks=False)
        if (value.st_dev, value.st_ino, value.st_size) != expected:
            raise ProtocolError("CBPUPR quarantine moved bytes drifted.")


def _is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


@contextmanager
def _exclusive_existing_terminal_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("CBPUPR terminal-lineage run lock is absent or unsafe.")
    flags = (
        os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProtocolError("CBPUPR terminal-lineage run lock is not regular.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            raise ProtocolError(
                "CBPUPR terminal diagnostic is active; quarantine is forbidden."
            ) from exc
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = (
    "FAILED_TERMINAL_LINEAGE_ARTIFACT_DIRECTORIES",
    "FAILED_TERMINAL_LINEAGE_ERROR",
    "FAILED_TERMINAL_LINEAGE_FILES",
    "FAILED_TERMINAL_LINEAGE_FINAL_MEMBERS",
    "FAILED_TERMINAL_LINEAGE_SCRATCH_DIRECTORIES",
    "FAILED_TERMINAL_LINEAGE_SCRATCH_FILES",
    "audit_failed_terminal_lineage_for_quarantine",
    "quarantine_failed_terminal_lineage",
)
