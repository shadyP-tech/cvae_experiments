"""Durable content indexing and the single HARP v18 completion commit."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_hash
from ....runtime.artifact_io import atomic_json, read_json, sha256_file
from ....runtime.harp_v18_execution.durability import durable_barrier
from ..identity import PUBLICATION_STATUS, TERMINAL_DECISION
from ..source_train_label_access_fence import (
    SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER,
    load_source_train_label_access_fence,
)


def write_content_index(root: Path) -> Path:
    """Write the immutable member inventory, excluding itself and final commit."""

    fence = root / SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER
    if not fence.is_file() or fence.is_symlink():
        raise ProtocolError("HARP v18 content index requires the source-train-label fence.")
    load_source_train_label_access_fence(root).reauthenticate()
    members = []
    excluded = {"manifests/content_index.json", "reports/run_state.json"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            members.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    body = {
        "schema_version": "midogpp_harp_v18_content_index_v1",
        "members": members,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "may_feed_another_experiment": False,
        "run_state_excluded_as_final_commit": True,
    }
    path = root / "manifests/content_index.json"
    atomic_json(path, {**body, "content_index_hash": canonical_hash(body)})
    return path


def validate_content_index(root: Path, path: Path) -> None:
    """Recompute the complete member inventory before committing COMPLETE."""

    payload = read_json(path)
    stored_hash = payload.get("content_index_hash")
    body = {key: value for key, value in payload.items() if key != "content_index_hash"}
    raw_members = payload.get("members")
    if (
        payload.get("schema_version") != "midogpp_harp_v18_content_index_v1"
        or stored_hash != canonical_hash(body)
        or not isinstance(raw_members, list)
    ):
        raise ProtocolError("HARP v18 content index self-binding drifted.")
    observed: list[dict[str, str]] = []
    excluded = {"manifests/content_index.json", "reports/run_state.json"}
    for member in sorted(root.rglob("*")):
        if member.is_symlink():
            raise ProtocolError("HARP v18 content index encountered a symlink.")
        relative = member.relative_to(root).as_posix()
        if member.is_file() and relative not in excluded:
            observed.append({"path": relative, "sha256": sha256_file(member)})
    if raw_members != observed:
        raise ProtocolError("HARP v18 content index member coverage drifted.")
    if not any(
        isinstance(member, dict)
        and member.get("path") == SOURCE_TRAIN_LABEL_ACCESS_FENCE_MEMBER
        for member in raw_members
    ):
        raise ProtocolError("HARP v18 content index omits the source-train-label fence.")
    load_source_train_label_access_fence(root).reauthenticate()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProtocolError("HARP v18 completion directory fsync failed.") from exc


def commit_completion_state(
    root: Path,
    run_state_path: Path,
    payload: Mapping[str, object],
    *,
    durable_members: tuple[Path, ...],
) -> None:
    """Commit COMPLETE only after every indexed terminal member is durable."""

    if run_state_path.exists() or run_state_path.is_symlink():
        raise ProtocolError("HARP v18 completion marker already exists.")
    pending = run_state_path.with_name(f".run_state.{os.getpid()}.pending.json")
    if pending.exists() or pending.is_symlink():
        raise ProtocolError("HARP v18 completion transaction has stale pending state.")
    atomic_json(pending, payload)
    try:
        if read_json(pending) != dict(payload):
            raise ProtocolError("HARP v18 pending completion state changed bytes.")
        durable_barrier((*durable_members, pending))
        os.replace(pending, run_state_path)
        # The first barrier makes the inode and all dependencies durable; the
        # atomic rename is the commit point. The second barrier and directory
        # fsyncs make that commit marker itself durable.
        durable_barrier((*durable_members, run_state_path))
        _fsync_directory(run_state_path.parent)
        _fsync_directory(root / "manifests")
        _fsync_directory(root)
    except BaseException:
        pending.unlink(missing_ok=True)
        if run_state_path.exists() and not run_state_path.is_symlink():
            run_state_path.unlink(missing_ok=True)
            _fsync_directory(run_state_path.parent)
        raise


__all__ = (
    "commit_completion_state",
    "validate_content_index",
    "write_content_index",
)
