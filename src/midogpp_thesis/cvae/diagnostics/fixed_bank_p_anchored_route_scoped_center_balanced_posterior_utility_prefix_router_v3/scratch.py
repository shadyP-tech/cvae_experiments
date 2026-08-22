"""Fail-closed, launch-local scratch selection and cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import (
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_LOCK_MEMBER,
    load_frozen_source_streams,
)
from .constants import SCRATCH_ROOT
from .experiment_contracts import EXPECTED_GENERATION_LOCK_HASH


SOURCE_DIRECTORY = "source_generation"
PREDICTION_DIRECTORY = "prediction_cache"


@dataclass(frozen=True)
class ScratchLease:
    root: Path
    role: str


def probe_scratch(root: Path, runtime: Mapping[str, object]) -> dict[str, object]:
    lease = select_scratch(root, runtime)
    if lease.root.exists() or lease.root.is_symlink():
        raise ProtocolError("CBPUPR prior-run or foreign scratch is forbidden.")
    parent = lease.root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("CBPUPR scratch parent is absent or unsafe.")
    free_bytes = int(shutil.disk_usage(parent).free)
    if free_bytes < int(runtime["minimum_artifact_disk_free_bytes"]):
        raise ProtocolError("CBPUPR scratch reserve is too low.")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".cbpupr-write-probe-", dir=parent
        ) as probe:
            marker = Path(probe) / "probe"
            marker.write_bytes(b"cbpupr\n")
            with marker.open("r+b") as handle:
                os.fsync(handle.fileno())
    except OSError as exc:
        raise ProtocolError("CBPUPR scratch parent is not writable.") from exc
    return {
        "scratch_root_id": lease.root.name,
        "scratch_role": lease.role,
        "scratch_absent_at_launch": True,
        "scratch_parent_writable": True,
        "scratch_free_bytes_at_launch": free_bytes,
    }


def select_scratch(root: Path, runtime: Mapping[str, object]) -> ScratchLease:
    if tuple(runtime.get("scratch_preference", ())) != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("CBPUPR scratch preference drifted.")
    dedicated = Path(SCRATCH_ROOT)
    if not dedicated.is_absolute() or str(dedicated) != SCRATCH_ROOT:
        raise ProtocolError("CBPUPR dedicated scratch is not literal.")
    if dedicated.exists() or dedicated.is_symlink():
        raise ProtocolError("CBPUPR dedicated scratch contains prior state.")
    if dedicated.parent.is_dir() and not dedicated.parent.is_symlink():
        return ScratchLease(dedicated, "dedicated_local")
    fallback = Path(root).resolve().parent / f".{Path(root).name}.cbpupr-scratch"
    if fallback.exists() or fallback.is_symlink():
        raise ProtocolError("CBPUPR fallback scratch contains prior state.")
    return ScratchLease(fallback, "artifact_parent")


def create_scratch(root: Path, runtime: Mapping[str, object]) -> ScratchLease:
    lease = select_scratch(root, runtime)
    lease.root.mkdir(parents=True, exist_ok=False)
    return lease


def cleanup_scratch(
    lease: ScratchLease,
    *,
    config: object,
    artifact_root: Path,
) -> None:
    base = lease.root
    source_root = base / SOURCE_DIRECTORY
    prediction_root = base / PREDICTION_DIRECTORY
    if (
        not base.is_dir()
        or base.is_symlink()
        or not source_root.is_dir()
        or source_root.is_symlink()
        or not prediction_root.is_dir()
        or prediction_root.is_symlink()
        or any(path.is_symlink() for path in base.rglob("*"))
    ):
        raise ProtocolError("CBPUPR scratch tree is unsafe to clean.")
    local = load_frozen_source_streams(
        source_root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    canonical = load_frozen_source_streams(
        artifact_root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    if dict(local.lock_payload) != dict(canonical.lock_payload):
        raise ProtocolError("CBPUPR scratch/canonical source seals differ.")
    _remove_empty_owned_checkpoint_parent(source_root, role="source")
    _require_exact_source_inventory(source_root)
    _remove_empty_owned_checkpoint_parent(prediction_root, role="prediction")
    if any(prediction_root.iterdir()):
        raise ProtocolError("CBPUPR prediction scratch was not sealed and cleaned.")
    shutil.rmtree(base)


def _remove_empty_owned_checkpoint_parent(path: Path, *, role: str) -> None:
    """Normalize the one empty parent intentionally left by neutral runtimes."""

    checkpoint_root = path / "checkpoints"
    if not checkpoint_root.exists():
        if checkpoint_root.is_symlink():
            raise ProtocolError(
                f"CBPUPR {role} checkpoint parent is a dangling symlink."
            )
        return
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ProtocolError(f"CBPUPR {role} checkpoint parent is unsafe.")
    if any(checkpoint_root.iterdir()):
        raise ProtocolError(
            f"CBPUPR completed {role} checkpoint parent is not empty."
        )
    checkpoint_root.rmdir()


def _require_exact_source_inventory(path: Path) -> None:
    expected_files = {SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_LOCK_MEMBER}
    expected_directories = {"arrays", "manifests"}
    members = tuple(path.rglob("*"))
    observed_files = {
        member.relative_to(path).as_posix()
        for member in members
        if member.is_file()
    }
    observed_directories = {
        member.relative_to(path).as_posix()
        for member in members
        if member.is_dir()
    }
    if (
        any(member.is_symlink() for member in members)
        or observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ProtocolError("CBPUPR local source inventory drifted.")


__all__ = (
    "PREDICTION_DIRECTORY",
    "SOURCE_DIRECTORY",
    "ScratchLease",
    "cleanup_scratch",
    "create_scratch",
    "probe_scratch",
    "select_scratch",
)
