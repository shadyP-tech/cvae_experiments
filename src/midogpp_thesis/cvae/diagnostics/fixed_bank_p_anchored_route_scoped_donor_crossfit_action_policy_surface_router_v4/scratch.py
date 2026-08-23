"""Absent-at-launch, no-recovery scratch admission for P-DCAPS v4."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError


SOURCE_DIRECTORY = "source_generation"
PREDICTION_DIRECTORY = "prediction_cache"
OUTER_DIRECTORY = "outer_H_chunks"
CANONICAL_SCRATCH_ROOT = (
    "/data/local/fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_v4"
)


@dataclass(frozen=True)
class ScratchLease:
    root: Path
    role: str

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute() or self.role not in {
            "dedicated_local",
            "artifact_parent",
        }:
            raise ProtocolError("P-DCAPS v4 scratch lease drifted.")
        object.__setattr__(self, "root", root)


def _canonical_scratch_root(runtime: Mapping[str, object]) -> Path:
    preference = tuple(runtime.get("scratch_preference", ()))
    if len(preference) != 2 or preference[1] != "artifact_parent":
        raise ProtocolError("P-DCAPS v4 scratch preference drifted.")
    dedicated = Path(str(preference[0]))
    if (
        not dedicated.is_absolute()
        or str(dedicated) != CANONICAL_SCRATCH_ROOT
    ):
        raise ProtocolError("P-DCAPS v4 dedicated scratch identity drifted.")
    return dedicated


def select_scratch(root: Path, runtime: Mapping[str, object]) -> ScratchLease:
    dedicated = _canonical_scratch_root(runtime)
    if dedicated.exists() or dedicated.is_symlink():
        raise ProtocolError("P-DCAPS v4 dedicated scratch contains prior state.")
    if dedicated.parent.is_dir() and not dedicated.parent.is_symlink():
        return ScratchLease(dedicated, "dedicated_local")
    artifact = Path(root).resolve()
    fallback = artifact.parent / f".{artifact.name}.pdcaps-v4-scratch"
    if fallback.exists() or fallback.is_symlink():
        raise ProtocolError("P-DCAPS v4 fallback scratch contains prior state.")
    return ScratchLease(fallback, "artifact_parent")


def assert_scratch_absent(lease: ScratchLease) -> None:
    if not isinstance(lease, ScratchLease):
        raise ProtocolError("P-DCAPS v4 scratch lease type drifted.")
    if lease.root.exists() or lease.root.is_symlink():
        raise ProtocolError("P-DCAPS v4 scratch recovery or reuse is forbidden.")
    parent = lease.root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("P-DCAPS v4 scratch parent is absent or unsafe.")


def probe_scratch(root: Path, runtime: Mapping[str, object]) -> dict[str, object]:
    """Perform the post-authorization free-space and fsync probe."""

    lease = select_scratch(root, runtime)
    assert_scratch_absent(lease)
    parent = lease.root.parent
    free_bytes = int(shutil.disk_usage(parent).free)
    if free_bytes < int(runtime["minimum_scratch_disk_free_bytes"]):
        raise ProtocolError("P-DCAPS v4 scratch reserve is too low.")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".pdcaps-v4-write-probe-", dir=parent
        ) as probe:
            marker = Path(probe) / "probe"
            marker.write_bytes(b"pdcaps-v4\n")
            with marker.open("r+b") as handle:
                os.fsync(handle.fileno())
    except OSError as exc:
        raise ProtocolError("P-DCAPS v4 scratch parent is not writable.") from exc
    return {
        "scratch_root_id": lease.root.name,
        "scratch_role": lease.role,
        "scratch_absent_at_launch": True,
        "scratch_parent_writable": True,
        "scratch_free_bytes_at_launch": free_bytes,
        "scratch_recovery_used": False,
        "v1_scratch_or_checkpoint_used": False,
    }


def create_scratch(root: Path, runtime: Mapping[str, object]) -> ScratchLease:
    """Create a new lease only; an existing path is never resumed or repaired."""

    lease = select_scratch(root, runtime)
    assert_scratch_absent(lease)
    lease.root.mkdir(parents=False, exist_ok=False)
    return lease


def cleanup_scratch(lease: ScratchLease, *, artifact_root: Path) -> None:
    """Remove only the exact lease created for this run.

    Cleanup is deliberately not a recovery operation: the lease must still be
    the canonical dedicated path or this artifact's exact sibling fallback,
    and it must be a real directory rather than an alias or prior-run path.
    """

    if not isinstance(lease, ScratchLease):
        raise ProtocolError("P-DCAPS v4 scratch cleanup lease drifted.")
    artifact = Path(artifact_root)
    if not artifact.is_absolute() or artifact.is_symlink():
        raise ProtocolError("P-DCAPS v4 scratch cleanup artifact root is unsafe.")
    artifact = artifact.resolve()
    dedicated = Path(CANONICAL_SCRATCH_ROOT)
    fallback = artifact.parent / f".{artifact.name}.pdcaps-v4-scratch"
    expected = {
        "dedicated_local": dedicated,
        "artifact_parent": fallback,
    }.get(lease.role)
    if expected is None or lease.root != expected:
        raise ProtocolError("P-DCAPS v4 scratch cleanup target drifted.")
    if lease.root.is_symlink() or not lease.root.is_dir():
        raise ProtocolError("P-DCAPS v4 scratch cleanup target is unsafe.")
    shutil.rmtree(lease.root)


__all__ = (
    "CANONICAL_SCRATCH_ROOT",
    "OUTER_DIRECTORY",
    "PREDICTION_DIRECTORY",
    "SOURCE_DIRECTORY",
    "ScratchLease",
    "assert_scratch_absent",
    "cleanup_scratch",
    "create_scratch",
    "probe_scratch",
    "select_scratch",
)
