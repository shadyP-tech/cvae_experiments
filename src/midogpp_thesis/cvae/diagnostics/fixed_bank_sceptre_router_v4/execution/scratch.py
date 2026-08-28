"""Absent-at-launch SCEPTRE v4 scratch with no recovery semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Mapping

from ....protocol import ProtocolError
from .authorization_lease import AuthorizationLease
from ..experiment_contracts import CANONICAL_SCRATCH_ROOT


SOURCE_DIRECTORY = "source_generation"
PREDICTION_DIRECTORY = "prediction_surface"
ROUTER_DIRECTORY = "router_state"


@dataclass(frozen=True, slots=True)
class ScratchLease:
    root: Path
    role: str

    def __post_init__(self) -> None:
        if (
            not Path(self.root).is_absolute()
            or self.role not in {"dedicated_local", "artifact_parent"}
        ):
            raise ProtocolError("SCEPTRE v4 scratch lease drifted.")


def select_scratch(root: Path, runtime: Mapping[str, object]) -> ScratchLease:
    preference = tuple(runtime.get("scratch_preference", ()))
    if preference != (CANONICAL_SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("SCEPTRE v4 scratch preference drifted.")
    dedicated = Path(CANONICAL_SCRATCH_ROOT)
    if dedicated.exists() or dedicated.is_symlink():
        raise ProtocolError("SCEPTRE v4 dedicated scratch contains prior state.")
    if dedicated.parent.is_dir() and not dedicated.parent.is_symlink():
        return ScratchLease(dedicated, "dedicated_local")
    artifact = Path(root)
    if not artifact.is_absolute():
        raise ProtocolError("SCEPTRE v4 artifact root must be absolute.")
    artifact = artifact.resolve()
    fallback = artifact.parent / f".{artifact.name}.sceptre-v4-scratch"
    if fallback.exists() or fallback.is_symlink():
        raise ProtocolError("SCEPTRE v4 fallback scratch contains prior state.")
    return ScratchLease(fallback, "artifact_parent")


def assert_scratch_absent(lease: ScratchLease) -> None:
    if not isinstance(lease, ScratchLease):
        raise ProtocolError("SCEPTRE v4 scratch lease type drifted.")
    if lease.root.exists() or lease.root.is_symlink():
        raise ProtocolError("SCEPTRE v4 scratch recovery or reuse is forbidden.")
    if lease.root.parent.is_symlink() or not lease.root.parent.is_dir():
        raise ProtocolError("SCEPTRE v4 scratch parent is absent or unsafe.")


def create_scratch(
    root: Path,
    runtime: Mapping[str, object],
    *,
    authorization_lease: AuthorizationLease,
    admitted: ScratchLease | None = None,
) -> ScratchLease:
    if (
        not isinstance(authorization_lease, AuthorizationLease)
        or authorization_lease.status != "CLAIMED_IN_PROGRESS"
    ):
        raise ProtocolError("SCEPTRE v4 scratch creation must follow lease claim.")
    lease = select_scratch(root, runtime)
    if admitted is not None and (
        not isinstance(admitted, ScratchLease) or lease != admitted
    ):
        raise ProtocolError("SCEPTRE v4 scratch differs from read-only admission.")
    assert_scratch_absent(lease)
    lease.root.mkdir(mode=0o700, parents=False, exist_ok=False)
    return lease


def cleanup_scratch(lease: ScratchLease, *, artifact_root: Path) -> None:
    artifact = Path(artifact_root)
    if not artifact.is_absolute() or artifact.is_symlink():
        raise ProtocolError("SCEPTRE v4 scratch cleanup artifact root is unsafe.")
    artifact = artifact.resolve()
    expected = {
        "dedicated_local": Path(CANONICAL_SCRATCH_ROOT),
        "artifact_parent": artifact.parent / f".{artifact.name}.sceptre-v4-scratch",
    }.get(lease.role)
    if expected is None or lease.root != expected:
        raise ProtocolError("SCEPTRE v4 scratch cleanup target drifted.")
    if lease.root.is_symlink() or not lease.root.is_dir():
        raise ProtocolError("SCEPTRE v4 scratch cleanup target is unsafe.")
    shutil.rmtree(lease.root)


__all__ = (
    "PREDICTION_DIRECTORY",
    "ROUTER_DIRECTORY",
    "SOURCE_DIRECTORY",
    "ScratchLease",
    "assert_scratch_absent",
    "cleanup_scratch",
    "create_scratch",
    "select_scratch",
)
