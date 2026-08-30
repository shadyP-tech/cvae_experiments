"""Closed-world path topology for one OE-PPUR v4 execution."""

from __future__ import annotations

import os
from pathlib import Path
import stat

from ...protocol import ProtocolError


WORKSPACE_ENVELOPE_FILES = (
    "COMMITTED",
    "config.resolved.yaml",
    "preparation/execution_launch_authority.json",
    "preparation/final_authorization_envelope.json",
    "preparation/sealed_execution_replay.json",
    "provenance/input_artifacts.json",
)
WORKSPACE_ENVELOPE_DIRECTORIES = (
    "preparation",
    "provenance",
)


def validate_absolute_path(value: str | Path, *, role: str) -> Path:
    path = Path(value)
    candidate = Path(os.path.abspath(path))
    if (
        not path.is_absolute()
        or path != candidate
        or path == Path(path.anchor)
        or ".." in path.parts
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} is unsafe.")
    return path


def assert_no_symlink_chain(
    path: Path,
    *,
    allow_missing_leaf: bool = False,
) -> None:
    value = validate_absolute_path(path, role="path")
    current = Path(value.anchor)
    parts = value.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf:
                return
            raise ProtocolError("OE-PPUR v4 input path is absent.") from None
        except OSError as exc:
            raise ProtocolError("OE-PPUR v4 input path cannot be inspected.") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError("OE-PPUR v4 path contains a symlink.")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ProtocolError("OE-PPUR v4 path parent is not a directory.")


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def is_exact_workspace_launch_envelope(root: Path) -> bool:
    if not root.exists() or root.is_symlink() or not root.is_dir():
        return False
    members = tuple(root.rglob("*"))
    if any(member.is_symlink() for member in members):
        return False
    files = tuple(
        sorted(
            member.relative_to(root).as_posix()
            for member in members
            if member.is_file()
        )
    )
    directories = tuple(
        sorted(
            member.relative_to(root).as_posix()
            for member in members
            if member.is_dir()
        )
    )
    return (
        files == WORKSPACE_ENVELOPE_FILES
        and directories == WORKSPACE_ENVELOPE_DIRECTORIES
    )


def validate_launch_roots(
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> tuple[Path, Path]:
    """Validate the pristine, lease-before-output launch topology."""

    artifact = validate_absolute_path(artifact_root, role="artifact root")
    scratch = validate_absolute_path(scratch_root, role="scratch root")
    assert_no_symlink_chain(artifact, allow_missing_leaf=True)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    if (
        os.path.lexists(artifact)
        or os.path.lexists(scratch)
        or artifact.parent.is_symlink()
        or not artifact.parent.is_dir()
        or scratch.parent.is_symlink()
        or not scratch.parent.is_dir()
        or paths_overlap(artifact, scratch)
    ):
        raise ProtocolError("OE-PPUR v4 launch-root topology drifted.")
    return artifact, scratch


__all__ = (
    "WORKSPACE_ENVELOPE_DIRECTORIES",
    "WORKSPACE_ENVELOPE_FILES",
    "assert_no_symlink_chain",
    "is_exact_workspace_launch_envelope",
    "paths_overlap",
    "validate_absolute_path",
    "validate_launch_roots",
)
