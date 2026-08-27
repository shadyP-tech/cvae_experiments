"""Shared fail-closed validation for OE-PPUR v2 run roots."""

from __future__ import annotations

import os
from pathlib import Path
import stat

from ...protocol import ProtocolError


WORKSPACE_ENVELOPE_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
)
WORKSPACE_ENVELOPE_DIRECTORIES = (
    "manifests",
    "provenance",
    "reports",
    "tables",
)


def validate_requested_run_root(
    value: str | Path,
    *,
    role: str,
    allow_workspace_envelope: bool,
) -> Path:
    """Accept an absent root, or only the exact workspace launch envelope."""

    path = validate_absolute_path(value, role=role)
    assert_no_symlink_chain(path, allow_missing_leaf=True)
    path = path.resolve(strict=False)
    if not path.exists():
        return path
    if allow_workspace_envelope and is_exact_workspace_launch_envelope(path):
        return path
    raise ProtocolError(f"OE-PPUR v2 {role} already exists.")


def validate_absolute_path(value: str | Path, *, role: str) -> Path:
    """Apply the common lexical safety fence without touching the path."""

    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor) or ".." in path.parts:
        raise ProtocolError(f"OE-PPUR v2 {role} is unsafe.")
    return path


def assert_no_symlink_chain(
    path: Path,
    *,
    allow_missing_leaf: bool = False,
) -> None:
    """Reject symlinks and non-directory parents along an absolute path."""

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
            raise ProtocolError("OE-PPUR v2 input path is absent.") from None
        except OSError as exc:
            raise ProtocolError("OE-PPUR v2 input path cannot be inspected.") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError("OE-PPUR v2 path contains a symlink.")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ProtocolError("OE-PPUR v2 path parent is not a directory.")


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either canonical path contains the other."""

    return left == right or left in right.parents or right in left.parents


def is_exact_workspace_launch_envelope(root: Path) -> bool:
    """Recognize only the files MIDOG++ workspace writes before its runner."""

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


__all__ = (
    "WORKSPACE_ENVELOPE_DIRECTORIES",
    "WORKSPACE_ENVELOPE_FILES",
    "assert_no_symlink_chain",
    "is_exact_workspace_launch_envelope",
    "paths_overlap",
    "validate_absolute_path",
    "validate_requested_run_root",
)
