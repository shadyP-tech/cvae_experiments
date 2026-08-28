"""Owned scratch topology and durability helpers for source preparation.

Only the dedicated source producer may create or remove these paths.  A
failed production attempt leaves its validated checkpoints in place; a later
attempt may consume only the exact members declared by the producer.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import stat
from typing import Iterable
import json

from ....protocol import ProtocolError


WORK_DIRECTORY_MEMBERS = (
    "held_prediction_checkpoints",
    "source_evaluation",
    "source_streams",
)
WORK_FILE_MEMBERS = ("resume_identity.json",)
_ATOMIC_TEMP = re.compile(r"^.+\.[0-9]+\.tmp$")


def prepare_resumable_work_root(
    scratch_parent: Path,
    work_root: str | Path,
) -> Path:
    """Create or validate one deterministic producer-owned work root."""

    parent = Path(scratch_parent)
    root = Path(os.path.abspath(Path(work_root)))
    if (
        not root.is_absolute()
        or root == Path(root.anchor)
        or root.parent != parent
        or root.name in {"", ".", ".."}
    ):
        raise ProtocolError("OE-PPUR v3 resumable source work root is unsafe.")
    _reject_symlink_chain(root, allow_missing_leaf=True)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ProtocolError("OE-PPUR v3 resumable source work root drifted.")
        observed = tuple(sorted(item.name for item in root.iterdir()))
        if any(
            item not in {*WORK_DIRECTORY_MEMBERS, *WORK_FILE_MEMBERS}
            for item in observed
        ):
            raise ProtocolError(
                "OE-PPUR v3 resumable source work inventory drifted."
            )
        for item in root.iterdir():
            expected_kind = (
                item.is_dir()
                if item.name in WORK_DIRECTORY_MEMBERS
                else item.is_file()
            )
            if item.is_symlink() or not expected_kind:
                raise ProtocolError(
                    "OE-PPUR v3 resumable source work member is unsafe."
                )
        return root
    root.mkdir(mode=0o750, exist_ok=False)
    fsync_directory(parent)
    return root


def bind_resume_identity(root: Path, payload: dict[str, object]) -> None:
    """Persist once and exactly match the complete retry identity."""

    path = Path(root) / WORK_FILE_MEMBERS[0]
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("OE-PPUR v3 resume identity is unsafe.")
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise ProtocolError("OE-PPUR v3 resume identity is unreadable.") from exc
        if observed != encoded:
            raise ProtocolError("OE-PPUR v3 resume identity drifted.")
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o640)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    fsync_file(path)
    fsync_directory(Path(root))


def prepare_exact_checkpoint_directory(
    root: Path,
    *,
    expected_members: Iterable[str],
) -> None:
    """Admit only exact final checkpoint names and owned atomic remnants."""

    path = Path(root)
    expected = frozenset(str(value) for value in expected_members)
    if len(expected) == 0 or any(
        Path(value).name != value or value in {".", ".."} for value in expected
    ):
        raise ProtocolError("OE-PPUR v3 checkpoint inventory is unsafe.")
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ProtocolError("OE-PPUR v3 checkpoint root is unsafe.")
    else:
        path.mkdir(parents=True, mode=0o750, exist_ok=False)
        fsync_directory(path.parent)
    _reject_symlink_chain(path)
    for member in tuple(path.iterdir()):
        if member.is_symlink() or not member.is_file():
            raise ProtocolError("OE-PPUR v3 checkpoint member is unsafe.")
        if member.name in expected:
            continue
        if _is_owned_atomic_temp(member.name, expected=expected):
            member.unlink()
            continue
        raise ProtocolError("OE-PPUR v3 checkpoint member inventory drifted.")
    fsync_directory(path)


def validate_exact_directory_tree(
    root: Path,
    *,
    expected_directories: Iterable[str],
    expected_files: Iterable[str],
) -> None:
    """Reject symlinks and any undeclared member in a resumable mini-tree."""

    path = Path(root)
    directories = frozenset(str(value) for value in expected_directories)
    files = frozenset(str(value) for value in expected_files)
    if path.is_symlink() or not path.is_dir():
        raise ProtocolError("OE-PPUR v3 resumable mini-tree root is unsafe.")
    _reject_symlink_chain(path)
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    for member in path.rglob("*"):
        relative = member.relative_to(path).as_posix()
        if member.is_symlink():
            raise ProtocolError("OE-PPUR v3 resumable mini-tree has a symlink.")
        if member.is_dir():
            observed_directories.add(relative)
        elif member.is_file():
            observed_files.add(relative)
        else:
            raise ProtocolError("OE-PPUR v3 resumable mini-tree is unsafe.")
    if not observed_directories.issubset(directories):
        raise ProtocolError("OE-PPUR v3 resumable directory inventory drifted.")
    for relative in tuple(observed_files):
        if relative in files:
            continue
        name = Path(relative).name
        parent = Path(relative).parent.as_posix()
        eligible = frozenset(
            Path(value).name
            for value in files
            if Path(value).parent.as_posix() == parent
        )
        if _is_owned_atomic_temp(name, expected=eligible):
            (path / relative).unlink()
            observed_files.remove(relative)
            continue
        raise ProtocolError("OE-PPUR v3 resumable file inventory drifted.")
    fsync_directory(path)


def remove_owned_work_root(root: Path, *, scratch_parent: Path) -> None:
    """Remove only the exact narrow producer-owned root after success."""

    path = Path(root)
    parent = Path(scratch_parent)
    if path.parent != parent or path == Path(path.anchor):
        raise ProtocolError("OE-PPUR v3 owned source cleanup escaped scratch.")
    _reject_symlink_chain(path)
    if path.is_symlink() or not path.is_dir():
        raise ProtocolError("OE-PPUR v3 owned source cleanup root drifted.")
    shutil.rmtree(path)
    fsync_directory(parent)


def fsync_file(path: Path) -> None:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise ProtocolError("OE-PPUR v3 durable source member is unsafe.")
    descriptor = os.open(value, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    value = Path(path)
    if value.is_symlink() or not value.is_dir():
        raise ProtocolError("OE-PPUR v3 durable source directory is unsafe.")
    descriptor = os.open(
        value,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_owned_atomic_temp(name: str, *, expected: frozenset[str]) -> bool:
    if _ATOMIC_TEMP.fullmatch(name) is None:
        return False
    return any(name.startswith(value + ".") for value in expected)


def _reject_symlink_chain(path: Path, *, allow_missing_leaf: bool = False) -> None:
    value = Path(path)
    current = value
    while True:
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf:
                current = current.parent
                continue
            raise ProtocolError("OE-PPUR v3 resumable source path is absent.") from None
        except OSError as exc:
            raise ProtocolError(
                "OE-PPUR v3 resumable source path cannot be inspected."
            ) from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError("OE-PPUR v3 resumable source path has a symlink.")
        if current != value and not stat.S_ISDIR(mode):
            raise ProtocolError(
                "OE-PPUR v3 resumable source parent is not a directory."
            )
        if current == current.parent:
            return
        current = current.parent


__all__ = (
    "WORK_DIRECTORY_MEMBERS",
    "WORK_FILE_MEMBERS",
    "bind_resume_identity",
    "fsync_directory",
    "fsync_file",
    "prepare_exact_checkpoint_directory",
    "prepare_resumable_work_root",
    "remove_owned_work_root",
    "validate_exact_directory_tree",
)
