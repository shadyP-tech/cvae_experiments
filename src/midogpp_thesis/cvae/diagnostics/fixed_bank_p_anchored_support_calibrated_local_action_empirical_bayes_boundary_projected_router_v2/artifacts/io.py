"""Fail-closed atomic persistence for the single-use SCALE-BP v2 bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

from ..protocol import GovernanceError
from .hashing import canonical_json_bytes, sha256_file


def safe_member(value: str | Path) -> str:
    """Normalize a bundle-relative member without resolving through symlinks."""

    text = Path(value).as_posix() if isinstance(value, Path) else str(value)
    member = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or member.is_absolute()
        or any(part in {"", ".", ".."} for part in member.parts)
        or "\\" in text
    ):
        raise GovernanceError("SCALE-BP v2 bundle member is not a safe relative path.")
    return member.as_posix()


def member_path(root: str | Path, member: str | Path) -> Path:
    base = Path(root)
    relative = safe_member(member)
    if base.is_symlink():
        raise GovernanceError("SCALE-BP v2 bundle root cannot be a symlink.")
    path = base.joinpath(*PurePosixPath(relative).parts)
    current = base
    for part in PurePosixPath(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise GovernanceError("SCALE-BP v2 bundle parent cannot be a symlink.")
    return path


def read_json_object(path: str | Path) -> dict[str, object]:
    member = Path(path)
    if member.is_symlink() or not member.is_file():
        raise GovernanceError("SCALE-BP v2 JSON member is absent or unsafe.")
    try:
        value = json.loads(member.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("SCALE-BP v2 JSON member is unreadable.") from exc
    if not isinstance(value, dict):
        raise GovernanceError("SCALE-BP v2 JSON member must contain an object.")
    return value


def atomic_json(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    replace: bool = False,
) -> None:
    """Atomically create an immutable JSON member or replace explicit state.

    All scientific members use ``replace=False``.  Only run state is expected
    to opt into replacement, and its immutable transition records remain in
    the content-addressed bundle.
    """

    member = Path(path)
    if member.is_symlink():
        raise GovernanceError("SCALE-BP v2 refuses to write through a symlink.")
    member.parent.mkdir(parents=True, exist_ok=True)
    if member.parent.is_symlink():
        raise GovernanceError("SCALE-BP v2 refuses a symlink artifact directory.")
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    if member.exists() and not replace:
        if not member.is_file() or member.read_bytes() != encoded:
            raise GovernanceError("SCALE-BP v2 refuses artifact repair or overwrite.")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{member.name}.", suffix=".tmp", dir=member.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and member.exists():
            if member.read_bytes() != encoded:
                raise GovernanceError(
                    "SCALE-BP v2 concurrent artifact writer disagreed."
                )
            return
        os.replace(temporary, member)
        _fsync_directory(member.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def indexed_file_row(root: str | Path, member: str | Path) -> dict[str, object]:
    relative = safe_member(member)
    path = member_path(root, relative)
    if path.is_symlink() or not path.is_file():
        raise GovernanceError("SCALE-BP v2 indexed member is absent or unsafe.")
    return {
        "member": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "atomic_json",
    "indexed_file_row",
    "member_path",
    "read_json_object",
    "safe_member",
)
