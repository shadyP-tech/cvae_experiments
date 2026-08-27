"""Small atomic persistence boundary for OE-PPUR v2 receipts."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile

from ...protocol import ProtocolError
from .hashing import canonical_json_bytes


def atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    if target.is_symlink():
        raise ProtocolError("OE-PPUR v2 persistence target is a symlink.")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json_object(path: str | Path) -> dict[str, object]:
    target = Path(path)
    try:
        if target.is_symlink() or not target.is_file():
            raise OSError("unsafe member")
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v2 persisted JSON is unreadable.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("OE-PPUR v2 persisted JSON is not an object.")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ("atomic_json", "read_json_object")
