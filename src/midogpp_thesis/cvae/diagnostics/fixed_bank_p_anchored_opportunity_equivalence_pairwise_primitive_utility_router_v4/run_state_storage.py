"""Package-private persistence boundary for OE-PPUR v4 run state."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

from ...protocol import ProtocolError
from . import durable_io
from .run_paths import assert_no_symlink_chain


def read_json_regular_nofollow(path: Path) -> dict[str, object]:
    return durable_io.read_json_regular_nofollow(
        Path(path),
        role="run state",
    )


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    _ensure_parent_directory(target)
    durable_io.replace_canonical_json_atomic(
        target,
        payload,
        role="atomic state",
    )


def write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    _ensure_parent_directory(target)
    try:
        durable_io.write_canonical_json_exclusive(
            target,
            payload,
            role="exclusive state",
        )
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 exclusive state write failed.") from exc


def _ensure_parent_directory(path: Path) -> None:
    candidate = Path(os.path.abspath(path))
    if candidate != path or path == Path(path.anchor):
        raise ProtocolError("OE-PPUR v4 state path is unsafe.")
    missing: list[Path] = []
    current = path.parent
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        current = current.parent
    assert_no_symlink_chain(current)
    if not current.is_dir():
        raise ProtocolError("OE-PPUR v4 state parent is unsafe.")
    for directory in reversed(missing):
        try:
            directory.mkdir(exist_ok=False)
        except OSError as exc:
            raise ProtocolError(
                "OE-PPUR v4 state parent creation failed."
            ) from exc
        durable_io.fsync_directory(directory.parent)
    assert_no_symlink_chain(path.parent)


__all__ = ()
