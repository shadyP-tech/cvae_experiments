"""Targeted durability barriers; avoids repeated full-tree fsync/rehash."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from ...protocol import ProtocolError


def durable_barrier(paths: Iterable[Path]) -> None:
    files = tuple(dict.fromkeys(Path(path).resolve() for path in paths))
    if not files:
        raise ProtocolError("HARP v15 durability barrier cannot be empty.")
    directories: set[Path] = set()
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ProtocolError("HARP v15 durability barrier member is absent or unsafe.")
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ProtocolError("HARP v15 durability barrier failed.") from exc
        directories.add(path.parent)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ("durable_barrier",)
