"""Shared fail-closed reads for SCEPTRE v5 validation modules."""

from __future__ import annotations

from pathlib import Path

from ....protocol import ProtocolError
from ....runtime.artifact_io import read_json


def read_validation_object(path: Path) -> dict[str, object]:
    """Read one regular JSON object without following a terminal symlink."""

    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"SCEPTRE v5 validation member is absent: {path.name}.")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ProtocolError("SCEPTRE v5 validation member is not an object.")
    return value


__all__ = ("read_validation_object",)
