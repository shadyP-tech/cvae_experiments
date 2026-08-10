"""Package-owned non-repairing artifact helpers built on neutral runtime IO."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    expected = dict(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise ProtocolError(
                f"Existing actionability JSON differs and will not be repaired: {path}."
            )
        return
    atomic_json(path, expected)


def relative_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".run.lock"
        )
    )


__all__ = (
    "persist_or_validate_json",
    "read_json",
    "relative_files",
    "sha256_file",
)
