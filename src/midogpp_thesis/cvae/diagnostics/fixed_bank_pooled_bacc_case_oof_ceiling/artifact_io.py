"""Deterministic, non-repairing IO owned by the pooled-BACC v2 bundle."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    expected = dict(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise ProtocolError(f"Existing pooled-BACC JSON differs and will not be repaired: {path}.")
        return
    atomic_json(path, expected)


def persist_or_validate_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ProtocolError("Pooled-BACC CSV row columns drifted.")
        writer.writerow({column: row[column] for column in columns})
    expected = buffer.getvalue().encode("utf-8")
    if path.is_file():
        if path.read_bytes() != expected:
            raise ProtocolError(f"Existing pooled-BACC CSV differs and will not be repaired: {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_bytes(expected)
    os.replace(temporary, path)


def relative_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".run.lock"
        )
    )


__all__ = (
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "read_json",
    "relative_files",
    "sha256_file",
)
