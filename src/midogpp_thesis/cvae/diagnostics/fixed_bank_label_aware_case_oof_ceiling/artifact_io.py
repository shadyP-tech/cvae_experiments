"""Deterministic non-repairing artifact IO for the label-aware bundle."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise ProtocolError(f"Existing label-aware JSON differs and will not be repaired: {path}.")
        return
    atomic_json(path, payload)


def persist_or_validate_csv(
    path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> None:
    expected = _csv_bytes(rows, columns)
    if path.is_file():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(f"Cannot read label-aware CSV: {path}.") from exc
        if observed != expected:
            raise ProtocolError(f"Existing label-aware CSV differs and will not be repaired: {path}.")
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


def _csv_bytes(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ProtocolError("Label-aware CSV row columns drifted.")
        writer.writerow({column: row[column] for column in columns})
    return buffer.getvalue().encode("utf-8")


__all__ = (
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "read_json",
    "relative_files",
    "sha256_file",
)
