"""Small, schema-stable persistence helpers for the diagnostic bundle.

Scientific modules return immutable dataclasses.  This module is the only
place that translates those objects into JSON/CSV values, keeping the runner
and the replay validator independent from dataclass implementation details.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .artifact_io import persist_or_validate_json


def json_value(value: object) -> object:
    """Return a deterministic JSON-compatible representation."""

    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "to_payload"):
        return json_value(value.to_payload())
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def payload(value: object) -> dict[str, object]:
    """Serialize a typed product without exposing private dataclass state."""

    if isinstance(value, Mapping):
        raw: object = value
    elif hasattr(value, "to_payload"):
        raw = value.to_payload()
    else:
        raw = getattr(value, "__dict__", None)
    if not isinstance(raw, Mapping):
        raise TypeError("Artifact product must provide a mapping payload.")
    return {
        str(key): json_value(item)
        for key, item in raw.items()
        if not str(key).startswith("_")
    }


def persist_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Stream a fixed-schema CSV to an atomic temp and validate on resume.

    The seed-probability table has more than 1.6 million rows.  Streaming keeps
    publication memory bounded instead of constructing multiple full CSV and
    dictionary copies in RAM.
    """

    iterator = iter(rows)
    try:
        first = _canonical_row(next(iterator))
    except StopIteration as exc:
        raise ProtocolError(f"Diagnostic table cannot be empty: {path}.") from exc
    columns = tuple(first)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(columns),
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            _write_row(writer, first, columns, path)
            for raw in iterator:
                _write_row(writer, _canonical_row(raw), columns, path)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_file():
            if (
                path.stat().st_size != temporary.stat().st_size
                or sha256_file(path) != sha256_file(temporary)
            ):
                raise ProtocolError(
                    f"Existing diagnostic CSV differs and will not be repaired: {path}."
                )
            temporary.unlink()
        else:
            os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_row(row: Mapping[str, object]) -> dict[str, object]:
    return {str(key): json_value(value) for key, value in row.items()}


def _write_row(
    writer: csv.DictWriter[str],
    row: Mapping[str, object],
    columns: tuple[str, ...],
    path: Path,
) -> None:
    if tuple(row) != columns:
        raise ProtocolError(f"Diagnostic table schema drifted: {path}.")
    writer.writerow(
        {
            key: (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
    )


def persist_payload(path: Path, value: Mapping[str, object]) -> None:
    persist_or_validate_json(path, json_value(value))


def read_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Read a CSV while rejecting duplicate columns and malformed rows."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ProtocolError(f"Diagnostic CSV header drifted: {path}.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read diagnostic CSV: {path}.") from exc
    if not rows:
        raise ProtocolError(f"Diagnostic CSV is empty: {path}.")
    return rows


__all__ = ("json_value", "payload", "persist_payload", "persist_rows", "read_rows")
