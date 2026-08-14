"""Atomic, idempotent, nonrepairing JSON and canonical-CSV writers."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from .artifact_rows import reject_forbidden_persistence, rows_payload
from .hashing import json_native


def persist_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    allow_path_keys: bool = False,
) -> None:
    converted = json_native(payload)
    if not isinstance(converted, dict):
        raise ProtocolError("Dual-endpoint JSON product must be an object.")
    if not allow_path_keys:
        reject_forbidden_persistence(converted)
    if path.is_symlink():
        raise ProtocolError("Dual-endpoint JSON path is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != converted:
            raise ProtocolError(f"Dual-endpoint refuses repair of {path.name}.")
        return
    atomic_json(path, converted)


def persist_rows(
    path: Path,
    rows: Sequence[object],
    *,
    fields: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:
    payloads = rows_payload(rows)
    columns = tuple(fields or sorted(payloads[0]))
    if len(set(columns)) != len(columns) or any(set(row) != set(columns) for row in payloads):
        raise ProtocolError(f"Dual-endpoint table schema drifted: {path.name}.")
    expected = _csv_bytes(payloads, columns)
    if path.is_symlink():
        raise ProtocolError("Dual-endpoint table path is a symlink.")
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise ProtocolError(f"Dual-endpoint refuses repair: {path.name}.")
        return payloads
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_bytes(expected)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payloads


def read_rows(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"Dual-endpoint table absent: {path}.")
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if not fields or len(set(fields)) != len(fields):
                raise ProtocolError("Dual-endpoint CSV header drifted.")
            rows = []
            for raw in reader:
                row: dict[str, object] = {}
                for key in fields:
                    try:
                        row[key] = json.loads(raw[key])
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ProtocolError(
                            "Dual-endpoint CSV cell is not canonical JSON."
                        ) from exc
                reject_forbidden_persistence(row)
                rows.append(row)
    except OSError as exc:
        raise ProtocolError(f"Cannot read dual-endpoint table: {path}.") from exc
    if not rows or path.read_bytes() != _csv_bytes(tuple(rows), fields):
        raise ProtocolError("Dual-endpoint CSV bytes are not canonical.")
    return tuple(rows)


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(
                    json_native(row[key]),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                for key in fields
            }
        )
    return buffer.getvalue().encode("utf-8")


__all__ = ("persist_json", "persist_rows", "read_rows")
