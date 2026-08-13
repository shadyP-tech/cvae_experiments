"""Atomic, non-repairing JSON and CSV persistence helpers for S4."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file


def json_value(value: object) -> object:
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


def object_payload(value: object) -> dict[str, object]:
    raw = (
        value
        if isinstance(value, Mapping)
        else value.to_payload()
        if hasattr(value, "to_payload")
        else getattr(value, "__dict__", None)
    )
    if not isinstance(raw, Mapping):
        raise TypeError("S4 artifact object must expose a mapping payload.")
    return {
        str(key): json_value(item)
        for key, item in raw.items()
        if not str(key).startswith("_")
    }


def persist_json(path: Path, payload: Mapping[str, object]) -> None:
    expected = json_value(payload)
    if not isinstance(expected, Mapping):
        raise TypeError("S4 JSON payload must be a mapping.")
    if path.is_symlink():
        raise ProtocolError(f"S4 JSON path is a symlink: {path}.")
    if path.is_file():
        if read_json(path) != dict(expected):
            raise ProtocolError(f"Existing S4 JSON differs; refusing repair: {path}.")
    else:
        atomic_json(path, expected)


def persist_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fields: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolError(f"S4 CSV path is a symlink: {path}.")
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fields),
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for raw in rows:
                row = {key: json_value(value) for key, value in raw.items()}
                if tuple(row) != fields:
                    raise ProtocolError(
                        f"S4 table schema drifted at {path}: {tuple(row)}."
                    )
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise ProtocolError(f"S4 table cannot be empty: {path}.")
        if path.is_file():
            if (
                path.stat().st_size != temporary.stat().st_size
                or sha256_file(path) != sha256_file(temporary)
            ):
                raise ProtocolError(
                    f"Existing S4 CSV differs; refusing repair: {path}."
                )
            temporary.unlink()
        else:
            os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_rows(path: Path) -> tuple[dict[str, str], ...]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"S4 table is absent or unsafe: {path}.")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ProtocolError(f"S4 CSV header drifted: {path}.")
            rows = tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read S4 table: {path}.") from exc
    if not rows:
        raise ProtocolError(f"S4 table is empty: {path}.")
    return rows


__all__ = (
    "json_value",
    "object_payload",
    "persist_json",
    "persist_rows",
    "read_rows",
)
