"""Neutral byte-first artifact helpers for the endpoint-router bundle."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import yaml

from ...protocol import ProtocolError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read endpoint-router JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Endpoint-router JSON is not an object: {path}.")
    return payload


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    atomic_bytes(
        path,
        (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def atomic_yaml(path: str | Path, payload: Mapping[str, object]) -> None:
    atomic_bytes(
        path,
        yaml.safe_dump(dict(payload), sort_keys=False).encode("utf-8"),
    )


def csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ProtocolError("Endpoint-router CSV rows must be nonempty.")
    columns = tuple(str(key) for key in rows[0])
    if not columns or any(tuple(str(key) for key in row) != columns for row in rows):
        raise ProtocolError("Endpoint-router CSV column order drifted.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def atomic_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    atomic_bytes(path, csv_bytes(rows))


def persist_or_validate_json(
    path: str | Path, payload: Mapping[str, object]
) -> None:
    destination = Path(path)
    if destination.exists():
        if read_json(destination) != dict(payload):
            raise ProtocolError(f"Endpoint-router JSON checkpoint drifted: {destination}.")
        return
    atomic_json(destination, payload)


def persist_or_validate_csv(
    path: str | Path, rows: Sequence[Mapping[str, object]]
) -> None:
    destination = Path(path)
    payload = csv_bytes(rows)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ProtocolError(f"Endpoint-router CSV checkpoint drifted: {destination}.")
        return
    atomic_bytes(destination, payload)


def relative_files(root: str | Path) -> tuple[str, ...]:
    base = Path(root)
    if not base.exists():
        return ()
    unsafe = tuple(
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_symlink()
    )
    if unsafe:
        raise ProtocolError(
            f"Endpoint-router artifact tree contains symlinks: {sorted(unsafe)}."
        )
    return tuple(
        sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
    )


__all__ = (
    "atomic_bytes",
    "atomic_csv",
    "atomic_json",
    "atomic_yaml",
    "csv_bytes",
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "read_json",
    "relative_files",
    "sha256_file",
)
