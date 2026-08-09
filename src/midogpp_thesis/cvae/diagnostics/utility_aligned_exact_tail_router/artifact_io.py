"""Small package-local atomic and byte-exact artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


def json_ready(value: object, *, path: str = "$") -> object:
    """Detach frozen containers into a strict JSON-native value tree."""

    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ProtocolError(
                    f"Utility-aligned JSON mapping key is not a string at {path}."
                )
            output[key] = json_ready(item, path=f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            json_ready(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(
                f"Utility-aligned JSON contains a non-finite float at {path}."
            )
        return value
    raise ProtocolError(
        "Utility-aligned JSON contains an unsupported value "
        f"at {path}: {type(value).__name__}."
    )


def _json_object(payload: Mapping[str, object]) -> dict[str, object]:
    prepared = json_ready(payload)
    if not isinstance(prepared, dict):  # Defensive: the public contract is an object.
        raise ProtocolError("Utility-aligned JSON payload must be an object.")
    return prepared


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash utility-aligned artifact: {path}.") from exc
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility-aligned JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Utility-aligned JSON must be an object: {path}.")
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    prepared = _json_object(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                prepared,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def render_csv(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str]
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(columns):
            raise ProtocolError("Utility-aligned CSV row schema drifted.")
        writer.writerow({column: row[column] for column in columns})
    return output.getvalue()


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    rendered = render_csv(rows, columns).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    expected = _json_object(payload)
    if path.is_file():
        if read_json(path) != expected:
            raise ProtocolError(f"Utility-aligned resumed JSON drifted: {path}.")
        return
    atomic_json(path, expected)


def persist_or_validate_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    expected = render_csv(rows, columns).encode("utf-8")
    if path.is_file():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(f"Cannot read utility-aligned CSV: {path}.") from exc
        if observed != expected:
            raise ProtocolError(f"Utility-aligned resumed CSV drifted: {path}.")
        return
    atomic_csv(path, rows, columns)


def relative_files(root: Path) -> tuple[str, ...]:
    ignored = {".run.lock"}
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in ignored
            and ".tmp" not in path.name
            and not path.relative_to(root).as_posix().startswith("checkpoints/")
        )
    )


__all__ = (
    "atomic_csv",
    "atomic_json",
    "atomic_npy",
    "atomic_npz",
    "json_ready",
    "persist_or_validate_csv",
    "persist_or_validate_json",
    "read_json",
    "relative_files",
    "render_csv",
    "sha256_file",
)
