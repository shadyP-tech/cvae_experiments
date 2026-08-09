"""Small deterministic IO helpers shared by neutral runtime phases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from ..protocol import ProtocolError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash runtime member: {path}.") from exc
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read runtime JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Runtime JSON must be an object: {path}.")
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{key: np.ascontiguousarray(value) for key, value in arrays.items()},
        )
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path, *, expected_sha256: str) -> None:
    if not source.is_file():
        raise ProtocolError(f"Runtime source member is absent: {source}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ProtocolError("Runtime local-staging copy changed bytes.")
    os.replace(temporary, destination)


__all__ = (
    "atomic_copy",
    "atomic_json",
    "atomic_npy",
    "atomic_npz",
    "read_json",
    "sha256_array",
    "sha256_file",
)
