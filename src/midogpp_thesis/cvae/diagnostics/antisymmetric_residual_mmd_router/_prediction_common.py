"""Shared closed-world parsing, hashing, and atomic array helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Atomically persist one non-pickled NumPy array."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically persist a compressed, non-pickled NumPy archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_json_value(value: object) -> object:
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            "Antisymmetric prediction metadata JSON is malformed."
        ) from exc


def integer(value: object, role: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Antisymmetric {role} is not an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ProtocolError(f"Antisymmetric {role} is not integral.")
    return parsed


def truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def is_hash(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def require_mapping(
    payload: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Antisymmetric mapping is missing: {key}.")
    return value


__all__ = (
    "atomic_save_npy",
    "atomic_save_npz",
    "compact_json",
    "integer",
    "is_hash",
    "parse_json_value",
    "require_mapping",
    "sha256_array",
    "truthy",
)
