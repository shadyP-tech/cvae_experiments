"""Canonical, path-independent hashes for SCALE-BP v2 artifacts.

This module intentionally depends only on the v2 governance error and the
standard library.  Artifact hashes therefore remain reconstructible in fresh
CPU-only processes without importing any scientific fitting module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..protocol import GovernanceError


def json_native(value: object) -> Any:
    """Return a strict JSON-native value and reject nonfinite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GovernanceError("SCALE-BP v2 artifact contains a nonfinite float.")
        return value
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_native(dataclasses.asdict(value))
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return json_native(to_payload())
    if isinstance(value, Mapping):
        return {
            str(key): json_native(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [json_native(item) for item in value]
    raise GovernanceError(
        "SCALE-BP v2 artifact contains unsupported type "
        f"{type(value).__name__}."
    )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise GovernanceError(
            f"SCALE-BP v2 {str(role).strip() or 'hash'} is not a lowercase SHA-256."
        )
    return digest


def sha256_file(path: str | Path, *, offset: int = 0, length: int | None = None) -> str:
    """Hash a whole regular file or one exact byte slice."""

    member = Path(path)
    if member.is_symlink() or not member.is_file():
        raise GovernanceError("SCALE-BP v2 artifact member is absent or unsafe.")
    if type(offset) is not int or offset < 0 or (
        length is not None and (type(length) is not int or length < 0)
    ):
        raise GovernanceError("SCALE-BP v2 file hash extent is malformed.")
    digest = hashlib.sha256()
    remaining = length
    try:
        with member.open("rb") as handle:
            handle.seek(offset)
            while remaining is None or remaining:
                amount = 8 * 1024 * 1024 if remaining is None else min(
                    8 * 1024 * 1024, remaining
                )
                chunk = handle.read(amount)
                if not chunk:
                    break
                digest.update(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
    except OSError as exc:
        raise GovernanceError("SCALE-BP v2 artifact member is unreadable.") from exc
    if remaining not in (None, 0):
        raise GovernanceError("SCALE-BP v2 file hash slice is truncated.")
    return digest.hexdigest()


def sha256_array(values: object) -> str:
    """Hash dtype, shape, and C-order bytes of one numeric array."""

    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise GovernanceError("SCALE-BP v2 array hash requires finite numeric data.")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "canonical_hash",
    "canonical_json",
    "canonical_json_bytes",
    "json_native",
    "require_sha256",
    "sha256_array",
    "sha256_file",
)
