"""Canonical serialization helpers owned by the fixed-bank audit."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


def canonical_hash(payload: Mapping[str, object]) -> str:
    """Hash a canonical JSON-compatible mapping."""

    if not isinstance(payload, Mapping):
        raise ProtocolError("Canonical hash input must be a mapping.")
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_array_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ProtocolError("Canonical arrays must be finite.")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def require_exact_keys(
    payload: Mapping[str, object], expected: Sequence[str], role: str
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != set(expected):
        raise ProtocolError(f"{role} payload does not match the exact schema.")


def require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{name} must be a canonical lowercase SHA-256.")
    return value


def finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ProtocolError(f"{name} must be finite numeric data.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be finite numeric data.") from exc
    if not np.isfinite(result):
        raise ProtocolError(f"{name} must be finite numeric data.")
    return result


def probability_delta(value: object, name: str) -> float:
    result = finite(value, name)
    if result < -1.0 or result > 1.0:
        raise ProtocolError(f"{name} must be in [-1, 1].")
    return result


def synthetic_hash(token: str) -> str:
    """Deterministic helper used only for internal derived provenance."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


__all__ = (
    "canonical_array_hash",
    "canonical_hash",
    "finite",
    "probability_delta",
    "require_exact_keys",
    "require_sha256",
    "synthetic_hash",
)
