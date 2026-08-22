"""Canonical float32 probability primitives for the CBPUPR router.

The router treats the fixed-bank probability store as immutable input.  In
particular, abstention must return the portfolio probabilities byte for byte;
this module centralises that invariant so every runtime phase uses the same
conversion and hashing rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(payload: object) -> str:
    """Return a stable SHA-256 over a JSON-compatible payload."""

    normalised = _normalise_json(payload)
    encoded = json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, field_name: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ProtocolError(f"CBPUPR {field_name} must be a lowercase SHA-256 digest.")
    return digest


def canonical_float32_probabilities(
    values: object,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Validate probabilities and return an immutable contiguous float32 copy."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ProtocolError("CBPUPR probabilities must be one-dimensional.")
    if expected_length is not None and len(array) != int(expected_length):
        raise ProtocolError("CBPUPR probability vector length drifted.")
    if not len(array) or not np.isfinite(array).all():
        raise ProtocolError("CBPUPR probabilities must be finite and nonempty.")
    if bool(np.any((array < np.float32(0.0)) | (array > np.float32(1.0)))):
        raise ProtocolError("CBPUPR probabilities left the closed unit interval.")
    result = np.ascontiguousarray(array, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def probability_sha256(values: object) -> str:
    array = canonical_float32_probabilities(values)
    header = f"float32:{array.shape[0]}:".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def exact_p_fallback(portfolio_probabilities: object) -> np.ndarray:
    """Return a byte-identical float32 copy of canonical P."""

    portfolio = canonical_float32_probabilities(portfolio_probabilities)
    result = portfolio.copy(order="C")
    result.setflags(write=False)
    if result.dtype != np.dtype("float32") or result.tobytes() != portfolio.tobytes():
        raise ProtocolError("CBPUPR exact-P fallback changed canonical bytes.")
    return result


def require_byte_exact_p(candidate: object, portfolio_probabilities: object) -> None:
    actual = canonical_float32_probabilities(candidate)
    expected = canonical_float32_probabilities(portfolio_probabilities)
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ProtocolError("CBPUPR abstention is not byte-exact P.")


@dataclass(frozen=True)
class CanonicalProbabilityVector:
    """Spawn-safe, hash-bound representation of a float32 probability vector."""

    values: tuple[float, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        array = canonical_float32_probabilities(self.values)
        canonical_values = tuple(float(value) for value in array)
        object.__setattr__(self, "values", canonical_values)
        object.__setattr__(self, "sha256", probability_sha256(array))

    @classmethod
    def from_array(cls, values: object) -> "CanonicalProbabilityVector":
        array = canonical_float32_probabilities(values)
        return cls(tuple(float(value) for value in array))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CanonicalProbabilityVector":
        row = cls(tuple(float(value) for value in payload["values"]))  # type: ignore[index]
        if "sha256" in payload and str(payload["sha256"]) != row.sha256:
            raise ProtocolError("CBPUPR probability payload hash drifted.")
        return row

    def as_array(self) -> np.ndarray:
        return canonical_float32_probabilities(self.values)

    def to_payload(self) -> dict[str, object]:
        return {"values": list(self.values), "sha256": self.sha256, "dtype": "float32"}

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return CanonicalProbabilityVector, (self.values,)


def _normalise_json(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("CBPUPR canonical payload contains a nonfinite float.")
        return value
    if isinstance(value, np.generic):
        return _normalise_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalise_json(item) for item in value]
    if hasattr(value, "to_payload"):
        return _normalise_json(value.to_payload())  # type: ignore[union-attr]
    raise ProtocolError(f"CBPUPR cannot canonicalise payload type {type(value).__name__}.")


__all__ = (
    "CanonicalProbabilityVector",
    "canonical_float32_probabilities",
    "canonical_hash",
    "exact_p_fallback",
    "probability_sha256",
    "require_byte_exact_p",
    "require_sha256",
)
