"""Shared scalar, hash, and exact-nine validation helpers."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256
from .constants import EXACT_SEED_PAIR_COUNT


_HASH = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def hash_token(value: object, name: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a lowercase 16- or 64-hex hash.")
    return value


def sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a canonical lowercase SHA-256 hash.")
    return value


def hash_sequence(values: object, name: str, expected: int) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != expected:
        raise ProtocolError(f"{name} length drifted.")
    return tuple(hash_token(value, name) for value in values)


def hash_matrix(
    values: object, name: str, expected_cases: int
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(values, (tuple, list)) or len(values) != expected_cases:
        raise ProtocolError(f"{name} case coverage drifted.")
    result = tuple(
        tuple(sha256(value, name) for value in row)
        for row in values
        if isinstance(row, (tuple, list))
    )
    if len(result) != expected_cases or any(
        len(row) != EXACT_SEED_PAIR_COUNT for row in result
    ):
        raise ProtocolError(f"{name} requires exact-nine hashes per support case.")
    return result


def vector_hashes(
    values: Sequence[str], matrix: np.ndarray, name: str
) -> tuple[str, ...]:
    if not values:
        return tuple(array_sha256(matrix[index]) for index in range(len(matrix)))
    if len(values) != EXACT_SEED_PAIR_COUNT:
        raise ProtocolError(f"{name} requires exact-nine vector hashes.")
    return tuple(sha256(value, name) for value in values)


def probability_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).copy()
    if (
        matrix.ndim != 2
        or matrix.shape[0] != EXACT_SEED_PAIR_COUNT
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0.0)
        or np.any(matrix > 1.0)
    ):
        raise ProtocolError(f"{name} must have finite [0,1] shape (9, n_rows).")
    matrix.setflags(write=False)
    return matrix


def finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ProtocolError(f"{name} must be finite numeric data.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be finite numeric data.") from exc
    if not np.isfinite(result):
        raise ProtocolError(f"{name} must be finite numeric data.")
    return result


def bounded(value: object, name: str, lower: float, upper: float) -> float:
    result = finite(value, name)
    if result < lower - 1.0e-12 or result > upper + 1.0e-12:
        raise ProtocolError(f"{name} is outside [{lower}, {upper}].")
    return float(np.clip(result, lower, upper))


__all__ = (
    "bounded",
    "finite",
    "hash_matrix",
    "hash_sequence",
    "hash_token",
    "probability_matrix",
    "sha256",
    "vector_hashes",
)
