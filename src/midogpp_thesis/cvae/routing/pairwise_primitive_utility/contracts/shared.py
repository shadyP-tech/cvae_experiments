"""Shared scalar validation and hashing for neutral routing contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Iterable, Sequence

from ....protocol import ProtocolError


P_ACTION_ID = "P_PROTECTED"
UTILITY_METRICS = ("bacc", "brier", "log")
UNCERTAINTY_METRICS = (*UTILITY_METRICS, "pairwise")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, *, role: str) -> str:
    result = str(value).strip()
    if not result:
        raise ProtocolError(f"Pairwise primitive utility requires non-empty {role}.")
    return result


def _finite_tuple(values: Sequence[float], *, role: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ProtocolError(f"Pairwise primitive utility {role} is empty or non-finite.")
    return result


def _sorted_unique(values: Iterable[object], *, role: str) -> tuple[str, ...]:
    result = tuple(sorted({_text(value, role=role) for value in values}))
    if not result:
        raise ProtocolError(f"Pairwise primitive utility requires at least one {role}.")
    return result


_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def feature_name_tokens(name: object) -> tuple[str, ...]:
    return tuple(token for token in _TOKEN_SPLIT.split(str(name).lower()) if token)


__all__ = (
    "P_ACTION_ID",
    "ProtocolError",
    "UNCERTAINTY_METRICS",
    "UTILITY_METRICS",
    "canonical_sha256",
    "feature_name_tokens",
)
