"""Canonical, full-length hashes for scientific DTOs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from ...protocol import ProtocolError


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be finite numeric data.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be finite numeric data.") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite numeric data.")
    return result


def nonempty_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ProtocolError(f"{name} must be a non-empty string.")
    return value


def canonical_pairs(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((nonempty_text(key, "key"), finite_float(value, key)) for key, value in values.items()))


__all__ = ("canonical_hash", "canonical_pairs", "finite_float", "nonempty_text")
