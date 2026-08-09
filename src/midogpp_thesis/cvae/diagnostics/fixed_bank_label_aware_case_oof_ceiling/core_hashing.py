"""Canonical hashing helpers for the label-aware ceiling core."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

from ...protocol import ProtocolError


def canonical_hash(payload: Mapping[str, object]) -> str:
    """Return a full lowercase SHA-256 over canonical JSON data."""

    if not isinstance(payload, Mapping):
        raise ProtocolError("Canonical hash input must be a mapping.")
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{name} must be a canonical lowercase SHA-256.")
    return value


def finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be finite numeric data.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be finite numeric data.") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite numeric data.")
    return result


__all__ = ("canonical_hash", "finite", "require_sha256")
