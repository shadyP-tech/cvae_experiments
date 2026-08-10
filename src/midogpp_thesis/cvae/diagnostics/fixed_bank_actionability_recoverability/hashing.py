"""Canonical hashing and scalar validation used by the pure scientific core."""

from __future__ import annotations

import hashlib
import json
import math

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


def require_sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def nonempty_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ProtocolError(f"{name} must be a non-empty string.")
    return value


__all__ = ("canonical_hash", "finite", "nonempty_text", "require_sha256")
