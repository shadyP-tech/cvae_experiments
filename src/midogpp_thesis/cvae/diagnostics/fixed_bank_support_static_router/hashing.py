"""Canonical validation and hashing primitives for the diagnostic core."""

from __future__ import annotations

import hashlib
import json
import math

from ...protocol import ProtocolError


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha256(value: object, role: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ProtocolError(f"{role} is not a lowercase SHA-256 digest.")
    return text


def require_stable_hash(value: object, role: str) -> str:
    text = str(value)
    if (
        len(text) != 16
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ProtocolError(f"{role} is not a repository stable hash.")
    return text


def nonempty_text(value: object, role: str) -> str:
    text = str(value)
    if not text:
        raise ProtocolError(f"{role} must be non-empty.")
    return text


def finite(value: object, role: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{role} must be numeric.") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"{role} must be finite.")
    return number


def nonnegative_int(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{role} must be a non-negative integer.")
    return value


__all__ = (
    "canonical_hash",
    "finite",
    "nonempty_text",
    "nonnegative_int",
    "require_sha256",
    "require_stable_hash",
)
