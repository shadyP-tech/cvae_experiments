"""Canonical hashing helpers owned by the flip-router diagnostic."""

from __future__ import annotations

import hashlib
import json
import math

from ...protocol import ProtocolError


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or text != text.lower() or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(f"{role} is not a SHA-256 digest.")
    return text


def require_stable_hash(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 16 or text != text.lower() or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(f"{role} is not a repository stable hash.")
    return text


def finite(value: object, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{role} is not numeric.") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{role} is not finite.")
    return result


def nonempty_text(value: object, role: str) -> str:
    text = str(value)
    if not text:
        raise ProtocolError(f"{role} is empty.")
    return text


__all__ = (
    "canonical_hash", "finite", "nonempty_text", "require_sha256", "require_stable_hash",
)
