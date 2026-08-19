"""Canonical full-length hashes for scientific plans and seals."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from typing import Any

from ...protocol import ProtocolError


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json(payload: object) -> str:
    return json.dumps(
        json_native(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def json_native(value: Any) -> Any:
    """Convert immutable science contracts into strict JSON-native values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    converter = getattr(value, "to_payload", None)
    if callable(converter):
        return json_native(converter())
    if is_dataclass(value):
        return json_native(asdict(value))
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    raise ProtocolError(
        f"PDCB cannot convert {type(value).__name__} to JSON."
    )


def require_sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or text != text.lower() or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(f"{role} is not a lowercase SHA-256 digest.")
    return text


def require_digest(value: object, role: str) -> str:
    """Accept repository-native short hashes or full SHA-256 content hashes."""

    text = str(value)
    if len(text) not in {16, 64} or text != text.lower() or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ProtocolError(f"{role} is not a repository digest.")
    return text


__all__ = (
    "canonical_hash",
    "canonical_json",
    "json_native",
    "require_digest",
    "require_sha256",
)
