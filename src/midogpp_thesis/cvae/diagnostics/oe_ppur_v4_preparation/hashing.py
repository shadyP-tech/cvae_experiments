"""Deterministic primitive hashing for OE-PPUR v4 preparation.

This module deliberately knows nothing about experiment execution, labels, or
legacy authorization state.  It accepts JSON-compatible primitives only so
that every preparation commitment can be reconstructed in a fresh process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from ...protocol import ProtocolError


def require_sha256(value: object, role: str, *, allow_zero: bool = False) -> str:
    """Return a normalized SHA-256 digest or fail closed."""

    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_zero and value == "0" * 64)
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} is not an exact SHA-256 digest.")
    return value


def require_nonempty_text(value: object, role: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ProtocolError(f"OE-PPUR v4 {role} is malformed.")
    return value


def primitive(value: object, *, role: str = "commitment") -> Any:
    """Normalize a recursively primitive value without implicit coercion."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        raise ProtocolError(
            f"OE-PPUR v4 {role} contains a non-canonical floating-point value."
        )
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise ProtocolError(f"OE-PPUR v4 {role} has a non-string mapping key.")
        return {
            key: primitive(item, role=role)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [primitive(item, role=role) for item in value]
    raise ProtocolError(f"OE-PPUR v4 {role} is not recursively primitive.")


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    normalized = primitive(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    normalized = primitive(payload)
    return (
        json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def bytes_sha256(raw: bytes) -> str:
    if type(raw) is not bytes:
        raise ProtocolError("OE-PPUR v4 hashed bytes are untyped.")
    return hashlib.sha256(raw).hexdigest()


__all__ = (
    "bytes_sha256",
    "canonical_json_bytes",
    "payload_sha256",
    "pretty_json_bytes",
    "primitive",
    "require_nonempty_text",
    "require_sha256",
)
