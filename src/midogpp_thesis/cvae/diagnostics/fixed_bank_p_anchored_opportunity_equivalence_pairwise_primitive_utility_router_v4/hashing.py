"""Small canonical hashing boundary for OE-PPUR v4 contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ...protocol import ProtocolError


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 contract is not canonical JSON.") from exc


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} is not SHA-256.")
    return value


__all__ = ("canonical_bytes", "canonical_hash", "require_sha256")
