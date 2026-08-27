"""Canonical hashing helpers for the isolated OE-PPUR v3 package."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ...protocol import ProtocolError


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value with a single deterministic representation."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 payload is not canonical JSON.") from exc


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ProtocolError(f"OE-PPUR v3 {role} is not a lowercase SHA-256.")
    return digest


__all__ = ("canonical_hash", "canonical_json_bytes", "require_sha256")
