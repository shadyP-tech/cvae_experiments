"""Canonical identities for the in-memory disagreement-regret core."""

from __future__ import annotations

import hashlib
import json


def canonical_sha256(payload: object) -> str:
    """Return a full SHA-256 over canonical JSON-compatible content."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = ("canonical_sha256", "is_sha256")
