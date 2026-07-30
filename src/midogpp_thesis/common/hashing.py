"""Deterministic semantic hashes shared across repository capabilities."""

from __future__ import annotations

import hashlib
import json


def stable_hash(payload: object) -> str:
    """Return the canonical short SHA-256 identity for JSON-like payloads."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]

