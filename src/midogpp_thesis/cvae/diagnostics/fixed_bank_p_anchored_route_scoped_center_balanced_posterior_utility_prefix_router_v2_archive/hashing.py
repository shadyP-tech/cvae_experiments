"""Strict JSON hashes used only by the immutable archive utility."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def short_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def without(payload: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: value for name, value in payload.items() if name != key}


__all__ = ("canonical_hash", "short_hash", "without")
