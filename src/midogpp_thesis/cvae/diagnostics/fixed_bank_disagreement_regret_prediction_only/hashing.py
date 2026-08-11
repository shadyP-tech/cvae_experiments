"""Full-length canonical hashing for prediction-only scientific products."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def without_hash(payload: Mapping[str, object], name: str) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != name}


def hash_rows(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_hash([dict(row) for row in rows])


__all__ = ("canonical_hash", "hash_rows", "without_hash")
