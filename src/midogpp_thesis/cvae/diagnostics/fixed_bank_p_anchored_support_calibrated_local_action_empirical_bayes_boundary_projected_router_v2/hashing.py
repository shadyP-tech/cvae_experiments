"""Canonical full-width identities for the isolated SCALE-BP v2 package."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .identity import GovernanceError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise GovernanceError(
            "SCALE-BP v2 canonical payload contains a nonfinite float."
        )
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def canonical_json(payload: object) -> bytes:
    """Encode a JSON-like payload deterministically and reject NaN/Infinity."""

    try:
        return json.dumps(
            _finite(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GovernanceError("SCALE-BP v2 payload is not canonical JSON.") from exc


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise GovernanceError(
            f"SCALE-BP v2 {role} is not a canonical SHA-256 digest."
        )
    return digest


__all__ = ("canonical_hash", "canonical_json", "require_sha256")
