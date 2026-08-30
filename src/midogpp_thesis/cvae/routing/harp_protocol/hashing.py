"""Closed-world canonical hashing for HARP protocol artifacts.

The repository-wide short semantic hash is useful for human-facing identities,
but HARP seals cross a label-access boundary.  Those seals therefore use a
full SHA-256 over a deliberately small JSON value language.  Unsupported
objects are rejected instead of being stringified.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ...protocol import ProtocolError


def canonical_bytes(value: object) -> bytes:
    """Encode a JSON-like value with deterministic keys and number handling."""

    normalized = _normalize(value, path="$", active=set())
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ProtocolError(f"{name} must be a lowercase SHA-256.")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolError(f"{name} must be a lowercase SHA-256.")
    return value


def _normalize(value: object, *, path: str, active: set[int]) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(f"HARP canonical value at {path} is not finite.")
        return 0.0 if value == 0.0 else value

    identity = id(value)
    if identity in active:
        raise ProtocolError(f"HARP canonical value at {path} is recursive.")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str or key in normalized:
                    raise ProtocolError(
                        f"HARP canonical mapping at {path} requires unique string keys."
                    )
                normalized[key] = _normalize(
                    item, path=f"{path}.{key}", active=active
                )
            return normalized
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            return [
                _normalize(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
    finally:
        active.remove(identity)
    raise ProtocolError(
        f"HARP canonical value at {path} has unsupported type "
        f"{type(value).__name__}."
    )


__all__ = ("canonical_bytes", "canonical_hash", "require_sha256")
