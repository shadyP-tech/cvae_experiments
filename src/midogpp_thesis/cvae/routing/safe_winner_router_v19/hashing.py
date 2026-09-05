"""Canonical, process-stable identities for the independent HARP v19 router."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError


def canonical_value(value: object) -> object:
    """Return the JSON-safe value used by every v19 scientific identity."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError("HARP v19 hashes require finite floats.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, np.generic):
        return canonical_value(value.item())
    if isinstance(value, np.ndarray):
        return canonical_value(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            member.name: canonical_value(getattr(value, member.name))
            for member in fields(value)
            if not member.name.startswith("_")
        }
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, member in value.items():
            if type(key) is not str or not key:
                raise ProtocolError("HARP v19 hash keys must be nonempty strings.")
            output[key] = canonical_value(member)
        return output
    if isinstance(value, (tuple, list)):
        return [canonical_value(member) for member in value]
    raise ProtocolError(
        f"HARP v19 value is not canonically serializable: {type(value).__name__}."
    )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ProtocolError(f"HARP v19 {name} must be a SHA-256 digest.")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ProtocolError(f"HARP v19 {name} must be a SHA-256 digest.")
    return normalized


__all__ = ("canonical_bytes", "canonical_hash", "canonical_value", "require_sha256")
