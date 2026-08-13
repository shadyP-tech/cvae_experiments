"""Canonical JSON hashing for the isolated DCSE diagnostic."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError


def json_native(value: object) -> object:
    """Return a closed JSON-native tree and reject ambiguous numerics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("DCSE JSON payloads cannot contain non-finite values.")
        return value
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_native(asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if name in result:
                raise ProtocolError("DCSE JSON mapping keys collide after stringification.")
            result[name] = json_native(item)
        return result
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    # NumPy scalars expose ``item``; arrays are intentionally rejected because
    # persisted payloads must make list conversion explicit at the call site.
    item = getattr(value, "item", None)
    if callable(item) and not hasattr(value, "shape"):
        return json_native(item())
    raise ProtocolError(f"DCSE payload is not JSON-native: {type(value).__name__}.")


def canonical_json(payload: object) -> bytes:
    return json.dumps(
        json_native(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: object) -> str:
    """Return a full SHA-256 over canonical, JSON-native content."""

    return hashlib.sha256(canonical_json(payload)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return result


def require_stable_hash(value: object, role: str) -> str:
    result = str(value)
    if not result or any(char.isspace() for char in result):
        raise ProtocolError(f"{role} must be a non-empty stable hash.")
    return result


__all__ = (
    "canonical_hash",
    "canonical_json",
    "json_native",
    "require_sha256",
    "require_stable_hash",
)
