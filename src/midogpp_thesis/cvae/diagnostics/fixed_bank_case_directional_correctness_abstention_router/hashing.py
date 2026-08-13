"""Canonical JSON and SHA-256 helpers for the abstention router."""

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
    if is_dataclass(value) and not isinstance(value, type):
        return json_native(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("Abstention-router payload contains a non-finite float.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProtocolError(
        f"Abstention-router payload is not JSON-native: {type(value).__name__}."
    )


def canonical_json(value: object) -> bytes:
    return json.dumps(
        json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ProtocolError(f"Abstention-router {role} must be lowercase SHA-256.")
    return result


def require_stable_hash(value: object, role: str) -> str:
    result = str(value)
    if not result or len(result) > 128:
        raise ProtocolError(f"Abstention-router {role} is malformed.")
    return result


__all__ = (
    "canonical_hash",
    "canonical_json",
    "json_native",
    "require_sha256",
    "require_stable_hash",
)
