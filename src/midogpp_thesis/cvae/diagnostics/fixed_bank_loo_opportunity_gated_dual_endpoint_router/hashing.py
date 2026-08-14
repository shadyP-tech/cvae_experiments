"""Canonical JSON hashing for replay-sensitive science products."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from typing import Any

from ...protocol import ProtocolError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_native(value: Any) -> Any:
    """Recursively convert immutable science values into JSON-native objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if hasattr(value, "to_payload"):
        return json_native(value.to_payload())
    if is_dataclass(value):
        return json_native(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    raise ProtocolError(f"OGDE cannot convert {type(value).__name__} to JSON-native form.")


def require_sha256(value: object, role: str) -> str:
    result = str(value)
    if not _SHA256.fullmatch(result):
        raise ProtocolError(f"OGDE {role} is not a SHA-256 digest.")
    return result


def require_stable_hash(value: object, role: str) -> str:
    result = str(value)
    if not result:
        raise ProtocolError(f"OGDE {role} is empty.")
    return result


__all__ = ("canonical_hash", "canonical_json", "json_native", "require_sha256", "require_stable_hash")
