"""Canonical, path-independent hashing for OE-PPUR contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...protocol import ProtocolError


def canonical_json_bytes(value: object) -> bytes:
    """Encode the small contract value set deterministically."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ProtocolError(f"OE-PPUR {role} is not a lowercase SHA-256 digest.")
    return text


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("OE-PPUR canonical payload contains a nonfinite float.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        to_payload = getattr(value, "to_payload", None)
        return _json_value(to_payload() if callable(to_payload) else dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise ProtocolError(
        "OE-PPUR canonical payload contains unsupported type "
        f"{type(value).__name__}."
    )


__all__ = ("canonical_hash", "canonical_json_bytes", "require_sha256")
