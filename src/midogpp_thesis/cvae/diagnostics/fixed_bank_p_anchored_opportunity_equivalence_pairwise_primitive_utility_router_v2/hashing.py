"""Deterministic hashing primitives for the isolated OE-PPUR v2 adapter."""

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
    """Encode the small contract value set without platform-dependent bytes."""

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
    if not isinstance(value, str):
        raise ProtocolError(f"OE-PPUR v2 {role} is not a SHA-256 string.")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolError(
            f"OE-PPUR v2 {role} is not a lowercase SHA-256 digest."
        )
    return value


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("OE-PPUR v2 canonical payload is nonfinite.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        to_payload = getattr(value, "to_payload", None)
        return _json_value(
            to_payload() if callable(to_payload) else dataclasses.asdict(value)
        )
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    raise ProtocolError(
        "OE-PPUR v2 canonical payload contains unsupported type "
        f"{type(value).__name__}."
    )


__all__ = ("canonical_hash", "canonical_json_bytes", "require_sha256")
