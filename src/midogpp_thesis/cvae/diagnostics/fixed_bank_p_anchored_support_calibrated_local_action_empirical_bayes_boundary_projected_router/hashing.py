"""Strict path-independent canonical hashing for SCALE-BP contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...protocol import ProtocolError


def canonical_json_bytes(value: object) -> bytes:
    """Encode supported values deterministically and reject nonfinite floats."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: object) -> str:
    """Return the lowercase SHA-256 of strict canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def require_sha256(value: object, role: str) -> str:
    """Return ``value`` only when it is a full lowercase SHA-256 digest."""

    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ProtocolError(f"SCALE-BP {role} is not a lowercase SHA-256 digest.")
    return text


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError("SCALE-BP canonical payload contains a nonfinite float.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_json_value(item) for item in value]
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _json_value(to_payload())
    raise ProtocolError(
        "SCALE-BP canonical payload contains unsupported type "
        f"{type(value).__name__}."
    )


__all__ = ("canonical_hash", "canonical_json_bytes", "require_sha256")
