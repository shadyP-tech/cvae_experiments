"""Canonical hashing for the compatibility-conditioned directional router.

The scientific core persists hashes instead of Python object identities.  This
module deliberately accepts only a small JSON-compatible value language so a
receipt reconstructed in a fresh process has exactly the same digest.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


def canonical_value(value: object) -> object:
    """Return a recursively canonical JSON value.

    Dataclass fields whose names end in ``_hash`` are not treated specially;
    callers choose explicitly whether an object's derived hash belongs in a
    parent receipt.  Mapping keys must be strings and are sorted by the JSON
    encoder.
    """

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError("Directional-router hashes cannot contain non-finite values.")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return canonical_value(value.item())
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ProtocolError("Directional-router hashes cannot contain non-finite arrays.")
        return canonical_value(array.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonical_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, (Mapping, MappingProxyType)):
        output: dict[str, object] = {}
        for key, member in value.items():
            if type(key) is not str or not key:
                raise ProtocolError("Directional-router hash mappings require string keys.")
            if key in output:
                raise ProtocolError("Directional-router hash mapping keys are ambiguous.")
            output[key] = canonical_value(member)
        return output
    if isinstance(value, (tuple, list)):
        return [canonical_value(member) for member in value]
    raise ProtocolError(
        f"Directional-router value is not canonically serializable: {type(value).__name__}."
    )


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Directional-router value is not canonically serializable.") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ProtocolError(f"{name} must be a SHA-256 hexadecimal digest.")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ProtocolError(f"{name} must be a SHA-256 hexadecimal digest.")
    return normalized


def probability_bytes_hash(values: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for raw in values:
        if type(raw) is not bytes or len(raw) != 4:
            raise ProtocolError("Probability hashes require exact little-endian float32 cells.")
        digest.update(raw)
    return digest.hexdigest()


__all__ = (
    "canonical_bytes",
    "canonical_hash",
    "canonical_value",
    "probability_bytes_hash",
    "require_sha256",
)
