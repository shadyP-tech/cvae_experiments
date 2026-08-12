"""Exact provenance hashes and stable fitted-numeric fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence


def canonical_hash(payload: object) -> str:
    """Hash exact JSON semantics; use this for identities and provenance."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fitted_numeric_fingerprint(payload: object, *, significant_digits: int = 12) -> str:
    """Hash a deterministic, explicitly lossy view of derived fitted numerics.

    Raw fitted values remain persisted and are compared with a declared tolerance
    during replay.  This fingerprint is deliberately unsuitable for provenance,
    categorical decisions, topology, or terminal counts.
    """

    if significant_digits != 12:
        raise ValueError("The fitted-numeric fingerprint freezes 12 significant digits.")
    return canonical_hash(_quantized(payload, significant_digits=significant_digits))


def _quantized(value: object, *, significant_digits: int) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Fitted numerics must be finite.")
        normalized = 0.0 if value == 0.0 else value
        return {"__fitted_float__": format(normalized, f".{significant_digits}g")}
    if isinstance(value, Mapping):
        return {
            str(key): _quantized(item, significant_digits=significant_digits)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_quantized(item, significant_digits=significant_digits) for item in value]
    raise TypeError(f"Unsupported fitted-numeric payload type: {type(value).__name__}.")


__all__ = ("canonical_hash", "fitted_numeric_fingerprint")
