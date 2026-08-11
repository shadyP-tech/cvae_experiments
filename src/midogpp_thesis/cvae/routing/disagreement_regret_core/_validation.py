"""Shared scalar validation helpers for disagreement-regret contracts."""

from __future__ import annotations

import math

from ...protocol import ProtocolError


def _canonical_id(value: object, *, name: str) -> str:
    text = str(value)
    if not text or text.strip() != text:
        raise ProtocolError(f"{name} must be nonempty and canonical.")
    return text


def _finite_probability(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProtocolError(f"{name} must be finite and lie in [0, 1].")
    return result


__all__ = ()
