"""Center-blocked lower-tail and upper-tail donor safety summaries."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import TAIL_RISK_FRACTION


def lower_tail_mean(values: Sequence[float]) -> float:
    """Mean of the worst lower tail (two of eight donors at alpha=0.25)."""

    return _tail_mean(values, upper=False)


def upper_tail_mean(values: Sequence[float]) -> float:
    """Mean of the worst upper tail for loss deltas."""

    return _tail_mean(values, upper=True)


def _tail_mean(values: Sequence[float], *, upper: bool) -> float:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ProtocolError("PSSCUR donor tail-risk input drifted.")
    count = max(1, int(math.ceil(TAIL_RISK_FRACTION * len(array))))
    ordered = np.sort(array)
    tail = ordered[-count:] if upper else ordered[:count]
    return float(np.mean(tail, dtype=np.float64))


__all__ = ("lower_tail_mean", "upper_tail_mean")
