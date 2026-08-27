"""Deterministic case-then-center aggregation for pseudo admission."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def equal_center_mean(values: dict[str, list[float]]) -> float:
    if not values or any(not rows for rows in values.values()):
        return 0.0
    per_center = [float(np.mean(rows, dtype=np.float64)) for rows in values.values()]
    return float(np.mean(per_center, dtype=np.float64))


def equal_center_metric_mean(
    values: dict[str, list[tuple[float, float, float]]],
) -> tuple[float, float, float]:
    if not values or any(not rows for rows in values.values()):
        return 0.0, 0.0, 0.0
    per_center = [
        np.mean(np.asarray(rows, dtype=np.float64), axis=0, dtype=np.float64)
        for rows in values.values()
    ]
    result = np.mean(np.asarray(per_center), axis=0, dtype=np.float64)
    return float(result[0]), float(result[1]), float(result[2])


def spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = np.asarray(tuple(left), dtype=np.float64)
    y = np.asarray(tuple(right), dtype=np.float64)
    if len(x) < 2 or len(x) != len(y):
        return None
    xr = _average_ranks(x)
    yr = _average_ranks(y)
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return None
    value = float(np.corrcoef(xr, yr)[0, 1])
    return value if math.isfinite(value) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


__all__ = ("equal_center_mean", "equal_center_metric_mean", "spearman")
