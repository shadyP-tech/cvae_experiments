"""Target-support distributional-distance feature helpers."""

from __future__ import annotations


def distributional_distance_feature(value: float) -> dict[str, float]:
    return {"target_support_distributional_distance": float(value)}
