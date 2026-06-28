"""Support-NELBO feature helpers."""

from __future__ import annotations

from typing import Mapping


def support_nelbo_feature(scores_by_expert: Mapping[str, float], candidate_expert: str) -> dict[str, float]:
    return {"support_nelbo": float(scores_by_expert[str(candidate_expert)])}
