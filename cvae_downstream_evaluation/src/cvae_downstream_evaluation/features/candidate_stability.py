"""Candidate stability feature helpers."""

from __future__ import annotations


def candidate_stability_feature(value: float) -> dict[str, float]:
    return {"source_inner_candidate_stability": float(value)}
