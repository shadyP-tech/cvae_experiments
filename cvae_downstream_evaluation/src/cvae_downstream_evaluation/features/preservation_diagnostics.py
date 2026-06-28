"""Target-support/source-inner preservation diagnostic feature helpers."""

from __future__ import annotations


def preservation_diagnostic_feature(value: float) -> dict[str, float]:
    return {"support_preservation_diagnostic": float(value)}
