"""Split manifest helpers for support/evaluation separation."""

from __future__ import annotations

from typing import Iterable


def assert_disjoint_ids(support_ids: Iterable[str], evaluation_ids: Iterable[str]) -> None:
    """Ensure target support samples cannot leak into target evaluation."""

    overlap = sorted(set(support_ids).intersection(set(evaluation_ids)))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"Support/evaluation overlap detected: {preview}")

