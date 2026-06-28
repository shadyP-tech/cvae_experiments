"""Small summary helpers for diagnostic utility matrices."""

from __future__ import annotations

from statistics import mean
from typing import Sequence

from ..downstream import CandidateDownstreamRow


def mean_ok_bacc(rows: Sequence[CandidateDownstreamRow]) -> float:
    values = [float(row.bacc) for row in rows if row.status == "ok"]
    return mean(values) if values else float("nan")
