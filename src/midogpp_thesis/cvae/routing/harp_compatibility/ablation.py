"""Source-only, held-query ablation gate for optional HARP compatibility."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence

from ...protocol import ProtocolError


MAD_SCALE = 1.4826


@dataclass(frozen=True)
class CompatibilityAblationFold:
    heldout_query: str
    independent_case_count: int
    correctness_delta_with_minus_without: float
    brier_delta_with_minus_without: float
    log_loss_delta_with_minus_without: float

    def __post_init__(self) -> None:
        if (
            not str(self.heldout_query)
            or isinstance(self.independent_case_count, bool)
            or int(self.independent_case_count) <= 0
            or not all(
                math.isfinite(float(value))
                for value in (
                    self.correctness_delta_with_minus_without,
                    self.brier_delta_with_minus_without,
                    self.log_loss_delta_with_minus_without,
                )
            )
        ):
            raise ProtocolError("HARP compatibility ablation fold is invalid.")


@dataclass(frozen=True)
class CompatibilityAblationDecision:
    enabled: bool
    correctness_lower: float
    brier_upper: float
    log_loss_upper: float
    heldout_queries: tuple[str, ...]
    independent_case_count: int
    rejection_reasons: tuple[str, ...]
    target_support_labels_used: bool = False
    target_evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        if bool(self.target_support_labels_used) or bool(self.target_evaluation_labels_used):
            raise ProtocolError("HARP compatibility ablation used target labels.")
        if bool(self.enabled) == bool(self.rejection_reasons):
            raise ProtocolError("HARP compatibility ablation decision is inconsistent.")


def decide_compatibility_ablation(
    folds: Sequence[CompatibilityAblationFold],
    *,
    kappa: float = 1.0,
    minimum_query_count: int = 4,
    minimum_case_count: int = 16,
) -> CompatibilityAblationDecision:
    """Enable energy only when held-query replay improves safely.

    Each row is one held-query aggregate, so a large query cannot dominate the
    decision through patch or case count.  Compatibility is disabled on every
    tie, coverage failure, or proper-loss regression.
    """

    rows = tuple(folds)
    multiplier = float(kappa)
    if (
        not rows
        or any(not isinstance(row, CompatibilityAblationFold) for row in rows)
        or not math.isfinite(multiplier)
        or multiplier < 0.0
        or isinstance(minimum_query_count, bool)
        or int(minimum_query_count) < 2
        or isinstance(minimum_case_count, bool)
        or int(minimum_case_count) < 1
    ):
        raise ProtocolError("HARP compatibility ablation parameters are invalid.")
    queries = tuple(sorted(row.heldout_query for row in rows))
    if len(queries) != len(set(queries)):
        raise ProtocolError("HARP compatibility ablation duplicated a held query.")
    correctness_lower = _lower(
        tuple(row.correctness_delta_with_minus_without for row in rows), multiplier
    )
    brier_upper = _upper(
        tuple(row.brier_delta_with_minus_without for row in rows), multiplier
    )
    log_loss_upper = _upper(
        tuple(row.log_loss_delta_with_minus_without for row in rows), multiplier
    )
    total_cases = sum(int(row.independent_case_count) for row in rows)
    reasons: list[str] = []
    if len(rows) < int(minimum_query_count):
        reasons.append("insufficient_held_query_coverage")
    if total_cases < int(minimum_case_count):
        reasons.append("insufficient_independent_cases")
    if correctness_lower <= 0.0:
        reasons.append("no_conservative_correctness_improvement")
    if brier_upper > 0.0:
        reasons.append("brier_noninferiority_failed")
    if log_loss_upper > 0.0:
        reasons.append("log_loss_noninferiority_failed")
    return CompatibilityAblationDecision(
        enabled=not reasons,
        correctness_lower=correctness_lower,
        brier_upper=brier_upper,
        log_loss_upper=log_loss_upper,
        heldout_queries=queries,
        independent_case_count=total_cases,
        rejection_reasons=tuple(reasons),
    )


def _median_mad(values: tuple[float, ...]) -> tuple[float, float]:
    center = float(median(values))
    spread = MAD_SCALE * float(median(tuple(abs(value - center) for value in values)))
    return center, spread


def _lower(values: tuple[float, ...], kappa: float) -> float:
    center, spread = _median_mad(values)
    return center - kappa * spread


def _upper(values: tuple[float, ...], kappa: float) -> float:
    center, spread = _median_mad(values)
    return center + kappa * spread


__all__ = (
    "CompatibilityAblationDecision",
    "CompatibilityAblationFold",
    "decide_compatibility_ablation",
)
