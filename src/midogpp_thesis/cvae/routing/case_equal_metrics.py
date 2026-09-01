"""Stage-neutral case-equal classification endpoints.

The center estimand balances truth classes first, then gives each case that
supports a class equal mass within that class.  The per-case contribution is
therefore additive: its mean reconstructs the exact center-level BACC while
remaining aligned with case-level routing decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from ..protocol import ProtocolError


PRIMARY_ESTIMAND = (
    "equal_centers_equal_classes_equal_supporting_cases_recall_at_threshold_0_5"
)
PRIMARY_METRIC_NAME = "case_equal_bacc"
CASE_CONTRIBUTION_METRIC_NAME = "case_equal_bacc_contribution"
SINGLE_CLASS_CASE_RULE = (
    "sole_class_recall_weighted_by_total_cases_over_twice_class_supporting_cases"
)


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    case_equal_bacc_contribution: float
    brier: float
    log_loss: float

    def __post_init__(self) -> None:
        for name in ("case_equal_bacc_contribution", "brier", "log_loss"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ProtocolError("Case-equal metrics are nonfinite or negative.")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, float]:
        return {
            CASE_CONTRIBUTION_METRIC_NAME: self.case_equal_bacc_contribution,
            "brier": self.brier,
            "log_loss": self.log_loss,
        }


def _validated_case(
    probability: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probability, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if (
        values.ndim != 1
        or truth.ndim != 1
        or values.shape != truth.shape
        or not len(values)
        or not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        or np.any((truth != 0) & (truth != 1))
    ):
        raise ProtocolError("Case-equal metric inputs are invalid or misaligned.")
    return values, truth


def case_class_support_counts(
    case_label_rows: Sequence[np.ndarray | Sequence[int]],
) -> tuple[int, int]:
    rows = tuple(np.asarray(row, dtype=np.int64) for row in case_label_rows)
    if (
        not rows
        or any(
            row.ndim != 1
            or not len(row)
            or np.any((row != 0) & (row != 1))
            for row in rows
        )
    ):
        raise ProtocolError("Case class-support rows are malformed.")
    counts = tuple(
        sum(bool(np.any(row == label)) for row in rows) for label in (0, 1)
    )
    if any(value <= 0 for value in counts):
        raise ProtocolError("Exact case-equal BACC requires both center classes.")
    return counts  # type: ignore[return-value]


def case_metrics(
    probability: np.ndarray | Sequence[float],
    labels: np.ndarray | Sequence[int],
    *,
    total_case_count: int,
    class_support_case_counts: tuple[int, int],
) -> CaseMetrics:
    values, truth = _validated_case(probability, labels)
    counts = tuple(int(value) for value in class_support_case_counts)
    if (
        type(total_case_count) is not int
        or total_case_count < 1
        or type(class_support_case_counts) is not tuple
        or len(counts) != 2
        or any(value < 1 or value > total_case_count for value in counts)
    ):
        raise ProtocolError("Case-equal normalization is malformed.")
    prediction = values >= 0.5
    contribution = 0.0
    for label, support_count in zip((0, 1), counts, strict=True):
        mask = truth == label
        if np.any(mask):
            recall = float(np.mean(prediction[mask] == truth[mask]))
            contribution += 0.5 * total_case_count * recall / support_count
    clipped = np.clip(values, 1.0e-6, 1.0 - 1.0e-6)
    loss = -(truth * np.log(clipped) + (1 - truth) * np.log(1.0 - clipped))
    return CaseMetrics(
        case_equal_bacc_contribution=contribution,
        brier=float(np.mean((values - truth) ** 2, dtype=np.float64)),
        log_loss=float(np.mean(loss, dtype=np.float64)),
    )


def aggregate_case_equal_metrics(rows: Sequence[CaseMetrics]) -> dict[str, float]:
    values = tuple(rows)
    if not values or any(not isinstance(value, CaseMetrics) for value in values):
        raise ProtocolError("Case-equal aggregation requires typed cases.")
    result = {
        PRIMARY_METRIC_NAME: float(
            np.mean(
                [value.case_equal_bacc_contribution for value in values],
                dtype=np.float64,
            )
        ),
        "brier": float(np.mean([value.brier for value in values], dtype=np.float64)),
        "log_loss": float(
            np.mean([value.log_loss for value in values], dtype=np.float64)
        ),
    }
    if not 0.0 <= result[PRIMARY_METRIC_NAME] <= 1.0:
        raise ProtocolError("Reconstructed case-equal BACC is invalid.")
    return result


__all__ = (
    "CASE_CONTRIBUTION_METRIC_NAME",
    "PRIMARY_ESTIMAND",
    "PRIMARY_METRIC_NAME",
    "SINGLE_CLASS_CASE_RULE",
    "CaseMetrics",
    "aggregate_case_equal_metrics",
    "case_class_support_counts",
    "case_metrics",
)
