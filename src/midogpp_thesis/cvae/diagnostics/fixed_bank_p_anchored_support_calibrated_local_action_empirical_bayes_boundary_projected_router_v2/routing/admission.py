"""Primary routing admission metrics in predeclared thesis priority order."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, MetricVector


@dataclass(frozen=True, slots=True)
class AdmissionObservation:
    center: str
    case_id: str
    predicted_by_action: Mapping[str, MetricVector]
    realized_by_action: Mapping[str, MetricVector]
    selected_action_id: str | None

    def __post_init__(self) -> None:
        if (
            tuple(self.predicted_by_action) != ACTION_IDS
            or tuple(self.realized_by_action) != ACTION_IDS
            or (self.selected_action_id is not None and self.selected_action_id not in ACTION_IDS)
            or not self.center
            or not self.case_id
        ):
            raise GovernanceError("SCALE-BP v2 admission observation drifted.")


@dataclass(frozen=True, slots=True)
class AdmissionThresholds:
    minimum_opportunity_cases: int = 24
    minimum_represented_centers: int = 6
    minimum_within_case_spearman: float = 0.0
    maximum_normalized_oracle_gap: float = 1.0
    maximum_harmful_selected_policy_count: int = 0


@dataclass(frozen=True, slots=True)
class AdmissionMetrics:
    case_count: int
    represented_center_count: int
    top1_oracle_agreement: float
    mean_within_case_spearman: float
    mean_normalized_oracle_gap: float
    selected_count: int
    harmful_selected_count: int
    proper_safe_selected_count: int
    passed: bool
    failed_gates: tuple[str, ...]
    metrics_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.top1_oracle_agreement,
            self.mean_within_case_spearman,
            self.mean_normalized_oracle_gap,
        )
        if (
            self.case_count < 0
            or self.represented_center_count < 0
            or not all(math.isfinite(value) for value in values)
            or not 0.0 <= self.top1_oracle_agreement <= 1.0
            or self.mean_normalized_oracle_gap < 0.0
            or self.selected_count < 0
            or self.harmful_selected_count < 0
            or self.proper_safe_selected_count < 0
            or self.passed != (not self.failed_gates)
        ):
            raise GovernanceError("SCALE-BP v2 admission metrics drifted.")
        object.__setattr__(
            self,
            "metrics_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_admission_metrics_v1",
                    "case_count": self.case_count,
                    "represented_center_count": self.represented_center_count,
                    "top1_oracle_agreement": self.top1_oracle_agreement,
                    "mean_within_case_spearman": self.mean_within_case_spearman,
                    "mean_normalized_oracle_gap": self.mean_normalized_oracle_gap,
                    "selected_count": self.selected_count,
                    "harmful_selected_count": self.harmful_selected_count,
                    "proper_safe_selected_count": self.proper_safe_selected_count,
                    "passed": self.passed,
                    "failed_gates": self.failed_gates,
                    "metric_priority": (
                        "top1_oracle_agreement",
                        "spearman",
                        "normalized_oracle_gap",
                        "center_stability",
                    ),
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_v2_admission_metrics_v1",
            "case_count": self.case_count,
            "opportunity_case_count": self.case_count,
            "represented_center_count": self.represented_center_count,
            "top1_oracle_agreement": self.top1_oracle_agreement,
            "mean_within_case_spearman": self.mean_within_case_spearman,
            "mean_normalized_oracle_gap": self.mean_normalized_oracle_gap,
            "selected_count": self.selected_count,
            "harmful_selected_count": self.harmful_selected_count,
            "proper_safe_selected_count": self.proper_safe_selected_count,
            "passed": self.passed,
            "failed_gates": list(self.failed_gates),
            "case_then_equal_center_aggregation": True,
            "metrics_hash": self.metrics_hash,
        }


def evaluate_admission(
    observations: Sequence[AdmissionObservation],
    *,
    thresholds: AdmissionThresholds = AdmissionThresholds(),
) -> AdmissionMetrics:
    rows = tuple(observations)
    if not rows or len({(row.center, row.case_id) for row in rows}) != len(rows):
        raise GovernanceError("SCALE-BP v2 admission surface is empty or duplicated.")
    per_center: dict[str, list[tuple[float, float, float]]] = {}
    selected_count = harmful = proper_safe = 0
    for row in rows:
        if row.selected_action_id is not None:
            selected_count += 1
            realized_selected = row.realized_by_action[row.selected_action_id]
            harmful += int(realized_selected.bacc <= 0.0)
            proper_safe += int(
                realized_selected.bacc > 0.0
                and realized_selected.brier <= 0.0
                and realized_selected.log <= 0.0
            )
        predicted = np.asarray(
            [row.predicted_by_action[action].bacc for action in ACTION_IDS], dtype=np.float64
        )
        realized = np.asarray(
            [row.realized_by_action[action].bacc for action in ACTION_IDS], dtype=np.float64
        )
        # Complete matrices retain structural no-op cases, but learnability is
        # evaluated only where at least one action has nonzero realized value.
        realized_matrix = np.asarray(
            [row.realized_by_action[action].as_tuple() for action in ACTION_IDS],
            dtype=np.float64,
        )
        if not np.any(np.abs(realized_matrix) > 0.0):
            continue
        predicted_top = set(np.flatnonzero(predicted == np.max(predicted)))
        realized_top = set(np.flatnonzero(realized == np.max(realized)))
        top1 = float(bool(predicted_top & realized_top))
        rank = _spearman(predicted, realized)
        selected_value = (
            0.0
            if row.selected_action_id is None
            else row.realized_by_action[row.selected_action_id].bacc
        )
        oracle = max(0.0, float(np.max(realized)))
        gap = max(0.0, oracle - selected_value) / max(abs(oracle), 1.0e-12)
        per_center.setdefault(row.center, []).append((top1, rank, gap))
    opportunity_count = sum(len(values) for values in per_center.values())
    centers = tuple(sorted(per_center))
    failed: list[str] = []
    center_means = {
        center: np.mean(np.asarray(values, dtype=np.float64), axis=0, dtype=np.float64)
        for center, values in per_center.items()
    }
    if center_means:
        equal_center = np.mean(
            np.stack(tuple(center_means.values()), axis=0), axis=0, dtype=np.float64
        )
        mean_top1, mean_rank, mean_gap = (float(value) for value in equal_center)
    else:
        mean_top1 = mean_rank = mean_gap = 0.0
    if opportunity_count < thresholds.minimum_opportunity_cases:
        failed.append("INSUFFICIENT_OPPORTUNITY_CASES")
    if len(centers) < thresholds.minimum_represented_centers:
        failed.append("INSUFFICIENT_REPRESENTED_CENTERS")
    if mean_rank <= thresholds.minimum_within_case_spearman:
        failed.append("NONPOSITIVE_ACTION_RANK_ASSOCIATION")
    if mean_gap > thresholds.maximum_normalized_oracle_gap:
        failed.append("ORACLE_GAP_TOO_LARGE")
    if harmful > thresholds.maximum_harmful_selected_policy_count:
        failed.append("HARMFUL_SELECTED_POLICY")
    if selected_count and proper_safe != selected_count:
        failed.append("SELECTED_POLICY_PROPER_LOSS_UNSAFE")
    return AdmissionMetrics(
        opportunity_count,
        len(centers),
        mean_top1,
        mean_rank,
        mean_gap,
        selected_count,
        harmful,
        proper_safe,
        not failed,
        tuple(failed),
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return 0.0 if denominator == 0.0 else float(left_centered @ right_centered / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


__all__ = (
    "AdmissionMetrics",
    "AdmissionObservation",
    "AdmissionThresholds",
    "evaluate_admission",
)
