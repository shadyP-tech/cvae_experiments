"""Source-inner threshold selection and classifier decision helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .artifacts import stable_hash
from .downstream import balanced_accuracy
from .protocol import ProtocolError

THRESHOLD_RULE_ID = "source_inner_macro_center_bacc_one_se_closest_0_5_v1"
FIXED_THRESHOLD_RULE_ID = "fixed_threshold_v1"
THRESHOLD_TIE_BREAKER = "within_max_se_0_005_then_closest_0_5_then_larger"
THRESHOLD_MISSING_ROW_POLICY = "available_scoring_units_within_pseudo_target"
DEFAULT_THRESHOLD_GRID = tuple(round(value / 100.0, 2) for value in range(1, 100))
DEFAULT_MIN_VALID_PSEUDO_TARGETS = 3


@dataclass(frozen=True)
class ThresholdPredictionSet:
    """One source-inner scoring unit used for threshold selection."""

    pseudo_target_center: str
    y_true: tuple[int, ...]
    prob_pos: tuple[float, ...]
    scoring_unit_id: str = "primary"

    def __post_init__(self) -> None:
        if not self.pseudo_target_center:
            raise ProtocolError("Threshold prediction sets require a pseudo_target_center.")
        if not self.scoring_unit_id:
            raise ProtocolError("Threshold prediction sets require a scoring_unit_id.")
        if len(self.y_true) != len(self.prob_pos):
            raise ProtocolError("Threshold prediction set label/probability lengths differ.")

    def to_payload(self) -> dict[str, object]:
        return {
            "pseudo_target_center": self.pseudo_target_center,
            "scoring_unit_id": self.scoring_unit_id,
            "y_true": [int(value) for value in self.y_true],
            "prob_pos": [float(value) for value in self.prob_pos],
        }


@dataclass(frozen=True)
class ThresholdDecisionSpec:
    """Frozen classifier operating-point identity."""

    threshold_policy: str
    threshold_value: float
    threshold_policy_group_id: str
    threshold_grid_hash: str
    threshold_rule_id: str
    threshold_selection_source: str
    threshold_source_score_table_hash: str
    threshold_tie_breaker: str
    n_min: int
    n_valid_pseudo_targets: int
    selected_source_inner_score_vector: Mapping[str, float]
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        if self.threshold_policy not in {"fixed_0_5", "source_inner_selected"}:
            raise ProtocolError(f"Unsupported threshold policy: {self.threshold_policy!r}")
        value = float(self.threshold_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError(f"Threshold value must be finite in [0, 1], got {value!r}.")
        if not self.threshold_policy_group_id:
            raise ProtocolError("ThresholdDecisionSpec requires threshold_policy_group_id.")
        if int(self.n_min) < 1:
            raise ProtocolError("ThresholdDecisionSpec.n_min must be positive.")
        if int(self.n_valid_pseudo_targets) < 0:
            raise ProtocolError("ThresholdDecisionSpec.n_valid_pseudo_targets cannot be negative.")

    @property
    def decision_config_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "threshold_policy": self.threshold_policy,
            "threshold_value": float(self.threshold_value),
            "threshold_policy_group_id": self.threshold_policy_group_id,
            "threshold_grid_hash": self.threshold_grid_hash,
            "threshold_rule_id": self.threshold_rule_id,
            "threshold_selection_source": self.threshold_selection_source,
            "threshold_source_score_table_hash": self.threshold_source_score_table_hash,
            "threshold_tie_breaker": self.threshold_tie_breaker,
            "n_min": int(self.n_min),
            "n_valid_pseudo_targets": int(self.n_valid_pseudo_targets),
            "selected_source_inner_score_vector": dict(self.selected_source_inner_score_vector),
            "fallback_reason": self.fallback_reason,
            "probabilities_calibrated": False,
        }


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Source-inner threshold selection output."""

    decision: ThresholdDecisionSpec
    threshold_grid: tuple[float, ...]
    objective_by_threshold: Mapping[str, float]
    selected_threshold: float
    best_threshold: float
    best_mean_bacc: float
    selected_mean_bacc: float
    se_best: float
    valid_pseudo_target_centers: tuple[str, ...]

    def to_artifact_fields(self) -> dict[str, object]:
        return {
            **self.decision.to_payload(),
            "threshold_decision_config_hash": self.decision.decision_config_hash,
            "threshold_grid": list(self.threshold_grid),
            "threshold_objective_by_threshold": dict(self.objective_by_threshold),
            "threshold_best_value": float(self.best_threshold),
            "threshold_best_mean_bacc": float(self.best_mean_bacc),
            "threshold_selected_mean_bacc": float(self.selected_mean_bacc),
            "threshold_se_best": float(self.se_best),
            "threshold_valid_pseudo_target_centers": list(self.valid_pseudo_target_centers),
            "threshold_missing_row_policy": THRESHOLD_MISSING_ROW_POLICY,
            "target_eval_labels_used_for_threshold": False,
            "oracle_rows_used_for_threshold": False,
        }


def fixed_threshold_spec(
    threshold_value: float = 0.5,
    *,
    threshold_policy_group_id: str = "fixed_0_5",
) -> ThresholdDecisionSpec:
    """Return a fixed-threshold decision policy."""

    value = float(threshold_value)
    return ThresholdDecisionSpec(
        threshold_policy="fixed_0_5",
        threshold_value=value,
        threshold_policy_group_id=threshold_policy_group_id,
        threshold_grid_hash=threshold_grid_hash((value,)),
        threshold_rule_id=FIXED_THRESHOLD_RULE_ID,
        threshold_selection_source="fixed_0_5",
        threshold_source_score_table_hash=stable_hash({"source": "fixed_0_5", "threshold_value": value}),
        threshold_tie_breaker="none",
        n_min=1,
        n_valid_pseudo_targets=0,
        selected_source_inner_score_vector={},
        fallback_reason="",
    )


def select_threshold_source_inner_lodo(
    *,
    outer_target_center: str,
    prediction_sets: Sequence[ThresholdPredictionSet],
    threshold_policy_group_payload: Mapping[str, object],
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    n_min: int = DEFAULT_MIN_VALID_PSEUDO_TARGETS,
) -> ThresholdSelectionResult:
    """Select a BACC operating point from source-inner pseudo-target folds only."""

    grid = _validate_threshold_grid(threshold_grid)
    group_id = threshold_policy_group_id(threshold_policy_group_payload)
    source_hash = threshold_source_score_table_hash(prediction_sets)
    by_center: dict[str, list[ThresholdPredictionSet]] = {}
    for item in prediction_sets:
        if str(item.pseudo_target_center) == str(outer_target_center):
            raise ProtocolError("Outer target center cannot be used for source-inner threshold selection.")
        if _valid_prediction_set(item):
            by_center.setdefault(str(item.pseudo_target_center), []).append(item)
    valid_centers = tuple(sorted(by_center))
    if len(valid_centers) < int(n_min):
        decision = ThresholdDecisionSpec(
            threshold_policy="source_inner_selected",
            threshold_value=0.5,
            threshold_policy_group_id=group_id,
            threshold_grid_hash=threshold_grid_hash(grid),
            threshold_rule_id=THRESHOLD_RULE_ID,
            threshold_selection_source="source_inner_lodo",
            threshold_source_score_table_hash=source_hash,
            threshold_tie_breaker=THRESHOLD_TIE_BREAKER,
            n_min=int(n_min),
            n_valid_pseudo_targets=len(valid_centers),
            selected_source_inner_score_vector={},
            fallback_reason="insufficient_valid_pseudo_targets",
        )
        return ThresholdSelectionResult(
            decision=decision,
            threshold_grid=grid,
            objective_by_threshold={},
            selected_threshold=0.5,
            best_threshold=0.5,
            best_mean_bacc=math.nan,
            selected_mean_bacc=math.nan,
            se_best=math.nan,
            valid_pseudo_target_centers=valid_centers,
        )

    center_scores_by_threshold: dict[float, dict[str, float]] = {}
    objective_by_threshold: dict[str, float] = {}
    for threshold in grid:
        per_center = {
            center: _center_bacc_for_threshold(by_center[center], threshold)
            for center in valid_centers
        }
        center_scores_by_threshold[threshold] = per_center
        objective_by_threshold[_fmt_threshold(threshold)] = _mean(per_center.values())

    best_threshold = max(grid, key=lambda value: (objective_by_threshold[_fmt_threshold(value)], -value))
    best_scores = list(center_scores_by_threshold[best_threshold].values())
    se_best = _sample_std(best_scores) / math.sqrt(float(len(best_scores)))
    best_mean = objective_by_threshold[_fmt_threshold(best_threshold)]
    tolerance = max(float(se_best), 0.005)
    eligible = [
        threshold
        for threshold in grid
        if objective_by_threshold[_fmt_threshold(threshold)] >= best_mean - tolerance
    ]
    selected = min(eligible, key=lambda value: (abs(value - 0.5), -value))
    selected_scores = center_scores_by_threshold[selected]
    decision = ThresholdDecisionSpec(
        threshold_policy="source_inner_selected",
        threshold_value=float(selected),
        threshold_policy_group_id=group_id,
        threshold_grid_hash=threshold_grid_hash(grid),
        threshold_rule_id=THRESHOLD_RULE_ID,
        threshold_selection_source="source_inner_lodo",
        threshold_source_score_table_hash=source_hash,
        threshold_tie_breaker=THRESHOLD_TIE_BREAKER,
        n_min=int(n_min),
        n_valid_pseudo_targets=len(valid_centers),
        selected_source_inner_score_vector=selected_scores,
        fallback_reason="",
    )
    return ThresholdSelectionResult(
        decision=decision,
        threshold_grid=grid,
        objective_by_threshold=objective_by_threshold,
        selected_threshold=float(selected),
        best_threshold=float(best_threshold),
        best_mean_bacc=float(best_mean),
        selected_mean_bacc=float(objective_by_threshold[_fmt_threshold(selected)]),
        se_best=float(se_best),
        valid_pseudo_target_centers=valid_centers,
    )


def apply_threshold(prob_pos: Sequence[float], threshold_value: float) -> list[int]:
    """Convert positive-class probabilities into labels using a frozen threshold."""

    threshold = float(threshold_value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ProtocolError(f"Threshold value must be finite in [0, 1], got {threshold!r}.")
    predictions: list[int] = []
    for value in prob_pos:
        prob = float(value)
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ProtocolError(f"Invalid positive-class probability: {value!r}")
        predictions.append(1 if prob >= threshold else 0)
    return predictions


def threshold_policy_group_id(payload: Mapping[str, object]) -> str:
    return stable_hash(dict(payload))


def threshold_grid_hash(threshold_grid: Sequence[float]) -> str:
    return stable_hash([float(value) for value in threshold_grid])


def threshold_source_score_table_hash(prediction_sets: Sequence[ThresholdPredictionSet]) -> str:
    return stable_hash([item.to_payload() for item in prediction_sets])


def artifact_fields_for_decision(decision: ThresholdDecisionSpec) -> dict[str, object]:
    payload = decision.to_payload()
    return {
        **payload,
        "threshold_decision_config_hash": decision.decision_config_hash,
        "target_eval_labels_used_for_threshold": False,
        "oracle_rows_used_for_threshold": False,
    }


def _validate_threshold_grid(threshold_grid: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(sorted({round(float(value), 6) for value in threshold_grid}))
    if not grid:
        raise ProtocolError("Threshold grid cannot be empty.")
    for value in grid:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError(f"Threshold grid value must be finite in [0, 1], got {value!r}.")
    if 0.5 not in grid:
        raise ProtocolError("Threshold grid must include 0.5.")
    return grid


def _valid_prediction_set(item: ThresholdPredictionSet) -> bool:
    labels = [int(value) for value in item.y_true]
    probs = [float(value) for value in item.prob_pos]
    if len(labels) != len(probs) or not labels:
        return False
    if sorted(set(labels)) != [0, 1]:
        return False
    return all(math.isfinite(prob) and 0.0 <= prob <= 1.0 for prob in probs)


def _center_bacc_for_threshold(items: Sequence[ThresholdPredictionSet], threshold: float) -> float:
    scores = [
        balanced_accuracy(item.y_true, apply_threshold(item.prob_pos, threshold))
        for item in items
    ]
    return _mean(scores)


def _mean(values: Sequence[float] | object) -> float:
    vals = [float(value) for value in values]
    if not vals:
        raise ProtocolError("Cannot average an empty threshold score vector.")
    return sum(vals) / float(len(vals))


def _sample_std(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    if len(vals) < 2:
        return math.nan
    mean = sum(vals) / float(len(vals))
    variance = sum((value - mean) ** 2 for value in vals) / float(len(vals) - 1)
    return math.sqrt(variance)


def _fmt_threshold(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")
