"""Same-H support-only intercept and shared residual-shrinkage selection."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...protocol import ProtocolError
from .composition import calibrated_baseline_predictions, compose_probabilities
from .contracts import (
    BinaryLabel,
    CalibrationChoice,
    CaseClassWeights,
    PredictionRow,
    SampleActionProbability,
)
from .pooled_metrics import paired_whole_case_cluster_lcb, score_case_confusions
from .residuals import clipped_probability
from .scientific_constants import INTERCEPT_GRID, RESIDUAL_SCALE_GRID


def class_balanced_log_loss(
    predictions: Sequence[PredictionRow],
    labels: Sequence[BinaryLabel],
) -> float:
    """Fixed 0.5/0.5 class-weighted proper loss over the legal support scope."""

    prediction_by_sample = {row.sample_key: row.probability for row in predictions}
    label_by_sample = {row.sample_key: row.label for row in labels}
    if (
        not prediction_by_sample
        or len(prediction_by_sample) != len(tuple(predictions))
        or len(label_by_sample) != len(tuple(labels))
        or set(prediction_by_sample) != set(label_by_sample)
    ):
        raise ProtocolError("Class-balanced loss inputs must be unique, non-empty, and aligned.")
    losses: dict[int, list[float]] = {0: [], 1: []}
    for key in sorted(label_by_sample):
        label = label_by_sample[key]
        probability = clipped_probability(prediction_by_sample[key])
        losses[label].append(-math.log(probability if label else 1.0 - probability))
    if not losses[0] or not losses[1]:
        raise ProtocolError("Class-balanced support loss requires both pooled classes.")
    return 0.5 * (
        math.fsum(losses[0]) / len(losses[0])
        + math.fsum(losses[1]) / len(losses[1])
    )


def fit_baseline_intercept(
    probabilities: Sequence[SampleActionProbability],
    labels: Sequence[BinaryLabel],
    *,
    intercept_grid: Sequence[float] = INTERCEPT_GRID,
) -> CalibrationChoice:
    label_rows = tuple(labels)
    if not label_rows or any(row.label_scope != "target_support" for row in label_rows):
        raise ProtocolError("B_cal intercept fitting requires target-support labels only.")
    grid = tuple(float(value) for value in intercept_grid)
    if grid != INTERCEPT_GRID:
        raise ProtocolError("Intercept selection left the frozen support grid.")
    support_probabilities = _filter_probabilities_to_labels(probabilities, label_rows)
    scores = tuple(
        (
            intercept,
            class_balanced_log_loss(
                calibrated_baseline_predictions(support_probabilities, intercept=intercept),
                label_rows,
            ),
        )
        for intercept in grid
    )
    intercept, objective = min(scores, key=lambda item: (item[1], abs(item[0]), item[0]))
    return CalibrationChoice(
        method_id="B_cal",
        intercept=intercept,
        residual_scale=0.0,
        objective_value=objective,
        support_case_count=len({(row.target_center, row.case_id) for row in label_rows}),
        lcb_gain_over_baseline_calibrated=None,
    )


def fit_residual_scale(
    probabilities: Sequence[SampleActionProbability],
    weights: Sequence[CaseClassWeights],
    labels: Sequence[BinaryLabel],
    *,
    intercept: float,
    lambda_grid: Sequence[float] = RESIDUAL_SCALE_GRID,
    method_id: str = "R",
) -> CalibrationChoice:
    """Select R lambda by smooth loss, then enforce the exact-BACC LCB gate."""

    label_rows = tuple(labels)
    if not label_rows or any(row.label_scope != "target_support" for row in label_rows):
        raise ProtocolError("Residual-scale fitting requires target-support labels only.")
    grid = tuple(float(value) for value in lambda_grid)
    if grid != RESIDUAL_SCALE_GRID:
        raise ProtocolError("Residual-scale selection left the frozen support grid.")
    support_probabilities = _filter_probabilities_to_labels(probabilities, label_rows)
    support_cases = {(row.target_center, row.case_id) for row in label_rows}
    support_weights = tuple(row for row in weights if (row.target_center, row.case_id) in support_cases)
    if {(row.target_center, row.case_id, row.class_side) for row in support_weights} != {
        (target, case, side) for target, case in support_cases for side in (0, 1)
    }:
        raise ProtocolError("Residual-scale fitting lacks support-case class weights.")
    prediction_by_scale = {
        scale: compose_probabilities(
            support_probabilities,
            support_weights,
            intercept=intercept,
            residual_scale=scale,
            method_id=method_id,
        )
        for scale in grid
    }
    objectives = {
        scale: class_balanced_log_loss(prediction_by_scale[scale], label_rows) for scale in grid
    }
    proposed = min(grid, key=lambda scale: (objectives[scale], scale))
    baseline = calibrated_baseline_predictions(support_probabilities, intercept=intercept)
    candidate_counts = score_case_confusions(prediction_by_scale[proposed], label_rows)
    baseline_counts = score_case_confusions(baseline, label_rows)
    contrast = paired_whole_case_cluster_lcb(candidate_counts, baseline_counts)
    selected = proposed if proposed > 0.0 and contrast.lower_bound > 0.0 else 0.0
    return CalibrationChoice(
        method_id=method_id,
        intercept=float(intercept),
        residual_scale=selected,
        objective_value=objectives[selected],
        support_case_count=len(support_cases),
        lcb_gain_over_baseline_calibrated=contrast.lower_bound,
    )


def _filter_probabilities_to_labels(
    probabilities: Sequence[SampleActionProbability],
    labels: Sequence[BinaryLabel],
) -> tuple[SampleActionProbability, ...]:
    keys = {row.sample_key for row in labels}
    if len(keys) != len(tuple(labels)):
        raise ProtocolError("Support label surface contains duplicate sample rows.")
    rows = tuple(row for row in probabilities if row.sample_key in keys)
    if {row.sample_key for row in rows} != keys:
        raise ProtocolError("Support labels are not covered by the probability surface.")
    return rows


__all__ = (
    "class_balanced_log_loss",
    "fit_baseline_intercept",
    "fit_residual_scale",
)
