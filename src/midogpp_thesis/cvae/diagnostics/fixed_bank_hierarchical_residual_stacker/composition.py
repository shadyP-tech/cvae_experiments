"""Baseline-anchored residual composition with continuous class mixing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ...protocol import ProtocolError
from .contracts import CaseClassWeights, PredictionRow, SampleActionProbability
from .core_hashing import finite_float
from .residuals import logit_clip, residual_logit, sigmoid
from .scientific_constants import BASELINE_ACTION_ID, HARD_THRESHOLD, MAX_RESIDUAL_SCALE


def calibrated_baseline_probability(probability: float, intercept: float) -> float:
    return sigmoid(logit_clip(probability) + finite_float(intercept, "intercept"))


def soft_class_residual(
    negative_side_residual: float,
    positive_side_residual: float,
    calibrated_baseline: float,
) -> float:
    negative = finite_float(negative_side_residual, "negative_side_residual")
    positive = finite_float(positive_side_residual, "positive_side_residual")
    probability = finite_float(calibrated_baseline, "calibrated_baseline")
    if not 0.0 <= probability <= 1.0:
        raise ProtocolError("Calibrated baseline branch probability must lie in [0, 1].")
    return (1.0 - probability) * negative + probability * positive


def compose_probabilities(
    probabilities: Sequence[SampleActionProbability],
    weights: Sequence[CaseClassWeights],
    *,
    intercept: float,
    residual_scale: float,
    method_id: str,
) -> tuple[PredictionRow, ...]:
    """Compose one method; lambda zero returns B_cal bit-for-bit."""

    scale = finite_float(residual_scale, "residual_scale")
    if not 0.0 <= scale <= MAX_RESIDUAL_SCALE:
        raise ProtocolError("Residual scale must lie in the frozen [0, 0.25] interval.")
    action_by_sample: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in probabilities:
        if row.action_id in action_by_sample[row.sample_key]:
            raise ProtocolError("Composition probability surface contains duplicate actions.")
        action_by_sample[row.sample_key][row.action_id] = row.probability
    weight_lookup: dict[tuple[str, str, int], CaseClassWeights] = {}
    for row in weights:
        key = (row.target_center, row.case_id, row.class_side)
        if key in weight_lookup:
            raise ProtocolError("Composition weights contain a duplicate case/class row.")
        weight_lookup[key] = row
    output: list[PredictionRow] = []
    for (target, case, sample), actions in sorted(action_by_sample.items()):
        if BASELINE_ACTION_ID not in actions:
            raise ProtocolError("Composition sample is missing baseline B.")
        baseline = actions[BASELINE_ACTION_ID]
        baseline_calibrated = calibrated_baseline_probability(baseline, intercept)
        if scale == 0.0:
            # This early branch is the explicit bit-identity contract.
            composed = baseline_calibrated
        else:
            deltas: list[float] = []
            for side in (0, 1):
                weight_row = weight_lookup.get((target, case, side))
                if weight_row is None:
                    raise ProtocolError("Composition needs both class-side weights for every case.")
                residual_sum = 0.0
                for source, weight in weight_row.weights:
                    if source not in actions:
                        raise ProtocolError("Composition sample is missing a selected source action.")
                    residual_sum += weight * residual_logit(actions[source], baseline)
                deltas.append(residual_sum)
            delta = soft_class_residual(deltas[0], deltas[1], baseline_calibrated)
            composed = sigmoid(logit_clip(baseline) + float(intercept) + scale * delta)
        output.append(
            PredictionRow(
                method_id=str(method_id),
                target_center=target,
                case_id=case,
                sample_id=sample,
                probability=composed,
                hard_prediction=int(composed >= HARD_THRESHOLD),
            )
        )
    if not output:
        raise ProtocolError("Cannot compose an empty probability surface.")
    return tuple(output)


def baseline_predictions(
    probabilities: Sequence[SampleActionProbability],
    *,
    method_id: str = "B",
) -> tuple[PredictionRow, ...]:
    rows = tuple(row for row in probabilities if row.action_id == BASELINE_ACTION_ID)
    if not rows:
        raise ProtocolError("Baseline probability surface is empty.")
    return tuple(
        PredictionRow(
            method_id=method_id,
            target_center=row.target_center,
            case_id=row.case_id,
            sample_id=row.sample_id,
            probability=row.probability,
            hard_prediction=int(row.probability >= HARD_THRESHOLD),
        )
        for row in sorted(rows)
    )


def calibrated_baseline_predictions(
    probabilities: Sequence[SampleActionProbability],
    *,
    intercept: float,
    method_id: str = "B_cal",
) -> tuple[PredictionRow, ...]:
    rows = tuple(row for row in probabilities if row.action_id == BASELINE_ACTION_ID)
    if not rows:
        raise ProtocolError("Baseline probability surface is empty.")
    return tuple(
        PredictionRow(
            method_id=method_id,
            target_center=row.target_center,
            case_id=row.case_id,
            sample_id=row.sample_id,
            probability=calibrated_baseline_probability(row.probability, intercept),
            hard_prediction=int(calibrated_baseline_probability(row.probability, intercept) >= HARD_THRESHOLD),
        )
        for row in sorted(rows)
    )


__all__ = (
    "baseline_predictions",
    "calibrated_baseline_predictions",
    "calibrated_baseline_probability",
    "compose_probabilities",
    "soft_class_residual",
)
