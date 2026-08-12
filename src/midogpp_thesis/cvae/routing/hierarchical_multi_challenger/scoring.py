"""Expected BACC gains and joint epistemic/calibration uncertainty."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .calibration import calibrated_probability
from .contracts import (
    ActionScore,
    DIRECTIONS,
    DirectionalCalibration,
    DirectionalLogitModel,
    DirectionalPrediction,
)


def score_action_against_baseline(
    *,
    action_id: str,
    predictions: Mapping[str, DirectionalPrediction],
    models: Mapping[str, DirectionalLogitModel],
    calibrations: Mapping[str, DirectionalCalibration],
    flip_counts: Mapping[str, int],
    n_positive: int,
    n_negative: int,
) -> ActionScore:
    """Score B->action flips; no residual outcome variance is added."""

    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Action scoring requires both calibration classes.")
    if (
        set(predictions) != set(DIRECTIONS)
        or set(models) != set(DIRECTIONS)
        or set(calibrations) != set(DIRECTIONS)
        or set(flip_counts) != set(DIRECTIONS)
    ):
        raise ProtocolError("Action scoring direction topology drifted.")
    expected_gain = 0.0
    epistemic_variance = 0.0
    calibration_variance = 0.0
    model_gradients: dict[str, tuple[float, ...]] = {}
    calibration_gradients: dict[str, float] = {}
    for direction in DIRECTIONS:
        count = int(flip_counts[direction])
        if count < 0:
            raise ProtocolError("Flip counts cannot be negative.")
        prediction = predictions[direction]
        model = models[direction]
        calibration = calibrations[direction]
        if (
            prediction.model_fingerprint != model.fit_fingerprint
            or calibration.direction != direction
            or model.direction != direction
        ):
            raise ProtocolError("Action score model/calibration identity drifted.")
        probability = calibrated_probability(prediction, calibration)
        if direction == "0to1":
            per_flip = 0.5 * (
                probability / n_positive - (1.0 - probability) / n_negative
            )
        else:
            per_flip = 0.5 * (
                probability / n_negative - (1.0 - probability) / n_positive
            )
        expected_gain += count * per_flip
        derivative_probability = 0.5 * count * (
            1.0 / n_positive + 1.0 / n_negative
        )
        derivative_eta = derivative_probability * probability * (1.0 - probability)
        gradient = derivative_eta * np.asarray(prediction.design, dtype=np.float64)
        covariance = np.asarray(model.covariance, dtype=np.float64)
        epistemic_variance += max(float(gradient @ covariance @ gradient), 0.0)
        calibration_variance += (
            derivative_eta * derivative_eta * calibration.offset_variance
        )
        model_gradients[direction] = tuple(float(value) for value in gradient)
        calibration_gradients[direction] = float(derivative_eta)
    return ActionScore(
        action_id=str(action_id),
        expected_gain=float(expected_gain),
        epistemic_variance=float(epistemic_variance),
        calibration_variance=float(calibration_variance),
        model_gradients=model_gradients,
        calibration_gradients=calibration_gradients,
    )


def baseline_action_score(
    *,
    models: Mapping[str, DirectionalLogitModel],
) -> ActionScore:
    if set(models) != set(DIRECTIONS):
        raise ProtocolError("Baseline score model topology drifted.")
    return ActionScore(
        action_id="B",
        expected_gain=0.0,
        epistemic_variance=0.0,
        calibration_variance=0.0,
        model_gradients={
            direction: (0.0,) * models[direction].dimension for direction in DIRECTIONS
        },
        calibration_gradients={direction: 0.0 for direction in DIRECTIONS},
    )


__all__ = ("baseline_action_score", "score_action_against_baseline")
