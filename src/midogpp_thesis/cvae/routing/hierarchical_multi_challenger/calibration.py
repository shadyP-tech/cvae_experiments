"""Target-local Bayesian direction-offset calibration over a sealed menu."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import DirectionalCalibration, DirectionalPrediction


CALIBRATION_ALPHA = 4.0
CALIBRATION_MAX_ITERATIONS = 100
CALIBRATION_TOLERANCE = 1.0e-12


@dataclass(frozen=True, order=True)
class CalibrationObservation:
    """One menu-bound aggregate with no held-evaluation labels."""

    case_id: str
    action_id: str
    direction: str
    success_count: int
    trial_count: int
    base_probability: float
    model_fingerprint: str

    def __post_init__(self) -> None:
        if (
            not self.case_id
            or not self.action_id
            or self.direction not in {"0to1", "1to0"}
            or self.trial_count < 0
            or not 0 <= self.success_count <= self.trial_count
            or not math.isfinite(float(self.base_probability))
            or not 0.0 < float(self.base_probability) < 1.0
            or len(self.model_fingerprint) != 64
        ):
            raise ProtocolError("Calibration observation drifted.")


def build_calibration_observation(
    *,
    case_id: str,
    action_id: str,
    direction: str,
    success_count: int,
    trial_count: int,
    prediction: DirectionalPrediction,
) -> CalibrationObservation:
    return CalibrationObservation(
        case_id=str(case_id),
        action_id=str(action_id),
        direction=str(direction),
        success_count=int(success_count),
        trial_count=int(trial_count),
        base_probability=prediction.probability,
        model_fingerprint=prediction.model_fingerprint,
    )


def fit_direction_calibration(
    observations: Sequence[CalibrationObservation],
    *,
    direction: str,
    menu_hash: str,
    alpha: float = CALIBRATION_ALPHA,
) -> DirectionalCalibration:
    """Fit one MAP offset and its Laplace posterior variance.

    ``alpha`` is the precision of the frozen zero-centered Gaussian prior.
    Target labels update only this menu-bound offset; shared donor parameters
    remain frozen.
    """

    rows = tuple(observations)
    if direction not in {"0to1", "1to0"} or float(alpha) != CALIBRATION_ALPHA:
        raise ProtocolError("Directional calibration hyperparameters drifted.")
    if any(row.direction != direction for row in rows):
        raise ProtocolError("Calibration observation direction drifted.")
    informative = tuple(row for row in rows if row.trial_count > 0)
    if not informative:
        return DirectionalCalibration(
            direction=direction,
            offset=0.0,
            offset_variance=1.0 / CALIBRATION_ALPHA,
            success_count=0,
            trial_count=0,
            row_count=0,
            case_count=0,
            alpha=CALIBRATION_ALPHA,
            menu_hash=str(menu_hash),
            valid=True,
        )
    offset = 0.0
    curvature = CALIBRATION_ALPHA
    for _ in range(CALIBRATION_MAX_ITERATIONS):
        gradient = -CALIBRATION_ALPHA * offset
        curvature = CALIBRATION_ALPHA
        for row in informative:
            probability = _sigmoid(_logit(row.base_probability) + offset)
            gradient += row.success_count - row.trial_count * probability
            curvature += row.trial_count * probability * (1.0 - probability)
        step = gradient / curvature
        offset += step
        if abs(step) <= CALIBRATION_TOLERANCE:
            break
    else:
        raise ProtocolError("Directional calibration did not converge.")
    case_count = len({row.case_id for row in informative})
    # This is a hierarchical offset with a frozen N(0, 1/alpha) prior.  The
    # Laplace posterior variance remains defined under zero/one-case support,
    # unlike a cluster sandwich, and does not pretend sparse calibration is
    # certain.
    offset_variance = 1.0 / curvature
    return DirectionalCalibration(
        direction=direction,
        offset=offset,
        offset_variance=max(offset_variance, 0.0),
        success_count=sum(row.success_count for row in informative),
        trial_count=sum(row.trial_count for row in informative),
        row_count=len(informative),
        case_count=case_count,
        alpha=CALIBRATION_ALPHA,
        menu_hash=str(menu_hash),
        valid=True,
    )


def calibrated_probability(
    prediction: DirectionalPrediction,
    calibration: DirectionalCalibration,
) -> float:
    if not calibration.valid:
        raise ProtocolError("Invalid direction calibration cannot score a candidate.")
    return _sigmoid(_logit(prediction.probability) + calibration.offset)


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1.0e-15), 1.0 - 1.0e-15)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


__all__ = (
    "CALIBRATION_ALPHA",
    "CalibrationObservation",
    "build_calibration_observation",
    "calibrated_probability",
    "fit_direction_calibration",
)
