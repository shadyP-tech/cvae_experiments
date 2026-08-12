"""Calibration-fold-only direction-shared zero-intercept slopes."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CalibrationRow,
    CaseActionFeatures,
    ContributionTarget,
    DirectionSharedCalibration,
    TwoHeadPrediction,
)


def directional_raw_gains(
    prediction: TwoHeadPrediction,
    features: CaseActionFeatures,
    *,
    n_positive: int,
    n_negative: int,
) -> tuple[float, float, float]:
    """Map two additive heads to two label-free direction components.

    The two heads first form the exact calibration-prevalence-weighted utility
    estimate.  Because labels are unavailable at routing time, its only legal
    directional allocation is the observed 0->1/1->0 flip-count share.  The
    components therefore sum exactly to the uncalibrated utility estimate;
    calibration changes only their two shared zero-intercept slopes.
    """

    if prediction.case_id != features.case_id or prediction.action_id != features.action_id:
        raise ProtocolError("Prediction/feature identity drifted.")
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Directional gain weights require both calibration classes.")
    weight_tp = 0.5 / int(n_positive)
    weight_tn = 0.5 / int(n_negative)
    raw_gain = weight_tp * prediction.mean_delta_tp + weight_tn * prediction.mean_delta_tn
    raw_variance = (
        weight_tp * weight_tp * prediction.variance_delta_tp
        + weight_tn * weight_tn * prediction.variance_delta_tn
    )
    flip_total = features.flip_0to1_count + features.flip_1to0_count
    if flip_total == 0:
        return 0.0, 0.0, max(float(raw_variance), 0.0)
    share_01 = features.flip_0to1_count / flip_total
    return float(raw_gain * share_01), float(raw_gain * (1.0 - share_01)), float(raw_variance)


def build_calibration_row(
    *,
    prediction: TwoHeadPrediction,
    features: CaseActionFeatures,
    target: ContributionTarget,
    calibration_n_positive: int,
    calibration_n_negative: int,
) -> CalibrationRow:
    """Use only calibration-fold prevalence to construct a slope row."""

    if target.case_id != features.case_id or target.action_id != features.action_id:
        raise ProtocolError("Calibration target/feature identity drifted.")
    raw_01, raw_10, _ = directional_raw_gains(
        prediction,
        features,
        n_positive=calibration_n_positive,
        n_negative=calibration_n_negative,
    )
    exact = (
        0.5 * target.delta_tp / calibration_n_positive
        + 0.5 * target.delta_tn / calibration_n_negative
    )
    return CalibrationRow(features.case_id, features.action_id, raw_01, raw_10, exact)


def fit_direction_shared_calibration(
    rows: Sequence[CalibrationRow],
    *,
    calibration_n_positive: int,
    calibration_n_negative: int,
) -> DirectionSharedCalibration:
    """Fit exactly two shared slopes, with no intercept or source/action terms."""

    records = tuple(rows)
    if calibration_n_positive <= 0 or calibration_n_negative <= 0 or not records:
        return DirectionSharedCalibration(
            gamma_0to1=0.0,
            gamma_1to0=0.0,
            n_positive=max(int(calibration_n_positive), 0),
            n_negative=max(int(calibration_n_negative), 0),
            row_count=len(records),
            valid=False,
        )
    design = np.asarray(
        [[row.raw_gain_0to1, row.raw_gain_1to0] for row in records], dtype=np.float64
    )
    response = np.asarray([row.exact_gain for row in records], dtype=np.float64)
    if not np.isfinite(design).all() or not np.isfinite(response).all():
        raise ProtocolError("Calibration surface is non-finite.")
    gamma, *_ = np.linalg.lstsq(design, response, rcond=None)
    if not np.isfinite(gamma).all():
        raise ProtocolError("Direction-shared calibration is non-finite.")
    return DirectionSharedCalibration(
        gamma_0to1=float(gamma[0]),
        gamma_1to0=float(gamma[1]),
        n_positive=int(calibration_n_positive),
        n_negative=int(calibration_n_negative),
        row_count=len(records),
        valid=True,
    )


def calibrated_gain(
    calibration: DirectionSharedCalibration,
    prediction: TwoHeadPrediction,
    features: CaseActionFeatures,
) -> tuple[float, float]:
    """Return calibrated mean and standard error using calibration prevalence."""

    if not calibration.valid or not features.has_flips:
        return 0.0, 0.0
    raw_01, raw_10, raw_variance = directional_raw_gains(
        prediction,
        features,
        n_positive=calibration.n_positive,
        n_negative=calibration.n_negative,
    )
    total_flips = features.flip_0to1_count + features.flip_1to0_count
    share_01 = features.flip_0to1_count / total_flips
    scale = calibration.gamma_0to1 * share_01 + calibration.gamma_1to0 * (1.0 - share_01)
    mean = calibration.gamma_0to1 * raw_01 + calibration.gamma_1to0 * raw_10
    return float(mean), float(abs(scale) * sqrt(max(raw_variance, 0.0)))


__all__ = (
    "build_calibration_row",
    "calibrated_gain",
    "directional_raw_gains",
    "fit_direction_shared_calibration",
)
