"""Manual deterministic float64 weighted ridge for action calibration."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ..identity import RIDGE_ALPHA
from .contracts import ActionCalibrationModel


SCALE_FLOOR = 1.0e-12
PINV_RCOND = 1.0e-12


def _matrix(value: object, *, role: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0 or not np.isfinite(array).all():
        raise ProtocolError(f"P-DCAPS {role} matrix drifted.")
    return array


def _vector(value: object, *, length: int, role: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ProtocolError(f"P-DCAPS {role} vector drifted.")
    return array


def fit_weighted_ridge(
    features: object,
    response: object,
    weights: object,
    *,
    metric: str,
    excluded_outer_center: str,
    excluded_scored_center: str | None,
    training_centers: Sequence[str],
    feature_names: Sequence[str],
    training_response_hash: str,
    weight_audit_hash: str,
    ridge_alpha: float = RIDGE_ALPHA,
) -> ActionCalibrationModel:
    """Fit one unpenalized-intercept ridge, falling back to a fixed pinv."""

    x = _matrix(features, role="ridge feature")
    y = _vector(response, length=len(x), role="ridge response")
    weight = _vector(weights, length=len(x), role="ridge weight")
    names = tuple(str(name) for name in feature_names)
    if (
        len(names) != x.shape[1]
        or len(names) != len(set(names))
        or np.any(weight <= 0.0)
        or not np.isclose(np.sum(weight, dtype=np.float64), 1.0, atol=1.0e-12, rtol=0.0)
        or float(ridge_alpha) != RIDGE_ALPHA
    ):
        raise ProtocolError("P-DCAPS weighted-ridge fit contract drifted.")

    mean = np.sum(weight[:, None] * x, axis=0, dtype=np.float64)
    variance = np.sum(weight[:, None] * (x - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > SCALE_FLOOR, scale, 1.0)
    standardized = np.ascontiguousarray((x - mean) / scale, dtype=np.float64)
    design = np.ascontiguousarray(
        np.column_stack((np.ones(len(x), dtype=np.float64), standardized)),
        dtype=np.float64,
    )
    penalty = np.diag(
        np.asarray([0.0, *([float(ridge_alpha)] * x.shape[1])], dtype=np.float64)
    )
    system = design.T @ (weight[:, None] * design) + penalty
    target = design.T @ (weight * y)
    try:
        fitted = np.linalg.solve(system, target)
        solver = "solve"
    except np.linalg.LinAlgError:
        fitted = np.linalg.pinv(system, rcond=PINV_RCOND) @ target
        solver = "pinv"
    if fitted.shape != (x.shape[1] + 1,) or not np.isfinite(fitted).all():
        raise ProtocolError("P-DCAPS weighted ridge produced nonfinite coefficients.")
    return ActionCalibrationModel(
        str(metric),
        str(excluded_outer_center),
        None if excluded_scored_center is None else str(excluded_scored_center),
        tuple(str(center) for center in training_centers),
        names,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        float(fitted[0]),
        tuple(float(value) for value in fitted[1:]),
        float(ridge_alpha),
        len(x),
        str(training_response_hash),
        str(weight_audit_hash),
        solver,
    )


def predict_weighted_ridge(
    model: ActionCalibrationModel,
    features: object,
) -> float | np.ndarray:
    """Replay a serialized model without an estimator or mutable state."""

    raw = np.asarray(features, dtype=np.float64)
    scalar = raw.ndim == 1
    matrix = raw.reshape(1, -1) if scalar else raw
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(model.feature_names)
        or not np.isfinite(matrix).all()
    ):
        raise ProtocolError("P-DCAPS ridge replay descriptor drifted.")
    standardized = (
        matrix - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
    result = np.ascontiguousarray(
        model.intercept
        + standardized @ np.asarray(model.coefficients, dtype=np.float64),
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ProtocolError("P-DCAPS ridge replay produced a nonfinite result.")
    if scalar:
        return float(result[0])
    result.setflags(write=False)
    return result


__all__ = (
    "PINV_RCOND",
    "SCALE_FLOOR",
    "fit_weighted_ridge",
    "predict_weighted_ridge",
)
