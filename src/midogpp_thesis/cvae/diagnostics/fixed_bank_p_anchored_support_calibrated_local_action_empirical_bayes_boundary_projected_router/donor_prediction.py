"""Pure deterministic donor-prior prediction replay."""

from __future__ import annotations

import numpy as np

from .donor_contracts import DonorPriorModel, DonorPriorPrediction
from .identity import ACTION_IDS
from .influence.contracts import ActionDescriptor, ActionMetricVector
from .protocol import ProtocolError
from .replay_scope import DonorScope


def predict_metric_values(
    descriptor: ActionDescriptor,
    *,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Replay the frozen cell-plus-standardized-feature linear design."""

    cell = f"{descriptor.family}::{descriptor.direction}"
    design = np.zeros(len(ACTION_IDS) + len(feature_mean), dtype=np.float64)
    design[ACTION_IDS.index(cell)] = 1.0
    design[len(ACTION_IDS) :] = (
        np.asarray(descriptor.values, dtype=np.float64) - feature_mean
    ) / feature_scale
    result = coefficients @ design
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ProtocolError("SCALE-BP donor prior prediction is invalid.")
    return result


def predict_donor_prior(
    model: DonorPriorModel,
    descriptor: ActionDescriptor,
    *,
    scope: DonorScope,
) -> DonorPriorPrediction:
    if (
        descriptor.feature_names != model.feature_names
        or model.scope_hash != scope.scope_hash
        or model.fit_role != scope.fit_role
        or model.held_center != scope.prediction_center
        or descriptor.case_id != scope.held_case_id
    ):
        raise ProtocolError("SCALE-BP donor prediction feature schema drifted.")
    values = predict_metric_values(
        descriptor,
        feature_mean=np.asarray(model.feature_mean, dtype=np.float64),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float64),
        coefficients=np.asarray(model.coefficients, dtype=np.float64),
    )
    return DonorPriorPrediction(
        descriptor.descriptor_hash,
        ActionMetricVector.from_iterable(values),
        model.between_center_standard_error,
        model.model_hash,
        scope.scope_hash,
        scope.fit_role,
    )


__all__ = ("predict_donor_prior", "predict_metric_values")
