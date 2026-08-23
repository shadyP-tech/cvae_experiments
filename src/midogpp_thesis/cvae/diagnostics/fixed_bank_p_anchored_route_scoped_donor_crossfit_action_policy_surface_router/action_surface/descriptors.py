"""Fixed, label-free P-DCAPS action-response descriptors."""

from __future__ import annotations

import numpy as np

from ....protocol import ProtocolError
from ..identity import ACTION_STRATA, METRICS
from .contracts import ActionPrediction


def action_feature_names(metric: str) -> tuple[str, ...]:
    """Return the frozen 14-column feature identity for one response metric."""

    metric_id = str(metric)
    if metric_id not in METRICS:
        raise ProtocolError("P-DCAPS action descriptor metric drifted.")
    indicators = tuple(
        f"stratum__{family}__{direction}" for family, direction in ACTION_STRATA
    )
    interactions = tuple(
        f"predicted_{metric_id}_x_stratum__{family}__{direction}"
        for family, direction in ACTION_STRATA
    )
    return (
        f"predicted_favorable_{metric_id}",
        "crossing_fraction",
        *indicators,
        *interactions,
    )


def predicted_metric_value(prediction: ActionPrediction, metric: str) -> float:
    metric_id = str(metric)
    if metric_id == "bacc":
        return prediction.predicted_utility.bacc_gain
    if metric_id == "brier":
        return prediction.predicted_utility.brier_gain
    if metric_id == "log":
        return prediction.predicted_utility.log_gain
    raise ProtocolError("P-DCAPS action descriptor metric drifted.")


def build_action_descriptor(prediction: ActionPrediction, metric: str) -> np.ndarray:
    """Build a contiguous float64 descriptor from label-free sealed values."""

    predicted = float(predicted_metric_value(prediction, metric))
    stratum = prediction.key.stratum
    if stratum not in ACTION_STRATA:
        raise ProtocolError("P-DCAPS action descriptor stratum drifted.")
    indicator = np.asarray(
        [1.0 if candidate == stratum else 0.0 for candidate in ACTION_STRATA],
        dtype=np.float64,
    )
    values = np.ascontiguousarray(
        np.concatenate(
            (
                np.asarray([predicted, prediction.crossing_fraction], dtype=np.float64),
                indicator,
                predicted * indicator,
            )
        ),
        dtype=np.float64,
    )
    if values.shape != (len(action_feature_names(metric)),) or not np.isfinite(values).all():
        raise ProtocolError("P-DCAPS action descriptor vector drifted.")
    values.setflags(write=False)
    return values


def build_action_descriptor_matrix(
    predictions: tuple[ActionPrediction, ...] | list[ActionPrediction],
    metric: str,
) -> np.ndarray:
    rows = tuple(predictions)
    if not rows:
        raise ProtocolError("P-DCAPS action descriptor matrix is empty.")
    matrix = np.ascontiguousarray(
        np.stack([build_action_descriptor(row, metric) for row in rows]),
        dtype=np.float64,
    )
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ProtocolError("P-DCAPS action descriptor matrix drifted.")
    matrix.setflags(write=False)
    return matrix


__all__ = (
    "action_feature_names",
    "build_action_descriptor",
    "build_action_descriptor_matrix",
    "predicted_metric_value",
)
