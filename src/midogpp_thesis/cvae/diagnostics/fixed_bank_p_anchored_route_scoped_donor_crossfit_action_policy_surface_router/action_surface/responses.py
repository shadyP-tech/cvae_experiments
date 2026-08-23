"""Scoped pseudo-action responses using fixed whole-center denominators."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ..label_firewall import PseudoResponseLabelCapability
from .contracts import ActionPrediction, ActionResponse
from ..contracts import FavorableUtility


LOG_CLIP_EPSILON = 1.0e-12


def canonical_probabilities(value: object, *, expected_length: int | None = None) -> np.ndarray:
    """Return a read-only contiguous float32 probability vector."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if (
        array.ndim != 1
        or array.size <= 0
        or (expected_length is not None and len(array) != int(expected_length))
        or not np.isfinite(array).all()
        or bool(np.any(array < np.float32(0.0)))
        or bool(np.any(array > np.float32(1.0)))
    ):
        raise ProtocolError("P-DCAPS probability vector drifted.")
    array.setflags(write=False)
    return array


def probability_sha256(value: object) -> str:
    """Hash canonical float32 bytes with explicit dtype and shape framing."""

    array = canonical_probabilities(value)
    digest = hashlib.sha256()
    digest.update(b"pdcaps_probability_float32_v1\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def realized_favorable_utility(
    baseline_probabilities: object,
    action_probabilities: object,
    labels: Sequence[int],
    *,
    positive_denominator: int,
    negative_denominator: int,
    row_denominator: int,
    log_clip_epsilon: float = LOG_CLIP_EPSILON,
) -> FavorableUtility:
    """Compute one case contribution against exact P with frozen denominators."""

    baseline = canonical_probabilities(baseline_probabilities)
    action = canonical_probabilities(action_probabilities, expected_length=len(baseline))
    truth = np.ascontiguousarray(np.asarray(tuple(labels), dtype=np.int8))
    n_positive = int(positive_denominator)
    n_negative = int(negative_denominator)
    n_rows = int(row_denominator)
    epsilon = float(log_clip_epsilon)
    observed_positive = int(np.sum(truth, dtype=np.int64)) if truth.ndim == 1 else -1
    observed_negative = len(truth) - observed_positive
    if (
        truth.shape != baseline.shape
        or bool(np.any((truth != 0) & (truth != 1)))
        or n_positive <= 0
        or n_negative <= 0
        or n_rows != n_positive + n_negative
        or observed_positive > n_positive
        or observed_negative > n_negative
        or not math_is_strict_probability_clip(epsilon)
    ):
        raise ProtocolError("P-DCAPS response label/denominator contract drifted.")

    old_prediction = (baseline >= np.float32(0.5)).astype(np.float64)
    new_prediction = (action >= np.float32(0.5)).astype(np.float64)
    delta = new_prediction - old_prediction
    truth64 = truth.astype(np.float64)
    bacc_gain = 0.5 * float(
        np.sum(
            delta
            * (
                truth64 / n_positive
                - (1.0 - truth64) / n_negative
            ),
            dtype=np.float64,
        )
    )

    baseline64 = baseline.astype(np.float64, copy=False)
    action64 = action.astype(np.float64, copy=False)
    brier_gain = float(
        np.sum(
            baseline64 * baseline64
            - action64 * action64
            - 2.0 * truth64 * (baseline64 - action64),
            dtype=np.float64,
        )
        / n_rows
    )
    baseline_clip = np.clip(baseline64, epsilon, 1.0 - epsilon)
    action_clip = np.clip(action64, epsilon, 1.0 - epsilon)
    log_gain = float(
        np.sum(
            truth64 * np.log(action_clip / baseline_clip)
            + (1.0 - truth64)
            * np.log((1.0 - action_clip) / (1.0 - baseline_clip)),
            dtype=np.float64,
        )
        / n_rows
    )
    return FavorableUtility(bacc_gain, brier_gain, log_gain)


def math_is_strict_probability_clip(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0 and value < 0.5)


def build_action_response(
    prediction: ActionPrediction,
    *,
    baseline_probabilities: object,
    action_probabilities: object,
    label_capability: PseudoResponseLabelCapability,
    positive_denominator: int,
    negative_denominator: int,
    row_denominator: int,
) -> ActionResponse:
    """Open one scoped pseudo response after the complete action surface seal."""

    if (
        prediction.key.route_key.surface_role != "pseudo"
        or label_capability.route_key != prediction.key.route_key
    ):
        raise ProtocolError(
            "P-DCAPS cannot build an action-calibration response on target labels."
        )
    labels = label_capability.values
    baseline = canonical_probabilities(baseline_probabilities)
    action = canonical_probabilities(action_probabilities, expected_length=len(baseline))
    if probability_sha256(action) != prediction.key.probability_hash:
        raise ProtocolError("P-DCAPS action response probability lineage drifted.")
    utility = realized_favorable_utility(
        baseline,
        action,
        labels,
        positive_denominator=positive_denominator,
        negative_denominator=negative_denominator,
        row_denominator=row_denominator,
    )
    return ActionResponse(
        prediction.key,
        prediction.prediction_hash,
        utility,
        len(tuple(labels)),
        int(positive_denominator),
        int(negative_denominator),
        int(row_denominator),
        probability_sha256(baseline),
        label_capability.evaluation_row_hash,
    )


__all__ = (
    "LOG_CLIP_EPSILON",
    "build_action_response",
    "canonical_probabilities",
    "probability_sha256",
    "realized_favorable_utility",
)
