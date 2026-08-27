"""Exact per-sample BACC and proper-loss influence kernels."""

from __future__ import annotations

import math

import numpy as np

from ..action_geometry import HARD_THRESHOLD, canonical_probabilities
from ..protocol import ProtocolError
from .contracts import ActionMetricVector


LOG_CLIP_EPSILON = 1.0e-12


def _soft_labels(values: object, *, length: int, strict_binary: bool) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if (
        array.shape != (length,)
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
        or (strict_binary and not np.all((array == 0.0) | (array == 1.0)))
    ):
        raise ProtocolError("SCALE-BP influence label/probability vector drifted.")
    return array


def sample_metric_influences(
    portfolio: object,
    action: object,
    positive_probability: object,
    *,
    positive_denominator: float,
    negative_denominator: float,
    row_denominator: float,
) -> np.ndarray:
    """Return sample contributions in `(BACC gain, Brier delta, log delta)` order."""

    baseline = canonical_probabilities(portfolio)
    candidate = canonical_probabilities(action, expected_length=len(baseline))
    eta = _soft_labels(positive_probability, length=len(baseline), strict_binary=False)
    n_positive = float(positive_denominator)
    n_negative = float(negative_denominator)
    n_rows = float(row_denominator)
    if (
        not all(math.isfinite(value) for value in (n_positive, n_negative, n_rows))
        or n_positive <= 0.0
        or n_negative <= 0.0
        or n_rows <= 0.0
        or abs(n_positive + n_negative - n_rows) > 1.0e-7
    ):
        raise ProtocolError("SCALE-BP influence denominator geometry drifted.")

    baseline64 = baseline.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    hard_delta = (candidate >= HARD_THRESHOLD).astype(np.float64) - (
        baseline >= HARD_THRESHOLD
    ).astype(np.float64)
    bacc = 0.5 * hard_delta * (
        eta / n_positive - (1.0 - eta) / n_negative
    )
    brier = (
        candidate64 * candidate64
        - baseline64 * baseline64
        - 2.0 * eta * (candidate64 - baseline64)
    ) / n_rows
    p = np.clip(baseline64, LOG_CLIP_EPSILON, 1.0 - LOG_CLIP_EPSILON)
    a = np.clip(candidate64, LOG_CLIP_EPSILON, 1.0 - LOG_CLIP_EPSILON)
    log_delta = -(
        eta * np.log(a / p) + (1.0 - eta) * np.log((1.0 - a) / (1.0 - p))
    ) / n_rows
    result = np.ascontiguousarray(np.column_stack((bacc, brier, log_delta)), dtype=np.float64)
    if result.shape != (len(baseline), 3) or not np.isfinite(result).all():
        raise ProtocolError("SCALE-BP influence kernel produced nonfinite output.")
    result.setflags(write=False)
    return result


def aggregate_sample_influences(values: object) -> ActionMetricVector:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ProtocolError("SCALE-BP sample-influence matrix drifted.")
    return ActionMetricVector.from_iterable(
        np.sum(array, axis=0, dtype=np.float64)
    )


def expected_action_metrics(
    portfolio: object,
    action: object,
    posterior_eta: object,
    *,
    support_positive_count: float,
    support_negative_count: float,
    support_row_count: int,
) -> ActionMetricVector:
    """Compute label-free expected action value using support plus held eta."""

    baseline = canonical_probabilities(portfolio)
    eta = _soft_labels(posterior_eta, length=len(baseline), strict_binary=False)
    support_positive = float(support_positive_count)
    support_negative = float(support_negative_count)
    support_rows = int(support_row_count)
    if (
        not math.isfinite(support_positive)
        or not math.isfinite(support_negative)
        or support_positive < 0.0
        or support_negative < 0.0
        or support_rows < 0
        or abs(support_positive + support_negative - support_rows) > 1.0e-8
    ):
        raise ProtocolError("SCALE-BP expected-metric support counts drifted.")
    positive = support_positive + float(np.sum(eta, dtype=np.float64))
    negative = support_negative + float(np.sum(1.0 - eta, dtype=np.float64))
    total = support_rows + len(baseline)
    return aggregate_sample_influences(
        sample_metric_influences(
            baseline,
            action,
            eta,
            positive_denominator=positive,
            negative_denominator=negative,
            row_denominator=total,
        )
    )


def realized_action_metrics(
    portfolio: object,
    action: object,
    labels: object,
    *,
    positive_denominator: int,
    negative_denominator: int,
    row_denominator: int,
) -> ActionMetricVector:
    """Reduce support-only labels to an aggregate action response."""

    baseline = canonical_probabilities(portfolio)
    truth = _soft_labels(labels, length=len(baseline), strict_binary=True)
    positive = int(positive_denominator)
    negative = int(negative_denominator)
    total = int(row_denominator)
    if (
        positive <= 0
        or negative <= 0
        or total != positive + negative
        or len(truth) > total
    ):
        raise ProtocolError("SCALE-BP realized-metric denominator contract drifted.")
    return aggregate_sample_influences(
        sample_metric_influences(
            baseline,
            action,
            truth,
            positive_denominator=positive,
            negative_denominator=negative,
            row_denominator=total,
        )
    )


__all__ = (
    "LOG_CLIP_EPSILON",
    "aggregate_sample_influences",
    "expected_action_metrics",
    "realized_action_metrics",
    "sample_metric_influences",
)
