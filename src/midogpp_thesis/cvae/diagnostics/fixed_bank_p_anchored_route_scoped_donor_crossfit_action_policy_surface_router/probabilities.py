"""Canonical probability and posterior-utility kernels for P-DCAPS."""

from __future__ import annotations

import math

import numpy as np

from ...protocol import ProtocolError
from .contracts import FavorableUtility
from .identity import DIRECTIONS


HARD_THRESHOLD = np.float32(0.5)
LOG_CLIP_EPSILON = 1.0e-12


def canonical_probabilities(
    values: object,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    if (
        array.ndim != 1
        or not len(array)
        or (expected_length is not None and len(array) != int(expected_length))
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
    ):
        raise ProtocolError("P-DCAPS probability vector drifted.")
    array.setflags(write=False)
    return array


def directional_action(
    portfolio: object,
    alternative: object,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    p = canonical_probabilities(portfolio)
    alternative_values = canonical_probabilities(alternative, expected_length=len(p))
    if direction == DIRECTIONS[0]:
        crossing = (p < HARD_THRESHOLD) & (alternative_values >= HARD_THRESHOLD)
    elif direction == DIRECTIONS[1]:
        crossing = (p >= HARD_THRESHOLD) & (alternative_values < HARD_THRESHOLD)
    else:
        raise ProtocolError("P-DCAPS action direction drifted.")
    result = np.array(p, dtype=np.float32, copy=True, order="C")
    result[crossing] = alternative_values[crossing]
    result.setflags(write=False)
    crossing = np.ascontiguousarray(crossing, dtype=bool)
    crossing.setflags(write=False)
    return result, crossing


def expected_favorable_utility(
    portfolio: object,
    action: object,
    posterior_eta: object,
    *,
    support_n_positive: float,
    support_n_negative: float,
    support_row_count: int,
    crossing_mask: object | None = None,
) -> FavorableUtility:
    """Compute label-free expected downstream deltas versus exact P."""

    p = canonical_probabilities(portfolio)
    a = canonical_probabilities(action, expected_length=len(p))
    eta = np.ascontiguousarray(posterior_eta, dtype=np.float64)
    if (
        eta.shape != p.shape
        or not np.isfinite(eta).all()
        or np.any((eta < 0.0) | (eta > 1.0))
        or not math.isfinite(float(support_n_positive))
        or not math.isfinite(float(support_n_negative))
        or float(support_n_positive) < 0.0
        or float(support_n_negative) < 0.0
        or int(support_row_count) < 0
        or abs(
            float(support_n_positive)
            + float(support_n_negative)
            - int(support_row_count)
        )
        > 1.0e-8
    ):
        raise ProtocolError("P-DCAPS posterior utility inputs drifted.")
    actual_crossing = (p >= HARD_THRESHOLD) != (a >= HARD_THRESHOLD)
    if crossing_mask is None:
        crossing = actual_crossing
    else:
        crossing = np.asarray(crossing_mask, dtype=bool)
        if crossing.shape != p.shape or not np.array_equal(crossing, actual_crossing):
            raise ProtocolError("P-DCAPS crossing mask drifted.")
    if not np.any(crossing):
        return FavorableUtility.zeros()

    n_positive = float(support_n_positive) + float(np.sum(eta, dtype=np.float64))
    n_negative = float(support_n_negative) + float(
        np.sum(1.0 - eta, dtype=np.float64)
    )
    n_total = int(support_row_count) + len(eta)
    if n_positive <= 0.0 or n_negative <= 0.0 or n_total <= 0:
        raise ProtocolError("P-DCAPS posterior utility denominators are empty.")

    old_hard = (p >= HARD_THRESHOLD).astype(np.float64)
    new_hard = (a >= HARD_THRESHOLD).astype(np.float64)
    delta = new_hard - old_hard
    bacc = 0.5 * float(
        np.sum(
            delta[crossing]
            * (
                eta[crossing] / n_positive
                - (1.0 - eta[crossing]) / n_negative
            ),
            dtype=np.float64,
        )
    )
    p64 = p.astype(np.float64, copy=False)[crossing]
    a64 = a.astype(np.float64, copy=False)[crossing]
    eta_crossing = eta[crossing]
    brier = float(
        np.sum(
            p64 * p64 - a64 * a64 - 2.0 * eta_crossing * (p64 - a64),
            dtype=np.float64,
        )
        / n_total
    )
    p_clip = np.clip(p64, LOG_CLIP_EPSILON, 1.0 - LOG_CLIP_EPSILON)
    a_clip = np.clip(a64, LOG_CLIP_EPSILON, 1.0 - LOG_CLIP_EPSILON)
    log_gain = float(
        np.sum(
            eta_crossing * np.log(a_clip / p_clip)
            + (1.0 - eta_crossing)
            * np.log((1.0 - a_clip) / (1.0 - p_clip)),
            dtype=np.float64,
        )
        / n_total
    )
    return FavorableUtility(bacc, brier, log_gain)


__all__ = (
    "HARD_THRESHOLD",
    "LOG_CLIP_EPSILON",
    "canonical_probabilities",
    "directional_action",
    "expected_favorable_utility",
)
