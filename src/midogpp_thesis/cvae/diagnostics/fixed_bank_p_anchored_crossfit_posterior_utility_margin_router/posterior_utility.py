"""Analytic posterior expected utility for P-anchored directional actions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import directional_candidate
from .constants import (
    LOG_LOSS_CLIP_EPSILON,
    PORTFOLIO_METHOD_ID,
    ROBUST_MAD_MULTIPLIER,
    ROBUST_MAD_FLOOR,
    ROBUST_MAD_SCALE,
)
from .contracts import EndpointCasePrediction
from .posterior_contracts import RoutePosteriorEnsemble
from .utility_contracts import PosteriorUtilityPrediction, UtilityDescriptor


def score_posterior_utilities(
    endpoint: EndpointCasePrediction,
    descriptors: Sequence[UtilityDescriptor],
    ensemble: RoutePosteriorEnsemble,
) -> tuple[PosteriorUtilityPrediction, ...]:
    """Score all six actions without opening the held case's labels."""

    rows = tuple(descriptors)
    if (
        len(rows) != 6
        or len({row.key for row in rows}) != 6
        or endpoint.center != ensemble.target_center
        or endpoint.case_id != ensemble.held_case_id
        or endpoint.sample_ids != ensemble.held_sample_ids
        or any(
            row.target_center != endpoint.center
            or row.case_id != endpoint.case_id
            or row.endpoint_prediction_hash != endpoint.prediction_hash
            for row in rows
        )
    ):
        raise ProtocolError("PUMR posterior utility rectangle drifted.")

    portfolio = np.asarray(
        endpoint.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    results: list[PosteriorUtilityPrediction] = []
    for descriptor in rows:
        candidate, crossing = directional_candidate(
            endpoint, descriptor.alternative, descriptor.direction
        )
        crossing_ids = tuple(
            endpoint.sample_ids[int(index)] for index in np.flatnonzero(crossing)
        )
        if crossing_ids != descriptor.crossing_sample_ids:
            raise ProtocolError("PUMR posterior utility crossing identity drifted.")
        bacc_folds: list[float] = []
        brier_folds: list[float] = []
        log_folds: list[float] = []
        for probabilities in ensemble.held_natural_probabilities_by_fold:
            eta = np.asarray(probabilities, dtype=np.float64)
            n_positive = ensemble.support_n_positive + float(
                np.sum(eta, dtype=np.float64)
            )
            n_negative = ensemble.support_n_negative + float(
                np.sum(1.0 - eta, dtype=np.float64)
            )
            n_total = ensemble.support_row_count + len(eta)
            if min(n_positive, n_negative) <= 0.0 or n_total <= 0:
                raise ProtocolError("PUMR posterior utility denominator drifted.")
            positive_term = float(np.sum(eta[crossing], dtype=np.float64))
            negative_term = float(
                np.sum(1.0 - eta[crossing], dtype=np.float64)
            )
            sign = 1.0 if descriptor.direction == "zero_to_one" else -1.0
            bacc_folds.append(
                0.5
                * sign
                * (positive_term / n_positive - negative_term / n_negative)
            )
            p = portfolio[crossing]
            q = candidate[crossing]
            eta_crossing = eta[crossing]
            brier_folds.append(
                float(
                    np.sum(
                        q * q - p * p - 2.0 * eta_crossing * (q - p),
                        dtype=np.float64,
                    )
                    / n_total
                )
            )
            p_clip = np.clip(
                p, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
            )
            q_clip = np.clip(
                q, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
            )
            log_folds.append(
                float(
                    np.sum(
                        eta_crossing * np.log(p_clip / q_clip)
                        + (1.0 - eta_crossing)
                        * np.log((1.0 - p_clip) / (1.0 - q_clip)),
                        dtype=np.float64,
                    )
                    / n_total
                )
            )
        results.append(
            PosteriorUtilityPrediction(
                endpoint.center,
                endpoint.case_id,
                descriptor.alternative,
                descriptor.direction,
                ensemble.control_id,
                descriptor.crossing_count,
                tuple(bacc_folds),
                tuple(brier_folds),
                tuple(log_folds),
                _robust_lower(bacc_folds),
                _robust_upper(brier_folds),
                _robust_upper(log_folds),
                ensemble.oof_auc,
                ensemble.oof_brier_skill,
                ensemble.reliability_pass,
                descriptor.descriptor_hash,
                ensemble.ensemble_hash,
            )
        )
    output = tuple(sorted(results, key=lambda row: row.key))
    if len(output) != 6:
        raise ProtocolError("PUMR posterior utility output drifted.")
    return output


def _robust_lower(values: Sequence[float]) -> float:
    median, mad = _median_mad(values)
    return median - ROBUST_MAD_MULTIPLIER * ROBUST_MAD_SCALE * mad


def _robust_upper(values: Sequence[float]) -> float:
    median, mad = _median_mad(values)
    return median + ROBUST_MAD_MULTIPLIER * ROBUST_MAD_SCALE * mad


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ProtocolError("PUMR robust posterior utility input drifted.")
    median = float(np.median(array))
    mad = max(float(np.median(np.abs(array - median))), ROBUST_MAD_FLOOR)
    return median, mad


__all__ = ("score_posterior_utilities",)
