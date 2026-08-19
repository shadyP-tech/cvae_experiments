"""Label-free, action-specific descriptors for P-versus-endpoint crossings."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BASELINE_METHOD_ID,
    CROSSING_FEATURE_NAMES,
    HARD_THRESHOLD,
    IDENTIFICATION_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    ROBUST_METHOD_ID,
)
from .contracts import EndpointCasePrediction
from .crossing_contracts import CrossingDescriptor


def build_crossing_descriptors(
    prediction: EndpointCasePrediction,
) -> tuple[CrossingDescriptor, ...]:
    """Describe every actionable crossing without selecting one endpoint first."""

    portfolio = np.asarray(
        prediction.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    if portfolio.shape != (len(prediction.sample_ids),):
        raise ProtocolError("PDCB portfolio probability topology drifted.")
    portfolio_hard = portfolio >= HARD_THRESHOLD
    rows: list[CrossingDescriptor] = []
    for alternative in ALTERNATIVE_METHOD_IDS:
        candidate = np.asarray(prediction.probabilities[alternative], dtype=np.float64)
        candidate_hard = candidate >= HARD_THRESHOLD
        crossing = portfolio_hard != candidate_hard
        indices = np.flatnonzero(crossing)
        if not len(indices):
            continue
        up = (~portfolio_hard) & candidate_hard
        down = portfolio_hard & (~candidate_hard)
        crossing_rate = float(np.mean(crossing, dtype=np.float64))
        imbalance = float(np.mean(up, dtype=np.float64) - np.mean(down, dtype=np.float64))
        for index in indices:
            direction = "zero_to_one" if bool(up[index]) else "one_to_zero"
            p_value = float(portfolio[index])
            a_value = float(candidate[index])
            shift = a_value - p_value
            values = (
                p_value,
                a_value,
                abs(p_value - HARD_THRESHOLD),
                abs(a_value - HARD_THRESHOLD),
                shift,
                abs(shift),
                crossing_rate,
                imbalance,
                float(direction == "zero_to_one"),
                float(alternative == BASELINE_METHOD_ID),
                float(alternative == IDENTIFICATION_METHOD_ID),
                float(alternative == ROBUST_METHOD_ID),
            )
            rows.append(
                CrossingDescriptor(
                    prediction.center,
                    prediction.case_id,
                    prediction.sample_ids[int(index)],
                    alternative,
                    direction,
                    CROSSING_FEATURE_NAMES,
                    values,
                    prediction.prediction_hash,
                )
            )
    result = tuple(sorted(rows, key=lambda row: row.key))
    if len({row.key for row in result}) != len(result):
        raise ProtocolError("PDCB crossing descriptors are duplicated.")
    return result


def build_crossing_descriptor_surface(
    predictions: Sequence[EndpointCasePrediction],
) -> tuple[CrossingDescriptor, ...]:
    rows = tuple(
        descriptor
        for prediction in predictions
        for descriptor in build_crossing_descriptors(prediction)
    )
    if len({row.key for row in rows}) != len(rows):
        raise ProtocolError("PDCB crossing descriptor surface is duplicated.")
    return rows


__all__ = ("build_crossing_descriptor_surface", "build_crossing_descriptors")
