"""Complete label-free case/action/direction utility descriptors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import directional_candidate
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    DIRECTION_IDS,
    HARD_THRESHOLD,
    PORTFOLIO_METHOD_ID,
    UTILITY_FEATURE_NAMES,
)
from .contracts import EndpointCasePrediction
from .utility_contracts import UtilityDescriptor


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values, dtype=np.float64)) if len(values) else 0.0


def _entropy(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    clipped = np.clip(values, 1.0e-12, 1.0 - 1.0e-12)
    return _mean(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped)))


def build_utility_descriptors(
    prediction: EndpointCasePrediction,
) -> tuple[UtilityDescriptor, ...]:
    """Emit exactly six rows, retaining structural no-crossing candidates."""

    portfolio = np.asarray(
        prediction.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    p_hard = portfolio >= HARD_THRESHOLD
    rows: list[UtilityDescriptor] = []
    for alternative in ALTERNATIVE_METHOD_IDS:
        candidate = np.asarray(prediction.probabilities[alternative], dtype=np.float64)
        for direction in DIRECTION_IDS:
            _composed, crossing = directional_candidate(
                prediction, alternative, direction
            )
            branch = ~p_hard if direction == "zero_to_one" else p_hard
            crossing_ids = tuple(
                prediction.sample_ids[int(index)]
                for index in np.flatnonzero(crossing)
            )
            values = (
                float(np.log1p(len(portfolio))),
                _mean(branch.astype(np.float64)),
                _mean(crossing.astype(np.float64)),
                _mean(portfolio[branch]),
                _mean(candidate[branch]),
                _mean(np.abs(portfolio[branch] - HARD_THRESHOLD)),
                _mean(np.abs(candidate[branch] - HARD_THRESHOLD)),
                _mean((candidate - portfolio)[crossing]),
                _mean(np.abs(candidate - portfolio)[crossing]),
                _entropy(portfolio[crossing]),
                _entropy(candidate[crossing]),
                float(np.log1p(len(crossing_ids))),
            )
            rows.append(
                UtilityDescriptor(
                    prediction.center,
                    prediction.case_id,
                    alternative,
                    direction,
                    UTILITY_FEATURE_NAMES,
                    values,
                    crossing_ids,
                    prediction.prediction_hash,
                )
            )
    result = tuple(sorted(rows, key=lambda row: row.key))
    if len(result) != len(ALTERNATIVE_METHOD_IDS) * len(DIRECTION_IDS):
        raise ProtocolError("PCSI-PARC utility descriptor rectangle drifted.")
    return result


def build_utility_descriptor_surface(
    predictions: Sequence[EndpointCasePrediction],
) -> tuple[UtilityDescriptor, ...]:
    rows = tuple(
        descriptor
        for prediction in predictions
        for descriptor in build_utility_descriptors(prediction)
    )
    if len({row.key for row in rows}) != len(rows):
        raise ProtocolError("PCSI-PARC utility descriptor surface is duplicated.")
    return rows


__all__ = ("build_utility_descriptor_surface", "build_utility_descriptors")
