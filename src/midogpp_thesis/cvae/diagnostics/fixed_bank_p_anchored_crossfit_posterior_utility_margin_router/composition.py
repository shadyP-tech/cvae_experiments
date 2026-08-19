"""Branch-disjoint P-anchored composition from frozen PUMR decisions."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import directional_candidate
from .constants import DIRECTION_IDS, PORTFOLIO_METHOD_ID
from .contracts import EndpointCasePrediction
from .selection import select_directional_actions
from .utility_contracts import (
    ComposedCasePrediction,
    MarginCalibration,
    PosteriorUtilityPrediction,
    UtilityDescriptor,
)


def compose_case_probabilities(
    endpoint: EndpointCasePrediction,
    descriptors: Sequence[UtilityDescriptor],
    utility_predictions: Sequence[PosteriorUtilityPrediction],
    calibration: MarginCalibration,
    *,
    policy_id: str,
) -> ComposedCasePrediction:
    rows = tuple(descriptors)
    decisions = select_directional_actions(
        rows,
        utility_predictions,
        calibration,
        policy_id=policy_id,
    )
    portfolio = np.asarray(
        endpoint.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    output = portfolio.copy()
    occupied = np.zeros(len(output), dtype=bool)
    counts: list[tuple[str, int]] = []
    for decision in decisions:
        if decision.selected_alternative == PORTFOLIO_METHOD_ID:
            mask = np.zeros(len(output), dtype=bool)
        else:
            candidate, mask = directional_candidate(
                endpoint, decision.selected_alternative, decision.direction
            )
            if np.any(occupied & mask):
                raise ProtocolError("PUMR directional actions overlap.")
            output[mask] = candidate[mask]
            occupied |= mask
        counts.append((decision.direction, int(np.sum(mask, dtype=np.int64))))
    if tuple(direction for direction, _count in counts) != DIRECTION_IDS:
        raise ProtocolError("PUMR directional composition order drifted.")
    return ComposedCasePrediction(
        endpoint.center,
        endpoint.case_id,
        policy_id,
        endpoint.sample_ids,
        tuple(float(value) for value in output),
        decisions,
        tuple(counts),
        endpoint.prediction_hash,
    )


__all__ = ("compose_case_probabilities",)
