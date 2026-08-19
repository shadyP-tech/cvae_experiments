"""P-protected convex probability composition from crossing evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    FULL_ONLY_METHOD_ID,
    PORTFOLIO_METHOD_ID,
)
from .contracts import EndpointCasePrediction
from .crossing_contracts import (
    ComposedCasePrediction,
    CrossingDescriptor,
    CrossingPrediction,
)


def compose_case_probabilities(
    endpoint: EndpointCasePrediction,
    descriptors: Sequence[CrossingDescriptor],
    crossing_predictions: Sequence[CrossingPrediction],
    *,
    policy_id: str,
) -> ComposedCasePrediction:
    rows = tuple(descriptors)
    predictions = tuple(crossing_predictions)
    by_hash = {row.descriptor_hash: row for row in predictions}
    if (
        len(by_hash) != len(predictions)
        or {row.descriptor_hash for row in rows} != set(by_hash)
        or any(
            row.target_center != endpoint.center
            or row.case_id != endpoint.case_id
            or row.endpoint_prediction_hash != endpoint.prediction_hash
            for row in rows
        )
    ):
        raise ProtocolError("PDCB crossing composition inputs drifted.")
    sample_index = {
        sample_id: index for index, sample_id in enumerate(endpoint.sample_ids)
    }
    raw_by_sample: dict[int, dict[str, float]] = defaultdict(dict)
    for descriptor in rows:
        prediction = by_hash[descriptor.descriptor_hash]
        raw = (
            max(0.0, 2.0 * prediction.full_probability - 1.0)
            if policy_id == FULL_ONLY_METHOD_ID
            else prediction.raw_weight
        )
        raw_by_sample[sample_index[descriptor.sample_id]][descriptor.alternative] = raw

    portfolio = np.asarray(
        endpoint.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    output = portfolio.copy()
    p_weights = np.ones(len(output), dtype=np.float64)
    alternative_weights = {
        method: np.zeros(len(output), dtype=np.float64)
        for method in ALTERNATIVE_METHOD_IDS
    }
    for sample_index_value, raw_weights in raw_by_sample.items():
        total = float(sum(raw_weights.values()))
        denominator = 1.0 + total
        p_weight = 1.0 / denominator
        value = p_weight * portfolio[sample_index_value]
        for alternative in ALTERNATIVE_METHOD_IDS:
            alternative_weight = raw_weights.get(alternative, 0.0) / denominator
            alternative_weights[alternative][sample_index_value] = alternative_weight
            value += alternative_weight * float(
                endpoint.probabilities[alternative][sample_index_value]
            )
        output[sample_index_value] = value
        p_weights[sample_index_value] = p_weight
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ProtocolError("PDCB convex composition escaped probability bounds.")
    reconstructed = p_weights * portfolio
    for alternative in ALTERNATIVE_METHOD_IDS:
        reconstructed += alternative_weights[alternative] * np.asarray(
            endpoint.probabilities[alternative], dtype=np.float64
        )
    residual = float(np.max(np.abs(output - reconstructed), initial=0.0))
    return ComposedCasePrediction(
        endpoint.center,
        endpoint.case_id,
        policy_id,
        endpoint.sample_ids,
        tuple(float(value) for value in output),
        tuple(float(value) for value in p_weights),
        tuple(
            (
                alternative,
                tuple(float(value) for value in alternative_weights[alternative]),
            )
            for alternative in ALTERNATIVE_METHOD_IDS
        ),
        tuple(
            (
                alternative,
                float(np.mean(alternative_weights[alternative], dtype=np.float64)),
            )
            for alternative in ALTERNATIVE_METHOD_IDS
        ),
        tuple(by_hash[row.descriptor_hash].prediction_hash for row in rows),
        endpoint.prediction_hash,
        residual,
    )


__all__ = ("compose_case_probabilities",)
