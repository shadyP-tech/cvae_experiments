"""Cross-fitted target-local balanced-accuracy influence score."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import DIRECTION_IDS
from .sample_influence_contracts import (
    InfluencePrediction,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)
from .utility_contracts import UtilityDescriptor


def score_sample_influences(
    descriptors: Sequence[UtilityDescriptor],
    *,
    posterior: TargetLocalPosteriorPrediction,
    model: TargetLocalPosteriorModel,
) -> tuple[InfluencePrediction, ...]:
    """Score each P-to-candidate crossing set under the frozen H-c posterior."""

    rows = tuple(descriptors)
    if (
        len(rows) != 6
        or len({(row.alternative, row.direction) for row in rows}) != 6
        or any(
            row.target_center != posterior.target_center
            or row.case_id != posterior.case_id
            for row in rows
        )
        or model.model_hash != posterior.model_hash
        or model.target_center != posterior.target_center
        or model.held_case_id != posterior.case_id
    ):
        raise ProtocolError("PCSI influence rectangle/model binding drifted.")
    eta = dict(zip(posterior.sample_ids, posterior.natural_probabilities, strict=True))
    output: list[InfluencePrediction] = []
    for descriptor in rows:
        if any(sample_id not in eta for sample_id in descriptor.crossing_sample_ids):
            raise ProtocolError("PCSI influence crossing escaped held-case predictions.")
        sign = 1.0 if descriptor.direction == DIRECTION_IDS[0] else -1.0
        score = 0.5 * sum(
            sign
            * (
                eta[sample_id] / model.support_n_positive
                - (1.0 - eta[sample_id]) / model.support_n_negative
            )
            for sample_id in descriptor.crossing_sample_ids
        )
        output.append(
            InfluencePrediction(
                descriptor.descriptor_hash,
                descriptor.target_center,
                descriptor.case_id,
                descriptor.alternative,
                descriptor.direction,
                descriptor.crossing_count,
                float(score),
                posterior.prediction_hash,
            )
        )
    return tuple(output)


__all__ = ("score_sample_influences",)
