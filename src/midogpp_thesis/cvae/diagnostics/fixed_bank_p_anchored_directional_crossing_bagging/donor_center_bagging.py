"""Complete donor-center deletion aggregation without independence claims."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import CROSSING_HELPFUL_THRESHOLD
from .crossing_contracts import (
    CrossingDescriptor,
    CrossingHelpfulnessModel,
    CrossingPrediction,
)
from .crossing_model import predict_crossing_helpfulness


def predict_with_donor_center_bagging(
    descriptor: CrossingDescriptor,
    *,
    full_model: CrossingHelpfulnessModel,
    delete_models: Mapping[str, CrossingHelpfulnessModel],
) -> CrossingPrediction:
    if (
        descriptor.target_center != full_model.outer_target_center
        or tuple(delete_models) != full_model.training_centers
        or any(
            model.outer_target_center != descriptor.target_center
            or model.training_centers
            != tuple(
                center
                for center in full_model.training_centers
                if center != deleted
            )
            for deleted, model in delete_models.items()
        )
    ):
        raise ProtocolError("PDCB donor deletion topology drifted.")
    full_probability = predict_crossing_helpfulness(full_model, descriptor)
    deletion_probabilities = tuple(
        (
            deleted,
            predict_crossing_helpfulness(model, descriptor),
        )
        for deleted, model in delete_models.items()
    )
    values = np.asarray(
        [value for _deleted, value in deletion_probabilities], dtype=np.float64
    )
    robust = float(np.median(values))
    positive_fraction = float(
        np.mean(values > CROSSING_HELPFUL_THRESHOLD, dtype=np.float64)
    )
    raw_weight = float(max(0.0, 2.0 * robust - 1.0) * positive_fraction)
    return CrossingPrediction(
        descriptor.descriptor_hash,
        full_probability,
        deletion_probabilities,
        robust,
        positive_fraction,
        raw_weight,
        (
            full_model.model_hash,
            *(model.model_hash for model in delete_models.values()),
        ),
    )


def predict_crossing_surface(
    descriptors: Sequence[CrossingDescriptor],
    *,
    full_model: CrossingHelpfulnessModel,
    delete_models: Mapping[str, CrossingHelpfulnessModel],
) -> tuple[CrossingPrediction, ...]:
    return tuple(
        predict_with_donor_center_bagging(
            descriptor,
            full_model=full_model,
            delete_models=delete_models,
        )
        for descriptor in descriptors
    )


__all__ = ("predict_crossing_surface", "predict_with_donor_center_bagging")
