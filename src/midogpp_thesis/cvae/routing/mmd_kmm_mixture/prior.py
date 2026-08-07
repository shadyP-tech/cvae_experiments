"""Source-only responsibility handling for unlabeled target support."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import expit

from ...protocol import ProtocolError
from .config import PriorControlConfig
from .contracts import (
    MMDKMMProtocol,
    SourceOnlyPriorPrediction,
    readonly_probabilities,
)


def prepare_source_only_responsibilities(
    raw_probabilities: object,
    *,
    protocol: MMDKMMProtocol,
    prior_model_hash: str,
    prior_fit_pool_hash: str,
    config: PriorControlConfig,
) -> SourceOnlyPriorPrediction:
    """Temperature-scale and clip externally frozen source-only predictions.

    The classifier is not fitted here.  ``raw_probabilities`` must come from one
    pooled, target-excluded, equal-source/class model (or one pre-frozen
    all-replica ensemble), identified by ``prior_model_hash``.
    """

    probabilities = readonly_probabilities(raw_probabilities)
    positive = np.asarray(probabilities[:, 1], dtype=np.float64)
    logits = np.log(positive) - np.log1p(-positive)
    scaled = expit(logits / float(config.temperature))
    clipped = np.clip(
        scaled,
        float(config.probability_clip),
        1.0 - float(config.probability_clip),
    )
    output = np.column_stack((1.0 - clipped, clipped))
    output.setflags(write=False)
    model_hash = str(prior_model_hash)
    pool_hash = str(prior_fit_pool_hash)
    if not model_hash or not pool_hash:
        raise ProtocolError("Source-only responsibility provenance is incomplete.")
    return SourceOnlyPriorPrediction(
        target_center=protocol.target_center,
        candidate_sources=protocol.candidate_sources,
        common_frame_hash=protocol.common_frame_hash,
        probabilities=output,
        prior_model_hash=model_hash,
        prior_fit_pool_hash=pool_hash,
        temperature=float(config.temperature),
        probability_clip=float(config.probability_clip),
        sensitivity_positive_priors=config.sensitivity_positive_priors,
        reference_positive_prior=float(config.reference_positive_prior),
        sensitivity_positive_prior=None,
        fit_role=config.fit_role,
        target_labels_used=False,
    )


def shift_source_only_prior_prediction(
    prediction: SourceOnlyPriorPrediction,
    *,
    positive_prior: float,
    config: PriorControlConfig,
) -> SourceOnlyPriorPrediction:
    """Create one provenance-bound no-label prior-sensitivity prediction."""

    if (
        prediction.sensitivity_positive_prior is not None
        or not any(
            np.isclose(
                float(positive_prior),
                float(value),
                rtol=0.0,
                atol=0.0,
            )
            for value in config.sensitivity_positive_priors
        )
        or prediction.fit_role != config.fit_role
        or not np.isclose(
            prediction.temperature,
            float(config.temperature),
            rtol=0.0,
            atol=0.0,
        )
        or not np.isclose(
            prediction.probability_clip,
            float(config.probability_clip),
            rtol=0.0,
            atol=0.0,
        )
        or not np.isclose(
            prediction.reference_positive_prior,
            float(config.reference_positive_prior),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ProtocolError(
            "Prior-sensitivity prediction crossed a frozen prior state."
        )
    shifted = shift_binary_prior(
        prediction.probabilities,
        positive_prior=positive_prior,
        reference_positive_prior=config.reference_positive_prior,
        probability_clip=config.probability_clip,
    )
    return replace(
        prediction,
        probabilities=shifted,
        sensitivity_positive_prior=float(positive_prior),
    )


def shift_binary_prior(
    probabilities: object,
    *,
    positive_prior: float,
    reference_positive_prior: float,
    probability_clip: float,
) -> np.ndarray:
    """Apply a label-free binary prior-odds sensitivity shift."""

    values = readonly_probabilities(probabilities)
    target_prior = float(positive_prior)
    reference_prior = float(reference_positive_prior)
    clip = float(probability_clip)
    if (
        not 0.0 < target_prior < 1.0
        or not 0.0 < reference_prior < 1.0
        or not 0.0 < clip < 0.5
    ):
        raise ProtocolError("Binary prior-sensitivity parameters are invalid.")
    odds = values[:, 1] / values[:, 0]
    odds_multiplier = (target_prior / (1.0 - target_prior)) / (
        reference_prior / (1.0 - reference_prior)
    )
    shifted_positive = odds * odds_multiplier / (1.0 + odds * odds_multiplier)
    shifted_positive = np.clip(shifted_positive, clip, 1.0 - clip)
    shifted = np.column_stack((1.0 - shifted_positive, shifted_positive))
    shifted.setflags(write=False)
    return shifted


__all__ = (
    "prepare_source_only_responsibilities",
    "shift_source_only_prior_prediction",
    "shift_binary_prior",
)
