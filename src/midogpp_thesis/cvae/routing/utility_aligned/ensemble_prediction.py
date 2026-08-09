"""Target candidate prediction and seed-spread helpers."""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError
from .ensemble_feature_contracts import EnsembleFeatureSurface
from .ensemble_model_contracts import EnsembleUtilityModel
from .row_contracts import TARGET_CANDIDATE_COUNT


def predict_target_candidates(
    model: EnsembleUtilityModel, surface: EnsembleFeatureSurface
) -> dict[str, float]:
    if set(model.candidate_models) != set(surface.candidate_sources):
        raise ProtocolError("Ensemble target model candidate universe drifted.")
    try:
        feature_indices = tuple(
            surface.feature_names.index(name) for name in model.feature_names
        )
    except ValueError as exc:
        raise ProtocolError("Ensemble target feature columns do not match the model.") from exc
    predictions: dict[str, float] = {}
    for index, row in enumerate(surface.rows):
        source_model = model.candidate_models[row.candidate_source]
        values = np.asarray(
            surface.values[index : index + 1, list(feature_indices)], dtype=np.float64
        )
        predictions[row.candidate_source] = float(source_model.predict(values)[0])
    return predictions


def predict_target_candidate_distributions(
    model: EnsembleUtilityModel, surface: EnsembleFeatureSurface
) -> dict[str, tuple[float, float]]:
    if set(model.candidate_models) != set(surface.candidate_sources):
        raise ProtocolError("Ensemble target model candidate universe drifted.")
    try:
        feature_indices = tuple(
            surface.feature_names.index(name) for name in model.feature_names
        )
    except ValueError as exc:
        raise ProtocolError("Ensemble target feature columns do not match the model.") from exc
    predictions: dict[str, tuple[float, float]] = {}
    for index, row in enumerate(surface.rows):
        source_model = model.candidate_models[row.candidate_source]
        values = np.asarray(
            surface.values[index : index + 1, list(feature_indices)], dtype=np.float64
        )
        distribution = source_model.predict_with_uncertainty(
            values, include_residual_variance=True
        )
        predictions[row.candidate_source] = (
            float(distribution.mean[0]),
            float(distribution.standard_error[0]),
        )
    return predictions


def permuted_target_seed_spread(
    surface: EnsembleFeatureSurface, *, permutation_seed: int
) -> dict[str, float]:
    ordered_rows = tuple(sorted(surface.rows, key=lambda row: row.candidate_source))
    count = len(ordered_rows)
    if count != TARGET_CANDIDATE_COUNT:
        raise ProtocolError("Target permutation spread requires eight candidates.")
    original = np.asarray(
        [row.target_local_scalar_seed_standard_deviation for row in ordered_rows],
        dtype=np.float64,
    )
    if not np.isfinite(original).all():
        raise ProtocolError("Target permutation seed spread is absent or non-finite.")
    shift = 1 + (abs(int(permutation_seed)) % (count - 1))
    permuted = np.roll(original, -shift)
    return {
        row.candidate_source: float(value)
        for row, value in zip(ordered_rows, permuted)
    }




__all__ = (
    "permuted_target_seed_spread",
    "predict_target_candidate_distributions",
    "predict_target_candidates",
)

