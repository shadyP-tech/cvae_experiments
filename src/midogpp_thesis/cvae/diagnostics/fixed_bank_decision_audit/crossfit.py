"""Fixed-alpha known-bank crossfitting with shared H/q models."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.local_marginal_utility.ridge import fit_cluster_weighted_ridge
from .constants import (
    CENTERS,
    EXACT_BACC_DELTA,
    EXACT_FAMILY_IDS,
    RIDGE_ALPHA,
    SMOOTH_BACC_DELTA,
    SMOOTH_DESCRIPTIVE_FAMILY_IDS,
)
from .controls import (
    augmented_model_matrix,
    held_query_indices,
    legal_candidate_history_counts,
    strict_training_indices,
)
from .features import (
    build_exact_family_designs,
    build_smooth_descriptive_designs,
)
from .model_contracts import (
    CrossfitFoldAudit,
    CrossfitPredictionRow,
    ExactCrossfitResult,
    FamilyDesign,
    SmoothCrossfitResult,
)
from .row_contracts import FixedBankDataset
from .serialization import canonical_array_hash


def crossfit_exact_families(
    dataset: FixedBankDataset,
    designs: Mapping[str, FamilyDesign] | None = None,
    *,
    family_ids: Sequence[str] = EXACT_FAMILY_IDS,
) -> ExactCrossfitResult:
    """Crossfit terminal exact utility; smooth values are never read."""

    selected = _selected(family_ids, EXACT_FAMILY_IDS)
    frozen_designs = (
        build_exact_family_designs(dataset, selected) if designs is None else designs
    )
    observed = np.asarray(
        [row.exact_bacc_delta for row in dataset.response_rows], dtype=np.float64
    )
    predictions, folds = _crossfit(
        dataset,
        frozen_designs,
        selected,
        observed,
        response_name=EXACT_BACC_DELTA,
    )
    return ExactCrossfitResult(
        predictions=predictions,
        fold_audits=folds,
        family_ids=selected,
        feature_surface_hash=dataset.feature_surface_hash,
        exact_response_surface_hash=dataset.exact_response_surface_hash,
    )


def crossfit_smooth_descriptive(
    dataset: FixedBankDataset,
    designs: Mapping[str, FamilyDesign] | None = None,
    *,
    family_ids: Sequence[str] = SMOOTH_DESCRIPTIVE_FAMILY_IDS,
) -> SmoothCrossfitResult:
    """Produce a wholly separate smooth-response descriptive surface."""

    selected = _selected(family_ids, SMOOTH_DESCRIPTIVE_FAMILY_IDS)
    frozen_designs = (
        build_smooth_descriptive_designs(dataset, selected)
        if designs is None
        else designs
    )
    observed = np.asarray(
        [row.smooth_bacc_delta for row in dataset.response_rows], dtype=np.float64
    )
    predictions, folds = _crossfit(
        dataset,
        frozen_designs,
        selected,
        observed,
        response_name=SMOOTH_BACC_DELTA,
    )
    return SmoothCrossfitResult(
        predictions=predictions,
        fold_audits=folds,
        family_ids=selected,
        feature_surface_hash=dataset.feature_surface_hash,
        smooth_response_surface_hash=dataset.smooth_response_surface_hash,
    )


def _crossfit(
    dataset: FixedBankDataset,
    designs: Mapping[str, FamilyDesign],
    family_ids: tuple[str, ...],
    observed: np.ndarray,
    *,
    response_name: str,
) -> tuple[tuple[CrossfitPredictionRow, ...], tuple[CrossfitFoldAudit, ...]]:
    if not isinstance(dataset, FixedBankDataset):
        raise ProtocolError("Crossfit requires a typed fixed-bank dataset.")
    if set(designs) != set(family_ids):
        raise ProtocolError("Crossfit design coverage drifted.")
    predictions: list[CrossfitPredictionRow] = []
    folds: list[CrossfitFoldAudit] = []
    for family_id in family_ids:
        design = designs[family_id]
        if (
            design.spec.family_id != family_id
            or design.spec.response_name != response_name
            or design.row_keys != dataset.row_keys
            or design.feature_surface_hash != dataset.feature_surface_hash
        ):
            raise ProtocolError("Crossfit design provenance drifted.")
        # H/q reversal has the identical strict all-role training fold.  Cache
        # that fit while retaining separate ordered H/q audits and predictions.
        fit_cache: dict[frozenset[str], tuple[object, ...]] = {}
        for outer in CENTERS:
            for query in (value for value in CENTERS if value != outer):
                prediction_indices = held_query_indices(dataset, outer, query)
                prediction_matrix, prediction_names = augmented_model_matrix(
                    design, prediction_indices
                )
                cache_key = frozenset((outer, query))
                cached = fit_cache.get(cache_key)
                if cached is None:
                    training_indices = strict_training_indices(
                        dataset, outer, query
                    )
                    train_matrix, feature_names = augmented_model_matrix(
                        design, training_indices
                    )
                    training_rows = tuple(
                        dataset.row_keys[int(index)] for index in training_indices
                    )
                    clusters = tuple(
                        f"{row[0]}::{row[1]}" for row in training_rows
                    )
                    model = fit_cluster_weighted_ridge(
                        train_matrix,
                        observed[training_indices],
                        clusters,
                        alpha=RIDGE_ALPHA,
                        feature_names=feature_names,
                    )
                    fit_cache[cache_key] = (
                        training_indices,
                        training_rows,
                        feature_names,
                        model,
                    )
                else:
                    training_indices = cached[0]
                    training_rows = cached[1]
                    feature_names = cached[2]
                    model = cached[3]
                if prediction_names != feature_names:
                    raise ProtocolError("Training/prediction feature order drifted.")
                prediction = model.predict_with_uncertainty(
                    prediction_matrix,
                    include_residual_variance=True,
                )
                fold = CrossfitFoldAudit(
                    family_id=family_id,
                    response_name=response_name,
                    outer_target_id=outer,
                    query_id=query,
                    training_row_keys=training_rows,
                    legal_candidate_source_history_counts=(
                        legal_candidate_history_counts(
                            dataset, training_indices, outer, query
                        )
                    ),
                    feature_names=feature_names,
                    feature_mean=tuple(float(value) for value in model.feature_mean),
                    feature_scale=tuple(
                        float(value) for value in model.feature_scale
                    ),
                    intercept=model.intercept,
                    coefficients=tuple(
                        float(value) for value in model.coefficients
                    ),
                    coefficient_covariance_sha256=canonical_array_hash(
                        model.coefficient_covariance
                    ),
                    residual_variance=model.residual_variance,
                    family_design_hash=design.design_hash,
                )
                folds.append(fold)
                for position, row_index in enumerate(prediction_indices.tolist()):
                    key = dataset.row_keys[row_index]
                    predictions.append(
                        CrossfitPredictionRow(
                            family_id=family_id,
                            response_name=response_name,
                            outer_target_id=key[0],
                            query_id=key[1],
                            candidate_source=key[2],
                            predicted_delta=float(prediction.mean[position]),
                            prediction_standard_error=float(
                                prediction.standard_error[position]
                            ),
                            observed_delta=float(observed[row_index]),
                            fold_hash=fold.fold_hash,
                        )
                    )
    return tuple(predictions), tuple(folds)


def _selected(requested: Sequence[str], allowed: Sequence[str]) -> tuple[str, ...]:
    values = tuple(requested)
    if (
        not values
        or len(set(values)) != len(values)
        or any(value not in allowed for value in values)
    ):
        raise ProtocolError("Crossfit family selection drifted from predeclaration.")
    return values


__all__ = ("crossfit_exact_families", "crossfit_smooth_descriptive")
