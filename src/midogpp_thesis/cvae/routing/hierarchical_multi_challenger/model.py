"""Penalized pooled binomial direction models with epistemic covariance."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    DirectionalDonorRow,
    DirectionalLogitModel,
    DirectionalPrediction,
    MODEL_FAMILIES,
)
from .hashing import canonical_hash


FEATURE_ALPHA = 1.0
SOURCE_ALPHA = 4.0
QUERY_ALPHA = 4.0
INTERCEPT_ALPHA = 0.25
IRLS_MAX_ITERATIONS = 200
IRLS_TOLERANCE = 1.0e-10


def fit_directional_logit(
    rows: Sequence[DirectionalDonorRow],
    *,
    heldout_h: str,
    family: str,
    feature_alpha: float = FEATURE_ALPHA,
    source_alpha: float = SOURCE_ALPHA,
    query_alpha: float = QUERY_ALPHA,
    intercept_alpha: float = INTERCEPT_ALPHA,
) -> DirectionalLogitModel:
    """Fit one direction using strict H/q/e rows and binomial flip counts."""

    supplied = tuple(rows)
    if not supplied or family not in MODEL_FAMILIES:
        raise ProtocolError("Directional fitting requires rows and a frozen family.")
    if (
        float(feature_alpha) != FEATURE_ALPHA
        or float(source_alpha) != SOURCE_ALPHA
        or float(query_alpha) != QUERY_ALPHA
        or float(intercept_alpha) != INTERCEPT_ALPHA
    ):
        raise ProtocolError("Directional model penalties are frozen.")
    h = str(heldout_h)
    direction = supplied[0].direction
    names = supplied[0].feature_names
    if any(
        row.model_target != h
        or row.direction != direction
        or row.feature_names != names
        or row.query_center == h
        or row.candidate_source in {h, row.query_center}
        for row in supplied
    ):
        raise ProtocolError("Directional fitting rows violate H/q/e or schema scope.")
    ordered_all = tuple(
        sorted(
            supplied,
            key=lambda row: (
                row.query_center,
                row.case_id,
                row.action_id,
                row.candidate_source,
                row.feature_case_id,
            ),
        )
    )
    ordered = tuple(row for row in ordered_all if row.trial_count > 0)
    clusters = tuple(sorted({row.case_cluster for row in ordered}))
    if len(ordered) < 2 or len(clusters) < 2:
        raise ProtocolError("Directional fitting requires two informative donor cases.")
    matrix = np.asarray([row.values for row in ordered], dtype=np.float64)
    trials = np.asarray([row.trial_count for row in ordered], dtype=np.float64)
    successes = np.asarray([row.success_count for row in ordered], dtype=np.float64)
    trial_sum = float(np.sum(trials, dtype=np.float64))
    mean = np.sum(matrix * trials[:, None], axis=0, dtype=np.float64) / trial_sum
    centered = matrix - mean
    variance = (
        np.sum(centered * centered * trials[:, None], axis=0, dtype=np.float64)
        / trial_sum
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardized = centered / scale
    queries = tuple(sorted({row.query_center for row in ordered}))
    sources = tuple(sorted({row.candidate_source for row in ordered}))
    source_index = {source: ordinal for ordinal, source in enumerate(sources)}
    source_design = np.zeros((len(ordered), len(sources)), dtype=np.float64)
    for ordinal, row in enumerate(ordered):
        source_design[ordinal, source_index[row.candidate_source]] = 1.0
    query_index = {query: ordinal for ordinal, query in enumerate(queries)}
    query_design = np.zeros((len(ordered), len(queries)), dtype=np.float64)
    for ordinal, row in enumerate(ordered):
        query_design[ordinal, query_index[row.query_center]] = 1.0
    if family == "G":
        design = np.column_stack(
            (
                np.ones(len(ordered), dtype=np.float64),
                source_design,
                query_design,
            )
        )
        penalties = np.asarray(
            [
                INTERCEPT_ALPHA,
                *([SOURCE_ALPHA] * len(sources)),
                *([QUERY_ALPHA] * len(queries)),
            ],
            dtype=np.float64,
        )
    else:
        design = np.column_stack(
            (
                np.ones(len(ordered), dtype=np.float64),
                standardized,
                source_design,
                query_design,
            )
        )
        penalties = np.asarray(
            [
                INTERCEPT_ALPHA,
                *([FEATURE_ALPHA] * len(names)),
                *([SOURCE_ALPHA] * len(sources)),
                *([QUERY_ALPHA] * len(queries)),
            ],
            dtype=np.float64,
        )
    coefficients, covariance = _penalized_binomial_irls(
        design,
        successes,
        trials,
        penalties,
        clusters=tuple(row.case_cluster for row in ordered),
    )
    provenance_hash = canonical_hash(
        {
            "schema_version": "hierarchical_directional_donor_provenance_v2",
            "heldout_h": h,
            "family": family,
            "direction": direction,
            "strict_H_q_e_exclusion": True,
            "zero_trial_rows_retained_for_permutation_topology": True,
            "feature_alpha": FEATURE_ALPHA,
            "source_alpha": SOURCE_ALPHA,
            "query_alpha": QUERY_ALPHA,
            "intercept_alpha": INTERCEPT_ALPHA,
            "rows": [row.to_payload() for row in ordered_all],
        }
    )
    return DirectionalLogitModel(
        model_target=h,
        family=family,
        direction=direction,
        feature_names=names,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        candidate_sources=sources,
        query_centers=queries,
        coefficients=tuple(float(value) for value in coefficients),
        covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        feature_alpha=FEATURE_ALPHA,
        source_alpha=SOURCE_ALPHA,
        query_alpha=QUERY_ALPHA,
        intercept_alpha=INTERCEPT_ALPHA,
        training_row_count=len(ordered),
        training_trial_count=int(sum(row.trial_count for row in ordered)),
        training_case_clusters=clusters,
        provenance_hash=provenance_hash,
    )


def predict_direction(
    model: DirectionalLogitModel,
    *,
    candidate_source: str,
    feature_names: Sequence[str],
    values: Sequence[float],
    query_center: str | None = None,
) -> DirectionalPrediction:
    """Predict a target case; the held-out target query effect is exactly zero."""

    names = tuple(str(value) for value in feature_names)
    vector = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if names != model.feature_names or vector.shape != (len(model.feature_names),):
        raise ProtocolError("Directional prediction feature schema drifted.")
    source = str(candidate_source)
    if source not in model.candidate_sources:
        raise ProtocolError("Directional prediction candidate is outside trained topology.")
    source_design = np.zeros(len(model.candidate_sources), dtype=np.float64)
    source_design[model.candidate_sources.index(source)] = 1.0
    query = model.model_target if query_center is None else str(query_center)
    query_design = np.zeros(len(model.query_centers), dtype=np.float64)
    if query in model.query_centers:
        query_design[model.query_centers.index(query)] = 1.0
    elif query != model.model_target:
        raise ProtocolError("Directional prediction query is outside model topology.")
    if model.family == "G":
        design = np.asarray([1.0, *source_design, *query_design], dtype=np.float64)
    else:
        standardized = (
            vector - np.asarray(model.feature_mean, dtype=np.float64)
        ) / np.asarray(model.feature_scale, dtype=np.float64)
        design = np.asarray(
            [1.0, *standardized, *source_design, *query_design],
            dtype=np.float64,
        )
    eta = float(design @ np.asarray(model.coefficients, dtype=np.float64))
    probability = _sigmoid(eta)
    covariance = np.asarray(model.covariance, dtype=np.float64)
    parameter_variance = max(float(design @ covariance @ design), 0.0)
    return DirectionalPrediction(
        probability=probability,
        design=tuple(float(value) for value in design),
        parameter_variance=parameter_variance,
        model_fingerprint=model.fit_fingerprint,
    )


def _penalized_binomial_irls(
    design: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    penalties: np.ndarray,
    *,
    clusters: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if (
        design.ndim != 2
        or successes.shape != (design.shape[0],)
        or trials.shape != successes.shape
        or penalties.shape != (design.shape[1],)
        or len(clusters) != design.shape[0]
    ):
        raise ProtocolError("Directional IRLS geometry drifted.")
    penalty = np.diag(penalties)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    hessian = np.empty((design.shape[1], design.shape[1]), dtype=np.float64)
    for _ in range(IRLS_MAX_ITERATIONS):
        eta = np.clip(design @ coefficients, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        weights = np.maximum(trials * probabilities * (1.0 - probabilities), 1.0e-12)
        gradient = design.T @ (successes - trials * probabilities) - penalties * coefficients
        hessian = design.T @ (weights[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ProtocolError("Directional IRLS Hessian is singular.") from exc
        coefficients += step
        if float(np.max(np.abs(step))) <= IRLS_TOLERANCE:
            converged = True
            break
    if not converged:
        raise ProtocolError("Directional IRLS did not converge.")
    eta = np.clip(design @ coefficients, -35.0, 35.0)
    probabilities = 1.0 / (1.0 + np.exp(-eta))
    weights = np.maximum(trials * probabilities * (1.0 - probabilities), 1.0e-12)
    hessian = design.T @ (weights[:, None] * design) + penalty
    try:
        bread = np.linalg.solve(
            hessian, np.eye(hessian.shape[0], dtype=np.float64)
        )
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("Directional covariance Hessian is singular.") from exc
    cluster_array = np.asarray(tuple(str(value) for value in clusters), dtype=object)
    unique_clusters = tuple(sorted(set(clusters)))
    meat = np.zeros_like(hessian)
    residual = successes - trials * probabilities
    for cluster in unique_clusters:
        mask = cluster_array == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    correction = len(unique_clusters) / (len(unique_clusters) - 1)
    if len(successes) > design.shape[1]:
        correction *= (len(successes) - 1) / (
            len(successes) - design.shape[1]
        )
    covariance = correction * (bread @ meat @ bread.T)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (
        eigenvectors * np.maximum(eigenvalues, 0.0)
    ) @ eigenvectors.T
    if not np.isfinite(coefficients).all() or not np.isfinite(covariance).all():
        raise ProtocolError("Directional fit contains non-finite numerics.")
    return coefficients, covariance


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


__all__ = (
    "FEATURE_ALPHA",
    "INTERCEPT_ALPHA",
    "IRLS_MAX_ITERATIONS",
    "IRLS_TOLERANCE",
    "QUERY_ALPHA",
    "SOURCE_ALPHA",
    "fit_directional_logit",
    "predict_direction",
)
