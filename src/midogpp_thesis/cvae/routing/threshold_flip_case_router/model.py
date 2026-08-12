"""H-specific, case-cluster-balanced two-head ridge model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CaseActionFeatures,
    DonorRow,
    HeadModel,
    TwoHeadPrediction,
    TwoHeadRidgeModel,
    canonical_hash,
)


RIDGE_ALPHA = 1.0
VARIANCE_FLOOR = 1.0e-6


def fit_two_head_ridge(
    rows: Sequence[DonorRow],
    *,
    heldout_h: str,
    alpha: float = RIDGE_ALPHA,
    variance_floor: float = VARIANCE_FLOOR,
) -> TwoHeadRidgeModel:
    """Fit TP/TN heads using only strict q!=H and e!=H,q donor rows."""

    donors = tuple(rows)
    if not donors:
        raise ProtocolError("Two-head ridge requires donor rows.")
    if float(alpha) != RIDGE_ALPHA or float(variance_floor) < VARIANCE_FLOOR:
        raise ProtocolError("The router freezes alpha=1 and variance floor>=1e-6.")
    h = str(heldout_h)
    if any(row.model_target != h for row in donors):
        raise ProtocolError("An H-specific model cannot reuse rows prepared for another H.")
    # Recheck at the fitting boundary in addition to DonorRow construction.
    if any(
        row.query_center == h or row.candidate_source in {h, row.query_center}
        for row in donors
    ):
        raise ProtocolError("Donor rows violate strict H/q/e exclusion.")
    names = donors[0].feature_names
    if any(row.feature_names != names for row in donors):
        raise ProtocolError("Donor feature schema drifted.")
    ordered = tuple(
        sorted(
            donors,
            key=lambda row: (
                row.query_center,
                row.case_id,
                row.action_id,
                row.candidate_source,
                row.feature_case_id,
            ),
        )
    )
    matrix = np.asarray([row.values for row in ordered], dtype=np.float64)
    targets = np.asarray(
        [[row.target.delta_tp, row.target.delta_tn] for row in ordered], dtype=np.float64
    )
    clusters = tuple(row.case_cluster for row in ordered)
    unique_clusters = tuple(sorted(set(clusters)))
    if len(unique_clusters) < 2:
        raise ProtocolError("Case-cluster covariance requires at least two donor cases.")
    weights = _equal_cluster_weights(clusters)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    mean = np.sum(matrix * weights[:, None], axis=0, dtype=np.float64) / weight_sum
    centered = matrix - mean
    variance = np.sum(centered * centered * weights[:, None], axis=0, dtype=np.float64) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= np.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardized = centered / scale
    design = np.column_stack((np.ones(len(ordered), dtype=np.float64), standardized))
    penalty = np.diag(np.asarray([0.0, *([RIDGE_ALPHA] * len(names))], dtype=np.float64))
    gram = design.T @ (weights[:, None] * design)
    normal = gram + penalty
    try:
        bread = np.linalg.solve(normal, np.eye(normal.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("Two-head ridge normal equations are singular.") from exc
    heads = tuple(
        _fit_head(
            design=design,
            response=targets[:, ordinal],
            weights=weights,
            clusters=clusters,
            unique_clusters=unique_clusters,
            normal=normal,
            bread=bread,
            gram=gram,
            variance_floor=float(variance_floor),
        )
        for ordinal in range(2)
    )
    provenance_hash = canonical_hash(
        {
            "schema_version": "threshold_flip_two_head_donor_provenance_v1",
            "heldout_h": h,
            "strict_H_q_e_exclusion": True,
            "rows": [row.to_payload() for row in ordered],
        }
    )
    return TwoHeadRidgeModel(
        model_target=h,
        feature_names=names,
        feature_mean=tuple(float(v) for v in mean),
        feature_scale=tuple(float(v) for v in scale),
        alpha=RIDGE_ALPHA,
        variance_floor=float(variance_floor),
        tp_head=heads[0],
        tn_head=heads[1],
        training_case_clusters=unique_clusters,
        donor_query_centers=tuple(sorted({row.query_center for row in ordered})),
        donor_candidate_sources=tuple(sorted({row.candidate_source for row in ordered})),
        training_row_count=len(ordered),
        provenance_hash=provenance_hash,
    )


def predict_two_head(
    model: TwoHeadRidgeModel,
    features: CaseActionFeatures,
) -> TwoHeadPrediction:
    """Predict a target-H case; models are deliberately not cross-H reusable."""

    if features.target_center != model.model_target:
        raise ProtocolError("An H-specific two-head model cannot be reused for another H.")
    if features.feature_names != model.feature_names:
        raise ProtocolError("Prediction feature schema does not match the model.")
    values = np.asarray(features.values, dtype=np.float64)
    standardized = (values - np.asarray(model.feature_mean)) / np.asarray(model.feature_scale)
    design = np.asarray([1.0, *standardized], dtype=np.float64)
    tp_mean, tp_variance = _predict_head(model.tp_head, design, model.variance_floor)
    tn_mean, tn_variance = _predict_head(model.tn_head, design, model.variance_floor)
    return TwoHeadPrediction(
        model_target=model.model_target,
        case_id=features.case_id,
        action_id=features.action_id,
        mean_delta_tp=tp_mean,
        mean_delta_tn=tn_mean,
        variance_delta_tp=tp_variance,
        variance_delta_tn=tn_variance,
        model_hash=model.model_hash,
    )


def _equal_cluster_weights(clusters: Sequence[str]) -> np.ndarray:
    unique = tuple(sorted(set(clusters)))
    counts = {cluster: clusters.count(cluster) for cluster in unique}
    n_rows = len(clusters)
    result = np.asarray(
        [n_rows / (len(unique) * counts[cluster]) for cluster in clusters], dtype=np.float64
    )
    return result


def _fit_head(
    *,
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    clusters: tuple[str, ...],
    unique_clusters: tuple[str, ...],
    normal: np.ndarray,
    bread: np.ndarray,
    gram: np.ndarray,
    variance_floor: float,
) -> HeadModel:
    rhs = design.T @ (weights * response)
    theta = np.linalg.solve(normal, rhs)
    residual = response - design @ theta
    rank = int(np.linalg.matrix_rank(design))
    residual_variance = max(
        float(np.dot(weights, residual * residual) / max(len(response) - rank, 1)),
        variance_floor,
    )
    meat = np.zeros_like(normal)
    cluster_array = np.asarray(clusters, dtype=object)
    for cluster in unique_clusters:
        mask = cluster_array == cluster
        score = design[mask].T @ (weights[mask] * residual[mask])
        meat += np.outer(score, score)
    correction = len(unique_clusters) / (len(unique_clusters) - 1)
    if len(response) > rank:
        correction *= (len(response) - 1) / (len(response) - rank)
    covariance = correction * (bread @ meat @ bread.T)
    covariance = (covariance + covariance.T) * 0.5
    if not np.isfinite(covariance).all():
        covariance = residual_variance * (bread @ gram @ bread.T)
        covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return HeadModel(
        intercept=float(theta[0]),
        coefficients=tuple(float(v) for v in theta[1:]),
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        residual_variance=residual_variance,
    )


def _predict_head(head: HeadModel, design: np.ndarray, variance_floor: float) -> tuple[float, float]:
    mean = float(head.intercept + design[1:] @ np.asarray(head.coefficients))
    covariance = np.asarray(head.covariance, dtype=np.float64)
    parameter_variance = float(design @ covariance @ design)
    variance = max(parameter_variance + head.residual_variance, variance_floor)
    if not np.isfinite(mean) or not np.isfinite(variance):
        raise ProtocolError("Two-head prediction is non-finite.")
    return mean, variance


__all__ = (
    "RIDGE_ALPHA",
    "VARIANCE_FLOOR",
    "fit_two_head_ridge",
    "predict_two_head",
)
