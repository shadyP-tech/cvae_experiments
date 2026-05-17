"""Distributional fidelity diagnostics for generated embeddings.

These diagnostics are secondary evidence only. They must never choose the
routing method or override downstream utility gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FidelityScore:
    expert_domain: str
    metrics: Mapping[str, float]


def compute_fidelity_diagnostics(
    *,
    expert_domain: str,
    real_embeddings: Sequence[Sequence[float]],
    synthetic_embeddings: Sequence[Sequence[float]],
) -> FidelityScore:
    """Compute lightweight embedding-space fidelity diagnostics.

    Full kNN precision/recall and RBF MMD can be added later; this v1 utility
    provides stable mean/covariance-distance fields without external packages.
    """

    real = _as_matrix(real_embeddings)
    synthetic = _as_matrix(synthetic_embeddings)
    if not real or not synthetic:
        raise ValueError("Fidelity diagnostics require non-empty real and synthetic embeddings.")
    if len(real[0]) != len(synthetic[0]):
        raise ValueError("Real and synthetic embeddings must share dimensionality.")
    real_mean = _column_means(real)
    syn_mean = _column_means(synthetic)
    real_var = _column_variances(real, real_mean)
    syn_var = _column_variances(synthetic, syn_mean)
    mean_distance = _euclidean(real_mean, syn_mean)
    covariance_distance = _euclidean(real_var, syn_var)
    return FidelityScore(
        expert_domain=str(expert_domain),
        metrics={
            "mean_distance": mean_distance,
            "covariance_distance": covariance_distance,
            "frechet_embedding_distance": mean_distance + covariance_distance,
            "rbf_mmd": math.nan,
            "energy_distance": math.nan,
            "knn_precision": math.nan,
            "knn_recall": math.nan,
            "density": math.nan,
            "coverage": math.nan,
        },
    )


def _as_matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = [[float(v) for v in row] for row in values]
    widths = {len(row) for row in matrix}
    if len(widths) != 1:
        raise ValueError("Embedding rows must have a consistent width.")
    return matrix


def _column_means(matrix: Sequence[Sequence[float]]) -> list[float]:
    width = len(matrix[0])
    return [sum(row[i] for row in matrix) / float(len(matrix)) for i in range(width)]


def _column_variances(matrix: Sequence[Sequence[float]], means: Sequence[float]) -> list[float]:
    width = len(matrix[0])
    return [
        sum((row[i] - means[i]) ** 2 for row in matrix) / float(max(len(matrix) - 1, 1))
        for i in range(width)
    ]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
