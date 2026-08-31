"""Center-balanced partially pooled ridge used by HARP."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError


@dataclass(frozen=True)
class HarpRidgeModel:
    feature_names: tuple[str, ...]
    candidate_levels: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    normal_inverse: np.ndarray
    alpha: float
    training_query_ids: tuple[str, ...]
    training_source_ids: tuple[str, ...]
    training_case_ids: tuple[str, ...]
    excluded_donor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("feature_mean", "feature_scale", "coefficients", "normal_inverse"):
            try:
                value = np.array(getattr(self, name), dtype=np.float64, order="C", copy=True)
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Serialized HARP ridge arrays must be numeric.") from exc
            object.__setattr__(self, name, value)
        dimension = 1 + len(self.feature_names) + len(self.candidate_levels)
        identifiers = (*self.feature_names, *self.candidate_levels, *self.training_query_ids, *self.training_source_ids, *self.training_case_ids, *self.excluded_donor_ids)
        if (
            not self.feature_names
            or not self.candidate_levels
            or not self.training_query_ids
            or not self.training_source_ids
            or not self.training_case_ids
            or any(type(value) is not str or not value or value != value.strip() for value in identifiers)
            or len(set(self.feature_names)) != len(self.feature_names)
            or self.candidate_levels != tuple(sorted(set(self.candidate_levels)))
            or self.training_query_ids != tuple(sorted(set(self.training_query_ids)))
            or self.training_source_ids != tuple(sorted(set(self.training_source_ids)))
            or self.training_case_ids != tuple(sorted(set(self.training_case_ids)))
            or self.excluded_donor_ids != tuple(sorted(set(self.excluded_donor_ids)))
            or set(self.excluded_donor_ids).intersection(self.training_query_ids)
            or set(self.excluded_donor_ids).intersection(self.training_source_ids)
            or self.feature_mean.shape != (len(self.feature_names),)
            or self.feature_scale.shape != (len(self.feature_names),)
            or self.coefficients.shape != (dimension,)
            or self.normal_inverse.shape != (dimension, dimension)
            or any(not np.isfinite(value).all() for value in (self.feature_mean, self.feature_scale, self.coefficients, self.normal_inverse))
            or np.any(self.feature_scale <= 0)
            or not np.allclose(self.normal_inverse, self.normal_inverse.T, rtol=1e-10, atol=1e-12)
            or float(np.linalg.eigvalsh(self.normal_inverse).min()) < -1e-10
            or not math.isfinite(float(self.alpha))
            or self.alpha <= 0
        ):
            raise ProtocolError("Serialized HARP ridge model state is invalid.")
        for value in (self.feature_mean, self.feature_scale, self.coefficients, self.normal_inverse):
            value.setflags(write=False)

    def _design(self, features: Sequence[Sequence[float]] | np.ndarray, candidates: Sequence[str]) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        candidate_ids = tuple(str(value) for value in candidates)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names) or len(matrix) != len(candidate_ids) or not np.isfinite(matrix).all():
            raise ProtocolError("HARP ridge prediction inputs are invalid or misaligned.")
        standardized = (matrix - self.feature_mean) / self.feature_scale
        one_hot = np.zeros((len(matrix), len(self.candidate_levels)), dtype=np.float64)
        index = {candidate: idx for idx, candidate in enumerate(self.candidate_levels)}
        for row, candidate in enumerate(candidate_ids):
            if candidate in index:
                one_hot[row, index[candidate]] = 1.0
        return np.column_stack((np.ones(len(matrix)), standardized, one_hot))

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray, candidates: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        design = self._design(features, candidates)
        mean = design @ self.coefficients
        leverage = np.einsum("ij,jk,ik->i", design, self.normal_inverse, design)
        if not np.isfinite(mean).all() or not np.isfinite(leverage).all() or np.any(leverage < -1e-10):
            raise ProtocolError("HARP ridge produced invalid predictions or leverage.")
        mean = np.asarray(mean, dtype=np.float64)
        leverage = np.maximum(np.asarray(leverage, dtype=np.float64), 0.0)
        mean.setflags(write=False)
        leverage.setflags(write=False)
        return mean, leverage

    def predict_singleton_equivalent_batch(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        candidates: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch rows while preserving the legacy ``(1, p) @ (p,)`` bytes.

        A plain multirow matrix-vector product may select a different BLAS
        reduction and change low float64 bits.  The leading singleton axis
        retains the frozen per-row matmul geometry while sharing one validated
        design construction across the batch.
        """

        design = self._design(features, candidates)
        mean = np.matmul(
            design[:, np.newaxis, :], self.coefficients
        ).reshape(len(design))
        leverage = np.einsum(
            "ij,jk,ik->i", design, self.normal_inverse, design
        )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(leverage).all()
            or np.any(leverage < -1e-10)
        ):
            raise ProtocolError("HARP ridge produced invalid predictions or leverage.")
        mean = np.asarray(mean, dtype=np.float64)
        leverage = np.maximum(np.asarray(leverage, dtype=np.float64), 0.0)
        mean.setflags(write=False)
        leverage.setflags(write=False)
        return mean, leverage


def fit_partial_pool_ridge(
    features: Sequence[Sequence[float]] | np.ndarray,
    response: Sequence[float] | np.ndarray,
    query_ids: Sequence[str],
    case_ids: Sequence[str],
    candidate_ids: Sequence[str],
    *,
    feature_names: Sequence[str],
    alpha: float,
    excluded_donor_ids: Sequence[str] = (),
) -> HarpRidgeModel:
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(response, dtype=np.float64)
    queries = tuple(str(value) for value in query_ids)
    cases = tuple(str(value) for value in case_ids)
    candidates = tuple(str(value) for value in candidate_ids)
    names = tuple(str(value) for value in feature_names)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(names) or target.shape != (len(matrix),) or len(queries) != len(matrix) or len(cases) != len(matrix) or len(candidates) != len(matrix) or not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ProtocolError("HARP ridge training inputs are invalid or misaligned.")
    if any(not value for value in (*queries, *cases, *candidates)) or len(set(names)) != len(names):
        raise ProtocolError("HARP ridge identifiers and feature names must be canonical.")
    penalty = float(alpha)
    if not math.isfinite(penalty) or penalty <= 0:
        raise ProtocolError("HARP ridge alpha must be finite and positive.")
    query_levels = tuple(sorted(set(queries)))
    candidate_levels = tuple(sorted(set(candidates)))
    cases_by_query = {query: tuple(sorted({case for q, case in zip(queries, cases, strict=True) if q == query})) for query in query_levels}
    rows_by_query_case = {(query, case): sum(q == query and c == case for q, c in zip(queries, cases, strict=True)) for query in query_levels for case in cases_by_query[query]}
    # Equal total mass per query, then per independent case. Repeated patches or
    # seed cells inside a case cannot create pseudo-replication.
    weights = np.asarray([
        1.0 / (len(query_levels) * len(cases_by_query[query]) * rows_by_query_case[(query, case)])
        for query, case in zip(queries, cases, strict=True)
    ], dtype=np.float64)
    feature_mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - feature_mean) ** 2, axis=0, weights=weights)
    feature_scale = np.sqrt(np.maximum(variance, 0.0))
    feature_scale[feature_scale <= np.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardized = (matrix - feature_mean) / feature_scale
    one_hot = np.zeros((len(matrix), len(candidate_levels)), dtype=np.float64)
    candidate_index = {candidate: idx for idx, candidate in enumerate(candidate_levels)}
    for row, candidate in enumerate(candidates):
        one_hot[row, candidate_index[candidate]] = 1.0
    design = np.column_stack((np.ones(len(matrix)), standardized, one_hot))
    gram = design.T @ (weights[:, None] * design)
    penalty_matrix = np.diag(np.asarray([0.0, *([penalty] * (design.shape[1] - 1))]))
    normal = gram + penalty_matrix
    try:
        normal_inverse = np.linalg.inv(normal)
        coefficients = normal_inverse @ (design.T @ (weights * target))
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("HARP ridge normal equations are singular.") from exc
    for value in (feature_mean, feature_scale, coefficients, normal_inverse):
        if not np.isfinite(value).all():
            raise ProtocolError("HARP ridge model contains non-finite state.")
        value.setflags(write=False)
    exclusions = tuple(sorted(set(str(value) for value in excluded_donor_ids)))
    if set(exclusions).intersection(query_levels) or set(exclusions).intersection(candidate_levels):
        raise ProtocolError("Excluded donors leaked into a fitted HARP ridge model.")
    return HarpRidgeModel(
        feature_names=names,
        candidate_levels=candidate_levels,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        normal_inverse=normal_inverse,
        alpha=penalty,
        training_query_ids=query_levels,
        training_source_ids=candidate_levels,
        training_case_ids=tuple(sorted(set(cases))),
        excluded_donor_ids=exclusions,
    )


__all__ = ("HarpRidgeModel", "fit_partial_pool_ridge")
