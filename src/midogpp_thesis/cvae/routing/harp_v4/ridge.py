"""Shared-design, three-response partial-pool ridge for HARP v4."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import CaseTrainingObservation, Comparison, OUTCOME_NAMES


@dataclass(frozen=True)
class SharedDesignRidge:
    feature_names: tuple[str, ...]
    query_levels: tuple[str, ...]
    candidate_levels: tuple[str, ...]
    comparison_levels: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    normal_inverse: np.ndarray
    alpha: float
    training_query_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    training_case_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("feature_mean", "feature_scale", "coefficients", "normal_inverse"):
            try:
                value = np.array(getattr(self, name), dtype=np.float64, order="C", copy=True)
            except (TypeError, ValueError) as exc:
                raise ProtocolError("HARP v4 ridge arrays must be numeric.") from exc
            object.__setattr__(self, name, value)
        dimension = (
            1
            + len(self.feature_names)
            + len(self.query_levels)
            + len(self.candidate_levels)
            + len(self.comparison_levels)
        )
        canonical_sequences = (
            self.query_levels,
            self.candidate_levels,
            self.comparison_levels,
            self.training_query_ids,
            self.training_candidate_ids,
            self.training_case_ids,
            self.excluded_center_ids,
        )
        if (
            not self.feature_names
            or any(values != tuple(sorted(set(values))) for values in canonical_sequences)
            or len(set(self.feature_names)) != len(self.feature_names)
            or self.comparison_levels != tuple(sorted(value.value for value in Comparison))
            or self.feature_mean.shape != (len(self.feature_names),)
            or self.feature_scale.shape != (len(self.feature_names),)
            or self.coefficients.shape != (dimension, len(OUTCOME_NAMES))
            or self.normal_inverse.shape != (dimension, dimension)
            or any(
                not np.isfinite(value).all()
                for value in (
                    self.feature_mean,
                    self.feature_scale,
                    self.coefficients,
                    self.normal_inverse,
                )
            )
            or np.any(self.feature_scale <= 0)
            or not np.allclose(self.normal_inverse, self.normal_inverse.T, rtol=1e-10, atol=1e-12)
            or float(np.linalg.eigvalsh(self.normal_inverse).min()) < -1e-9
            or not math.isfinite(float(self.alpha))
            or self.alpha <= 0
            or set(self.excluded_center_ids).intersection(self.training_query_ids)
            or set(self.excluded_center_ids).intersection(self.training_candidate_ids)
        ):
            raise ProtocolError("Serialized HARP v4 shared ridge state is invalid.")
        for value in (
            self.feature_mean,
            self.feature_scale,
            self.coefficients,
            self.normal_inverse,
        ):
            value.setflags(write=False)

    def design(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        query_ids: Sequence[str],
        candidate_ids: Sequence[str | None],
        comparisons: Sequence[Comparison | str],
    ) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        queries = tuple(str(value) for value in query_ids)
        candidates = tuple(value if value is None else str(value) for value in candidate_ids)
        try:
            comparison_values = tuple(Comparison(value).value for value in comparisons)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("HARP v4 prediction comparison is invalid.") from exc
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.feature_names)
            or any(
                len(matrix) != length
                for length in (len(queries), len(candidates), len(comparison_values))
            )
            or not np.isfinite(matrix).all()
        ):
            raise ProtocolError("HARP v4 prediction inputs are invalid or misaligned.")
        standardized = (matrix - self.feature_mean) / self.feature_scale
        query_hot = _one_hot(queries, self.query_levels)
        candidate_hot = _one_hot(candidates, self.candidate_levels)
        comparison_hot = _one_hot(comparison_values, self.comparison_levels)
        return np.column_stack(
            (np.ones(len(matrix)), standardized, query_hot, candidate_hot, comparison_hot)
        )

    def predict(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        query_ids: Sequence[str],
        candidate_ids: Sequence[str | None],
        comparisons: Sequence[Comparison | str],
    ) -> tuple[np.ndarray, np.ndarray]:
        design = self.design(features, query_ids, candidate_ids, comparisons)
        means = design @ self.coefficients
        raw_leverage = np.einsum("ij,jk,ik->i", design, self.normal_inverse, design)
        if (
            not np.isfinite(means).all()
            or not np.isfinite(raw_leverage).all()
            or np.any(raw_leverage < -1e-9)
        ):
            raise ProtocolError("HARP v4 ridge produced invalid predictions or geometry.")
        means = np.asarray(means, dtype=np.float64)
        leverage = np.maximum(np.asarray(raw_leverage, dtype=np.float64), 0.0)
        means.setflags(write=False)
        leverage.setflags(write=False)
        return means, leverage


def _one_hot(values: Sequence[str | None], levels: tuple[str, ...]) -> np.ndarray:
    result = np.zeros((len(values), len(levels)), dtype=np.float64)
    index = {value: position for position, value in enumerate(levels)}
    for row, value in enumerate(values):
        if value in index:
            result[row, index[value]] = 1.0
    return result


def fit_shared_design_ridge(
    observations: Sequence[CaseTrainingObservation],
    *,
    alpha: float,
    excluded_center_ids: Sequence[str],
) -> SharedDesignRidge:
    rows = tuple(observations)
    if not rows or any(not isinstance(row, CaseTrainingObservation) for row in rows):
        raise ProtocolError("HARP v4 ridge requires typed source-development cases.")
    names = rows[0].feature_names
    outer_ids = {row.outer_target_id for row in rows}
    if len(outer_ids) != 1 or any(row.feature_names != names for row in rows):
        raise ProtocolError("HARP v4 ridge rows drifted in outer target or feature schema.")
    excluded = tuple(sorted({str(value) for value in excluded_center_ids}))
    if not excluded or any(
        row.pseudo_query_id in excluded or row.candidate_source_id in excluded for row in rows
    ):
        raise ProtocolError("Excluded query/candidate centers leaked into a HARP v4 fit.")
    penalty = float(alpha)
    if not math.isfinite(penalty) or penalty <= 0:
        raise ProtocolError("HARP v4 ridge alpha must be finite and positive.")

    features = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    responses = np.asarray([row.effects.as_tuple() for row in rows], dtype=np.float64)
    queries = tuple(row.pseudo_query_id for row in rows)
    candidates = tuple(row.candidate_source_id for row in rows)
    cases = tuple(row.case_id for row in rows)
    comparisons = tuple(row.comparison.value for row in rows)
    query_levels = tuple(sorted(set(queries)))
    candidate_levels = tuple(sorted({value for value in candidates if value is not None}))
    comparison_levels = tuple(sorted(value.value for value in Comparison))

    # Equal total mass per pseudo-query and then per independent case.  Multiple
    # candidate comparisons inside one case cannot manufacture replication.
    cases_by_query = {
        query: tuple(sorted({case for q, case in zip(queries, cases, strict=True) if q == query}))
        for query in query_levels
    }
    rows_by_query_case = {
        (query, case): sum(
            q == query and c == case for q, c in zip(queries, cases, strict=True)
        )
        for query in query_levels
        for case in cases_by_query[query]
    }
    weights = np.asarray(
        [
            1.0
            / (
                len(query_levels)
                * len(cases_by_query[query])
                * rows_by_query_case[(query, case)]
            )
            for query, case in zip(queries, cases, strict=True)
        ],
        dtype=np.float64,
    )
    feature_mean = np.average(features, axis=0, weights=weights)
    variance = np.average((features - feature_mean) ** 2, axis=0, weights=weights)
    feature_scale = np.sqrt(np.maximum(variance, 0.0))
    feature_scale[feature_scale <= np.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack(
        (
            np.ones(len(rows)),
            standardized,
            _one_hot(queries, query_levels),
            _one_hot(candidates, candidate_levels),
            _one_hot(comparisons, comparison_levels),
        )
    )
    gram = design.T @ (weights[:, None] * design)
    penalty_matrix = np.diag(
        np.asarray([0.0, *([penalty] * (design.shape[1] - 1))], dtype=np.float64)
    )
    normal = gram + penalty_matrix
    try:
        normal_inverse = np.linalg.inv(normal)
        normal_inverse = 0.5 * (normal_inverse + normal_inverse.T)
        coefficients = normal_inverse @ (design.T @ (weights[:, None] * responses))
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("HARP v4 shared ridge normal equations are singular.") from exc
    return SharedDesignRidge(
        feature_names=names,
        query_levels=query_levels,
        candidate_levels=candidate_levels,
        comparison_levels=comparison_levels,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        normal_inverse=normal_inverse,
        alpha=penalty,
        training_query_ids=tuple(sorted(set(queries))),
        training_candidate_ids=candidate_levels,
        training_case_ids=tuple(sorted(set(cases))),
        excluded_center_ids=excluded,
    )


__all__ = ("SharedDesignRidge", "fit_shared_design_ridge")
