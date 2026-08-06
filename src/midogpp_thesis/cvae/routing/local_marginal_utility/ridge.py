"""Query-cluster-balanced ridge models and leakage-safe nested LOQDO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


DEFAULT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
PSD_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RidgePrediction:
    mean: np.ndarray
    standard_error: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class ClusterWeightedRidgeModel:
    """A fitted ridge model with query-cluster sandwich covariance."""

    feature_names: tuple[str, ...]
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    residual_variance: float
    training_query_clusters: tuple[str, ...]
    observation_count: int
    effective_rank: int

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        matrix = _prediction_matrix(features, len(self.feature_names))
        standardized = (matrix - self.feature_mean) / self.feature_scale
        result = self.intercept + standardized @ self.coefficients
        if not np.isfinite(result).all():
            raise ProtocolError("Ridge prediction is non-finite.")
        result = np.asarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def predict_with_uncertainty(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        *,
        include_residual_variance: bool = False,
    ) -> RidgePrediction:
        matrix = _prediction_matrix(features, len(self.feature_names))
        standardized = (matrix - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(matrix), dtype=np.float64), standardized))
        covariance = design @ self.coefficient_covariance @ design.T
        covariance = 0.5 * (covariance + covariance.T)
        if include_residual_variance:
            covariance += np.eye(len(matrix), dtype=np.float64) * self.residual_variance
        covariance = validate_psd_covariance(
            covariance, dimension=len(matrix), name="ridge prediction covariance"
        )
        mean = self.predict(matrix)
        standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        standard_error.setflags(write=False)
        return RidgePrediction(
            mean=mean,
            standard_error=standard_error,
            covariance=covariance,
        )


@dataclass(frozen=True)
class LOQDOFold:
    heldout_query_cluster: str
    heldout_row_indices: tuple[int, ...]
    training_source_clusters: tuple[str, ...]
    strict_source_domain_exclusion: bool
    selected_alpha: float
    inner_mse_by_alpha: Mapping[float, float]
    model: ClusterWeightedRidgeModel
    prediction: RidgePrediction


@dataclass(frozen=True)
class NestedLOQDOResult:
    """Out-of-query predictions with all tuning nested inside each outer fold."""

    folds: tuple[LOQDOFold, ...]
    predictions: np.ndarray
    standard_errors: np.ndarray
    residuals: np.ndarray
    query_equal_mean_squared_error: float

    @property
    def selected_alpha_by_query(self) -> dict[str, float]:
        return {
            fold.heldout_query_cluster: fold.selected_alpha for fold in self.folds
        }


def fit_cluster_weighted_ridge(
    features: Sequence[Sequence[float]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    query_clusters: Sequence[str],
    *,
    alpha: float,
    feature_names: Sequence[str] | None = None,
) -> ClusterWeightedRidgeModel:
    """Fit ridge with equal total weight for every query cluster.

    Feature centering and scaling are learned from the supplied rows only.  The
    intercept is unpenalized.  Coefficient uncertainty uses a query-cluster
    sandwich estimator, with a model-based finite fallback only when a single
    training cluster makes a cluster sandwich unidentified.
    """

    matrix, response, clusters, names = _validated_training_inputs(
        features, utility, query_clusters, feature_names
    )
    penalty = float(alpha)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ProtocolError("Ridge alpha must be finite and strictly positive.")
    unique_clusters = tuple(sorted(set(clusters)))
    count_by_cluster = {
        cluster: sum(value == cluster for value in clusters) for cluster in unique_clusters
    }
    n_rows = len(matrix)
    n_clusters = len(unique_clusters)
    # Sum(weights) == n_rows while each cluster has total n_rows/n_clusters.
    weights = np.asarray(
        [n_rows / (n_clusters * count_by_cluster[cluster]) for cluster in clusters],
        dtype=np.float64,
    )
    weight_sum = float(weights.sum())
    mean = np.sum(matrix * weights[:, None], axis=0) / weight_sum
    centered = matrix - mean
    variance = np.sum(centered * centered * weights[:, None], axis=0) / weight_sum
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= np.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardized = centered / scale
    design = np.column_stack((np.ones(n_rows, dtype=np.float64), standardized))
    gram = design.T @ (weights[:, None] * design)
    penalty_matrix = np.diag(
        np.asarray([0.0, *([penalty] * matrix.shape[1])], dtype=np.float64)
    )
    normal = gram + penalty_matrix
    rhs = design.T @ (weights * response)
    try:
        theta = np.linalg.solve(normal, rhs)
        inverse_normal = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("Ridge normal equations are singular.") from exc
    fitted = design @ theta
    residuals = response - fitted
    rank = int(np.linalg.matrix_rank(design))
    residual_variance = float(
        np.dot(weights, residuals * residuals) / max(float(n_rows - rank), 1.0)
    )
    if n_clusters >= 2:
        meat = np.zeros_like(normal)
        for cluster in unique_clusters:
            mask = np.fromiter((value == cluster for value in clusters), dtype=bool)
            score = design[mask].T @ (weights[mask] * residuals[mask])
            meat += np.outer(score, score)
        correction = float(n_clusters / (n_clusters - 1))
        if n_rows > rank:
            correction *= float((n_rows - 1) / (n_rows - rank))
        covariance = correction * (inverse_normal @ meat @ inverse_normal)
    else:
        covariance = residual_variance * (inverse_normal @ gram @ inverse_normal)
    covariance = validate_psd_covariance(
        covariance,
        dimension=design.shape[1],
        name="ridge coefficient covariance",
    )
    for value in (mean, scale, theta):
        value.setflags(write=False)
    return ClusterWeightedRidgeModel(
        feature_names=names,
        alpha=penalty,
        feature_mean=mean,
        feature_scale=scale,
        intercept=float(theta[0]),
        coefficients=theta[1:],
        coefficient_covariance=covariance,
        residual_variance=residual_variance,
        training_query_clusters=unique_clusters,
        observation_count=n_rows,
        effective_rank=rank,
    )


def select_alpha_by_inner_loqdo(
    features: Sequence[Sequence[float]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    query_clusters: Sequence[str],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    feature_names: Sequence[str] | None = None,
    source_clusters: Sequence[str] | None = None,
) -> tuple[float, dict[float, float]]:
    """Select alpha by strict leave-one-domain-out mean squared error.

    When ``source_clusters`` is supplied, an inner-heldout domain is excluded
    from both roles: validation rows have that query domain and a different
    source, while training rows contain neither that query nor that source.
    Omitting ``source_clusters`` preserves query-row-only behavior for callers
    whose source role is not represented in the table.
    """

    matrix, response, clusters, names = _validated_training_inputs(
        features, utility, query_clusters, feature_names
    )
    unique_clusters = tuple(sorted(set(clusters)))
    if len(unique_clusters) < 2:
        raise ProtocolError("Inner LOQDO requires at least two query clusters.")
    candidates = _validated_alphas(alphas)
    losses: dict[float, float] = {}
    cluster_array = np.asarray(clusters, dtype=object)
    source_array = _validated_source_clusters(source_clusters, row_count=len(matrix))
    for alpha in candidates:
        fold_losses: list[float] = []
        for heldout in unique_clusters:
            query_test_mask = cluster_array == heldout
            if source_array is None:
                test_mask = query_test_mask
                train_mask = ~test_mask
            else:
                if np.any(query_test_mask & (source_array == heldout)):
                    raise ProtocolError(
                        "Strict LOQDO validation contains the held-out domain as source."
                    )
                test_mask = query_test_mask & (source_array != heldout)
                train_mask = (cluster_array != heldout) & (source_array != heldout)
            if not np.any(test_mask) or not np.any(train_mask):
                raise ProtocolError(
                    "Strict inner LOQDO produced an empty train or validation fold."
                )
            model = fit_cluster_weighted_ridge(
                matrix[train_mask],
                response[train_mask],
                tuple(cluster_array[train_mask]),
                alpha=alpha,
                feature_names=names,
            )
            residual = model.predict(matrix[test_mask]) - response[test_mask]
            fold_losses.append(float(np.mean(residual * residual, dtype=np.float64)))
        losses[alpha] = float(np.mean(fold_losses, dtype=np.float64))
    selected = min(candidates, key=lambda alpha: (losses[alpha], alpha))
    return selected, losses


def nested_loqdo_predictions(
    features: Sequence[Sequence[float]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    query_clusters: Sequence[str],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    feature_names: Sequence[str] | None = None,
    include_residual_variance: bool = False,
    source_clusters: Sequence[str] | None = None,
) -> NestedLOQDOResult:
    """Produce leakage-safe outer-LOQDO predictions and uncertainty.

    For each outer query, alpha selection is rerun by LOQDO using only the
    remaining query clusters.  When ``source_clusters`` is supplied, the held
    domain is removed from both query and expert-source roles in every outer
    and inner fold.  Thus neither its utility nor a row using its expert can
    influence fitting, standardization, covariance, or hyperparameters.
    """

    matrix, response, clusters, names = _validated_training_inputs(
        features, utility, query_clusters, feature_names
    )
    unique_clusters = tuple(sorted(set(clusters)))
    if len(unique_clusters) < 3:
        raise ProtocolError("Nested LOQDO requires at least three query clusters.")
    candidates = _validated_alphas(alphas)
    cluster_array = np.asarray(clusters, dtype=object)
    source_array = _validated_source_clusters(source_clusters, row_count=len(matrix))
    predictions = np.empty(len(matrix), dtype=np.float64)
    standard_errors = np.empty(len(matrix), dtype=np.float64)
    folds: list[LOQDOFold] = []
    fold_mse: list[float] = []
    for heldout in unique_clusters:
        query_test_mask = cluster_array == heldout
        if source_array is None:
            test_mask = query_test_mask
            train_mask = ~test_mask
            training_sources: tuple[str, ...] = ()
        else:
            if np.any(query_test_mask & (source_array == heldout)):
                raise ProtocolError(
                    "Strict LOQDO validation contains the held-out domain as source."
                )
            test_mask = query_test_mask & (source_array != heldout)
            train_mask = (cluster_array != heldout) & (source_array != heldout)
            training_sources = tuple(sorted(set(source_array[train_mask])))
        if not np.any(test_mask) or not np.any(train_mask):
            raise ProtocolError(
                "Strict outer LOQDO produced an empty train or validation fold."
            )
        selected, losses = select_alpha_by_inner_loqdo(
            matrix[train_mask],
            response[train_mask],
            tuple(cluster_array[train_mask]),
            alphas=candidates,
            feature_names=names,
            source_clusters=(
                tuple(source_array[train_mask]) if source_array is not None else None
            ),
        )
        model = fit_cluster_weighted_ridge(
            matrix[train_mask],
            response[train_mask],
            tuple(cluster_array[train_mask]),
            alpha=selected,
            feature_names=names,
        )
        prediction = model.predict_with_uncertainty(
            matrix[test_mask], include_residual_variance=include_residual_variance
        )
        predictions[test_mask] = prediction.mean
        standard_errors[test_mask] = prediction.standard_error
        heldout_indices = tuple(int(index) for index in np.flatnonzero(test_mask))
        fold_mse.append(
            float(np.mean((prediction.mean - response[test_mask]) ** 2, dtype=np.float64))
        )
        folds.append(
            LOQDOFold(
                heldout_query_cluster=heldout,
                heldout_row_indices=heldout_indices,
                training_source_clusters=training_sources,
                strict_source_domain_exclusion=source_array is not None,
                selected_alpha=selected,
                inner_mse_by_alpha=losses,
                model=model,
                prediction=prediction,
            )
        )
    residuals = response - predictions
    for value in (predictions, standard_errors, residuals):
        if not np.isfinite(value).all():
            raise ProtocolError("Nested LOQDO produced non-finite output.")
        value.setflags(write=False)
    return NestedLOQDOResult(
        folds=tuple(folds),
        predictions=predictions,
        standard_errors=standard_errors,
        residuals=residuals,
        query_equal_mean_squared_error=float(np.mean(fold_mse, dtype=np.float64)),
    )


def validate_psd_covariance(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dimension: int,
    name: str = "covariance",
) -> np.ndarray:
    """Validate a finite symmetric PSD covariance and remove roundoff only."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ProtocolError(f"{name} must be a finite {dimension}x{dimension} matrix.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ProtocolError(f"{name} must be symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -PSD_RELATIVE_TOLERANCE * scale:
        raise ProtocolError(f"{name} must be positive semidefinite.")
    clipped = np.maximum(eigenvalues, 0.0)
    result = (eigenvectors * clipped) @ eigenvectors.T
    result = 0.5 * (result + result.T)
    if not np.isfinite(result).all():
        raise ProtocolError(f"{name} PSD projection is non-finite.")
    result.setflags(write=False)
    return result


def _validated_training_inputs(
    features: Sequence[Sequence[float]] | np.ndarray,
    utility: Sequence[float] | np.ndarray,
    query_clusters: Sequence[str],
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    matrix = np.asarray(features, dtype=np.float64)
    response = np.asarray(utility, dtype=np.float64)
    clusters = tuple(str(value) for value in query_clusters)
    if (
        matrix.ndim != 2
        or not len(matrix)
        or response.shape != (len(matrix),)
        or len(clusters) != len(matrix)
        or any(not cluster for cluster in clusters)
        or not np.isfinite(matrix).all()
        or not np.isfinite(response).all()
    ):
        raise ProtocolError("Cluster-weighted ridge inputs are invalid or misaligned.")
    if feature_names is None:
        names = tuple(f"feature_{index}" for index in range(matrix.shape[1]))
    else:
        names = tuple(str(value) for value in feature_names)
    if (
        len(names) != matrix.shape[1]
        or len(set(names)) != len(names)
        or any(not value for value in names)
    ):
        raise ProtocolError("Ridge feature names must be unique and aligned.")
    return matrix, response, clusters, names


def _prediction_matrix(
    features: Sequence[Sequence[float]] | np.ndarray, dimension: int
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or not np.isfinite(matrix).all():
        raise ProtocolError("Ridge prediction features have invalid geometry.")
    return matrix


def _validated_source_clusters(
    values: Sequence[str] | None,
    *,
    row_count: int,
) -> np.ndarray | None:
    if values is None:
        return None
    normalized = tuple(str(value) for value in values)
    if len(normalized) != row_count or any(not value for value in normalized):
        raise ProtocolError("Source-cluster roles must be nonempty and row-aligned.")
    return np.asarray(normalized, dtype=object)


def _validated_alphas(values: Sequence[float]) -> tuple[float, ...]:
    candidates = tuple(sorted(set(float(value) for value in values)))
    if not candidates or any(not np.isfinite(value) or value <= 0.0 for value in candidates):
        raise ProtocolError("Nested LOQDO alphas must be finite and strictly positive.")
    return candidates


__all__ = (
    "DEFAULT_RIDGE_ALPHAS",
    "PSD_RELATIVE_TOLERANCE",
    "ClusterWeightedRidgeModel",
    "LOQDOFold",
    "NestedLOQDOResult",
    "RidgePrediction",
    "fit_cluster_weighted_ridge",
    "nested_loqdo_predictions",
    "select_alpha_by_inner_loqdo",
    "validate_psd_covariance",
)
