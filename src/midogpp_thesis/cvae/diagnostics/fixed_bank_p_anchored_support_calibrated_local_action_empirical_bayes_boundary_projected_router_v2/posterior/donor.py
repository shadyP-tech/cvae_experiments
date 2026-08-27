"""Center-balanced H/J/d-excluded donor action-value regression."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import (
    ACTION_IDS,
    CENTERS,
    FeatureVector,
    MetricVector,
    array_sha256,
)
from .contracts import DonorFitScope, DonorObservation, ScaleVector


@dataclass(frozen=True, slots=True)
class DonorDeleteCenterFold:
    """One independently reconstructed source-and-query-delete-center fold."""

    deleted_center: str
    base_scope: DonorFitScope
    training_scope: DonorFitScope
    training_observations: tuple[DonorObservation, ...]
    validation_observations: tuple[DonorObservation, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        deleted = str(self.deleted_center)
        training = tuple(self.training_observations)
        validation = tuple(self.validation_observations)
        expected_excluded = tuple(
            center
            for center in CENTERS
            if center in {*self.base_scope.source_excluded_centers, deleted}
        )
        expected_training_cases = {
            center: case_ids
            for center, case_ids in self.base_scope.training_case_ids_by_center.items()
            if center != deleted
        }
        if (
            deleted not in self.base_scope.training_centers
            or self.training_scope.outer_center != self.base_scope.outer_center
            or self.training_scope.prediction_center
            != self.base_scope.prediction_center
            or self.training_scope.held_case_id != self.base_scope.held_case_id
            or self.training_scope.role != self.base_scope.role
            or self.training_scope.source_excluded_centers != expected_excluded
            or dict(self.training_scope.training_case_ids_by_center)
            != expected_training_cases
            or not validation
            or any(
                row.query_center != deleted
                or row.scope_hash != self.base_scope.scope_hash
                or row.source_centers
                != tuple(
                    center
                    for center in self.base_scope.training_centers
                    if center != deleted
                )
                for row in validation
            )
        ):
            raise GovernanceError(
                "SCALE-BP v2 donor delete-center fold scope drifted."
            )
        _validate_observations(training, self.training_scope)
        expected_validation = {
            (deleted, case_id, action_id)
            for case_id in self.base_scope.training_case_ids_by_center[deleted]
            for action_id in ACTION_IDS
        }
        if {
            (row.query_center, row.case_id, row.action_id) for row in validation
        } != expected_validation or len(validation) != len(expected_validation):
            raise GovernanceError(
                "SCALE-BP v2 donor delete-center validation rectangle drifted."
            )
        object.__setattr__(self, "deleted_center", deleted)
        object.__setattr__(self, "training_observations", training)
        object.__setattr__(self, "validation_observations", validation)
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_donor_delete_center_fold_v1",
                    "deleted_center": deleted,
                    "base_scope_hash": self.base_scope.scope_hash,
                    "training_scope_hash": self.training_scope.scope_hash,
                    "training_observation_hashes": tuple(
                        row.observation_hash for row in training
                    ),
                    "validation_observation_hashes": tuple(
                        row.observation_hash for row in validation
                    ),
                    "deleted_center_absent_as_query_and_source_in_training": True,
                    "validation_labels_absent_from_training": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class DonorActionModel:
    scope: DonorFitScope
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    inverse_information: np.ndarray
    residual_variance: np.ndarray
    transport_rmse: ScaleVector
    heterogeneity: ScaleVector
    training_centers: tuple[str, ...]
    training_case_count: int
    training_row_count: int
    ridge_alpha: float
    delete_center_fold_hashes: tuple[str, ...]
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        mean = np.ascontiguousarray(self.feature_mean, dtype=np.float64)
        scale = np.ascontiguousarray(self.feature_scale, dtype=np.float64)
        coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        inverse = np.ascontiguousarray(self.inverse_information, dtype=np.float64)
        residual_variance = np.ascontiguousarray(
            self.residual_variance, dtype=np.float64
        )
        width = len(ACTION_IDS) + len(names)
        if (
            not names
            or len(names) != len(set(names))
            or mean.shape != (len(names),)
            or scale.shape != (len(names),)
            or np.any(scale <= 0.0)
            or coefficients.shape != (3, width)
            or inverse.shape != (width, width)
            or residual_variance.shape != (3,)
            or not all(
                np.isfinite(array).all()
                for array in (mean, scale, coefficients, inverse, residual_variance)
            )
            or np.any(residual_variance < 0.0)
            or self.training_centers != self.scope.training_centers
            or self.training_case_count <= 0
            or self.training_row_count != self.training_case_count * len(ACTION_IDS)
            or len(self.delete_center_fold_hashes) != len(self.training_centers)
            or len(set(self.delete_center_fold_hashes))
            != len(self.delete_center_fold_hashes)
            or not math.isfinite(self.ridge_alpha)
            or self.ridge_alpha <= 0.0
        ):
            raise GovernanceError("SCALE-BP v2 donor action model drifted.")
        for array in (mean, scale, coefficients, inverse, residual_variance):
            array.setflags(write=False)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "inverse_information", inverse)
        object.__setattr__(self, "residual_variance", residual_variance)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_donor_action_model_v2",
                    "scope_hash": self.scope.scope_hash,
                    "feature_names": names,
                    "feature_mean_sha256": array_sha256(mean),
                    "feature_scale_sha256": array_sha256(scale),
                    "coefficients_sha256": array_sha256(coefficients),
                    "inverse_information_sha256": array_sha256(inverse),
                    "residual_variance_sha256": array_sha256(residual_variance),
                    "transport_rmse": self.transport_rmse.to_payload(),
                    "heterogeneity": self.heterogeneity.to_payload(),
                    "training_centers": self.training_centers,
                    "training_case_count": self.training_case_count,
                    "training_row_count": self.training_row_count,
                    "ridge_alpha": self.ridge_alpha,
                    "delete_center_fold_hashes": self.delete_center_fold_hashes,
                    "equal_total_weight_per_center": True,
                    "equal_total_weight_per_case": True,
                    "action_intercepts_unpenalized": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DonorPrediction:
    action_id: str
    descriptor_hash: str
    mean: MetricVector
    transport_rmse: ScaleVector
    heterogeneity: ScaleVector
    estimator_se: ScaleVector
    model_hash: str
    scope_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.action_id not in ACTION_IDS or not all(
            (self.descriptor_hash, self.model_hash, self.scope_hash)
        ):
            raise GovernanceError("SCALE-BP v2 donor prediction identity drifted.")
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_donor_prediction_v1",
                    "action_id": self.action_id,
                    "descriptor_hash": self.descriptor_hash,
                    "mean": self.mean.to_payload(),
                    "transport_rmse": self.transport_rmse.to_payload(),
                    "heterogeneity": self.heterogeneity.to_payload(),
                    "estimator_se": self.estimator_se.to_payload(),
                    "model_hash": self.model_hash,
                    "scope_hash": self.scope_hash,
                    "uncertainty_components_collapsed": False,
                }
            ),
        )


def fit_donor_action_model(
    observations: Sequence[DonorObservation],
    *,
    scope: DonorFitScope,
    delete_center_folds: Sequence[DonorDeleteCenterFold],
    ridge_alpha: float = 1.0,
) -> DonorActionModel:
    rows = tuple(observations)
    _validate_observations(rows, scope)
    names = rows[0].descriptor.names
    mean, scale, coefficients, inverse, variance = _fit_parameters(
        rows, names, ridge_alpha=float(ridge_alpha)
    )

    folds = tuple(delete_center_folds)
    if (
        tuple(fold.deleted_center for fold in folds) != scope.training_centers
        or any(fold.base_scope.scope_hash != scope.scope_hash for fold in folds)
        or len({fold.fold_hash for fold in folds}) != len(folds)
    ):
        raise GovernanceError("SCALE-BP v2 donor delete-center fold universe drifted.")
    center_mse: list[np.ndarray] = []
    center_bias: list[np.ndarray] = []
    for fold in folds:
        local = _fit_parameters(
            fold.training_observations,
            names,
            ridge_alpha=float(ridge_alpha),
        )
        errors: list[np.ndarray] = []
        for row in fold.validation_observations:
            if row.descriptor.names != names:
                raise GovernanceError(
                    "SCALE-BP v2 donor fold descriptor schema drifted."
                )
            predicted, _ = _predict_parameters(
                row.action_id, row.descriptor, *local[:2], local[2], local[3], local[4]
            )
            errors.append(row.realized.as_array() - predicted)
        error_array = np.asarray(errors, dtype=np.float64)
        center_mse.append(
            np.mean(error_array * error_array, axis=0, dtype=np.float64)
        )
        center_bias.append(np.mean(error_array, axis=0, dtype=np.float64))
    transport_rmse = np.sqrt(
        np.mean(center_mse, axis=0, dtype=np.float64)
    )
    heterogeneity = np.std(
        np.asarray(center_bias, dtype=np.float64),
        axis=0,
        ddof=0,
        dtype=np.float64,
    )
    return DonorActionModel(
        scope,
        names,
        mean,
        scale,
        coefficients,
        inverse,
        variance,
        ScaleVector.from_values(transport_rmse),
        ScaleVector.from_values(heterogeneity),
        scope.training_centers,
        len({(row.query_center, row.case_id) for row in rows}),
        len(rows),
        float(ridge_alpha),
        tuple(fold.fold_hash for fold in folds),
    )


def predict_donor_action(
    model: DonorActionModel,
    *,
    action_id: object,
    descriptor: FeatureVector,
) -> DonorPrediction:
    action = str(action_id)
    if descriptor.names != model.feature_names or action not in ACTION_IDS:
        raise GovernanceError("SCALE-BP v2 donor prediction schema drifted.")
    predicted, estimator_se = _predict_parameters(
        action,
        descriptor,
        model.feature_mean,
        model.feature_scale,
        model.coefficients,
        model.inverse_information,
        model.residual_variance,
    )
    return DonorPrediction(
        action,
        descriptor.feature_hash,
        MetricVector.from_array(predicted),
        model.transport_rmse,
        model.heterogeneity,
        ScaleVector.from_values(estimator_se),
        model.model_hash,
        model.scope.scope_hash,
    )


def assess_donor_support(
    model: DonorActionModel,
    descriptor: FeatureVector,
    *,
    maximum_abs_standardized_feature: float = 4.0,
    minimum_independent_centers: int = 6,
) -> tuple[bool, bool]:
    """Return explicit descriptor-support and independent-center viability gates."""

    limit = float(maximum_abs_standardized_feature)
    if (
        descriptor.names != model.feature_names
        or not math.isfinite(limit)
        or limit <= 0.0
        or minimum_independent_centers < 3
    ):
        raise GovernanceError("SCALE-BP v2 donor support gate inputs drifted.")
    standardized = np.abs(
        (descriptor.as_array() - model.feature_mean) / model.feature_scale
    )
    within_support = bool(np.all(standardized <= limit))
    bank_viable = bool(
        len(model.training_centers) >= int(minimum_independent_centers)
        and all(
            math.isfinite(value)
            for value in (
                *model.transport_rmse.as_tuple(),
                *model.heterogeneity.as_tuple(),
            )
        )
    )
    return within_support, bank_viable


def _validate_observations(
    rows: tuple[DonorObservation, ...], scope: DonorFitScope
) -> None:
    if not rows:
        raise GovernanceError("SCALE-BP v2 donor observations are empty.")
    expected = {
        (center, case_id, action_id)
        for center, case_ids in scope.training_case_ids_by_center.items()
        for case_id in case_ids
        for action_id in ACTION_IDS
    }
    actual = {(row.query_center, row.case_id, row.action_id) for row in rows}
    excluded = set(scope.source_excluded_centers)
    names = rows[0].descriptor.names
    if (
        len(actual) != len(rows)
        or actual != expected
        or any(row.scope_hash != scope.scope_hash for row in rows)
        or any(row.descriptor.names != names for row in rows)
        or any(set(row.source_centers) & excluded for row in rows)
        or any(row.query_center in excluded for row in rows)
        or any(
            row.source_centers
            != tuple(
                center
                for center in scope.training_centers
                if center != row.query_center
            )
            for row in rows
        )
    ):
        raise GovernanceError("SCALE-BP v2 donor H/J/d observation rectangle drifted.")


def _fit_parameters(
    rows: Sequence[DonorObservation],
    names: tuple[str, ...],
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise GovernanceError("SCALE-BP v2 donor ridge alpha is invalid.")
    features = np.asarray([row.descriptor.values for row in rows], dtype=np.float64)
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    design = _design(rows, mean, scale)
    response = np.asarray([row.realized.as_array() for row in rows], dtype=np.float64)
    weights = _center_case_action_weights(rows)
    root = np.sqrt(weights)[:, None]
    weighted_design = design * root
    penalty = np.diag(
        np.asarray(
            [1.0e-12] * len(ACTION_IDS) + [ridge_alpha] * len(names),
            dtype=np.float64,
        )
    )
    information = weighted_design.T @ weighted_design + penalty
    try:
        inverse = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise GovernanceError("SCALE-BP v2 donor ridge information is singular.") from exc
    coefficients = (inverse @ weighted_design.T @ (response * root)).T
    residual = response - design @ coefficients.T
    residual_variance = np.average(residual * residual, axis=0, weights=weights)
    if not all(
        np.isfinite(value).all()
        for value in (mean, scale, inverse, coefficients, residual_variance)
    ):
        raise GovernanceError("SCALE-BP v2 donor ridge is nonfinite.")
    return mean, scale, coefficients, inverse, residual_variance


def _design(
    rows: Sequence[DonorObservation], mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(rows), len(ACTION_IDS) + len(mean)), dtype=np.float64)
    for index, row in enumerate(rows):
        result[index, ACTION_IDS.index(row.action_id)] = 1.0
        result[index, len(ACTION_IDS) :] = (row.descriptor.as_array() - mean) / scale
    return result


def _center_case_action_weights(rows: Sequence[DonorObservation]) -> np.ndarray:
    centers = tuple(dict.fromkeys(row.query_center for row in rows))
    weights = np.zeros(len(rows), dtype=np.float64)
    for center in centers:
        cases = tuple(dict.fromkeys(row.case_id for row in rows if row.query_center == center))
        for case in cases:
            positions = [
                index
                for index, row in enumerate(rows)
                if row.query_center == center and row.case_id == case
            ]
            value = 1.0 / (len(centers) * len(cases) * len(positions))
            weights[positions] = value
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0):
        raise GovernanceError("SCALE-BP v2 donor weights drifted.")
    return weights


def _predict_parameters(
    action_id: str,
    descriptor: FeatureVector,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    inverse: np.ndarray,
    residual_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.zeros(len(ACTION_IDS) + len(mean), dtype=np.float64)
    design[ACTION_IDS.index(action_id)] = 1.0
    design[len(ACTION_IDS) :] = (descriptor.as_array() - mean) / scale
    predicted = coefficients @ design
    leverage = max(0.0, float(design @ inverse @ design))
    estimator_se = np.sqrt(np.maximum(0.0, residual_variance * leverage))
    if not np.isfinite(predicted).all() or not np.isfinite(estimator_se).all():
        raise GovernanceError("SCALE-BP v2 donor prediction is nonfinite.")
    return predicted, estimator_se


__all__ = (
    "assess_donor_support",
    "DonorActionModel",
    "DonorDeleteCenterFold",
    "DonorPrediction",
    "fit_donor_action_model",
    "predict_donor_action",
)
