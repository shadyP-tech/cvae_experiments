"""Route-local H-minus-c four-fold OOF residual correction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, FeatureVector, MetricVector, array_sha256
from .contracts import LocalResidualObservation, ScaleVector
from .donor import DonorPrediction
from ..utility.metrics import ScoredActionRectangle


LOCAL_FOLD_COUNT = 4


@dataclass(frozen=True, slots=True, eq=False)
class LocalResidualModel:
    target_center: str
    route_case_id: str
    support_case_ids: tuple[str, ...]
    support_scope_hash: str
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    inverse_information: np.ndarray
    residual_variance: np.ndarray
    oof_rmse: ScaleVector
    fold_heterogeneity: ScaleVector
    fold_assignments: tuple[tuple[str, int], ...]
    ridge_alpha: float
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        mean = np.ascontiguousarray(self.feature_mean, dtype=np.float64)
        scale = np.ascontiguousarray(self.feature_scale, dtype=np.float64)
        coefficients = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        inverse = np.ascontiguousarray(self.inverse_information, dtype=np.float64)
        variance = np.ascontiguousarray(self.residual_variance, dtype=np.float64)
        width = len(ACTION_IDS) + len(names)
        if (
            not self.target_center
            or not self.route_case_id
            or self.route_case_id in self.support_case_ids
            or len(self.support_case_ids) < LOCAL_FOLD_COUNT
            or len(self.support_case_ids) != len(set(self.support_case_ids))
            or not self.support_scope_hash
            or mean.shape != (len(names),)
            or scale.shape != (len(names),)
            or coefficients.shape != (3, width)
            or inverse.shape != (width, width)
            or variance.shape != (3,)
            or np.any(scale <= 0.0)
            or np.any(variance < 0.0)
            or not all(np.isfinite(array).all() for array in (mean, scale, coefficients, inverse, variance))
            or tuple(case for case, _ in self.fold_assignments) != self.support_case_ids
            or {fold for _, fold in self.fold_assignments} != set(range(LOCAL_FOLD_COUNT))
            or not math.isfinite(self.ridge_alpha)
            or self.ridge_alpha <= 0.0
        ):
            raise GovernanceError("SCALE-BP v2 local residual model drifted.")
        for array in (mean, scale, coefficients, inverse, variance):
            array.setflags(write=False)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "inverse_information", inverse)
        object.__setattr__(self, "residual_variance", variance)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_local_residual_model_v1",
                    "target_center": self.target_center,
                    "route_case_id": self.route_case_id,
                    "support_case_ids": self.support_case_ids,
                    "support_scope_hash": self.support_scope_hash,
                    "feature_names": names,
                    "feature_mean_sha256": array_sha256(mean),
                    "feature_scale_sha256": array_sha256(scale),
                    "coefficients_sha256": array_sha256(coefficients),
                    "inverse_information_sha256": array_sha256(inverse),
                    "residual_variance_sha256": array_sha256(variance),
                    "oof_rmse": self.oof_rmse.to_payload(),
                    "fold_heterogeneity": self.fold_heterogeneity.to_payload(),
                    "fold_assignments": self.fold_assignments,
                    "fold_count": LOCAL_FOLD_COUNT,
                    "route_local_only": True,
                    "shared_model_updated": False,
                    "ridge_alpha": self.ridge_alpha,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalResidualPrediction:
    action_id: str
    descriptor_hash: str
    correction: MetricVector
    oof_rmse: ScaleVector
    fold_heterogeneity: ScaleVector
    estimator_se: ScaleVector
    model_hash: str
    support_scope_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.action_id not in ACTION_IDS or not all(
            (self.descriptor_hash, self.model_hash, self.support_scope_hash)
        ):
            raise GovernanceError("SCALE-BP v2 local prediction identity drifted.")
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_local_residual_prediction_v1",
                    "action_id": self.action_id,
                    "descriptor_hash": self.descriptor_hash,
                    "correction": self.correction.to_payload(),
                    "oof_rmse": self.oof_rmse.to_payload(),
                    "fold_heterogeneity": self.fold_heterogeneity.to_payload(),
                    "estimator_se": self.estimator_se.to_payload(),
                    "model_hash": self.model_hash,
                    "support_scope_hash": self.support_scope_hash,
                }
            ),
        )


def fit_route_local_residual(
    observations: Sequence[LocalResidualObservation],
    *,
    ridge_alpha: float = 1.0,
) -> LocalResidualModel:
    rows = tuple(observations)
    target, route_case, support_cases, scope_hash, names = _validate_rows(rows)
    fold_assignments = assign_local_support_folds(support_cases)
    fold_by_case = dict(fold_assignments)
    oof_error_by_fold: list[np.ndarray] = []
    all_oof_errors: list[np.ndarray] = []
    for fold in range(LOCAL_FOLD_COUNT):
        training = tuple(
            row for row in rows if fold_by_case[row.support_case_id] != fold
        )
        validation = tuple(
            row for row in rows if fold_by_case[row.support_case_id] == fold
        )
        if not training or not validation:
            raise GovernanceError("SCALE-BP v2 local OOF fold is empty.")
        fitted = _fit(training, names, float(ridge_alpha))
        errors = []
        for row in validation:
            predicted, _ = _predict(row.action_id, row.descriptor, *fitted)
            error = row.residual.as_array() - predicted
            errors.append(error)
            all_oof_errors.append(error)
        oof_error_by_fold.append(np.mean(errors, axis=0, dtype=np.float64))
    errors = np.asarray(all_oof_errors, dtype=np.float64)
    fold_errors = np.asarray(oof_error_by_fold, dtype=np.float64)
    oof_rmse = np.sqrt(np.mean(errors * errors, axis=0, dtype=np.float64))
    heterogeneity = np.std(fold_errors, axis=0, ddof=0, dtype=np.float64)
    mean, scale, coefficients, inverse, variance = _fit(
        rows, names, float(ridge_alpha)
    )
    return LocalResidualModel(
        target,
        route_case,
        support_cases,
        scope_hash,
        names,
        mean,
        scale,
        coefficients,
        inverse,
        variance,
        ScaleVector.from_values(oof_rmse),
        ScaleVector.from_values(heterogeneity),
        fold_assignments,
        float(ridge_alpha),
    )


def build_local_residual_observations(
    scored_rectangles: Sequence[ScoredActionRectangle],
    donor_predictions: Mapping[tuple[str, str], DonorPrediction],
    *,
    target_center: object,
    route_case_id: object,
    support_scope_hash: object,
) -> tuple[LocalResidualObservation, ...]:
    target, route_case = str(target_center), str(route_case_id)
    scored = tuple(scored_rectangles)
    if (
        not scored
        or any(
            row.rectangle.target_center != target
            or row.rectangle.case_id == route_case
            for row in scored
        )
        or len({row.rectangle.case_id for row in scored}) != len(scored)
        or not str(support_scope_hash)
    ):
        raise GovernanceError("SCALE-BP v2 local support rectangles drifted.")
    expected = {
        (row.rectangle.case_id, action_id)
        for row in scored
        for action_id in ACTION_IDS
    }
    if set(donor_predictions) != expected:
        raise GovernanceError("SCALE-BP v2 local donor prediction rectangle drifted.")
    ordered_cases = tuple(
        sorted(row.rectangle.case_id for row in scored)
    )
    fold_by_case = dict(assign_local_support_folds(ordered_cases))
    for row in scored:
        evaluation_case = row.rectangle.case_id
        expected_excluded = {
            route_case,
            *(
                case
                for case in ordered_cases
                if fold_by_case[case] == fold_by_case[evaluation_case]
            ),
        }
        if (
            row.rectangle.outer_held_case_id != route_case
            or set(row.rectangle.support_excluded_case_ids) != expected_excluded
        ):
            raise GovernanceError(
                "SCALE-BP v2 local rectangle used labels from its own fold."
            )
    output: list[LocalResidualObservation] = []
    for row in sorted(scored, key=lambda value: value.rectangle.case_id):
        realized = {value.action_id: value for value in row.values}
        for action_id in ACTION_IDS:
            cell = row.rectangle.cell(action_id)
            donor = donor_predictions[(row.rectangle.case_id, action_id)]
            if (
                donor.action_id != action_id
                or donor.descriptor_hash != cell.evidence.descriptor.feature_hash
            ):
                raise GovernanceError("SCALE-BP v2 local donor lineage drifted.")
            residual = realized[action_id].value.as_array() - donor.mean.as_array()
            output.append(
                LocalResidualObservation(
                    target,
                    route_case,
                    row.rectangle.case_id,
                    action_id,
                    cell.evidence.descriptor,
                    MetricVector.from_array(residual),
                    donor.prediction_hash,
                    str(support_scope_hash),
                    row.rectangle.endpoint_plan_hash,
                    row.rectangle.support_excluded_case_ids,
                    row.rectangle.outer_held_case_id,
                )
            )
    return tuple(output)


def predict_local_residual(
    model: LocalResidualModel,
    *,
    action_id: object,
    descriptor: FeatureVector,
) -> LocalResidualPrediction:
    action = str(action_id)
    if action not in ACTION_IDS or descriptor.names != model.feature_names:
        raise GovernanceError("SCALE-BP v2 local prediction schema drifted.")
    prediction, estimator_se = _predict(
        action,
        descriptor,
        model.feature_mean,
        model.feature_scale,
        model.coefficients,
        model.inverse_information,
        model.residual_variance,
    )
    return LocalResidualPrediction(
        action,
        descriptor.feature_hash,
        MetricVector.from_array(prediction),
        model.oof_rmse,
        model.fold_heterogeneity,
        ScaleVector.from_values(estimator_se),
        model.model_hash,
        model.support_scope_hash,
    )


def _validate_rows(
    rows: tuple[LocalResidualObservation, ...]
) -> tuple[str, str, tuple[str, ...], str, tuple[str, ...]]:
    if not rows:
        raise GovernanceError("SCALE-BP v2 local observations are empty.")
    target = rows[0].target_center
    route_case = rows[0].route_case_id
    scope_hash = rows[0].support_scope_hash
    names = rows[0].descriptor.names
    support_cases = tuple(sorted({row.support_case_id for row in rows}))
    fold_by_case = dict(assign_local_support_folds(support_cases))
    expected = {(case, action) for case in support_cases for action in ACTION_IDS}
    actual = {(row.support_case_id, row.action_id) for row in rows}
    if (
        len(support_cases) < LOCAL_FOLD_COUNT
        or len(actual) != len(rows)
        or actual != expected
        or any(
            row.target_center != target
            or row.route_case_id != route_case
            or row.support_scope_hash != scope_hash
            or row.descriptor.names != names
            or row.outer_held_case_id != route_case
            or set(row.support_excluded_case_ids)
            != {
                route_case,
                *(
                    case
                    for case in support_cases
                    if fold_by_case[case] == fold_by_case[row.support_case_id]
                ),
            }
            for row in rows
        )
    ):
        raise GovernanceError("SCALE-BP v2 H-minus-c local rectangle drifted.")
    return target, route_case, support_cases, scope_hash, names


def assign_local_support_folds(
    support_case_ids: Sequence[object],
) -> tuple[tuple[str, int], ...]:
    """Return the canonical deterministic whole-case four-fold assignment."""

    cases = tuple(sorted(str(value) for value in support_case_ids))
    if (
        len(cases) < LOCAL_FOLD_COUNT
        or len(cases) != len(set(cases))
        or any(not case for case in cases)
    ):
        raise GovernanceError("SCALE-BP v2 local support-case inventory drifted.")
    return tuple(
        (case_id, index % LOCAL_FOLD_COUNT)
        for index, case_id in enumerate(cases)
    )


def _fit(
    rows: Sequence[LocalResidualObservation],
    names: tuple[str, ...],
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise GovernanceError("SCALE-BP v2 local ridge alpha is invalid.")
    features = np.asarray([row.descriptor.values for row in rows], dtype=np.float64)
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    design = np.zeros((len(rows), len(ACTION_IDS) + len(names)), dtype=np.float64)
    for index, row in enumerate(rows):
        design[index, ACTION_IDS.index(row.action_id)] = 1.0
        design[index, len(ACTION_IDS) :] = (row.descriptor.as_array() - mean) / scale
    response = np.asarray([row.residual.as_array() for row in rows], dtype=np.float64)
    weights = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
    root = np.sqrt(weights)[:, None]
    weighted = design * root
    penalty = np.diag(
        np.asarray(
            [1.0e-12] * len(ACTION_IDS) + [ridge_alpha] * len(names),
            dtype=np.float64,
        )
    )
    information = weighted.T @ weighted + penalty
    try:
        inverse = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise GovernanceError("SCALE-BP v2 local ridge is singular.") from exc
    coefficients = (inverse @ weighted.T @ (response * root)).T
    residual = response - design @ coefficients.T
    variance = np.average(residual * residual, axis=0, weights=weights)
    return mean, scale, coefficients, inverse, variance


def _predict(
    action_id: str,
    descriptor: FeatureVector,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    inverse: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.zeros(len(ACTION_IDS) + len(mean), dtype=np.float64)
    design[ACTION_IDS.index(action_id)] = 1.0
    design[len(ACTION_IDS) :] = (descriptor.as_array() - mean) / scale
    predicted = coefficients @ design
    leverage = max(0.0, float(design @ inverse @ design))
    estimator_se = np.sqrt(np.maximum(0.0, variance * leverage))
    if not np.isfinite(predicted).all() or not np.isfinite(estimator_se).all():
        raise GovernanceError("SCALE-BP v2 local prediction is nonfinite.")
    return predicted, estimator_se


__all__ = (
    "LOCAL_FOLD_COUNT",
    "LocalResidualModel",
    "LocalResidualPrediction",
    "assign_local_support_folds",
    "build_local_residual_observations",
    "fit_route_local_residual",
    "predict_local_residual",
)
