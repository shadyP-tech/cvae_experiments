"""Center/case-balanced deterministic fitting for SCALE-BP donor priors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .donor_contracts import (
    DonorDeleteCenterFold,
    DonorObservation,
    DonorPriorModel,
)
from .donor_prediction import predict_metric_values
from .donor_scope import validate_delete_center_folds, validate_scope_rows
from .identity import ACTION_IDS, CENTERS, RIDGE_ALPHA
from .influence.contracts import MetricStandardError
from .protocol import ProtocolError
from .replay_scope import DonorScope


def fit_donor_prior(
    observations: Sequence[DonorObservation],
    *,
    scope: DonorScope,
    delete_center_folds: Sequence[DonorDeleteCenterFold],
) -> DonorPriorModel:
    """Fit a fixed ridge and estimate transport scale by leave-center replay."""

    rows = tuple(observations)
    target = scope.prediction_center
    if target not in CENTERS or not rows:
        raise ProtocolError("SCALE-BP donor prior input is empty or unknown.")
    validate_scope_rows(rows, scope)
    names = rows[0].descriptor.feature_names
    if any(row.descriptor.feature_names != names for row in rows):
        raise ProtocolError("SCALE-BP donor feature schema drifted.")
    identities = {
        (row.query_center, row.case_id, row.descriptor.action_id) for row in rows
    }
    if len(identities) != len(rows):
        raise ProtocolError("SCALE-BP donor observation rectangle is duplicated.")
    centers = tuple(
        center for center in CENTERS if any(row.query_center == center for row in rows)
    )
    if len(centers) < 3:
        raise ProtocolError("SCALE-BP donor prior lacks independent centers.")

    folds = validate_delete_center_folds(delete_center_folds, rows, scope)
    mean, scale, coefficients = fit_parameters(rows, names)
    heldout_errors: list[tuple[float, float, float]] = []
    for fold in folds:
        deleted = fold.deleted_center
        training = fold.training_observations
        validation = tuple(row for row in rows if row.query_center == deleted)
        if len({row.query_center for row in training}) < 2:
            raise ProtocolError("SCALE-BP delete-center donor fit is degenerate.")
        local_mean, local_scale, local_coefficients = fit_parameters(training, names)
        case_errors: dict[str, list[np.ndarray]] = {}
        for row in validation:
            predicted = predict_metric_values(
                row.descriptor,
                feature_mean=local_mean,
                feature_scale=local_scale,
                coefficients=local_coefficients,
            )
            case_errors.setdefault(row.case_id, []).append(
                np.asarray(row.realized.as_tuple(), dtype=np.float64) - predicted
            )
        center_case_errors = [
            np.mean(np.asarray(values), axis=0, dtype=np.float64)
            for values in case_errors.values()
        ]
        heldout_errors.append(
            tuple(
                float(value)
                for value in np.mean(
                    np.asarray(center_case_errors), axis=0, dtype=np.float64
                )
            )
        )
    error_array = np.asarray(heldout_errors, dtype=np.float64)
    center_scale = np.sqrt(
        np.mean(error_array * error_array, axis=0, dtype=np.float64)
    )
    return DonorPriorModel(
        target,
        scope.scope_hash,
        scope.fit_role,
        centers,
        tuple(sorted({row.case_id for row in rows})),
        names,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(tuple(float(value) for value in row) for row in coefficients),
        MetricStandardError.from_iterable(center_scale),
        len({(row.query_center, row.case_id) for row in rows}),
        len(rows),
        tuple(fold.fold_hash for fold in folds),
    )


def fit_parameters(
    rows: Sequence[DonorObservation],
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([row.descriptor.values for row in rows], dtype=np.float64)
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    cells = np.zeros((len(rows), len(ACTION_IDS)), dtype=np.float64)
    for index, row in enumerate(rows):
        cells[index, ACTION_IDS.index(row.cell_id)] = 1.0
    design = np.column_stack((cells, (features - mean) / scale))
    response = np.asarray(
        [row.realized.as_tuple() for row in rows], dtype=np.float64
    )
    weights = center_case_weights(rows)
    root_weight = np.sqrt(weights)[:, None]
    weighted_design = design * root_weight
    penalty = np.diag(
        np.asarray(
            [1.0e-12] * len(ACTION_IDS) + [RIDGE_ALPHA] * len(names),
            dtype=np.float64,
        )
    )
    system = weighted_design.T @ weighted_design + penalty
    target = weighted_design.T @ (response * root_weight)
    try:
        coefficients = np.linalg.solve(system, target).T
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("SCALE-BP donor prior ridge is singular.") from exc
    if not np.isfinite(coefficients).all():
        raise ProtocolError("SCALE-BP donor prior ridge is nonfinite.")
    return mean, scale, coefficients


def center_case_weights(rows: Sequence[DonorObservation]) -> np.ndarray:
    centers = tuple(dict.fromkeys(row.query_center for row in rows))
    weights = np.zeros(len(rows), dtype=np.float64)
    for center in centers:
        cases = tuple(
            dict.fromkeys(row.case_id for row in rows if row.query_center == center)
        )
        for case in cases:
            positions = [
                index
                for index, row in enumerate(rows)
                if row.query_center == center and row.case_id == case
            ]
            value = 1.0 / (len(centers) * len(cases) * len(positions))
            weights[positions] = value
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0):
        raise ProtocolError("SCALE-BP donor center/case weights drifted.")
    return weights


__all__ = ("fit_donor_prior",)
