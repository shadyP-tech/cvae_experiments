"""Deterministic H/J/K/L/d cross-fitting and label-free inference."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    RowPosteriorModel,
    RowPosteriorObservation,
    RowPosteriorOOFPrediction,
    RowPosteriorPrediction,
    SourceScopeReceipt,
)
from .row_posterior_features import assert_label_free_feature_names
from .row_posterior_fit import ROW_POSTERIOR_PROBABILITY_FLOOR, fit_source_row_posterior, sigmoid


def _role_map(
    source_centers: Sequence[str], explicit: Mapping[str, tuple[str, str]] | None
) -> dict[str, tuple[str, str]]:
    centers = tuple(sorted(set(str(center) for center in source_centers)))
    if len(centers) < 5:
        raise ProtocolError("H/J/K/L row-posterior cross-fitting requires at least five source centers.")
    if explicit is not None:
        result = {
            str(query): (str(values[0]), str(values[1]))
            for query, values in explicit.items()
        }
        if set(result) != set(centers):
            raise ProtocolError("Explicit row-posterior K/L role map is incomplete.")
    else:
        result = {}
        for ordinal, query in enumerate(centers):
            alternatives = tuple(center for center in centers if center != query)
            result[query] = (
                alternatives[ordinal % len(alternatives)],
                alternatives[(ordinal + 1) % len(alternatives)],
            )
            if result[query][0] == result[query][1]:
                result[query] = (alternatives[0], alternatives[1])
    for query, (hyperparameter, calibration) in result.items():
        if len({query, hyperparameter, calibration}) != 3 or not {
            hyperparameter, calibration
        }.issubset(centers):
            raise ProtocolError("Row-posterior K/L role map has colliding or unknown centers.")
    return result


def crossfit_source_row_posterior(
    observations: Sequence[RowPosteriorObservation],
    *,
    outer_target_center: object,
    role_centers_by_query: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[RowPosteriorOOFPrediction, ...]:
    """Emit deterministic leave-center-out and case-excluded source predictions."""

    h = str(outer_target_center).strip()
    rows = tuple(
        sorted(tuple(observations), key=lambda row: (row.center_id, row.case_id, row.row_id))
    )
    if not h or not rows or any(row.center_id == h for row in rows):
        raise ProtocolError("Source cross-fitting cannot contain outer-target H rows.")
    keys = tuple((row.center_id, row.case_id, row.row_id) for row in rows)
    if len(set(keys)) != len(keys):
        raise ProtocolError("Source cross-fitting row identities are duplicated.")
    centers = tuple(sorted({row.center_id for row in rows}))
    roles = _role_map(centers, role_centers_by_query)
    grouped: dict[tuple[str, str], list[RowPosteriorObservation]] = {}
    for row in rows:
        grouped.setdefault((row.center_id, row.case_id), []).append(row)
    output: list[RowPosteriorOOFPrediction] = []
    for (query, held_case), held_rows in sorted(grouped.items()):
        hyperparameter, calibration = roles[query]
        training_rows = tuple(
            row
            for row in rows
            if row.center_id not in {query, hyperparameter, calibration}
            and (row.center_id, row.case_id) != (query, held_case)
        )
        receipt = SourceScopeReceipt(
            outer_target_center=h,
            query_center=query,
            hyperparameter_center=hyperparameter,
            calibration_center=calibration,
            heldout_case_center=query,
            heldout_case_id=held_case,
            training_center_ids=tuple(sorted({row.center_id for row in training_rows})),
            training_case_keys=tuple(
                sorted({(row.center_id, row.case_id) for row in training_rows})
            ),
        )
        model = fit_source_row_posterior(training_rows, scope=receipt)
        for row in held_rows:
            prediction = predict_source_row_posterior(
                model,
                feature_names=row.feature_names,
                feature_values=row.feature_values,
            )
            output.append(
                RowPosteriorOOFPrediction(
                    center_id=row.center_id,
                    case_id=row.case_id,
                    row_id=row.row_id,
                    eta=prediction.eta,
                    model_hash=prediction.model_hash,
                    source_scope_receipt_hash=prediction.source_scope_receipt_hash,
                )
            )
    return tuple(sorted(output, key=lambda row: (row.center_id, row.case_id, row.row_id)))


def predict_source_row_posterior(
    model: RowPosteriorModel,
    *,
    feature_names: Sequence[str],
    feature_values: Sequence[float],
) -> RowPosteriorPrediction:
    """Predict eta from a label-free target row without identity fields."""

    names = assert_label_free_feature_names(feature_names)
    values = np.asarray(tuple(float(value) for value in feature_values), dtype=np.float64)
    if names != model.feature_names or values.shape != (len(names),) or not np.isfinite(values).all():
        raise ProtocolError("Row-posterior prediction feature schema drifted.")
    standardized = (values - np.asarray(model.feature_mean)) / np.asarray(model.feature_scale)
    linear = float(model.intercept + standardized @ np.asarray(model.coefficients))
    eta = float(sigmoid(np.asarray([linear], dtype=np.float64))[0])
    eta = min(max(eta, ROW_POSTERIOR_PROBABILITY_FLOOR), 1.0 - ROW_POSTERIOR_PROBABILITY_FLOOR)
    return RowPosteriorPrediction(
        eta=eta,
        model_hash=model.model_hash,
        source_scope_receipt_hash=model.source_scope_receipt_hash,
    )


__all__ = ("crossfit_source_row_posterior", "predict_source_row_posterior")
