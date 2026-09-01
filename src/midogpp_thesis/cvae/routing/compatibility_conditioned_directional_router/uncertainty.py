"""Exact action/comparator, endpoint-specific one-sided source-OOF bounds."""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ActionKind,
    ActionPrediction,
    BoundedActionEvidence,
    EndpointBounds,
    EndpointCalibration,
    EndpointCalibrationCell,
    EndpointEffects,
    OOFEndpointRow,
    SourceOOFPrediction,
)
from .hashing import canonical_hash
from .pairwise import action_key


DEFAULT_QUANTILE = 0.9


def _difference(left: EndpointEffects, right: EndpointEffects) -> EndpointEffects:
    return EndpointEffects(
        bacc_gain=left.bacc_gain - right.bacc_gain,
        brier_delta=left.brier_delta - right.brier_delta,
        log_delta=left.log_delta - right.log_delta,
    )


def build_oof_endpoint_rows(
    predictions: Sequence[SourceOOFPrediction],
) -> tuple[OOFEndpointRow, ...]:
    """Build B and available action-comparator rows from strict source OOF."""

    source_rows = tuple(predictions)
    if not source_rows or any(not isinstance(row, SourceOOFPrediction) for row in source_rows):
        raise ProtocolError("Endpoint uncertainty requires typed source OOF predictions.")
    grouped: dict[tuple[str, str], list[SourceOOFPrediction]] = defaultdict(list)
    output: list[OOFEndpointRow] = []
    for row in source_rows:
        feature = row.prediction.feature
        key = action_key(feature)
        output.append(
            OOFEndpointRow(
                query_center_id=row.held_center_id,
                case_id=feature.case_id,
                action_key=key,
                comparator_key=ActionKind.B.value,
                predicted=row.prediction.predicted_effects,
                observed=row.observed,
                fold_model_hash=row.fold_hash,
            )
        )
        grouped[(row.held_center_id, feature.case_id)].append(row)
    for (center, case), rows in sorted(grouped.items()):
        # Ordered pairs make the comparator identity explicit.  Same generalized
        # action keys are skipped rather than pooled across expert candidates.
        for left, right in permutations(rows, 2):
            left_key = action_key(left.prediction.feature)
            right_key = action_key(right.prediction.feature)
            if left_key == right_key:
                continue
            output.append(
                OOFEndpointRow(
                    query_center_id=center,
                    case_id=case,
                    action_key=left_key,
                    comparator_key=right_key,
                    predicted=_difference(
                        left.prediction.predicted_effects,
                        right.prediction.predicted_effects,
                    ),
                    observed=_difference(left.observed, right.observed),
                    fold_model_hash=left.fold_hash,
                )
            )
    return tuple(
        sorted(
            output,
            key=lambda row: (
                row.action_key,
                row.comparator_key,
                row.query_center_id,
                row.case_id,
                row.fold_model_hash,
            ),
        )
    )


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("Endpoint residual quantile input is invalid.")
    try:
        result = np.quantile(array, quantile, method="higher")
    except TypeError:  # NumPy < 1.22 compatibility.
        result = np.quantile(array, quantile, interpolation="higher")
    return max(0.0, float(result))


def calibrate_endpoint_uncertainty(
    rows: Sequence[OOFEndpointRow],
    *,
    quantile: float = DEFAULT_QUANTILE,
    minimum_centers_per_cell: int = 2,
) -> EndpointCalibration:
    """Calibrate separate clustered one-sided radii for every exact cell.

    Each source center contributes its worst residual to a cell before the
    quantile is taken.  Cells with insufficient centers are omitted.  Inference
    on an omitted cell fails closed; there is deliberately no pooled fallback.
    """

    source_rows = tuple(rows)
    if (
        not source_rows
        or any(not isinstance(row, OOFEndpointRow) for row in source_rows)
        or not 0.5 < float(quantile) < 1.0
        or minimum_centers_per_cell < 2
    ):
        raise ProtocolError("Endpoint uncertainty calibration inputs are invalid.")
    grouped: dict[tuple[str, str], list[OOFEndpointRow]] = defaultdict(list)
    for row in source_rows:
        grouped[(row.action_key, row.comparator_key)].append(row)
    cells: list[EndpointCalibrationCell] = []
    for (action, comparator), group in sorted(grouped.items()):
        by_center: dict[str, list[OOFEndpointRow]] = defaultdict(list)
        for row in group:
            by_center[row.query_center_id].append(row)
        if len(by_center) < minimum_centers_per_cell:
            continue
        bacc_residuals: list[float] = []
        brier_residuals: list[float] = []
        log_residuals: list[float] = []
        for center_rows in by_center.values():
            bacc_residuals.append(
                max(row.predicted.bacc_gain - row.observed.bacc_gain for row in center_rows)
            )
            brier_residuals.append(
                max(row.observed.brier_delta - row.predicted.brier_delta for row in center_rows)
            )
            log_residuals.append(
                max(row.observed.log_delta - row.predicted.log_delta for row in center_rows)
            )
        cells.append(
            EndpointCalibrationCell(
                action_key=action,
                comparator_key=comparator,
                bacc_overprediction_quantile=_higher_quantile(bacc_residuals, quantile),
                brier_underprediction_quantile=_higher_quantile(brier_residuals, quantile),
                log_underprediction_quantile=_higher_quantile(log_residuals, quantile),
                source_center_ids=tuple(sorted(by_center)),
                row_count=len(group),
            )
        )
    if not cells:
        raise ProtocolError("No endpoint action/comparator cell has enough source centers.")
    source_hash = canonical_hash(
        tuple(
            (
                row.query_center_id,
                row.case_id,
                row.action_key,
                row.comparator_key,
                row.predicted.as_tuple(),
                row.observed.as_tuple(),
                row.fold_model_hash,
            )
            for row in sorted(
                source_rows,
                key=lambda value: (
                    value.query_center_id,
                    value.case_id,
                    value.action_key,
                    value.comparator_key,
                    value.fold_model_hash,
                ),
            )
        )
    )
    return EndpointCalibration(
        quantile=float(quantile),
        cells=tuple(cells),
        source_oof_hash=source_hash,
    )


def apply_endpoint_bounds(
    predicted: EndpointEffects,
    *,
    action_key: str,
    comparator_key: str,
    calibration: EndpointCalibration,
) -> EndpointBounds:
    if not isinstance(predicted, EndpointEffects) or not isinstance(
        calibration, EndpointCalibration
    ):
        raise ProtocolError("Endpoint bounds require typed prediction and calibration.")
    cell = calibration.cell(action_key, comparator_key)
    return EndpointBounds(
        bacc_lcb=predicted.bacc_gain - cell.bacc_overprediction_quantile,
        brier_ucb=predicted.brier_delta + cell.brier_underprediction_quantile,
        log_ucb=predicted.log_delta + cell.log_underprediction_quantile,
    )


def bound_action_vs_baseline(
    prediction: ActionPrediction, *, calibration: EndpointCalibration
) -> BoundedActionEvidence:
    """Bind a target action to its exact safe-vs-B uncertainty cell."""

    if not isinstance(prediction, ActionPrediction):
        raise ProtocolError("Baseline-bound evidence requires a typed action prediction.")
    bounds = apply_endpoint_bounds(
        prediction.predicted_effects,
        action_key=action_key(prediction.feature),
        comparator_key=ActionKind.B.value,
        calibration=calibration,
    )
    return BoundedActionEvidence(
        prediction=prediction,
        comparator_key=ActionKind.B.value,
        bounds=bounds,
        uncertainty_calibration_hash=calibration.calibration_hash,
    )


__all__ = (
    "DEFAULT_QUANTILE",
    "apply_endpoint_bounds",
    "bound_action_vs_baseline",
    "build_oof_endpoint_rows",
    "calibrate_endpoint_uncertainty",
)
