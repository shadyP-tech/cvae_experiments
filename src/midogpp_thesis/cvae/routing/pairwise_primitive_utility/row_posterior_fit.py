"""Deterministic fixed-capacity fitting for the source-only row posterior."""

from __future__ import annotations

from collections import Counter
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import RowPosteriorModel, RowPosteriorObservation, SourceScopeReceipt, canonical_sha256
from .row_posterior_features import assert_label_free_feature_names


ROW_POSTERIOR_RIDGE_ALPHA = 1.0
ROW_POSTERIOR_MAX_ITERATIONS = 64
ROW_POSTERIOR_PROBABILITY_FLOOR = 1.0e-6


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _center_case_balanced_weights(rows: Sequence[RowPosteriorObservation]) -> np.ndarray:
    centers = Counter(row.center_id for row in rows)
    rows_by_case = Counter((row.center_id, row.case_id) for row in rows)
    cases_by_center = Counter(center for center, _ in rows_by_case)
    raw = np.asarray(
        [
            1.0
            / (
                len(centers)
                * cases_by_center[row.center_id]
                * rows_by_case[(row.center_id, row.case_id)]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return raw * (len(rows) / float(np.sum(raw, dtype=np.float64)))


def fit_model(
    observations: Sequence[RowPosteriorObservation], *, provenance_hash: str
) -> RowPosteriorModel:
    rows = tuple(
        sorted(
            tuple(observations),
            key=lambda row: (
                row.center_id, row.case_id, row.row_id, row.feature_values, row.outcome
            ),
        )
    )
    if not rows:
        raise ProtocolError("Row-posterior fitting requires source observations.")
    row_keys = tuple((row.center_id, row.case_id, row.row_id) for row in rows)
    if len(set(row_keys)) != len(row_keys):
        raise ProtocolError("Row-posterior source row identities are duplicated.")
    feature_names = assert_label_free_feature_names(rows[0].feature_names)
    if any(row.feature_names != feature_names for row in rows):
        raise ProtocolError("Row-posterior feature schema drifted across source rows.")
    centers = tuple(sorted({row.center_id for row in rows}))
    case_keys = tuple(sorted({(row.center_id, row.case_id) for row in rows}))
    if len(centers) < 2 or len(case_keys) < 2 or len({row.outcome for row in rows}) != 2:
        raise ProtocolError("Row-posterior fitting requires multiple centers, cases, and classes.")
    if len(rows) <= len(feature_names) + 1:
        raise ProtocolError("Row-posterior fitting is underdetermined for its frozen capacity.")

    matrix = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    target = np.asarray([row.outcome for row in rows], dtype=np.float64)
    weights = _center_case_balanced_weights(rows)
    weight_total = float(np.sum(weights, dtype=np.float64))
    mean = np.sum(weights[:, None] * matrix, axis=0, dtype=np.float64) / weight_total
    variance = (
        np.sum(weights[:, None] * (matrix - mean) ** 2, axis=0, dtype=np.float64)
        / weight_total
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    design = np.column_stack((np.ones(len(rows), dtype=np.float64), (matrix - mean) / scale))

    prevalence = float(np.dot(weights, target) / weight_total)
    prevalence = min(
        max(prevalence, ROW_POSTERIOR_PROBABILITY_FLOOR),
        1.0 - ROW_POSTERIOR_PROBABILITY_FLOOR,
    )
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.diag(
        np.asarray([0.0, *([ROW_POSTERIOR_RIDGE_ALPHA] * len(feature_names))], dtype=np.float64)
    )
    for _ in range(ROW_POSTERIOR_MAX_ITERATIONS):
        probability = np.clip(
            sigmoid(design @ coefficients),
            ROW_POSTERIOR_PROBABILITY_FLOOR,
            1.0 - ROW_POSTERIOR_PROBABILITY_FLOOR,
        )
        curvature = weights * probability * (1.0 - probability)
        gradient = design.T @ (weights * (target - probability)) - penalty @ coefficients
        hessian = design.T @ (curvature[:, None] * design) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients += step
        if float(np.max(np.abs(step))) <= 1.0e-10:
            break
    if not np.isfinite(coefficients).all():
        raise ProtocolError("Source-only row-posterior fit is non-finite.")
    return RowPosteriorModel(
        feature_names=feature_names,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        intercept=float(coefficients[0]),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        ridge_alpha=ROW_POSTERIOR_RIDGE_ALPHA,
        training_row_count=len(rows),
        training_center_count=len(centers),
        training_case_count=len(case_keys),
        source_scope_receipt_hash=provenance_hash,
    )


def fit_source_row_posterior(
    observations: Sequence[RowPosteriorObservation], *, scope: SourceScopeReceipt
) -> RowPosteriorModel:
    """Fit the fixed ridge-logistic row posterior on exact source-only rows."""

    if not isinstance(scope, SourceScopeReceipt):
        raise ProtocolError("Row-posterior fitting requires a typed source scope receipt.")
    rows = tuple(observations)
    centers = tuple(sorted({row.center_id for row in rows}))
    case_keys = tuple(sorted({(row.center_id, row.case_id) for row in rows}))
    if centers != scope.training_center_ids or case_keys != scope.training_case_keys:
        raise ProtocolError("Row-posterior observations drifted from the source scope receipt.")
    return fit_model(rows, provenance_hash=scope.receipt_hash)


def fit_final_source_row_posterior(
    observations: Sequence[RowPosteriorObservation],
    *,
    outer_target_center: object,
    fixed_capacity_receipt_hash: object,
) -> RowPosteriorModel:
    """Refit fixed capacity once on every legal C-minus-H source row."""

    h = str(outer_target_center).strip()
    capacity_hash = str(fixed_capacity_receipt_hash).strip()
    rows = tuple(observations)
    if not h or not capacity_hash or not rows or any(row.center_id == h for row in rows):
        raise ProtocolError("Final row posterior requires a fixed capacity and source-only C-minus-H rows.")
    centers = tuple(sorted({row.center_id for row in rows}))
    case_keys = tuple(sorted({(row.center_id, row.case_id) for row in rows}))
    receipt_hash = canonical_sha256(
        {
            "schema": "final_source_row_posterior_receipt_v2",
            "outer_target_H": h,
            "training_centers": centers,
            "training_case_keys": case_keys,
            "fixed_capacity_receipt_hash": capacity_hash,
            "ridge_alpha": ROW_POSTERIOR_RIDGE_ALPHA,
            "target_labels_used": False,
        }
    )
    return fit_model(rows, provenance_hash=receipt_hash)


__all__ = (
    "ROW_POSTERIOR_MAX_ITERATIONS",
    "ROW_POSTERIOR_PROBABILITY_FLOOR",
    "ROW_POSTERIOR_RIDGE_ALPHA",
    "fit_final_source_row_posterior",
    "fit_source_row_posterior",
)
