"""Center-balanced multivariate ridge for direct signed routing utility."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    RIDGE_ALPHA,
    UTILITY_CELL_IDS,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash
from .utility_contracts import DonorUtilityRow, SignedUtilityModel, UtilityDescriptor


def _cell_id(row: UtilityDescriptor | DonorUtilityRow) -> str:
    value = f"{row.alternative}::{row.direction}"
    if value not in UTILITY_CELL_IDS:
        raise ProtocolError("PCSI-PARC utility cell identity drifted.")
    return value


def fit_signed_utility_model(
    rows: Sequence[DonorUtilityRow],
    *,
    outer_target_center: str,
    training_centers: Sequence[str],
    ridge_alpha: float = RIDGE_ALPHA,
) -> SignedUtilityModel:
    """Fit all three responses with unpenalized action-direction intercepts."""

    outer = str(outer_target_center)
    centers = tuple(str(value) for value in training_centers)
    selected = tuple(
        row
        for row in rows
        if row.outer_target_center == outer and row.donor_center in centers
    )
    if (
        outer not in CENTERS
        or outer in centers
        or len(centers) != len(set(centers))
        or any(center not in CENTERS for center in centers)
        or ridge_alpha <= 0.0
        or any(
            row.outer_target_center != outer or row.donor_center == outer
            for row in rows
        )
    ):
        raise ProtocolError("PCSI-PARC signed utility fit lacks legal donor support.")
    counts = Counter(row.donor_center for row in selected)
    if not selected or any(counts[center] <= 0 for center in centers):
        raise ProtocolError("PCSI-PARC signed utility fit is missing a donor center.")
    observed_cells = Counter(_cell_id(row) for row in selected)
    if any(observed_cells[cell] <= 0 for cell in UTILITY_CELL_IDS):
        raise ProtocolError("PCSI-PARC signed utility fit lacks a complete cell rectangle.")

    raw = np.asarray([row.feature_values for row in selected], dtype=np.float64)
    response = np.asarray(
        [
            [row.response(response_id) for response_id in UTILITY_RESPONSE_IDS]
            for row in selected
        ],
        dtype=np.float64,
    )
    row_counts_by_case = Counter(
        (row.donor_center, row.case_id) for row in selected
    )
    cases_by_center = Counter(donor for donor, _case in set(row_counts_by_case))
    weights = np.asarray(
        [
            1.0
            / len(centers)
            / cases_by_center[row.donor_center]
            / row_counts_by_case[(row.donor_center, row.case_id)]
            for row in selected
        ],
        dtype=np.float64,
    )
    if abs(float(np.sum(weights, dtype=np.float64)) - 1.0) > 1.0e-12:
        raise ProtocolError("PCSI-PARC center-balanced weights drifted.")
    mean = np.sum(weights[:, None] * raw, axis=0, dtype=np.float64)
    variance = np.sum(
        weights[:, None] * (raw - mean) ** 2, axis=0, dtype=np.float64
    )
    scale = np.where(np.sqrt(variance) > 1.0e-12, np.sqrt(variance), 1.0)
    cells = np.zeros((len(selected), len(UTILITY_CELL_IDS)), dtype=np.float64)
    cell_order = {cell: index for index, cell in enumerate(UTILITY_CELL_IDS)}
    for row_index, row in enumerate(selected):
        cells[row_index, cell_order[_cell_id(row)]] = 1.0
    design = np.column_stack((cells, (raw - mean) / scale))
    penalty = np.diag(
        np.asarray(
            [
                *([0.0] * len(UTILITY_CELL_IDS)),
                *([float(ridge_alpha)] * len(UTILITY_FEATURE_NAMES)),
            ]
        )
    )
    system = design.T @ (weights[:, None] * design) + penalty
    target = design.T @ (weights[:, None] * response)
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(system, rcond=1.0e-12) @ target
    if not np.isfinite(coefficients).all():
        raise ProtocolError("PCSI-PARC signed utility ridge produced nonfinite coefficients.")
    split = len(UTILITY_CELL_IDS)
    intercepts = tuple(
        (
            response_id,
            tuple(float(value) for value in coefficients[:split, response_index]),
        )
        for response_index, response_id in enumerate(UTILITY_RESPONSE_IDS)
    )
    slopes = tuple(
        (
            response_id,
            tuple(float(value) for value in coefficients[split:, response_index]),
        )
        for response_index, response_id in enumerate(UTILITY_RESPONSE_IDS)
    )
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_signed_utility_model_v1",
        "outer_target_center": outer,
        "response_ids": list(UTILITY_RESPONSE_IDS),
        "training_centers": list(centers),
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "cell_ids": list(UTILITY_CELL_IDS),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "cell_intercepts": dict(intercepts),
        "slope_coefficients": dict(slopes),
        "ridge_alpha": float(ridge_alpha),
        "training_row_count_by_center": {
            center: counts[center] for center in centers
        },
        "equal_total_weight_per_donor_center": True,
        "equal_total_weight_per_case_within_donor_center": True,
        "cell_intercepts_penalized": False,
        "center_dummy_effects_used": False,
        "structural_zero_rows_used": True,
        "training_response_hash": canonical_hash(
            [
                {
                    "descriptor_hash": row.descriptor_hash,
                    "responses": {
                        response_id: row.response(response_id)
                        for response_id in UTILITY_RESPONSE_IDS
                    },
                }
                for row in selected
            ]
        ),
    }
    return SignedUtilityModel(
        outer,
        centers,
        UTILITY_FEATURE_NAMES,
        UTILITY_CELL_IDS,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        intercepts,
        slopes,
        float(ridge_alpha),
        MappingProxyType({center: counts[center] for center in centers}),
        canonical_hash(payload),
        "FIT",
    )


def predict_signed_utility(
    model: SignedUtilityModel,
    descriptor: UtilityDescriptor | DonorUtilityRow,
    response_id: str,
) -> float:
    if response_id not in UTILITY_RESPONSE_IDS:
        raise ProtocolError("PCSI-PARC requested an unknown utility response.")
    standardized = (
        np.asarray(descriptor.feature_values, dtype=np.float64)
        - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
    cell_index = model.cell_ids.index(_cell_id(descriptor))
    prediction = float(
        dict(model.cell_intercepts)[response_id][cell_index]
        + standardized
        @ np.asarray(dict(model.slope_coefficients)[response_id], dtype=np.float64)
    )
    if not np.isfinite(prediction):
        raise ProtocolError("PCSI-PARC signed utility prediction drifted.")
    return prediction


def fit_response_model_family(
    rows: Sequence[DonorUtilityRow],
    *,
    outer_target_center: str,
) -> tuple[SignedUtilityModel, Mapping[str, SignedUtilityModel]]:
    """Refit one multivariate surface for full and all donor deletions."""

    outer = str(outer_target_center)
    donors = tuple(center for center in CENTERS if center != outer)
    full = fit_signed_utility_model(
        rows, outer_target_center=outer, training_centers=donors
    )
    deleted = MappingProxyType(
        {
            donor: fit_signed_utility_model(
                rows,
                outer_target_center=outer,
                training_centers=tuple(
                    center for center in donors if center != donor
                ),
            )
            for donor in donors
        }
    )
    return full, deleted


__all__ = (
    "fit_response_model_family",
    "fit_signed_utility_model",
    "predict_signed_utility",
)
