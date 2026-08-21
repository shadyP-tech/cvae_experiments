"""Two-direction, twelve-descriptor multivariate ridge for PCSI-RACR."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ACTION_GEOMETRY_IDS,
    CENTERS,
    DIRECTION_IDS,
    RIDGE_ALPHA,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash
from .projected_contracts import (
    ProjectedDonorUtilityRow,
    ProjectedUtilityDescriptor,
    ProjectedUtilityModel,
)


def fit_projected_utility_model(
    rows: Sequence[ProjectedDonorUtilityRow],
    *,
    outer_target_center: str,
    geometry_id: str,
    training_centers: Sequence[str],
    ridge_alpha: float = RIDGE_ALPHA,
) -> ProjectedUtilityModel:
    outer = str(outer_target_center)
    centers = tuple(str(value) for value in training_centers)
    selected = tuple(
        row
        for row in rows
        if row.outer_target_center == outer
        and row.geometry_id == geometry_id
        and row.donor_center in centers
    )
    if (
        outer not in CENTERS
        or geometry_id not in ACTION_GEOMETRY_IDS
        or outer in centers
        or not centers
        or len(centers) != len(set(centers))
        or any(center not in CENTERS for center in centers)
        or ridge_alpha <= 0.0
    ):
        raise ProtocolError("PCSI-RACR projected fit lacks legal donor support.")
    counts = Counter(row.donor_center for row in selected)
    if not selected or any(counts[center] <= 0 for center in centers):
        raise ProtocolError("PCSI-RACR projected fit is missing a donor center.")
    direction_counts = Counter(row.direction for row in selected)
    if any(direction_counts[direction] <= 0 for direction in DIRECTION_IDS):
        raise ProtocolError("PCSI-RACR projected fit lacks both direction columns.")

    raw = np.asarray([row.feature_values for row in selected], dtype=np.float64)
    response = np.asarray(
        [[row.response(response_id) for response_id in UTILITY_RESPONSE_IDS] for row in selected],
        dtype=np.float64,
    )
    row_counts_by_case = Counter((row.donor_center, row.case_id) for row in selected)
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
        raise ProtocolError("PCSI-RACR donor/case/class weights drifted.")
    mean = np.sum(weights[:, None] * raw, axis=0, dtype=np.float64)
    variance = np.sum(weights[:, None] * (raw - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.where(np.sqrt(variance) > 1.0e-12, np.sqrt(variance), 1.0)

    directions = np.zeros((len(selected), len(DIRECTION_IDS)), dtype=np.float64)
    for index, row in enumerate(selected):
        directions[index, DIRECTION_IDS.index(row.direction)] = 1.0
    design = np.column_stack((directions, (raw - mean) / scale))
    penalty = np.diag(
        np.asarray(
            [*([0.0] * len(DIRECTION_IDS)), *([float(ridge_alpha)] * len(UTILITY_FEATURE_NAMES))],
            dtype=np.float64,
        )
    )
    system = design.T @ (weights[:, None] * design) + penalty
    target = design.T @ (weights[:, None] * response)
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(system, rcond=1.0e-12) @ target
    if not np.isfinite(coefficients).all():
        raise ProtocolError("PCSI-RACR projected ridge produced nonfinite coefficients.")
    split = len(DIRECTION_IDS)
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
        "schema_version": "fixed_bank_pcsi_racr_projected_model_v1",
        "outer_target_center": outer,
        "geometry_id": geometry_id,
        "training_centers": list(centers),
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "direction_ids": list(DIRECTION_IDS),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "direction_intercepts": dict(intercepts),
        "slope_coefficients": dict(slopes),
        "ridge_alpha": float(ridge_alpha),
        "training_row_count_by_center": {center: counts[center] for center in centers},
        "intercept_count": 2,
        "center_dummy_effects_used": False,
        "direction_intercepts_penalized": False,
        "equal_total_weight_per_donor_center": True,
        "equal_total_weight_per_case_within_donor_center": True,
        "equal_total_weight_per_equivalence_class_within_case": True,
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
    return ProjectedUtilityModel(
        outer,
        geometry_id,
        centers,
        UTILITY_FEATURE_NAMES,
        DIRECTION_IDS,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        intercepts,
        slopes,
        float(ridge_alpha),
        MappingProxyType({center: counts[center] for center in centers}),
        canonical_hash(payload),
    )


def predict_projected_utility(
    model: ProjectedUtilityModel,
    descriptor: ProjectedUtilityDescriptor | ProjectedDonorUtilityRow,
    response_id: str,
) -> float:
    if response_id not in UTILITY_RESPONSE_IDS or descriptor.geometry_id != model.geometry_id:
        raise ProtocolError("PCSI-RACR projected prediction identity drifted.")
    standardized = (
        np.asarray(descriptor.feature_values, dtype=np.float64)
        - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
    direction_index = DIRECTION_IDS.index(descriptor.direction)
    value = float(
        dict(model.direction_intercepts)[response_id][direction_index]
        + standardized @ np.asarray(dict(model.slope_coefficients)[response_id], dtype=np.float64)
    )
    if not np.isfinite(value):
        raise ProtocolError("PCSI-RACR projected prediction is nonfinite.")
    return value


def fit_projected_model_family(
    rows: Sequence[ProjectedDonorUtilityRow],
    *,
    outer_target_center: str,
    geometry_id: str,
    training_centers: Sequence[str],
) -> tuple[ProjectedUtilityModel, Mapping[str, ProjectedUtilityModel]]:
    centers = tuple(str(value) for value in training_centers)
    full = fit_projected_utility_model(
        rows,
        outer_target_center=outer_target_center,
        geometry_id=geometry_id,
        training_centers=centers,
    )
    deleted = MappingProxyType(
        {
            donor: fit_projected_utility_model(
                rows,
                outer_target_center=outer_target_center,
                geometry_id=geometry_id,
                training_centers=tuple(center for center in centers if center != donor),
            )
            for donor in centers
        }
    )
    return full, deleted


__all__ = (
    "fit_projected_model_family",
    "fit_projected_utility_model",
    "predict_projected_utility",
)
