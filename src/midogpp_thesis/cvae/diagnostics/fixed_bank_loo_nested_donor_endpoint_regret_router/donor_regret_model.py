"""Equal-center weighted, partial-pooled Ridge models for paired regret."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import CENTERS, CENTER_EFFECT_ALPHA, REGRET_FEATURE_NAMES, RIDGE_ALPHA
from .contracts import CenterBalancedRidgeModel, DonorRegretRow
from .hashing import canonical_hash


def _response(row: DonorRegretRow, name: str) -> float:
    if name == "bacc_regret":
        return float(row.bacc_regret)
    if name == "log_loss_delta":
        return float(row.log_loss_delta)
    raise ProtocolError("Unknown donor-regret response.")


def fit_center_balanced_ridge(
    rows: Sequence[DonorRegretRow],
    *,
    response_name: str,
    training_centers: Sequence[str],
    ridge_alpha: float = RIDGE_ALPHA,
    center_effect_alpha: float = CENTER_EFFECT_ALPHA,
) -> CenterBalancedRidgeModel:
    """Fit slopes plus shrunken center effects with equal center mass."""

    centers = tuple(str(value) for value in training_centers)
    selected = tuple(row for row in rows if row.donor_center in centers)
    counts = Counter(row.donor_center for row in selected)
    unique_cases = {
        center: {row.case_id for row in selected if row.donor_center == center}
        for center in centers
    }
    if (
        not selected
        or len(centers) != len(set(centers))
        or set(counts) != set(centers)
        or any(row.donor_center not in centers for row in selected)
        or any(
            counts[center] != len(unique_cases[center])
            or any(
                row.center_case_count != counts[center]
                for row in selected
                if row.donor_center == center
            )
            for center in centers
        )
        or ridge_alpha <= 0.0
        or center_effect_alpha <= 0.0
    ):
        raise ProtocolError("Center-balanced Ridge lacks every legal donor center.")
    raw = np.asarray([row.feature_values for row in selected], dtype=np.float64)
    response = np.asarray([_response(row, response_name) for row in selected], dtype=np.float64)
    weights = np.asarray(
        [1.0 / counts[row.donor_center] for row in selected], dtype=np.float64
    )
    weights /= float(np.sum(weights, dtype=np.float64))
    mean = np.sum(weights[:, None] * raw, axis=0, dtype=np.float64)
    variance = np.sum(weights[:, None] * (raw - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.where(np.sqrt(variance) > 1.0e-12, np.sqrt(variance), 1.0)
    standardized = (raw - mean) / scale
    center_columns = np.zeros((len(selected), len(centers)), dtype=np.float64)
    for row_index, row in enumerate(selected):
        center_columns[row_index, centers.index(row.donor_center)] = 1.0
    design = np.column_stack(
        (np.ones(len(selected), dtype=np.float64), standardized, center_columns)
    )
    root_weight = np.sqrt(weights * len(centers))
    weighted_design = design * root_weight[:, None]
    weighted_response = response * root_weight
    penalty = np.diag(
        np.asarray(
            [
                0.0,
                *([float(ridge_alpha)] * len(REGRET_FEATURE_NAMES)),
                *([float(center_effect_alpha)] * len(centers)),
            ],
            dtype=np.float64,
        )
    )
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_response
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs, rcond=1.0e-12) @ rhs
    if not np.isfinite(coefficients).all():
        raise ProtocolError("Center-balanced Ridge produced nonfinite coefficients.")
    payload = {
        "schema_version": "fixed_bank_center_balanced_partial_pool_ridge_v1",
        "response_name": response_name,
        "training_centers": list(centers),
        "feature_names": list(REGRET_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_alpha": float(ridge_alpha),
        "center_effect_alpha": float(center_effect_alpha),
        "training_row_count_by_center": {center: counts[center] for center in centers},
        "training_rows": [
            {
                "donor_center": row.donor_center,
                "case_id": row.case_id,
                "alternative": row.alternative,
                "descriptor_hash": row.descriptor_hash,
                "feature_values": list(row.feature_values),
                "response": _response(row, response_name),
            }
            for row in selected
        ],
        "equal_total_weight_per_center": True,
        "unseen_target_center_effect": 0.0,
    }
    return CenterBalancedRidgeModel(
        response_name,
        centers,
        REGRET_FEATURE_NAMES,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in coefficients),
        float(ridge_alpha),
        float(center_effect_alpha),
        MappingProxyType({center: counts[center] for center in centers}),
        canonical_hash(payload),
    )


def predict_unseen_center(
    model: CenterBalancedRidgeModel, feature_values: Sequence[float]
) -> float:
    values = np.asarray(tuple(feature_values), dtype=np.float64)
    if values.shape != (len(REGRET_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ProtocolError("Donor-regret prediction feature vector drifted.")
    standardized = (
        values - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
    design = np.concatenate(
        (
            np.ones(1, dtype=np.float64),
            standardized,
            np.zeros(len(model.training_centers), dtype=np.float64),
        )
    )
    result = float(design @ np.asarray(model.coefficients, dtype=np.float64))
    if not np.isfinite(result):
        raise ProtocolError("Donor-regret prediction is nonfinite.")
    return result


def fit_response_pair(
    rows: Sequence[DonorRegretRow], *, training_centers: Sequence[str]
) -> Mapping[str, CenterBalancedRidgeModel]:
    return MappingProxyType(
        {
            response: fit_center_balanced_ridge(
                rows, response_name=response, training_centers=training_centers
            )
            for response in ("bacc_regret", "log_loss_delta")
        }
    )


def fit_full_and_delete_donor_models(
    rows: Sequence[DonorRegretRow], *, outer_target_center: object
) -> tuple[
    Mapping[str, CenterBalancedRidgeModel],
    Mapping[str, Mapping[str, CenterBalancedRidgeModel]],
]:
    target = str(outer_target_center)
    donors = tuple(center for center in CENTERS if center != target)
    selected = tuple(row for row in rows if row.donor_center in donors)
    if target in donors or len(donors) != 8:
        raise ProtocolError("Outer target exclusion must leave exactly eight donor centers.")
    return fit_models_for_training_centers(selected, training_centers=donors)


def fit_models_for_training_centers(
    rows: Sequence[DonorRegretRow], *, training_centers: Sequence[str]
) -> tuple[
    Mapping[str, CenterBalancedRidgeModel],
    Mapping[str, Mapping[str, CenterBalancedRidgeModel]],
]:
    """Fit a full model and refit preprocessing after each center deletion."""

    donors = tuple(str(value) for value in training_centers)
    if (
        len(donors) < 2
        or len(donors) != len(set(donors))
        or any(center not in CENTERS for center in donors)
    ):
        raise ProtocolError("Donor model requires distinct canonical centers.")
    selected = tuple(row for row in rows if row.donor_center in donors)
    full = fit_response_pair(selected, training_centers=donors)
    deleted = {
        donor: fit_response_pair(
            selected,
            training_centers=tuple(center for center in donors if center != donor),
        )
        for donor in donors
    }
    return full, MappingProxyType(deleted)


__all__ = (
    "fit_center_balanced_ridge",
    "fit_full_and_delete_donor_models",
    "fit_models_for_training_centers",
    "fit_response_pair",
    "predict_unseen_center",
)
