"""Held-donor residual correction and complete donor-deletion stability."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    DIRECTION_IDS,
    RESIDUAL_MARGIN_MULTIPLIER,
    UTILITY_RESPONSE_IDS,
)
from .utility_contracts import (
    DonorUtilityRow,
    SignedUtilityModel,
    UtilityDescriptor,
    UtilityPrediction,
)
from .utility_model import predict_signed_utility


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def held_donor_residual_calibration(
    rows: Sequence[DonorUtilityRow],
    *,
    delete_models: Mapping[str, SignedUtilityModel],
) -> Mapping[tuple[str, str, str], tuple[float, float]]:
    """Estimate action/direction bias without predicting any row used in its fit."""

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for response_id in UTILITY_RESPONSE_IDS:
            model = delete_models[row.donor_center]
            residual = row.response(response_id) - predict_signed_utility(
                model, row, response_id
            )
            grouped[
                (response_id, row.alternative, row.direction, row.donor_center)
            ].append(float(residual))
    result: dict[tuple[str, str, str], tuple[float, float]] = {}
    for response_id in UTILITY_RESPONSE_IDS:
        for alternative in ALTERNATIVE_METHOD_IDS:
            for direction in DIRECTION_IDS:
                center_residuals = [
                    np.asarray(
                        grouped[(response_id, alternative, direction, donor)],
                        dtype=np.float64,
                    )
                    for donor in delete_models
                ]
                if any(len(values) == 0 for values in center_residuals):
                    raise ProtocolError(
                        "PCSI-PARC residual calibration lacks a donor response cell."
                    )
                center_means = [
                    float(np.mean(values, dtype=np.float64))
                    for values in center_residuals
                ]
                bias = _median(center_means)
                scale = float(
                    np.sqrt(
                        np.mean(
                            [
                                np.mean((values - bias) ** 2, dtype=np.float64)
                                for values in center_residuals
                            ],
                            dtype=np.float64,
                        )
                    )
                )
                result[(response_id, alternative, direction)] = (bias, scale)
    return result


def predict_utility_surface(
    descriptors: Sequence[UtilityDescriptor],
    *,
    donor_rows: Sequence[DonorUtilityRow],
    full_model: SignedUtilityModel,
    delete_models: Mapping[str, SignedUtilityModel],
) -> tuple[UtilityPrediction, ...]:
    calibration = held_donor_residual_calibration(
        donor_rows, delete_models=delete_models
    )
    output: list[UtilityPrediction] = []
    for descriptor in descriptors:
        full_values: list[tuple[str, float]] = []
        deletion_values: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        residual_bias: list[tuple[str, float]] = []
        residual_scale: list[tuple[str, float]] = []
        robust_values: list[tuple[str, float]] = []
        fractions: list[tuple[str, float]] = []
        if (
            descriptor.target_center != full_model.outer_target_center
            or tuple(delete_models) != full_model.training_centers
        ):
            raise ProtocolError("PCSI-PARC donor deletion topology drifted.")
        for response_id in UTILITY_RESPONSE_IDS:
            full_value = predict_signed_utility(
                full_model, descriptor, response_id
            )
            values = tuple(
                (donor, predict_signed_utility(model, descriptor, response_id))
                for donor, model in delete_models.items()
            )
            bias, scale = calibration[
                (response_id, descriptor.alternative, descriptor.direction)
            ]
            corrected = np.asarray(
                [value + bias for _donor, value in values], dtype=np.float64
            )
            robust = float(np.median(corrected))
            if response_id == "bacc_contribution_delta":
                fraction = float(
                    np.mean(
                        corrected > RESIDUAL_MARGIN_MULTIPLIER * scale,
                        dtype=np.float64,
                    )
                )
            else:
                fraction = float(np.mean(corrected <= 0.0, dtype=np.float64))
            full_values.append((response_id, full_value))
            deletion_values.append((response_id, values))
            residual_bias.append((response_id, bias))
            residual_scale.append((response_id, scale))
            robust_values.append((response_id, robust))
            fractions.append((response_id, fraction))
        output.append(
            UtilityPrediction(
                descriptor.descriptor_hash,
                tuple(full_values),
                tuple(deletion_values),
                tuple(residual_bias),
                tuple(residual_scale),
                tuple(robust_values),
                tuple(fractions),
                (
                    full_model.model_hash,
                    *(model.model_hash for model in delete_models.values()),
                ),
            )
        )
    return tuple(output)


__all__ = ("held_donor_residual_calibration", "predict_utility_surface")
