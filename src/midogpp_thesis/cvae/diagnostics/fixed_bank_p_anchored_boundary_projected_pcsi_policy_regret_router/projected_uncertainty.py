"""Bias-corrected donor-deletion predictions for PCSI-PARC surfaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, UTILITY_RESPONSE_IDS
from .projected_contracts import (
    ProjectedDonorUtilityRow,
    ProjectedUtilityDescriptor,
    ProjectedUtilityModel,
    ProjectedUtilityPrediction,
)
from .projected_model import predict_projected_utility


def held_donor_direction_calibration(
    rows: Sequence[ProjectedDonorUtilityRow],
    *,
    delete_models: Mapping[str, ProjectedUtilityModel],
) -> Mapping[tuple[str, str], tuple[float, float]]:
    """Calibrate by response/direction so B/I/R provenance cannot become utility."""

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.donor_center not in delete_models:
            continue
        model = delete_models[row.donor_center]
        expected_training = tuple(
            center
            for center in delete_models
            if center != row.donor_center
        )
        if (
            model.outer_target_center != row.outer_target_center
            or model.geometry_id != row.geometry_id
            or model.training_centers != expected_training
        ):
            raise ProtocolError("PCSI-PARC delete-donor model topology drifted.")
        for response_id in UTILITY_RESPONSE_IDS:
            grouped[
                (response_id, row.direction, row.donor_center, row.case_id)
            ].append(
                row.response(response_id)
                - predict_projected_utility(model, row, response_id)
            )
    result: dict[tuple[str, str], tuple[float, float]] = {}
    for response_id in UTILITY_RESPONSE_IDS:
        for direction in DIRECTION_IDS:
            donor_means: list[float] = []
            donor_case_rows: list[tuple[np.ndarray, ...]] = []
            for donor in delete_models:
                cases = sorted(
                    {
                        case_id
                        for grouped_response, grouped_direction, grouped_donor, case_id in grouped
                        if grouped_response == response_id
                        and grouped_direction == direction
                        and grouped_donor == donor
                    }
                )
                case_rows = tuple(
                    np.asarray(
                        grouped[(response_id, direction, donor, case_id)],
                        dtype=np.float64,
                    )
                    for case_id in cases
                )
                case_means = np.asarray(
                    [np.mean(values, dtype=np.float64) for values in case_rows],
                    dtype=np.float64,
                )
                if (
                    not len(cases)
                    or any(not len(values) for values in case_rows)
                    or not np.isfinite(case_means).all()
                ):
                    raise ProtocolError("PCSI-PARC residual calibration lacks a direction block.")
                donor_case_rows.append(case_rows)
                donor_means.append(float(np.mean(case_means, dtype=np.float64)))
            bias = float(np.median(np.asarray(donor_means, dtype=np.float64)))
            scale = float(
                np.sqrt(
                    np.mean(
                        [
                            np.mean(
                                [
                                    np.mean((values - bias) ** 2, dtype=np.float64)
                                    for values in case_rows
                                ],
                                dtype=np.float64,
                            )
                            for case_rows in donor_case_rows
                        ],
                        dtype=np.float64,
                    )
                )
            )
            result[(response_id, direction)] = (bias, scale)
    return result


def predict_projected_surface(
    descriptors: Sequence[ProjectedUtilityDescriptor],
    *,
    donor_rows: Sequence[ProjectedDonorUtilityRow],
    full_model: ProjectedUtilityModel,
    delete_models: Mapping[str, ProjectedUtilityModel],
    candidate_center: str | None = None,
) -> tuple[ProjectedUtilityPrediction, ...]:
    candidate = (
        full_model.outer_target_center
        if candidate_center is None
        else str(candidate_center)
    )
    training = tuple(delete_models)
    excluded = set(CENTERS).difference(training)
    if (
        full_model.training_centers != training
        or full_model.outer_target_center in training
        or candidate in training
        or full_model.outer_target_center not in excluded
        or candidate not in excluded
        or len(excluded) not in (1, 2)
        or (len(excluded) == 1 and candidate != full_model.outer_target_center)
        or (len(excluded) == 2 and candidate == full_model.outer_target_center)
    ):
        raise ProtocolError("PCSI-PARC actual/pseudo prediction scope drifted.")
    calibration = held_donor_direction_calibration(
        donor_rows,
        delete_models=delete_models,
    )
    output: list[ProjectedUtilityPrediction] = []
    for descriptor in descriptors:
        if (
            descriptor.target_center != candidate
            or descriptor.geometry_id != full_model.geometry_id
            or tuple(delete_models) != full_model.training_centers
        ):
            raise ProtocolError("PCSI-PARC model/descriptor topology drifted.")
        full_values: list[tuple[str, float]] = []
        deletion_values: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        residual_bias: list[tuple[str, float]] = []
        residual_scale: list[tuple[str, float]] = []
        robust_values: list[tuple[str, float]] = []
        for response_id in UTILITY_RESPONSE_IDS:
            full_value = predict_projected_utility(full_model, descriptor, response_id)
            values = tuple(
                (donor, predict_projected_utility(model, descriptor, response_id))
                for donor, model in delete_models.items()
            )
            bias, scale = calibration[(response_id, descriptor.direction)]
            corrected = np.asarray([value + bias for _donor, value in values], dtype=np.float64)
            robust = float(np.median(corrected))
            full_values.append((response_id, full_value))
            deletion_values.append((response_id, values))
            residual_bias.append((response_id, bias))
            residual_scale.append((response_id, scale))
            robust_values.append((response_id, robust))
        output.append(
            ProjectedUtilityPrediction(
                descriptor.descriptor_hash,
                descriptor.geometry_id,
                tuple(full_values),
                tuple(deletion_values),
                tuple(residual_bias),
                tuple(residual_scale),
                tuple(robust_values),
                (
                    full_model.model_hash,
                    *(model.model_hash for model in delete_models.values()),
                ),
            )
        )
    return tuple(output)


__all__ = (
    "held_donor_direction_calibration",
    "predict_projected_surface",
)
