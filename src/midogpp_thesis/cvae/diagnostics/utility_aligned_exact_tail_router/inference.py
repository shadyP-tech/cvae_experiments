"""Center-level paired inference for terminal consumed-data contrasts."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_DELTA_ACTION_ID,
    PERMUTATION_ACTION_ID,
    R2_ACTION_ID,
    UNIFORM_ACTION_ID,
    expected_target_action_ids,
)


PRIMARY_CONTRASTS = (
    ("R2-B", R2_ACTION_ID, BASE_ACTION_ID),
    ("R2-G_delta", R2_ACTION_ID, GLOBAL_DELTA_ACTION_ID),
    ("R2-U", R2_ACTION_ID, UNIFORM_ACTION_ID),
    ("R2-P", R2_ACTION_ID, PERMUTATION_ACTION_ID),
)
SECONDARY_CONTRASTS = (
    ("U-B", UNIFORM_ACTION_ID, BASE_ACTION_ID),
    ("G_delta-B", GLOBAL_DELTA_ACTION_ID, BASE_ACTION_ID),
)
ALL_CONTRASTS = (*PRIMARY_CONTRASTS, *SECONDARY_CONTRASTS)

CENTER_CONTRAST_COLUMNS = (
    "schema_version",
    "target_center",
    "contrast_id",
    "left_action_id",
    "right_action_id",
    "left_bacc",
    "right_bacc",
    "paired_bacc_delta",
    "contrast_role",
    "inference_unit",
    "diagnostic_only",
)
CONTRAST_INFERENCE_COLUMNS = (
    "schema_version",
    "contrast_id",
    "left_action_id",
    "right_action_id",
    "contrast_role",
    "center_count",
    "degrees_of_freedom",
    "mean_paired_bacc_delta",
    "sample_standard_deviation",
    "standard_error",
    "t_critical_975",
    "ci95_lower",
    "ci95_upper",
    "two_sided_p_value",
    "inference_unit",
    "technical_seed_cells_are_independent_units",
    "consumed_data_diagnostic_only",
)


def build_center_contrasts(
    ensemble_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    values = tuple(ensemble_rows)
    metrics = {
        (str(row["target_center"]), str(row["action_id"])): float(
            row["balanced_accuracy"]
        )
        for row in values
    }
    expected_keys = {
        (target, action_id)
        for target in CENTERS
        for action_id in expected_target_action_ids(target)
    }
    if len(values) != len(expected_keys) or set(metrics) != expected_keys:
        raise ProtocolError("Utility-aligned ensemble metric coverage drifted.")
    output: list[dict[str, object]] = []
    primary_ids = {item[0] for item in PRIMARY_CONTRASTS}
    for target in CENTERS:
        for contrast_id, left, right in ALL_CONTRASTS:
            try:
                left_value = metrics[(target, left)]
                right_value = metrics[(target, right)]
            except KeyError as exc:
                raise ProtocolError("Utility-aligned contrast action is absent.") from exc
            output.append(
                {
                    "schema_version": "midogpp_utility_aligned_stage90_center_contrast_v1",
                    "target_center": target,
                    "contrast_id": contrast_id,
                    "left_action_id": left,
                    "right_action_id": right,
                    "left_bacc": left_value,
                    "right_bacc": right_value,
                    "paired_bacc_delta": left_value - right_value,
                    "contrast_role": (
                        "primary_predeclared" if contrast_id in primary_ids else "secondary_predeclared"
                    ),
                    "inference_unit": "target_center",
                    "diagnostic_only": True,
                }
            )
    return tuple(output)


def infer_center_contrasts(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    values = tuple(rows)
    if len(values) != len(CENTERS) * len(ALL_CONTRASTS):
        raise ProtocolError("Utility-aligned center-contrast coverage drifted.")
    output: list[dict[str, object]] = []
    primary_ids = {item[0] for item in PRIMARY_CONTRASTS}
    for contrast_id, left, right in ALL_CONTRASTS:
        selected = tuple(
            row for row in values if str(row.get("contrast_id")) == contrast_id
        )
        if (
            tuple(str(row.get("target_center")) for row in selected) != CENTERS
            or any(
                row.get("inference_unit") != "target_center"
                or row.get("diagnostic_only") is not True
                for row in selected
            )
        ):
            raise ProtocolError("Utility-aligned contrast inference units drifted.")
        delta = np.asarray(
            [float(row["paired_bacc_delta"]) for row in selected], dtype=np.float64
        )
        if not np.isfinite(delta).all():
            raise ProtocolError("Utility-aligned contrast values must be finite.")
        count = len(delta)
        degrees = count - 1
        mean = float(np.mean(delta, dtype=np.float64))
        standard_deviation = float(np.std(delta, ddof=1))
        standard_error = standard_deviation / math.sqrt(float(count))
        critical = float(student_t.ppf(0.975, degrees))
        margin = critical * standard_error
        if standard_error == 0.0:
            p_value = 0.0 if mean != 0.0 else 1.0
        else:
            statistic = mean / standard_error
            p_value = float(2.0 * student_t.sf(abs(statistic), degrees))
        output.append(
            {
                "schema_version": "midogpp_utility_aligned_stage90_contrast_inference_v1",
                "contrast_id": contrast_id,
                "left_action_id": left,
                "right_action_id": right,
                "contrast_role": (
                    "primary_predeclared" if contrast_id in primary_ids else "secondary_predeclared"
                ),
                "center_count": count,
                "degrees_of_freedom": degrees,
                "mean_paired_bacc_delta": mean,
                "sample_standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "t_critical_975": critical,
                "ci95_lower": mean - margin,
                "ci95_upper": mean + margin,
                "two_sided_p_value": p_value,
                "inference_unit": "target_center",
                "technical_seed_cells_are_independent_units": False,
                "consumed_data_diagnostic_only": True,
            }
        )
    return tuple(output)


__all__ = (
    "ALL_CONTRASTS",
    "CENTER_CONTRAST_COLUMNS",
    "CONTRAST_INFERENCE_COLUMNS",
    "PRIMARY_CONTRASTS",
    "SECONDARY_CONTRASTS",
    "build_center_contrasts",
    "infer_center_contrasts",
)
