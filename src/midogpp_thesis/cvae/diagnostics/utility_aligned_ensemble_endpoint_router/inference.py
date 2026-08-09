"""Target-center paired inference for the terminal ensemble endpoint."""

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
    ROUTED_ENSEMBLE_ACTION_ID,
    UNIFORM_ACTION_ID,
)
from .scoring import TargetEnsembleEndpointScoreSet


PRIMARY_CONTRASTS = (
    ("R2E-B", ROUTED_ENSEMBLE_ACTION_ID, BASE_ACTION_ID),
    ("R2E-G_delta", ROUTED_ENSEMBLE_ACTION_ID, GLOBAL_DELTA_ACTION_ID),
    ("R2E-U", ROUTED_ENSEMBLE_ACTION_ID, UNIFORM_ACTION_ID),
    ("R2E-P", ROUTED_ENSEMBLE_ACTION_ID, PERMUTATION_ACTION_ID),
)
SECONDARY_CONTRASTS = (
    ("U-B", UNIFORM_ACTION_ID, BASE_ACTION_ID),
    ("G_delta-B", GLOBAL_DELTA_ACTION_ID, BASE_ACTION_ID),
)
ALL_CONTRASTS = (*PRIMARY_CONTRASTS, *SECONDARY_CONTRASTS)


def build_center_contrasts(
    scores: TargetEnsembleEndpointScoreSet,
) -> tuple[dict[str, object], ...]:
    if not isinstance(scores, TargetEnsembleEndpointScoreSet):
        raise ProtocolError("Center contrasts require the typed target score set.")
    metric = scores.by_key
    primary = {item[0] for item in PRIMARY_CONTRASTS}
    rows: list[dict[str, object]] = []
    for target in CENTERS:
        for contrast_id, left, right in ALL_CONTRASTS:
            left_value = metric[(target, left)].balanced_accuracy
            right_value = metric[(target, right)].balanced_accuracy
            rows.append(
                {
                    "schema_version": "midogpp_utility_aligned_stage90_ensemble_center_contrast_v1",
                    "target_center": target,
                    "contrast_id": contrast_id,
                    "left_action_id": left,
                    "right_action_id": right,
                    "left_bacc": left_value,
                    "right_bacc": right_value,
                    "paired_bacc_delta": left_value - right_value,
                    "contrast_role": (
                        "primary_predeclared"
                        if contrast_id in primary
                        else "secondary_predeclared"
                    ),
                    "inference_unit": "target_center",
                    "technical_seed_cells_are_independent_units": False,
                    "consumed_data_diagnostic_only": True,
                }
            )
    return tuple(rows)


def infer_center_contrasts(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    values = tuple(rows)
    if len(values) != len(CENTERS) * len(ALL_CONTRASTS):
        raise ProtocolError("Ensemble center-contrast coverage drifted.")
    output: list[dict[str, object]] = []
    primary = {item[0] for item in PRIMARY_CONTRASTS}
    for contrast_id, left, right in ALL_CONTRASTS:
        selected = tuple(
            row for row in values if str(row.get("contrast_id")) == contrast_id
        )
        if (
            tuple(str(row.get("target_center")) for row in selected) != CENTERS
            or any(
                row.get("inference_unit") != "target_center"
                or row.get("technical_seed_cells_are_independent_units") is not False
                or row.get("consumed_data_diagnostic_only") is not True
                for row in selected
            )
        ):
            raise ProtocolError("Ensemble contrast inference units drifted.")
        delta = np.asarray(
            [float(row["paired_bacc_delta"]) for row in selected], dtype=np.float64
        )
        if not np.isfinite(delta).all():
            raise ProtocolError("Ensemble contrast values must be finite.")
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
            p_value = float(
                2.0 * student_t.sf(abs(mean / standard_error), degrees)
            )
        output.append(
            {
                "schema_version": "midogpp_utility_aligned_stage90_ensemble_contrast_inference_v1",
                "contrast_id": contrast_id,
                "left_action_id": left,
                "right_action_id": right,
                "contrast_role": (
                    "primary_predeclared"
                    if contrast_id in primary
                    else "secondary_predeclared"
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
    "PRIMARY_CONTRASTS",
    "SECONDARY_CONTRASTS",
    "build_center_contrasts",
    "infer_center_contrasts",
)
