"""Fixed 3/5 identification + 2/5 robust probability-level portfolio."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import PORTFOLIO_IDENTIFICATION_WEIGHT, PORTFOLIO_ROBUST_WEIGHT
from .prediction_products import MethodPrediction


def _compose_weighted(
    left_predictions: Sequence[MethodPrediction],
    robust_predictions: Sequence[MethodPrediction],
    *,
    method_id: str,
    endpoint_identity: str,
    reason: str,
) -> tuple[MethodPrediction, ...]:
    left = {row.key: row for row in left_predictions}
    robust = {row.key: row for row in robust_predictions}
    if (
        not left
        or len(left) != len(left_predictions)
        or len(robust) != len(robust_predictions)
        or set(left) != set(robust)
    ):
        raise ProtocolError("OGDE portfolio endpoints are empty, duplicated, or unaligned.")
    output: list[MethodPrediction] = []
    for key in sorted(left):
        a = left[key]
        r = robust[key]
        if a.baseline_hard_prediction != r.baseline_hard_prediction:
            raise ProtocolError("OGDE portfolio endpoints disagree on baseline branch identity.")
        probability = float(
            np.float64(float(PORTFOLIO_IDENTIFICATION_WEIGHT)) * np.float64(a.probability)
            + np.float64(float(PORTFOLIO_ROBUST_WEIGHT)) * np.float64(r.probability)
        )
        output.append(
            MethodPrediction(
                a.target_center,
                a.case_id,
                a.sample_id,
                method_id,
                probability,
                int(probability >= 0.5),
                a.baseline_hard_prediction,
                endpoint_identity,
                a.selected_source,
                (*a.selected_sources_by_arm, *r.selected_sources_by_arm),
                reason,
            )
        )
    return tuple(output)


def compose_portfolio_predictions(
    identification_predictions: Sequence[MethodPrediction],
    robust_predictions: Sequence[MethodPrediction],
    *,
    method_id: str = "OGDE_PORTFOLIO",
) -> tuple[MethodPrediction, ...]:
    expected_left = "I_FEATURE_BLOCK_PERMUTED" if method_id == "OGDE_FEATURE_BLOCK_PERMUTED" else "I_OPPORTUNITY_GATED"
    if (
        {row.method_id for row in identification_predictions} != {expected_left}
        or {row.method_id for row in robust_predictions} != {"R_NINE_ARM_ROBUST"}
    ):
        raise ProtocolError("OGDE portfolio endpoint identities drifted.")
    return _compose_weighted(
        identification_predictions,
        robust_predictions,
        method_id=method_id,
        endpoint_identity=f"portfolio::3/5::{expected_left}::2/5::R_NINE_ARM_ROBUST",
        reason="fixed_float64_probability_score_ensemble_before_sole_threshold",
    )


def compose_calibration_only_predictions(
    baseline_predictions: Sequence[MethodPrediction],
    robust_predictions: Sequence[MethodPrediction],
) -> tuple[MethodPrediction, ...]:
    if (
        {row.method_id for row in baseline_predictions} != {"B"}
        or {row.method_id for row in robust_predictions} != {"R_NINE_ARM_ROBUST"}
    ):
        raise ProtocolError("OGDE calibration-only endpoint identities drifted.")
    return _compose_weighted(
        baseline_predictions,
        robust_predictions,
        method_id="CALIBRATION_ONLY_B_R",
        endpoint_identity="control::3/5::B::2/5::R_NINE_ARM_ROBUST",
        reason="calibration_only_fixed_probability_score_ensemble",
    )


__all__ = ("compose_calibration_only_predictions", "compose_portfolio_predictions")
