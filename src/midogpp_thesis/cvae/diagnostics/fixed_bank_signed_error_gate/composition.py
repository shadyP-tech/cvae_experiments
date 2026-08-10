"""Baseline-anchored composition for signed raw and safe corrections."""

from __future__ import annotations

import math
from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    PredictionRow,
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.residuals import logit_clip, sigmoid
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    BASELINE_ACTION_ID,
)
from .constants import (
    HARD_THRESHOLD,
    INTERCEPT_GRID,
    LAMBDA_GRID,
    MARGIN_BANDWIDTH_LOGIT,
)
from .contracts import CorrectionRow


def margin_gate(calibrated_baseline_probability: float) -> float:
    margin = abs(logit_clip(calibrated_baseline_probability))
    return math.exp(-((margin / MARGIN_BANDWIDTH_LOGIT) ** 2))


def compose_signed_predictions(
    probabilities: Sequence[SampleActionProbability],
    corrections: Sequence[CorrectionRow],
    *,
    intercept: float,
    residual_scale: float,
    method_id: str,
    safe: bool,
) -> tuple[PredictionRow, ...]:
    intercept_value = float(intercept)
    scale = float(residual_scale)
    method_contract = {
        "G": ("G", True),
        "R_raw": ("R", False),
        "R_safe": ("R", True),
        "P": ("P", True),
    }
    correction_families = {row.family for row in corrections}
    if (
        intercept_value not in INTERCEPT_GRID
        or scale not in LAMBDA_GRID
        or method_id not in method_contract
        or type(safe) is not bool
        or len(correction_families) != 1
        or method_contract.get(method_id)
        != (next(iter(correction_families), ""), safe)
    ):
        raise ProtocolError("Signed composition left its frozen method/grid contract.")
    baseline_rows = tuple(
        row
        for row in probabilities
        if row.action_id == BASELINE_ACTION_ID
    )
    baseline = {row.sample_key: row.probability for row in baseline_rows}
    correction_by_key = {row.sample_key: row for row in corrections}
    if (
        not baseline
        or len(baseline_rows) != len(baseline)
        or len(correction_by_key) != len(tuple(corrections))
    ):
        raise ProtocolError("Signed composition inputs are empty or duplicated.")
    if set(baseline) != set(correction_by_key):
        raise ProtocolError("Signed composition baseline and correction rows are not aligned.")
    output: list[PredictionRow] = []
    for target, case, sample in sorted(baseline):
        key = (target, case, sample)
        baseline_logit = logit_clip(baseline[key]) + intercept_value
        calibrated = sigmoid(baseline_logit)
        if scale == 0.0:
            probability = calibrated
        else:
            correction = correction_by_key[key]
            signed = correction.safe_correction if safe else correction.raw_correction
            probability = sigmoid(
                baseline_logit
                + scale * margin_gate(calibrated) * signed
            )
        output.append(
            PredictionRow(
                method_id,
                target,
                case,
                sample,
                probability,
                int(probability >= HARD_THRESHOLD),
            )
        )
    return tuple(output)


def threshold_crossing_count(
    candidate: Sequence[PredictionRow], reference: Sequence[PredictionRow]
) -> int:
    candidate_by_key = {row.sample_key: row.hard_prediction for row in candidate}
    reference_by_key = {row.sample_key: row.hard_prediction for row in reference}
    if set(candidate_by_key) != set(reference_by_key):
        raise ProtocolError("Threshold-crossing surfaces are not aligned.")
    return sum(candidate_by_key[key] != reference_by_key[key] for key in candidate_by_key)


__all__ = ("compose_signed_predictions", "margin_gate", "threshold_crossing_count")
