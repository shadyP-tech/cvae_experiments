"""Support-only shrinkage path with the unchanged exact-BACC LCB gate."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.calibration import (
    class_balanced_log_loss,
    fit_baseline_intercept,
)
from ..fixed_bank_hierarchical_residual_stacker.composition import (
    calibrated_baseline_predictions,
)
from ..fixed_bank_hierarchical_residual_stacker.contracts import (
    BinaryLabel,
    SampleActionProbability,
)
from ..fixed_bank_hierarchical_residual_stacker.pooled_metrics import (
    paired_whole_case_cluster_lcb,
    score_case_confusions,
)
from .composition import compose_signed_predictions, threshold_crossing_count
from .constants import LAMBDA_GRID
from .contracts import CorrectionRow, LambdaPathRow, SignedGateDecision


def fit_signed_gate_decision(
    probabilities: Sequence[SampleActionProbability],
    corrections: Sequence[CorrectionRow],
    support_labels: Sequence[BinaryLabel],
    *,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
) -> SignedGateDecision:
    """Choose lambda by proper loss, then retain the existing conservative LCB."""

    labels = tuple(support_labels)
    if not labels or any(row.label_scope != "target_support" for row in labels):
        raise ProtocolError("Signed gate calibration requires target-support labels only.")
    keys = {row.sample_key for row in labels}
    support_probabilities = tuple(row for row in probabilities if row.sample_key in keys)
    support_corrections = tuple(row for row in corrections if row.sample_key in keys)
    if {row.sample_key for row in support_corrections} != keys:
        raise ProtocolError("Signed correction surface does not cover target support.")
    intercept_choice = fit_baseline_intercept(support_probabilities, labels)
    baseline = calibrated_baseline_predictions(
        support_probabilities, intercept=intercept_choice.intercept
    )
    grid = tuple(float(value) for value in lambda_grid)
    if grid != LAMBDA_GRID:
        raise ProtocolError("Signed gate lambda selection left the frozen grid.")
    predictions = {
        scale: compose_signed_predictions(
            support_probabilities,
            support_corrections,
            intercept=intercept_choice.intercept,
            residual_scale=scale,
            method_id="R_safe",
            safe=True,
        )
        for scale in grid
    }
    losses = {scale: class_balanced_log_loss(predictions[scale], labels) for scale in grid}
    path = tuple(
        LambdaPathRow(
            scale,
            losses[scale],
            losses[scale] - losses[0.0],
            threshold_crossing_count(predictions[scale], baseline),
        )
        for scale in grid
    )
    proposed = min(grid, key=lambda value: (losses[value], value))
    contrast = paired_whole_case_cluster_lcb(
        score_case_confusions(predictions[proposed], labels),
        score_case_confusions(baseline, labels),
    )
    admitted_count = sum(row.uncertainty_admitted for row in support_corrections)
    if admitted_count == 0:
        selected = 0.0
        fallback = "no_uncertainty_admissible_corrections"
    elif proposed == 0.0:
        selected = 0.0
        fallback = "support_proper_loss_selected_zero"
    elif contrast.lower_bound <= 0.0:
        selected = 0.0
        fallback = "support_exact_bacc_lcb_not_strictly_positive"
    else:
        selected = proposed
        fallback = None
    return SignedGateDecision(
        intercept_choice.intercept,
        proposed,
        selected,
        contrast.lower_bound,
        fallback,
        path,
    )


__all__ = ("fit_signed_gate_decision",)
