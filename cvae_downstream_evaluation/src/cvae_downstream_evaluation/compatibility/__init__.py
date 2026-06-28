"""Source-inner learned downstream utility selection primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..protocol import ProtocolError


@dataclass(frozen=True)
class CompatibilityPrediction:
    candidate_id: str
    predicted_primary_utility: float
    support_nelbo: float
    source_inner_stability: float


def select_top1(predictions: Sequence[CompatibilityPrediction]) -> CompatibilityPrediction:
    if not predictions:
        raise ProtocolError("No compatibility predictions available.")
    return sorted(
        predictions,
        key=lambda pred: (
            _invalid_score(pred.predicted_primary_utility),
            -float(pred.predicted_primary_utility) if math.isfinite(float(pred.predicted_primary_utility)) else 0.0,
            float(pred.support_nelbo),
            -float(pred.source_inner_stability),
            pred.candidate_id,
        ),
    )[0]


def topk_uniform(predictions: Sequence[CompatibilityPrediction], *, k: int) -> dict[str, float]:
    if k <= 0:
        raise ProtocolError("top-k aggregation requires k > 0.")
    selected = sorted(
        predictions,
        key=lambda pred: (
            _invalid_score(pred.predicted_primary_utility),
            -float(pred.predicted_primary_utility) if math.isfinite(float(pred.predicted_primary_utility)) else 0.0,
            float(pred.support_nelbo),
            pred.candidate_id,
        ),
    )[: int(k)]
    if not selected:
        raise ProtocolError("No candidates selected for top-k aggregation.")
    weight = 1.0 / float(len(selected))
    return {pred.candidate_id: weight for pred in selected}


def softmax_weights(
    predictions: Sequence[CompatibilityPrediction],
    *,
    tau: float,
    fallback_to_support_nelbo: bool = True,
) -> dict[str, float]:
    if tau <= 0.0:
        raise ProtocolError("Soft aggregation temperature tau must be positive.")
    finite = [pred for pred in predictions if math.isfinite(float(pred.predicted_primary_utility))]
    if not finite:
        if not fallback_to_support_nelbo:
            raise ProtocolError("No finite predicted utilities for soft aggregation.")
        ordered = sorted(predictions, key=lambda pred: (float(pred.support_nelbo), pred.candidate_id))
        return {pred.candidate_id: 1.0 if idx == 0 else 0.0 for idx, pred in enumerate(ordered)}
    max_score = max(float(pred.predicted_primary_utility) for pred in finite)
    numerators = {
        pred.candidate_id: math.exp((float(pred.predicted_primary_utility) - max_score) / float(tau))
        for pred in finite
    }
    denom = sum(numerators.values())
    if denom <= 0.0:
        raise ProtocolError("Soft aggregation weights are degenerate.")
    return {candidate_id: value / denom for candidate_id, value in numerators.items()}


def assert_source_inner_training_labels(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject real held-out target downstream labels in estimator training rows."""

    forbidden = {
        "target_bacc",
        "target_macro_f1",
        "heldout_target_bacc",
        "heldout_target_macro_f1",
        "real_target_downstream_label",
        "downstream_oracle_expert",
        "oracle_rank",
    }
    for idx, row in enumerate(rows):
        present = sorted(forbidden.intersection(row))
        if present:
            raise ProtocolError(f"Estimator training row {idx} contains forbidden target labels: {present}")
        if str(row.get("fold_role", "")) != "source_inner_pseudo_target":
            raise ProtocolError(
                f"Estimator training row {idx} must be source_inner_pseudo_target; "
                f"got {row.get('fold_role')!r}"
            )


def _invalid_score(value: float) -> int:
    return 0 if math.isfinite(float(value)) else 1
