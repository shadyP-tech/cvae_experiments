"""Thin public facade for the modular pooled-BACC terminal evaluation."""

from .evaluation_contracts import (
    ActionSelectionMetricRow,
    CeilingEvaluationResult,
    CenterEvaluationMetric,
    EqualCenterInferenceRow,
    FoldEvaluationMetric,
    PermutationNullSummaryRow,
    PooledCeilingEvaluationResult,
)
from .evaluation_metrics import evaluate_decision_seal, evaluate_statistics_seal


__all__ = (
    "ActionSelectionMetricRow",
    "CeilingEvaluationResult",
    "CenterEvaluationMetric",
    "EqualCenterInferenceRow",
    "FoldEvaluationMetric",
    "PermutationNullSummaryRow",
    "PooledCeilingEvaluationResult",
    "evaluate_decision_seal",
    "evaluate_statistics_seal",
)
