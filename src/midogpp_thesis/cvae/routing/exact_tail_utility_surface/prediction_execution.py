"""Stable facade for exact-tail CPU prediction execution and sealing."""

from .prediction_contracts import (
    GLOBAL_SEAL_MEMBER,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    CoarsePredictionRecord,
    PredictionExecutionResult,
)
from .prediction_orchestration import materialize_exact_tail_predictions


__all__ = (
    "GLOBAL_SEAL_MEMBER",
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_MEMBER",
    "CoarsePredictionRecord",
    "PredictionExecutionResult",
    "materialize_exact_tail_predictions",
)
