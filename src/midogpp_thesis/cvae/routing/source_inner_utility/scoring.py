"""Compatibility facade for source-inner prediction and metric scoring."""

from .metric_scoring import (
    CASE_CONFUSION_COLUMNS,
    UTILITY_COLUMNS,
    reconstruct_metrics_from_case_confusions,
    score_prediction_pass,
)
from .prediction import (
    FIT_COLUMNS,
    PredictionPass,
    array_sha256,
    generated_block_sha256,
    run_label_free_prediction_pass,
)


__all__ = (
    "CASE_CONFUSION_COLUMNS",
    "FIT_COLUMNS",
    "PredictionPass",
    "UTILITY_COLUMNS",
    "array_sha256",
    "generated_block_sha256",
    "reconstruct_metrics_from_case_confusions",
    "run_label_free_prediction_pass",
    "score_prediction_pass",
)
