"""Source-inner estimator training entry points."""

from __future__ import annotations

from typing import Mapping, Sequence

from . import assert_source_inner_training_labels
from .estimators import LinearUtilityEstimator, MeanUtilityEstimator


def train_mean_utility_estimator(rows: Sequence[Mapping[str, object]]) -> MeanUtilityEstimator:
    assert_source_inner_training_labels(rows)
    return MeanUtilityEstimator.fit(rows)


def train_linear_utility_estimator(
    rows: Sequence[Mapping[str, object]],
    *,
    feature_columns: Sequence[str],
    label: str = "source_inner_heldout_bacc",
    ridge_lambda: float = 1e-6,
) -> LinearUtilityEstimator:
    assert_source_inner_training_labels(rows)
    return LinearUtilityEstimator.fit(
        rows,
        feature_columns=feature_columns,
        label=label,
        ridge_lambda=ridge_lambda,
    )
