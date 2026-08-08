"""Shared coverage validation for sealed Stage-70 metric surfaces."""

from __future__ import annotations

from ...protocol import ProtocolError
from .contracts import (
    EXPECTED_ENSEMBLE_METRIC_COUNT,
    EXPECTED_PLAN_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    PRIMARY_ENDPOINT,
    ScoredEvaluation,
)


def validate_scored_evaluation(scored: ScoredEvaluation) -> None:
    if not isinstance(scored, ScoredEvaluation):
        raise ProtocolError("Fresh Stage-70 scored input is invalid.")
    if (
        scored.primary_endpoint != PRIMARY_ENDPOINT
        or len(scored.seed_cell_metrics) != EXPECTED_PLAN_CELL_COUNT
        or len(scored.ensemble_metrics) != EXPECTED_ENSEMBLE_METRIC_COUNT
        or len({row.key for row in scored.seed_cell_metrics})
        != EXPECTED_PLAN_CELL_COUNT
        or len({row.key for row in scored.ensemble_metrics})
        != EXPECTED_ENSEMBLE_METRIC_COUNT
        or any(
            row.prediction_seal_hash != scored.prediction_seal_hash
            or row.endpoint_role
            != "paired_seed_cell_mean_bacc_descriptive_only"
            or row.descriptive_only is not True
            for row in scored.seed_cell_metrics
        )
        or any(
            row.prediction_seal_hash != scored.prediction_seal_hash
            or row.endpoint != PRIMARY_ENDPOINT
            or row.primary_endpoint is not True
            or row.seed_cell_count != EXPECTED_SEED_CELL_COUNT
            for row in scored.ensemble_metrics
        )
    ):
        raise ProtocolError("Fresh Stage-70 scored endpoint/coverage drifted.")


__all__ = ("validate_scored_evaluation",)
