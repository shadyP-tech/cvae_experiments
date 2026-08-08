"""Immutable contracts for Stage-70 prediction planning and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    GENERATION_SEEDS,
    PredictionCell,
    TRAINING_SEEDS,
)


PREDICTION_TASK_SCHEMA = "midogpp_utility_aligned_prediction_task_v1"
PREDICTION_CACHE_SCHEMA = "midogpp_utility_aligned_prediction_cache_v1"
EXPECTED_PREDICTION_TASK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)
PREDICTION_INDEX_COLUMNS = (
    "schema_version",
    "target_center",
    "training_seed",
    "generation_seed",
    "action_id",
    "action_hash",
    "composition_hash",
    "evaluation_row_ids_hash",
    "probability_member",
    "probability_row",
    "probability_sha256",
)


@dataclass(frozen=True)
class PredictionTaskSpec:
    payload: Mapping[str, object]


@dataclass(frozen=True)
class PredictionTaskRecord:
    task_id: str
    task_hash: str
    metadata_member: str
    probability_member: str
    metadata_sha256: str
    probability_sha256: str
    unique_composition_fit_count: int


@dataclass(frozen=True)
class PredictionCache:
    root: Path
    plan_hash: str
    source_cache_hash: str
    generation_lock_hash: str
    records: tuple[PredictionTaskRecord, ...]
    predictions: tuple[PredictionCell, ...]
    cache_hash: str
    unique_composition_fit_count: int

    def __post_init__(self) -> None:
        if (
            len(self.records) != EXPECTED_PREDICTION_TASK_COUNT
            or len(self.predictions) != EXPECTED_LOGICAL_PREDICTION_COUNT
        ):
            raise ProtocolError("Utility-aligned prediction-cache coverage drifted.")


PredictionTaskExecutor = Callable[[Sequence[PredictionTaskSpec]], None]


__all__ = (
    "EXPECTED_PREDICTION_TASK_COUNT",
    "PREDICTION_CACHE_SCHEMA",
    "PREDICTION_INDEX_COLUMNS",
    "PREDICTION_TASK_SCHEMA",
    "PredictionCache",
    "PredictionTaskExecutor",
    "PredictionTaskRecord",
    "PredictionTaskSpec",
)
