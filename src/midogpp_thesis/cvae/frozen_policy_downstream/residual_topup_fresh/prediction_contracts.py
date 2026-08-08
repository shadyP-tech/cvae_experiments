"""Data contracts shared by fresh Stage-70 prediction workers and caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    EXPECTED_PLAN_CELL_COUNT,
    GENERATION_SEEDS,
    PredictionCell,
    TRAINING_SEEDS,
)


PREDICTION_TASK_SCHEMA = "midogpp_residual_topup_fresh_prediction_task_v1"
PREDICTION_CACHE_SCHEMA = "midogpp_residual_topup_fresh_prediction_cache_v1"
PREDICTION_CELL_SCHEMA = "midogpp_residual_topup_fresh_prediction_cell_v1"
PREDICTION_INDEX_COLUMNS = (
    "target_center",
    "training_seed",
    "generation_seed",
    "action_id",
    "action_hash",
    "task_id",
    "task_hash",
    "row_count",
    "probability_sha256",
    "prediction_sha256",
    "composition_hash",
    "classifier_config_hash",
    "scaler_state_hash",
    "classifier_converged",
    "labels_available_to_fit_or_predict",
)
EXPECTED_PREDICTION_TASK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)


@dataclass(frozen=True)
class PredictionTaskRecord:
    target_center: str
    training_seed: int
    generation_seed: int
    task_id: str
    task_hash: str
    metadata_member: str
    probability_member: str
    prediction_member: str
    metadata_sha256: str
    probability_file_sha256: str
    prediction_file_sha256: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.target_center, self.training_seed, self.generation_seed

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "metadata_member": self.metadata_member,
            "probability_member": self.probability_member,
            "prediction_member": self.prediction_member,
            "metadata_sha256": self.metadata_sha256,
            "probability_file_sha256": self.probability_file_sha256,
            "prediction_file_sha256": self.prediction_file_sha256,
        }


@dataclass(frozen=True)
class PredictionCache:
    root: Path
    plan_hash: str
    source_cache_hash: str
    generation_lock_hash: str
    records: tuple[PredictionTaskRecord, ...]
    index_rows: tuple[Mapping[str, object], ...]
    predictions: tuple[PredictionCell, ...]
    cache_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.records) != EXPECTED_PREDICTION_TASK_COUNT
            or len({record.key for record in self.records})
            != EXPECTED_PREDICTION_TASK_COUNT
            or len(self.index_rows) != EXPECTED_PLAN_CELL_COUNT
            or len(self.predictions) != EXPECTED_PLAN_CELL_COUNT
        ):
            raise ProtocolError("Fresh prediction-cache coverage drifted.")


@dataclass(frozen=True)
class PredictionTaskSpec:
    payload: Mapping[str, object]

    @property
    def task_id(self) -> str:
        return str(self.payload["task_id"])


PredictionTaskExecutor = Callable[[Sequence[PredictionTaskSpec]], None]
ClassifierFitter = Callable[
    [np.ndarray, np.ndarray, np.ndarray, ClassifierSpec, int],
    Mapping[str, object],
]


__all__ = (
    "ClassifierFitter",
    "EXPECTED_PREDICTION_TASK_COUNT",
    "PREDICTION_CACHE_SCHEMA",
    "PREDICTION_CELL_SCHEMA",
    "PREDICTION_INDEX_COLUMNS",
    "PREDICTION_TASK_SCHEMA",
    "PredictionCache",
    "PredictionTaskExecutor",
    "PredictionTaskRecord",
    "PredictionTaskSpec",
)
