"""Immutable contracts for exact-tail CPU prediction execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .runtime import CoarsePredictionTask
from .scoring import SealedPredictionSurface
from .seals import GlobalPredictionSeal, PredictionCellSeal
from .source_contracts import SourceBlockRecord


PREDICTION_ARRAY_MEMBER = "arrays/exact_tail_predictions.npz"
PREDICTION_INDEX_MEMBER = "manifests/prediction_index.json"
GLOBAL_SEAL_MEMBER = "manifests/global_prediction_seal.json"
CHECKPOINT_SCHEMA = "midogpp_exact_tail_prediction_checkpoint_v2"
PREDICTION_INDEX_SCHEMA = "midogpp_exact_tail_prediction_index_v2"


@dataclass(frozen=True)
class CoarsePredictionRecord:
    task: CoarsePredictionTask
    checkpoint_relative_path: str
    checkpoint_file_sha256: str
    evaluation_row_count: int
    action_composition_sha256: Mapping[str, str]
    action_scaler_state_hash: Mapping[str, str]
    checkpoint_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_composition_sha256",
            MappingProxyType(dict(self.action_composition_sha256)),
        )
        object.__setattr__(
            self,
            "action_scaler_state_hash",
            MappingProxyType(dict(self.action_scaler_state_hash)),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Rebuild immutable hash maps after a process-pool result transfer."""

        return (
            type(self),
            (
                self.task,
                self.checkpoint_relative_path,
                self.checkpoint_file_sha256,
                self.evaluation_row_count,
                dict(self.action_composition_sha256),
                dict(self.action_scaler_state_hash),
                self.checkpoint_hash,
            ),
        )


@dataclass(frozen=True)
class PredictionExecutionResult:
    predictions: SealedPredictionSurface
    seal: GlobalPredictionSeal
    seal_path: Path
    prediction_index_path: Path
    prediction_arrays_path: Path
    task_records: tuple[CoarsePredictionRecord, ...]


@dataclass(frozen=True)
class PredictionWorkerInput:
    task: CoarsePredictionTask
    cache_root: str
    source_records: tuple[SourceBlockRecord, ...]
    evaluation_array_path: str
    evaluation_row_identity_hash: str
    partition_hash: str
    source_cache_hash: str
    classifier_payload: Mapping[str, object]
    checkpoint_root: str
    support_array_path: str = ""
    support_row_identity_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "classifier_payload",
            MappingProxyType(dict(self.classifier_payload)),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Rebuild the immutable classifier payload in a spawned CPU worker."""

        return (
            type(self),
            (
                self.task,
                self.cache_root,
                self.source_records,
                self.evaluation_array_path,
                self.evaluation_row_identity_hash,
                self.partition_hash,
                self.source_cache_hash,
                dict(self.classifier_payload),
                self.checkpoint_root,
                self.support_array_path,
                self.support_row_identity_hash,
            ),
        )


@dataclass(frozen=True)
class ConsolidatedPredictionArtifacts:
    predictions_by_key: Mapping[tuple[str, str, str, int, int], np.ndarray]
    probabilities_by_key: Mapping[tuple[str, str, str, int, int], np.ndarray]
    support_probabilities_by_key: Mapping[
        tuple[str, str, str, int, int], np.ndarray
    ]
    cells: tuple[PredictionCellSeal, ...]
    prediction_index_path: Path
    prediction_arrays_path: Path
    prediction_index_sha256: str
    prediction_arrays_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predictions_by_key",
            MappingProxyType(dict(self.predictions_by_key)),
        )
        object.__setattr__(
            self,
            "probabilities_by_key",
            MappingProxyType(dict(self.probabilities_by_key)),
        )
        object.__setattr__(
            self,
            "support_probabilities_by_key",
            MappingProxyType(dict(self.support_probabilities_by_key)),
        )


__all__ = (
    "CHECKPOINT_SCHEMA",
    "GLOBAL_SEAL_MEMBER",
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_MEMBER",
    "PREDICTION_INDEX_SCHEMA",
    "CoarsePredictionRecord",
    "ConsolidatedPredictionArtifacts",
    "PredictionExecutionResult",
    "PredictionWorkerInput",
)
