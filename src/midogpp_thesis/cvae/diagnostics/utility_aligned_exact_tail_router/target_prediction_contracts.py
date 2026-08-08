"""Immutable target-prediction and seal contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import FrozenExactTailActionLibrary
from .contracts import (
    CENTERS,
    EXPECTED_FROZEN_TARGET_ACTION_COUNT,
    EXPECTED_TARGET_ACTION_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    expected_target_action_ids,
)


TARGET_PREDICTION_ARRAY_MEMBER = "arrays/target_action_predictions.npz"
TARGET_PREDICTION_INDEX_MEMBER = "tables/target_prediction_index.csv"
TARGET_PREDICTION_CACHE_MEMBER = "manifests/target_prediction_cache.json"
TARGET_PREDICTION_SEAL_MEMBER = "manifests/global_target_prediction_seal.json"
TARGET_CHECKPOINT_DIRECTORY = "checkpoints/target_predictions"
EXPECTED_TARGET_PREDICTION_CELL_COUNT = (
    len(CENTERS)
    * len(TRAINING_SEEDS)
    * len(GENERATION_SEEDS)
    * EXPECTED_TARGET_ACTION_COUNT
)


TARGET_PREDICTION_INDEX_COLUMNS = (
    "schema_version",
    "cell_ordinal",
    "target_center",
    "action_id",
    "action_hash",
    "training_seed",
    "generation_seed",
    "evaluation_row_count",
    "evaluation_row_identity_hash",
    "prediction_sha256",
    "probability_sha256",
    "composition_sha256",
    "scaler_state_hash",
    "array_start",
    "array_stop",
    "aliased_fit",
    "labels_available",
)


@dataclass(frozen=True)
class TargetPredictionCell:
    target_center: str
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    evaluation_row_identity_hash: str
    predictions: np.ndarray
    probabilities: np.ndarray
    composition_sha256: str
    scaler_state_hash: str
    aliased_fit: bool

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.target_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )


@dataclass(frozen=True)
class TargetPredictionStore:
    cells: tuple[TargetPredictionCell, ...]
    action_library_hash: str
    source_cache_lock_hash: str
    case_fold_lock_hash: str
    unique_classifier_fit_count: int
    store_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.cells) != EXPECTED_TARGET_PREDICTION_CELL_COUNT
            or len({cell.key for cell in self.cells}) != len(self.cells)
            or self.unique_classifier_fit_count <= 0
            or self.unique_classifier_fit_count > EXPECTED_TARGET_PREDICTION_CELL_COUNT
        ):
            raise ProtocolError("Utility-aligned target prediction store coverage drifted.")
        for cell in self.cells:
            pred = np.asarray(cell.predictions)
            prob = np.asarray(cell.probabilities)
            if (
                pred.ndim != 1
                or pred.dtype != np.uint8
                or prob.shape != pred.shape
                or prob.dtype != np.float32
                or not np.isin(pred, (0, 1)).all()
                or not np.isfinite(prob).all()
                or np.any(prob < 0.0)
                or np.any(prob > 1.0)
            ):
                raise ProtocolError("Utility-aligned target prediction cell drifted.")
            pred.setflags(write=False)
            prob.setflags(write=False)
        if self.store_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Utility-aligned target prediction store hash drifted.")

    @property
    def by_key(self) -> Mapping[tuple[str, str, int, int], TargetPredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_utility_aligned_stage90_target_prediction_store_v1",
            "action_library_hash": self.action_library_hash,
            "source_cache_lock_hash": self.source_cache_lock_hash,
            "case_fold_lock_hash": self.case_fold_lock_hash,
            "cell_count": len(self.cells),
            "cell_keys": [list(cell.key) for cell in self.cells],
            "cell_action_hashes": [cell.action_hash for cell in self.cells],
            "cell_prediction_hashes": [array_sha256(cell.predictions) for cell in self.cells],
            "cell_probability_hashes": [array_sha256(cell.probabilities) for cell in self.cells],
            "composition_hashes": [cell.composition_sha256 for cell in self.cells],
            "unique_classifier_fit_count": self.unique_classifier_fit_count,
            "labels_stored": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "store_hash": self.store_hash}


def canonical_target_cell_keys(
    library: FrozenExactTailActionLibrary,
) -> tuple[tuple[str, str, int, int], ...]:
    if (
        not isinstance(library, FrozenExactTailActionLibrary)
        or library.action_count != EXPECTED_FROZEN_TARGET_ACTION_COUNT
    ):
        raise ProtocolError("Target cell planning requires a frozen action library.")
    keys = tuple(
        (target, action_id, training_seed, generation_seed)
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for action_id in expected_target_action_ids(target)
    )
    if len(keys) != EXPECTED_TARGET_PREDICTION_CELL_COUNT:
        raise ProtocolError("Utility-aligned target cell arithmetic drifted.")
    return keys


def array_sha256(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "EXPECTED_TARGET_PREDICTION_CELL_COUNT",
    "TARGET_CHECKPOINT_DIRECTORY",
    "TARGET_PREDICTION_ARRAY_MEMBER",
    "TARGET_PREDICTION_CACHE_MEMBER",
    "TARGET_PREDICTION_INDEX_COLUMNS",
    "TARGET_PREDICTION_INDEX_MEMBER",
    "TARGET_PREDICTION_SEAL_MEMBER",
    "TargetPredictionCell",
    "TargetPredictionStore",
    "array_sha256",
    "canonical_target_cell_keys",
)
