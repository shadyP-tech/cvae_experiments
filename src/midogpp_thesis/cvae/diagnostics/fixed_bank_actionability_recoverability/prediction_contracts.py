"""Immutable prediction DTOs, topology, and canonical hash validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .actions import actions_for_target
from .hashing import canonical_hash as stable_hash


PHYSICAL_ACTION_COUNT_PER_TARGET = 18
EXPECTED_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_CELL_COUNT = EXPECTED_TASK_COUNT * PHYSICAL_ACTION_COUNT_PER_TARGET

PREDICTION_ARRAY_MEMBER = "arrays/actionability_action_probabilities.npz"
PREDICTION_INDEX_MEMBER = "manifests/actionability_prediction_index.json"
GLOBAL_PREDICTION_SEAL_MEMBER = "manifests/actionability_prediction_seal.json"
CHECKPOINT_DIRECTORY = "checkpoints/actionability_action_predictions"

PredictionCellKey = tuple[str, str, int, int]


class ActionSpecLike(Protocol):
    target_center: str
    action_id: str
    action_hash: str

    def to_payload(self) -> Mapping[str, object]: ...


class ActionPredictionConfig(Protocol):
    contract_hash: str
    classifier: object
    runtime: Mapping[str, object]


@lru_cache(maxsize=1)
def canonical_cell_keys() -> tuple[PredictionCellKey, ...]:
    """Return the only permitted 1,458 cells in persisted task order."""

    return tuple(
        (target, action.action_id, training, generation)
        for target in CENTERS
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in actions_for_target(target)
    )


@lru_cache(maxsize=1)
def canonical_action_hashes() -> Mapping[PredictionCellKey, str]:
    """Bind every canonical cell key to its frozen action-library hash."""

    values = {
        (target, action.action_id, training, generation): action.action_hash
        for target in CENTERS
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in actions_for_target(target)
    }
    if tuple(values) != canonical_cell_keys():
        raise AssertionError("Canonical action hashes lost prediction-cell order.")
    return MappingProxyType(values)


@dataclass(frozen=True)
class PredictionCell:
    target_center: str
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    row_identity_hash: str
    probabilities: np.ndarray
    probability_sha256: str
    predictions_sha256: str
    composition_hash: str
    scaler_state_hash: str
    fit_provenance_hash: str

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        key = self.key
        if (
            key not in canonical_action_hashes()
            or canonical_action_hashes()[key] != self.action_hash
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or sha256_array(values) != self.probability_sha256
            or sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            != self.predictions_sha256
            or not all(
                hash_like(value)
                for value in (
                    self.row_identity_hash,
                    self.composition_hash,
                    self.scaler_state_hash,
                    self.fit_provenance_hash,
                )
            )
        ):
            raise ProtocolError("Actionability prediction cell drifted.")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)

    @property
    def key(self) -> PredictionCellKey:
        return (
            self.target_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def index_payload(self, *, array_member: str) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_identity_hash": self.row_identity_hash,
            "array_member": array_member,
            "probability_sha256": self.probability_sha256,
            "predictions_sha256": self.predictions_sha256,
            "composition_hash": self.composition_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "fit_provenance_hash": self.fit_provenance_hash,
        }


@dataclass(frozen=True)
class ActionPredictionStore:
    cells: tuple[PredictionCell, ...]
    rows_by_center: Mapping[str, tuple[str, ...]]
    case_ids_by_center: Mapping[str, tuple[str, ...]]
    source_stream_lock_hash: str
    action_library_hash: str
    target_cache_binding_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        rows = {
            str(center): tuple(str(value) for value in values)
            for center, values in self.rows_by_center.items()
        }
        cases = {
            str(center): tuple(str(value) for value in values)
            for center, values in self.case_ids_by_center.items()
        }
        expected_keys = canonical_cell_keys()
        if (
            tuple(rows) != CENTERS
            or tuple(cases) != CENTERS
            or len(self.cells) != EXPECTED_CELL_COUNT
            or len({cell.key for cell in self.cells}) != EXPECTED_CELL_COUNT
            or tuple(cell.key for cell in self.cells) != expected_keys
            or any(
                len(rows[center]) != len(cases[center]) or not rows[center]
                for center in CENTERS
            )
            or any(
                len(cell.probabilities) != len(rows[cell.target_center])
                for cell in self.cells
            )
            or not external_hash_like(self.source_stream_lock_hash)
            or not all(
                hash_like(value)
                for value in (
                    self.action_library_hash,
                    self.target_cache_binding_hash,
                    self.store_hash,
                )
            )
        ):
            raise ProtocolError("Actionability prediction-store topology drifted.")
        expected = prediction_store_hash(
            self.cells,
            rows_by_center=rows,
            case_ids_by_center=cases,
            source_stream_lock_hash=self.source_stream_lock_hash,
            action_library_hash=self.action_library_hash,
            target_cache_binding_hash=self.target_cache_binding_hash,
        )
        if self.store_hash != expected:
            raise ProtocolError("Actionability prediction-store hash drifted.")
        object.__setattr__(self, "rows_by_center", MappingProxyType(rows))
        object.__setattr__(self, "case_ids_by_center", MappingProxyType(cases))

    @cached_property
    def by_key(self) -> Mapping[PredictionCellKey, PredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def probabilities(
        self,
        target_center: str,
        action_id: str,
        training_seed: int,
        generation_seed: int,
    ) -> np.ndarray:
        try:
            return self.by_key[
                (
                    str(target_center),
                    str(action_id),
                    int(training_seed),
                    int(generation_seed),
                )
            ].probabilities
        except KeyError as exc:
            raise ProtocolError("Actionability prediction cell is absent.") from exc

    def exact_nine_mean(self, target_center: str, action_id: str) -> np.ndarray:
        values = np.stack(
            [
                self.probabilities(target_center, action_id, training, generation)
                for training in TRAINING_SEEDS
                for generation in GENERATION_SEEDS
            ]
        ).astype(np.float64, copy=False)
        return np.mean(values, axis=0, dtype=np.float64)


@dataclass(frozen=True)
class GlobalActionPredictionSeal:
    store: ActionPredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "global_prediction_seal_hash"
        }
        if (
            payload.get("global_prediction_seal_hash") != stable_hash(unhashed)
            or payload.get("schema_version")
            != "midogpp_actionability_global_prediction_seal_v1"
            or payload.get("status")
            != "SEALED_ALL_1458_LABEL_FREE_ACTIONABILITY_CELLS"
            or payload.get("prediction_store_hash") != self.store.store_hash
            or payload.get("source_stream_lock_hash")
            != self.store.source_stream_lock_hash
            or payload.get("action_library_hash")
            != self.store.action_library_hash
            or payload.get("target_cache_binding_hash")
            != self.store.target_cache_binding_hash
            or payload.get("cell_count") != EXPECTED_CELL_COUNT
            or payload.get("task_count") != EXPECTED_TASK_COUNT
            or payload.get("physical_action_count_per_target")
            != PHYSICAL_ACTION_COUNT_PER_TARGET
            or payload.get("labels_opened") is not False
            or payload.get("target_expert_used") is not False
            or payload.get("seed_selection_used") is not False
            or payload.get("a1_sample_weight_scope")
            != "logistic_regression_fit_only"
            or payload.get("scaler_fit_used_sample_weight") is not False
        ):
            raise ProtocolError("Actionability global prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["global_prediction_seal_hash"])


def validate_action_library(
    action_library: Mapping[str, Sequence[ActionSpecLike]],
) -> tuple[dict[str, list[dict[str, object]]], str]:
    if tuple(action_library) != CENTERS:
        raise ProtocolError("Actionability action-library target order drifted.")
    payload: dict[str, list[dict[str, object]]] = {}
    for target in CENTERS:
        actions = tuple(action_library[target])
        expected_payloads = [action.to_payload() for action in actions_for_target(target)]
        if (
            len(actions) != PHYSICAL_ACTION_COUNT_PER_TARGET
            or len({action.action_id for action in actions}) != len(actions)
            or any(action.target_center != target for action in actions)
            or any(not hash_like(action.action_hash) for action in actions)
            or [action.to_payload() for action in actions] != expected_payloads
        ):
            raise ProtocolError("Actionability physical action menu drifted.")
        payload[target] = [dict(action.to_payload()) for action in actions]
    return payload, stable_hash(payload)


def prediction_store_hash(
    cells: Sequence[PredictionCell],
    *,
    rows_by_center: Mapping[str, Sequence[str]],
    case_ids_by_center: Mapping[str, Sequence[str]],
    source_stream_lock_hash: str,
    action_library_hash: str,
    target_cache_binding_hash: str,
) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_actionability_prediction_store_v1",
            "source_stream_lock_hash": source_stream_lock_hash,
            "action_library_hash": action_library_hash,
            "target_cache_binding_hash": target_cache_binding_hash,
            "rows_by_center": {
                center: list(rows_by_center[center]) for center in CENTERS
            },
            "case_ids_by_center": {
                center: list(case_ids_by_center[center]) for center in CENTERS
            },
            "cells": [
                {
                    "key": list(cell.key),
                    "action_hash": cell.action_hash,
                    "row_identity_hash": cell.row_identity_hash,
                    "probability_sha256": cell.probability_sha256,
                    "predictions_sha256": cell.predictions_sha256,
                    "composition_hash": cell.composition_hash,
                    "scaler_state_hash": cell.scaler_state_hash,
                    "fit_provenance_hash": cell.fit_provenance_hash,
                }
                for cell in cells
            ],
            "labels_consumed": False,
        }
    )


def hash_like(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


def external_hash_like(value: object) -> bool:
    """Accept the neutral runtime's legacy digest only at its source boundary."""

    text = str(value)
    return len(text) in {16, 64} and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


def package_scaler_state_hash(neutral_scaler_state_hash: object) -> str:
    """Bind the neutral classifier digest into package-owned SHA-256 provenance."""

    raw = str(neutral_scaler_state_hash)
    if not external_hash_like(raw):
        raise ProtocolError("Neutral scaler-state hash is malformed.")
    return stable_hash(
        {
            "schema_version": "midogpp_actionability_scaler_binding_v1",
            "neutral_scaler_state_hash": raw,
        }
    )


__all__ = (
    "ActionPredictionConfig",
    "ActionPredictionStore",
    "ActionSpecLike",
    "CHECKPOINT_DIRECTORY",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_TASK_COUNT",
    "GLOBAL_PREDICTION_SEAL_MEMBER",
    "GlobalActionPredictionSeal",
    "PHYSICAL_ACTION_COUNT_PER_TARGET",
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_MEMBER",
    "PredictionCell",
    "PredictionCellKey",
    "canonical_action_hashes",
    "canonical_cell_keys",
    "external_hash_like",
    "hash_like",
    "package_scaler_state_hash",
    "prediction_store_hash",
    "validate_action_library",
)
