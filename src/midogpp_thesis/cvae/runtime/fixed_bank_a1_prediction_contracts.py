"""Contracts and topology for neutral fixed-bank A1 predictions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..protocol import ProtocolError
from .artifact_io import sha256_array


ACTION_COUNT_PER_TARGET = 10
EXPECTED_TASK_COUNT = 81
EXPECTED_CELL_COUNT = 810
CHECKPOINT_DIRECTORY = "checkpoints/fixed_bank_a1_action_predictions"
PREDICTION_ARRAY_MEMBER = "arrays/fixed_bank_a1_action_probabilities.npz"
PREDICTION_INDEX_MEMBER = "manifests/fixed_bank_a1_prediction_index.json"
PREDICTION_SEAL_MEMBER = "manifests/fixed_bank_a1_prediction_seal.json"


class PredictionConfig(Protocol):
    contract_hash: str
    classifier: object
    runtime: Mapping[str, object]


@dataclass(frozen=True)
class PredictionCell:
    target_center: str
    action_id: str
    training_seed: int
    generation_seed: int
    probabilities: np.ndarray
    action_hash: str
    row_identity_hash: str
    probability_sha256: str
    prediction_sha256: str
    fit_provenance_hash: str

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        if (
            self.target_center not in CENTERS
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or self.probability_sha256 != sha256_array(values)
            or self.prediction_sha256
            != sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            or not sha256_digest(self.action_hash)
            or not stable_digest(self.row_identity_hash)
            or not stable_digest(self.fit_provenance_hash)
        ):
            raise ProtocolError("Fixed-bank A1 prediction cell drifted.")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.target_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )


@dataclass(frozen=True)
class PredictionStore:
    cells: tuple[PredictionCell, ...]
    rows_by_center: Mapping[str, tuple[str, ...]]
    case_ids_by_center: Mapping[str, tuple[str, ...]]
    source_stream_lock_hash: str
    action_library_hash: str
    target_cache_binding_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        rows = {str(key): tuple(value) for key, value in self.rows_by_center.items()}
        cases = {
            str(key): tuple(value) for key, value in self.case_ids_by_center.items()
        }
        expected_keys = tuple(
            (target, action, training, generation)
            for target in CENTERS
            for training in TRAINING_SEEDS
            for generation in GENERATION_SEEDS
            for action in (
                "B",
                "U",
                *(f"A1::source={source}" for source in CENTERS if source != target),
            )
        )
        if (
            tuple(rows) != CENTERS
            or tuple(cases) != CENTERS
            or len(self.cells) != EXPECTED_CELL_COUNT
            or len({cell.key for cell in self.cells}) != EXPECTED_CELL_COUNT
            or tuple(cell.key for cell in self.cells) != expected_keys
            or any(len(rows[center]) != len(cases[center]) for center in CENTERS)
            or any(
                len(cell.probabilities) != len(rows[cell.target_center])
                for cell in self.cells
            )
            or not stable_digest(self.source_stream_lock_hash)
            or not stable_digest(self.action_library_hash)
            or not sha256_digest(self.target_cache_binding_hash)
            or not stable_digest(self.store_hash)
            or self.store_hash
            != prediction_store_hash(
                self.cells,
                rows,
                cases,
                self.source_stream_lock_hash,
                self.action_library_hash,
                self.target_cache_binding_hash,
            )
        ):
            raise ProtocolError("Fixed-bank A1 prediction store drifted.")
        object.__setattr__(self, "rows_by_center", MappingProxyType(rows))
        object.__setattr__(self, "case_ids_by_center", MappingProxyType(cases))

    @cached_property
    def by_key(self) -> Mapping[tuple[str, str, int, int], PredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def probabilities(
        self, target: str, action: str, training: int, generation: int
    ) -> np.ndarray:
        try:
            return self.by_key[
                (str(target), str(action), int(training), int(generation))
            ].probabilities
        except KeyError as exc:
            raise ProtocolError("Fixed-bank A1 probability cell is absent.") from exc

    def exact_nine(self, target: str, action: str) -> np.ndarray:
        values = np.stack(
            [
                self.probabilities(target, action, training, generation)
                for training in TRAINING_SEEDS
                for generation in GENERATION_SEEDS
            ]
        ).astype(np.float64, copy=False)
        return np.mean(values, axis=0, dtype=np.float64)


@dataclass(frozen=True)
class GlobalPredictionSeal:
    store: PredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["global_prediction_seal_hash"])


def validate_action_library(
    action_library: Mapping[str, Sequence[object]],
) -> tuple[dict[str, list[dict[str, object]]], str]:
    if tuple(action_library) != CENTERS:
        raise ProtocolError("Fixed-bank A1 action target order drifted.")
    payload: dict[str, list[dict[str, object]]] = {}
    for target in CENTERS:
        actions = tuple(action_library[target])
        rows = [dict(getattr(action, "to_payload")()) for action in actions]
        if (
            len(rows) != ACTION_COUNT_PER_TARGET
            or len({str(row.get("action_id")) for row in rows})
            != ACTION_COUNT_PER_TARGET
            or any(
                str(row.get("target_center")) != target
                or row.get("target_expert_excluded") is not True
                for row in rows
            )
        ):
            raise ProtocolError("Fixed-bank A1 action menu drifted.")
        payload[target] = rows
    return payload, stable_hash(payload)


def prediction_store_hash(
    cells: Sequence[PredictionCell],
    rows: Mapping[str, Sequence[str]],
    cases: Mapping[str, Sequence[str]],
    source_hash: str,
    library_hash: str,
    binding: str,
) -> str:
    return stable_hash(
        {
            "schema_version": "fixed_bank_a1_prediction_store_v1",
            "source_stream_lock_hash": source_hash,
            "action_library_hash": library_hash,
            "target_cache_binding_hash": binding,
            "rows_by_center": {key: list(value) for key, value in rows.items()},
            "case_ids_by_center": {key: list(value) for key, value in cases.items()},
            "cells": [
                {
                    "key": list(cell.key),
                    "action_hash": cell.action_hash,
                    "row_identity_hash": cell.row_identity_hash,
                    "probability_sha256": cell.probability_sha256,
                    "prediction_sha256": cell.prediction_sha256,
                    "fit_provenance_hash": cell.fit_provenance_hash,
                }
                for cell in cells
            ],
            "labels_consumed": False,
        }
    )


def assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("scientific_reductions_dtype") != "float64"
        or int(runtime.get("target_task_count", -1)) != EXPECTED_TASK_COUNT
        or int(runtime.get("target_probability_cell_count", -1))
        != EXPECTED_CELL_COUNT
        or int(runtime.get("maximum_total_classifier_fit_count", -1))
        != EXPECTED_CELL_COUNT
    ):
        raise ProtocolError("Fixed-bank A1 prediction workstation contract drifted.")


def digest_like(value: object) -> bool:
    text = str(value)
    return len(text) in {16, 64} and text == text.lower() and all(
        character in "0123456789abcdef" for character in text
    )


def stable_digest(value: object) -> bool:
    text = str(value)
    return len(text) == 16 and digest_like(text)


def sha256_digest(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and digest_like(text)


__all__ = (
    "ACTION_COUNT_PER_TARGET",
    "CHECKPOINT_DIRECTORY",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_TASK_COUNT",
    "GlobalPredictionSeal",
    "PREDICTION_ARRAY_MEMBER",
    "PREDICTION_INDEX_MEMBER",
    "PREDICTION_SEAL_MEMBER",
    "PredictionCell",
    "PredictionConfig",
    "PredictionStore",
    "assert_runtime",
    "digest_like",
    "sha256_digest",
    "stable_digest",
    "prediction_store_hash",
    "validate_action_library",
)
