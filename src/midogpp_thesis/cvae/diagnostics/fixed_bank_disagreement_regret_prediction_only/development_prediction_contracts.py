"""Contracts for the strict source-OOF development classifier bank."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array, sha256_file
from .constants import (
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    FEATURE_DIM,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from .development_actions import (
    DEVELOPMENT_ACTION_COUNT_PER_TASK,
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
    development_actions_for,
)
from .hashing import canonical_hash
from .prediction_contracts import classifier_parameter_sha256


DEVELOPMENT_ACTION_LIBRARY_MEMBER = "manifests/source_oof_action_library.json"
DEVELOPMENT_CLASSIFIER_MEAN_MEMBER = "arrays/source_oof_classifier_scaler_mean.npy"
DEVELOPMENT_CLASSIFIER_SCALE_MEMBER = "arrays/source_oof_classifier_scaler_scale.npy"
DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER = "arrays/source_oof_classifier_coefficients.npy"
DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER = "arrays/source_oof_classifier_intercepts.npy"
DEVELOPMENT_CLASSIFIER_INDEX_MEMBER = "manifests/source_oof_classifier_bank_index.json"
DEVELOPMENT_CLASSIFIER_SEAL_MEMBER = "manifests/source_oof_classifier_bank_seal.json"
DEVELOPMENT_PREDICTION_ARRAY_MEMBER = "arrays/source_oof_action_probabilities.npz"
DEVELOPMENT_PREDICTION_INDEX_MEMBER = "manifests/source_oof_prediction_index.json"
DEVELOPMENT_PREDICTION_SEAL_MEMBER = "manifests/source_oof_prediction_seal.json"
COMPOSITE_PRELABEL_SEAL_MEMBER = "manifests/prelabel_prediction_composite_seal.json"
DEVELOPMENT_CHECKPOINT_DIRECTORY = "checkpoints/disagreement_regret_strict_source_oof"

DEVELOPMENT_CLASSIFIER_STATUS = (
    "SEALED_5184_PHYSICAL_STRICT_SOURCE_OOF_CLASSIFIERS"
)
DEVELOPMENT_PREDICTION_STATUS = (
    "SEALED_5184_PHYSICAL_STRICT_SOURCE_OOF_FITS_10368_LOGICAL_PREDICTIONS"
)
COMPOSITE_PRELABEL_STATUS = (
    "SEALED_STRICT_SOURCE_OOF_AND_TARGET_CLASSIFIER_BANK_BEFORE_LABELS"
)

PhysicalCellKey = tuple[str, str, str, int, int]
LogicalCellKey = tuple[str, str, str, int, int]


class DevelopmentPredictionConfig(Protocol):
    contract_hash: str
    classifier: object
    runtime: Mapping[str, object]


def _sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return text


@lru_cache(maxsize=1)
def canonical_physical_cell_keys() -> tuple[PhysicalCellKey, ...]:
    return tuple(
        (left, right, action.action_id, training, generation)
        for left, right in combinations(CENTERS, 2)
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in development_actions_for(left, right)
    )


@lru_cache(maxsize=1)
def canonical_logical_cell_keys() -> tuple[LogicalCellKey, ...]:
    return tuple(
        (target, query, action.action_id, training, generation)
        for target in CENTERS
        for query in CENTERS
        if query != target
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in development_actions_for(target, query)
    )


@lru_cache(maxsize=1)
def canonical_physical_action_hashes() -> Mapping[PhysicalCellKey, str]:
    result = {
        (left, right, action.action_id, training, generation): action.action_hash
        for left, right in combinations(CENTERS, 2)
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in development_actions_for(left, right)
    }
    if tuple(result) != canonical_physical_cell_keys():
        raise AssertionError("Strict source-OOF physical action order drifted.")
    return MappingProxyType(result)


@lru_cache(maxsize=1)
def canonical_logical_action_hashes() -> Mapping[LogicalCellKey, tuple[str, str]]:
    result = {
        (target, query, action.action_id, training, generation): (
            action.action_hash,
            action.orientation_hash,
        )
        for target in CENTERS
        for query in CENTERS
        if query != target
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in development_actions_for(target, query)
    }
    if tuple(result) != canonical_logical_cell_keys():
        raise AssertionError("Strict source-OOF logical action order drifted.")
    return MappingProxyType(result)


@dataclass(frozen=True)
class DevelopmentClassifierCell:
    cell_ordinal: int
    excluded_pair: tuple[str, str]
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    composition_hash: str
    scaler_state_hash: str
    parameter_sha256: str
    fit_provenance_hash: str
    classifier_config_hash: str
    n_iter: tuple[int, ...]
    converged: bool

    def __post_init__(self) -> None:
        pair = tuple(str(value) for value in self.excluded_pair)
        if (
            type(self.cell_ordinal) is not int
            or self.cell_ordinal < 0
            or len(pair) != 2
            or pair != tuple(sorted(pair))
            or self.key not in canonical_physical_action_hashes()
            or canonical_physical_action_hashes()[self.key] != self.action_hash
            or not all(
                len(str(value)) in (16, 64)
                for value in (
                    self.composition_hash,
                    self.scaler_state_hash,
                    self.classifier_config_hash,
                )
            )
            or len(self.parameter_sha256) != 64
            or len(self.fit_provenance_hash) != 64
            or not self.n_iter
            or any(type(value) is not int or value < 0 for value in self.n_iter)
            or self.converged is not True
        ):
            raise ProtocolError("Strict source-OOF classifier cell drifted.")
        object.__setattr__(self, "excluded_pair", pair)

    @property
    def key(self) -> PhysicalCellKey:
        return (
            self.excluded_pair[0],
            self.excluded_pair[1],
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "cell_ordinal": self.cell_ordinal,
            "excluded_pair": list(self.excluded_pair),
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "composition_hash": self.composition_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "parameter_sha256": self.parameter_sha256,
            "fit_provenance_hash": self.fit_provenance_hash,
            "classifier_config_hash": self.classifier_config_hash,
            "n_iter": list(self.n_iter),
            "converged": self.converged,
        }


@dataclass(frozen=True)
class DevelopmentClassifierBank:
    root: Path
    cells: tuple[DevelopmentClassifierCell, ...]
    source_stream_lock_hash: str
    action_library_hash: str
    source_cache_binding_hash: str
    config_contract_hash: str
    bank_hash: str
    seal_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if (
            len(cells) != DEVELOPMENT_CLASSIFIER_FIT_COUNT
            or tuple(cell.key for cell in cells) != canonical_physical_cell_keys()
            or tuple(cell.cell_ordinal for cell in cells)
            != tuple(range(DEVELOPMENT_CLASSIFIER_FIT_COUNT))
            or len({cell.fit_provenance_hash for cell in cells}) != len(cells)
        ):
            raise ProtocolError("Strict source-OOF classifier-bank topology drifted.")
        for value, role in (
            (self.source_stream_lock_hash, "source stream lock hash"),
            (self.action_library_hash, "action library hash"),
            (self.source_cache_binding_hash, "source cache binding hash"),
            (self.bank_hash, "classifier bank hash"),
        ):
            _sha256(value, role)
        arrays = self._arrays()
        expected = (DEVELOPMENT_CLASSIFIER_FIT_COUNT, FEATURE_DIM)
        if (
            any(array.shape != expected for array in arrays[:3])
            or arrays[3].shape != (DEVELOPMENT_CLASSIFIER_FIT_COUNT,)
            or any(array.dtype != np.float64 for array in arrays)
            or not all(np.isfinite(array).all() for array in arrays)
            or np.any(arrays[1] <= 0.0)
        ):
            raise ProtocolError("Strict source-OOF classifier arrays drifted.")
        mean, scale, coefficient, intercept = arrays
        for cell in cells:
            index = cell.cell_ordinal
            if cell.parameter_sha256 != classifier_parameter_sha256(
                mean[index], scale[index], coefficient[index], intercept[index]
            ):
                raise ProtocolError("Strict source-OOF classifier parameters drifted.")
        payload = dict(self.seal_payload)
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "development_classifier_bank_seal_hash"
        }
        if (
            payload.get("development_classifier_bank_seal_hash")
            != canonical_hash(unhashed)
            or payload.get("status") != DEVELOPMENT_CLASSIFIER_STATUS
            or payload.get("classifier_bank_hash") != self.bank_hash
            or payload.get("physical_fit_count") != DEVELOPMENT_CLASSIFIER_FIT_COUNT
            or payload.get("source_labels_available_during_fit") is not False
            or payload.get("test_cache_admitted") is not False
            or any(
                payload.get(name) != sha256_file(path)
                for name, path in zip(
                    (
                        "scaler_mean_file_sha256",
                        "scaler_scale_file_sha256",
                        "coefficient_file_sha256",
                        "intercept_file_sha256",
                    ),
                    self.parameter_paths,
                    strict=True,
                )
            )
        ):
            raise ProtocolError("Strict source-OOF classifier-bank seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def parameter_paths(self) -> tuple[Path, Path, Path, Path]:
        return tuple(
            self.root / member
            for member in (
                DEVELOPMENT_CLASSIFIER_MEAN_MEMBER,
                DEVELOPMENT_CLASSIFIER_SCALE_MEMBER,
                DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER,
                DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER,
            )
        )  # type: ignore[return-value]

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.load(path, mmap_mode="r", allow_pickle=False)
            for path in self.parameter_paths
        )  # type: ignore[return-value]

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["development_classifier_bank_seal_hash"])

    @cached_property
    def by_key(self) -> Mapping[PhysicalCellKey, DevelopmentClassifierCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})


@dataclass(frozen=True)
class DevelopmentPredictionCell:
    outer_target: str
    query_center: str
    action_id: str
    action_hash: str
    orientation_hash: str
    training_seed: int
    generation_seed: int
    row_identity_hash: str
    probabilities: np.ndarray
    probability_sha256: str
    predictions_sha256: str
    classifier_parameter_sha256: str

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        expected = canonical_logical_action_hashes().get(self.key)
        if (
            expected != (self.action_hash, self.orientation_hash)
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or sha256_array(values) != self.probability_sha256
            or sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            != self.predictions_sha256
        ):
            raise ProtocolError("Strict source-OOF prediction cell drifted.")
        _sha256(self.row_identity_hash, "source row identity hash")
        _sha256(self.classifier_parameter_sha256, "classifier parameter hash")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)

    @property
    def key(self) -> LogicalCellKey:
        return (
            self.outer_target,
            self.query_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def index_payload(self, *, array_member: str) -> dict[str, object]:
        return {
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "orientation_hash": self.orientation_hash,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_identity_hash": self.row_identity_hash,
            "array_member": array_member,
            "probability_sha256": self.probability_sha256,
            "predictions_sha256": self.predictions_sha256,
            "classifier_parameter_sha256": self.classifier_parameter_sha256,
        }


@dataclass(frozen=True)
class DevelopmentPredictionStore:
    cells: tuple[DevelopmentPredictionCell, ...]
    rows_by_query: Mapping[str, tuple[str, ...]]
    case_ids_by_query: Mapping[str, tuple[str, ...]]
    frame_cache_binding_hash: str
    action_library_hash: str
    development_classifier_bank_seal_hash: str
    store_hash: str
    frame_role: str = "source"

    def __post_init__(self) -> None:
        rows = {str(key): tuple(map(str, value)) for key, value in self.rows_by_query.items()}
        cases = {
            str(key): tuple(map(str, value)) for key, value in self.case_ids_by_query.items()
        }
        if (
            self.frame_role != "source"
            or tuple(rows) != CENTERS
            or tuple(cases) != CENTERS
            or tuple(cell.key for cell in self.cells) != canonical_logical_cell_keys()
            or len(self.cells) != DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
            or any(not rows[q] or len(rows[q]) != len(cases[q]) for q in CENTERS)
            or any(len(cell.probabilities) != len(rows[cell.query_center]) for cell in self.cells)
        ):
            raise ProtocolError("Strict source-OOF prediction-store topology drifted.")
        for value, role in (
            (self.frame_cache_binding_hash, "frame cache binding hash"),
            (self.action_library_hash, "action library hash"),
            (self.development_classifier_bank_seal_hash, "classifier bank seal hash"),
            (self.store_hash, "prediction store hash"),
        ):
            _sha256(value, role)
        if self.store_hash != development_prediction_store_hash(
            self.cells,
            rows_by_query=rows,
            case_ids_by_query=cases,
            frame_cache_binding_hash=self.frame_cache_binding_hash,
            action_library_hash=self.action_library_hash,
            development_classifier_bank_seal_hash=self.development_classifier_bank_seal_hash,
        ):
            raise ProtocolError("Strict source-OOF prediction-store hash drifted.")
        object.__setattr__(self, "rows_by_query", MappingProxyType(rows))
        object.__setattr__(self, "case_ids_by_query", MappingProxyType(cases))

    @cached_property
    def by_key(self) -> Mapping[LogicalCellKey, DevelopmentPredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def probabilities(
        self,
        outer_target: str,
        query_center: str,
        action_id: str,
        training_seed: int,
        generation_seed: int,
    ) -> np.ndarray:
        try:
            return self.by_key[
                (
                    str(outer_target),
                    str(query_center),
                    str(action_id),
                    int(training_seed),
                    int(generation_seed),
                )
            ].probabilities
        except KeyError as exc:
            raise ProtocolError("Strict source-OOF probability cell is absent.") from exc

    def exact_nine_summary(
        self, outer_target: str, query_center: str, action_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.stack(
            [
                self.probabilities(outer_target, query_center, action_id, training, generation)
                for training in TRAINING_SEEDS
                for generation in GENERATION_SEEDS
            ]
        ).astype(np.float64, copy=False)
        mean = np.mean(values, axis=0, dtype=np.float64)
        variance = np.mean((values - mean) ** 2, axis=0, dtype=np.float64)
        positive = np.mean(values >= 0.5, axis=0, dtype=np.float64)
        return mean, np.sqrt(np.maximum(0.0, variance)), np.maximum(positive, 1.0 - positive)


@dataclass(frozen=True)
class DevelopmentSourcePredictionSeal:
    classifier_bank: DevelopmentClassifierBank
    source_store: DevelopmentPredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "source_prediction_seal_hash"
        }
        if (
            self.source_store.action_library_hash
            != self.classifier_bank.action_library_hash
            or self.source_store.development_classifier_bank_seal_hash
            != self.classifier_bank.seal_hash
            or self.source_store.frame_cache_binding_hash
            != self.classifier_bank.source_cache_binding_hash
            or payload.get("config_contract_hash")
            != self.classifier_bank.config_contract_hash
            or payload.get("source_prediction_seal_hash") != canonical_hash(unhashed)
            or payload.get("status") != DEVELOPMENT_PREDICTION_STATUS
            or payload.get("classifier_bank_seal_hash") != self.classifier_bank.seal_hash
            or payload.get("source_prediction_store_hash") != self.source_store.store_hash
            or payload.get("source_prediction_array_sha256")
            != sha256_file(self.arrays_path)
            or payload.get("source_prediction_index_sha256")
            != sha256_file(self.index_path)
            or payload.get("physical_fit_count") != DEVELOPMENT_CLASSIFIER_FIT_COUNT
            or payload.get("logical_source_prediction_cell_count")
            != DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
            or payload.get("source_labels_opened") is not False
            or payload.get("test_cache_admitted") is not False
            or payload.get("target_labels_available") is not False
        ):
            raise ProtocolError("Strict source-OOF prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["source_prediction_seal_hash"])

    @property
    def action_library_hash(self) -> str:
        return self.source_store.action_library_hash


@dataclass(frozen=True)
class CompositePrelabelPredictionSeal:
    """Durable gate requiring both development and target-compatible banks."""

    strict_source_predictions: DevelopmentSourcePredictionSeal
    target_classifier_bank: object
    seal_payload: Mapping[str, object]
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        target_seal_hash = getattr(self.target_classifier_bank, "seal_hash", None)
        target_source_binding = getattr(
            self.target_classifier_bank, "source_cache_binding_hash", None
        )
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "composite_prelabel_prediction_seal_hash"
        }
        if (
            payload.get("composite_prelabel_prediction_seal_hash")
            != canonical_hash(unhashed)
            or payload.get("status") != COMPOSITE_PRELABEL_STATUS
            or payload.get("strict_source_prediction_seal_hash")
            != self.strict_source_predictions.seal_hash
            or payload.get("strict_source_oof_classifier_bank_seal_hash")
            != self.strict_source_predictions.classifier_bank.seal_hash
            or payload.get("strict_source_oof_prediction_store_hash")
            != self.strict_source_predictions.source_store.store_hash
            or payload.get("target_classifier_bank_seal_hash") != target_seal_hash
            or target_source_binding
            != self.strict_source_predictions.source_store.frame_cache_binding_hash
            or payload.get("source_cache_binding_hash")
            != self.strict_source_predictions.source_store.frame_cache_binding_hash
            or payload.get("strict_source_physical_fit_count")
            != DEVELOPMENT_CLASSIFIER_FIT_COUNT
            or payload.get("strict_source_logical_prediction_cell_count")
            != DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
            or payload.get("target_classifier_fit_count")
            != EXPECTED_CLASSIFIER_FIT_COUNT
            or payload.get("total_physical_classifier_fit_count")
            != DEVELOPMENT_CLASSIFIER_FIT_COUNT + EXPECTED_CLASSIFIER_FIT_COUNT
            or payload.get("query_excluded_from_every_source_composition") is not True
            or payload.get("outer_target_excluded_from_every_source_composition")
            is not True
            or payload.get("unordered_excluded_pair_fit_reuse") is not True
            or payload.get("source_labels_opened") is not False
            or payload.get("test_cache_admitted") is not False
            or payload.get("target_labels_available") is not False
        ):
            raise ProtocolError("Composite prelabel prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["composite_prelabel_prediction_seal_hash"])

    @property
    def source_store(self) -> DevelopmentPredictionStore:
        return self.strict_source_predictions.source_store

    @property
    def classifier_bank(self) -> DevelopmentClassifierBank:
        """Compatibility alias for the strict source-OOF classifier bank."""

        return self.strict_source_predictions.classifier_bank


def development_prediction_store_hash(
    cells: Sequence[DevelopmentPredictionCell],
    *,
    rows_by_query: Mapping[str, Sequence[str]],
    case_ids_by_query: Mapping[str, Sequence[str]],
    frame_cache_binding_hash: str,
    action_library_hash: str,
    development_classifier_bank_seal_hash: str,
) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_strict_source_oof_prediction_store_v1",
            "frame_cache_binding_hash": frame_cache_binding_hash,
            "action_library_hash": action_library_hash,
            "development_classifier_bank_seal_hash": development_classifier_bank_seal_hash,
            "rows_by_query": {query: list(rows_by_query[query]) for query in CENTERS},
            "case_ids_by_query": {
                query: list(case_ids_by_query[query]) for query in CENTERS
            },
            "cells": [
                cell.index_payload(array_member=f"cell_{ordinal:05d}")
                for ordinal, cell in enumerate(cells)
            ],
        }
    )


__all__ = tuple(
    name
    for name in globals()
    if name.startswith("DEVELOPMENT_")
    or name.startswith("COMPOSITE_PRELABEL_")
    or name
    in {
        "CompositePrelabelPredictionSeal",
        "DevelopmentClassifierBank",
        "DevelopmentClassifierCell",
        "DevelopmentPredictionCell",
        "DevelopmentPredictionConfig",
        "DevelopmentPredictionStore",
        "DevelopmentSourcePredictionSeal",
        "LogicalCellKey",
        "PhysicalCellKey",
        "canonical_logical_action_hashes",
        "canonical_logical_cell_keys",
        "canonical_physical_action_hashes",
        "canonical_physical_cell_keys",
        "development_prediction_store_hash",
    }
)
