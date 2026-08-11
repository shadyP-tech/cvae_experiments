"""Immutable classifier-bank and source/test probability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array, sha256_file
from .actions import action_library_payload, actions_for_target
from .constants import (
    CENTERS,
    CLASSIFIER_COEFFICIENT_MEMBER,
    CLASSIFIER_INDEX_MEMBER,
    CLASSIFIER_INTERCEPT_MEMBER,
    CLASSIFIER_MEAN_MEMBER,
    CLASSIFIER_SCALE_MEMBER,
    CLASSIFIER_SEAL_MEMBER,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_ROWS_BY_CENTER,
    EXPECTED_TASK_COUNT,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FEATURE_DIM,
    GENERATION_SEEDS,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_SEAL_MEMBER,
    TEST_ARRAY_MEMBER,
    TEST_INDEX_MEMBER,
    TEST_SEAL_MEMBER,
    TRAINING_SEEDS,
)
from .hashing import canonical_hash
from .input_contracts import TestInferenceAdmission


PredictionCellKey = tuple[str, str, int, int]


class ActionPredictionConfig(Protocol):
    contract_hash: str
    classifier: object
    runtime: Mapping[str, object]


def _sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return text


@lru_cache(maxsize=1)
def canonical_cell_keys() -> tuple[PredictionCellKey, ...]:
    return tuple(
        (target, action.action_id, training, generation)
        for target in CENTERS
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in actions_for_target(target)
    )


@lru_cache(maxsize=1)
def canonical_action_hashes() -> Mapping[PredictionCellKey, str]:
    result = {
        (target, action.action_id, training, generation): action.action_hash
        for target in CENTERS
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
        for action in actions_for_target(target)
    }
    if tuple(result) != canonical_cell_keys():
        raise AssertionError("Prediction-only action hash order drifted.")
    return MappingProxyType(result)


@dataclass(frozen=True)
class ClassifierBankCell:
    cell_ordinal: int
    target_center: str
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
        if (
            type(self.cell_ordinal) is not int
            or self.cell_ordinal < 0
            or self.key not in canonical_action_hashes()
            or canonical_action_hashes()[self.key] != self.action_hash
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
            raise ProtocolError("Prediction-only classifier cell drifted.")

    @property
    def key(self) -> PredictionCellKey:
        return (
            self.target_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "cell_ordinal": self.cell_ordinal,
            "target_center": self.target_center,
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


def classifier_parameter_sha256(
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
) -> str:
    return canonical_hash(
        {
            "mean_sha256": sha256_array(np.asarray(mean, dtype=np.float64)),
            "scale_sha256": sha256_array(np.asarray(scale, dtype=np.float64)),
            "coefficient_sha256": sha256_array(
                np.asarray(coefficient, dtype=np.float64)
            ),
            "intercept": float(intercept),
        }
    )


@dataclass(frozen=True)
class ActionClassifierBank:
    root: Path
    cells: tuple[ClassifierBankCell, ...]
    source_stream_lock_hash: str
    action_library_hash: str
    source_cache_binding_hash: str
    config_contract_hash: str
    bank_hash: str
    seal_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if (
            len(cells) != EXPECTED_CLASSIFIER_FIT_COUNT
            or tuple(cell.key for cell in cells) != canonical_cell_keys()
            or tuple(cell.cell_ordinal for cell in cells)
            != tuple(range(EXPECTED_CLASSIFIER_FIT_COUNT))
            or len({cell.fit_provenance_hash for cell in cells}) != len(cells)
        ):
            raise ProtocolError("Prediction-only classifier-bank topology drifted.")
        for value, role in (
            (self.action_library_hash, "action library hash"),
            (self.source_cache_binding_hash, "source cache binding hash"),
            (self.bank_hash, "classifier bank hash"),
        ):
            _sha256(value, role)
        paths = self.parameter_paths
        arrays = self._arrays()
        mean, scale, coefficient, intercept = arrays
        expected_matrix = (EXPECTED_CLASSIFIER_FIT_COUNT, FEATURE_DIM)
        if (
            mean.shape != expected_matrix
            or scale.shape != expected_matrix
            or coefficient.shape != expected_matrix
            or intercept.shape != (EXPECTED_CLASSIFIER_FIT_COUNT,)
            or any(array.dtype != np.float64 for array in arrays)
            or not all(np.isfinite(array).all() for array in arrays)
            or np.any(scale <= 0.0)
        ):
            raise ProtocolError("Prediction-only classifier parameter arrays drifted.")
        for cell in cells:
            ordinal = cell.cell_ordinal
            if cell.parameter_sha256 != classifier_parameter_sha256(
                mean[ordinal], scale[ordinal], coefficient[ordinal], intercept[ordinal]
            ):
                raise ProtocolError("Prediction-only classifier parameter row drifted.")
        payload = dict(self.seal_payload)
        unhashed = {key: value for key, value in payload.items() if key != "classifier_bank_seal_hash"}
        if (
            payload.get("classifier_bank_seal_hash") != canonical_hash(unhashed)
            or payload.get("status") != "SEALED_1458_SOURCE_ONLY_ACTION_CLASSIFIERS"
            or payload.get("classifier_bank_hash") != self.bank_hash
            or payload.get("fit_count") != EXPECTED_CLASSIFIER_FIT_COUNT
            or payload.get("test_cache_admitted") is not False
            or payload.get("target_labels_available") is not False
            or payload.get("classifier_refit_required_for_test") is not False
            or any(
                payload.get(name) != sha256_file(path)
                for name, path in (
                    ("scaler_mean_file_sha256", paths[0]),
                    ("scaler_scale_file_sha256", paths[1]),
                    ("coefficient_file_sha256", paths[2]),
                    ("intercept_file_sha256", paths[3]),
                )
            )
        ):
            raise ProtocolError("Prediction-only classifier-bank seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def parameter_paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.root / CLASSIFIER_MEAN_MEMBER,
            self.root / CLASSIFIER_SCALE_MEMBER,
            self.root / CLASSIFIER_COEFFICIENT_MEMBER,
            self.root / CLASSIFIER_INTERCEPT_MEMBER,
        )

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.load(path, mmap_mode="r", allow_pickle=False)
            for path in self.parameter_paths
        )  # type: ignore[return-value]

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["classifier_bank_seal_hash"])

    @cached_property
    def by_key(self) -> Mapping[PredictionCellKey, ClassifierBankCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def parameters(
        self, key: PredictionCellKey
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        try:
            ordinal = self.by_key[key].cell_ordinal
        except KeyError as exc:
            raise ProtocolError("Prediction-only classifier cell is absent.") from exc
        mean, scale, coefficient, intercept = self._arrays()
        return (
            np.asarray(mean[ordinal], dtype=np.float64),
            np.asarray(scale[ordinal], dtype=np.float64),
            np.asarray(coefficient[ordinal], dtype=np.float64),
            float(intercept[ordinal]),
        )


@dataclass(frozen=True)
class PredictionCell:
    frame_role: str
    target_center: str
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    row_identity_hash: str
    probabilities: np.ndarray
    probability_sha256: str
    predictions_sha256: str
    classifier_parameter_sha256: str

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        if (
            self.frame_role not in ("source", "test")
            or self.key not in canonical_action_hashes()
            or canonical_action_hashes()[self.key] != self.action_hash
            or values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
            or sha256_array(values) != self.probability_sha256
            or sha256_array((values >= np.float32(0.5)).astype(np.uint8))
            != self.predictions_sha256
        ):
            raise ProtocolError("Prediction-only probability cell drifted.")
        for value, role in (
            (self.row_identity_hash, "row identity hash"),
            (self.classifier_parameter_sha256, "classifier parameter hash"),
        ):
            _sha256(value, role)
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
            "frame_role": self.frame_role,
            "target_center": self.target_center,
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "row_identity_hash": self.row_identity_hash,
            "array_member": array_member,
            "probability_sha256": self.probability_sha256,
            "predictions_sha256": self.predictions_sha256,
            "classifier_parameter_sha256": self.classifier_parameter_sha256,
        }


@dataclass(frozen=True)
class ActionPredictionStore:
    frame_role: str
    cells: tuple[PredictionCell, ...]
    rows_by_outer_target: Mapping[str, tuple[str, ...]]
    case_ids_by_outer_target: Mapping[str, tuple[str, ...]]
    query_ids_by_outer_target: Mapping[str, tuple[str, ...]]
    frame_cache_binding_hash: str
    action_library_hash: str
    action_classifier_bank_seal_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        if self.frame_role not in ("source", "test"):
            raise ProtocolError("Prediction-only store frame role drifted.")
        rows = {
            str(target): tuple(str(value) for value in values)
            for target, values in self.rows_by_outer_target.items()
        }
        cases = {
            str(target): tuple(str(value) for value in values)
            for target, values in self.case_ids_by_outer_target.items()
        }
        queries = {
            str(target): tuple(str(value) for value in values)
            for target, values in self.query_ids_by_outer_target.items()
        }
        if (
            tuple(rows) != CENTERS
            or tuple(cases) != CENTERS
            or tuple(queries) != CENTERS
            or tuple(cell.key for cell in self.cells) != canonical_cell_keys()
            or any(cell.frame_role != self.frame_role for cell in self.cells)
            or any(
                not rows[target]
                or len(rows[target]) != len(cases[target])
                or len(rows[target]) != len(queries[target])
                for target in CENTERS
            )
        ):
            raise ProtocolError("Prediction-only store topology drifted.")
        for target in CENTERS:
            # Source query H is retained as an unlabeled feature-only surface.
            # The one-way label capability, not the probability store, excludes
            # H from every response/model fit.
            expected_count = (
                EXPECTED_SOURCE_ROWS
                if self.frame_role == "source"
                else EXPECTED_TEST_ROWS_BY_CENTER[target]
            )
            expected_queries = set(CENTERS) if self.frame_role == "source" else {target}
            if len(rows[target]) != expected_count or set(queries[target]) != expected_queries:
                raise ProtocolError("Prediction-only store query coverage drifted.")
        if any(
            len(cell.probabilities) != len(rows[cell.target_center])
            for cell in self.cells
        ):
            raise ProtocolError("Prediction-only cell row coverage drifted.")
        for value, role in (
            (self.frame_cache_binding_hash, "frame cache binding hash"),
            (self.action_library_hash, "action library hash"),
            (self.action_classifier_bank_seal_hash, "classifier bank seal hash"),
            (self.store_hash, "prediction store hash"),
        ):
            _sha256(value, role)
        expected_hash = prediction_store_hash(
            self.frame_role,
            self.cells,
            rows_by_outer_target=rows,
            case_ids_by_outer_target=cases,
            query_ids_by_outer_target=queries,
            frame_cache_binding_hash=self.frame_cache_binding_hash,
            action_library_hash=self.action_library_hash,
            action_classifier_bank_seal_hash=self.action_classifier_bank_seal_hash,
        )
        if self.store_hash != expected_hash:
            raise ProtocolError("Prediction-only store hash drifted.")
        object.__setattr__(self, "rows_by_outer_target", MappingProxyType(rows))
        object.__setattr__(self, "case_ids_by_outer_target", MappingProxyType(cases))
        object.__setattr__(self, "query_ids_by_outer_target", MappingProxyType(queries))

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
            raise ProtocolError("Prediction-only probability cell is absent.") from exc

    def exact_nine_summary(
        self, target_center: str, action_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.stack(
            [
                self.probabilities(target_center, action_id, training, generation)
                for training in TRAINING_SEEDS
                for generation in GENERATION_SEEDS
            ],
            axis=0,
        ).astype(np.float64, copy=False)
        mean = np.mean(values, axis=0, dtype=np.float64)
        variance = np.mean((values - mean) ** 2, axis=0, dtype=np.float64)
        positive_fraction = np.mean(values >= 0.5, axis=0, dtype=np.float64)
        winning_fraction = np.maximum(positive_fraction, 1.0 - positive_fraction)
        return mean, np.sqrt(np.maximum(0.0, variance)), winning_fraction


@dataclass(frozen=True)
class GlobalSourcePredictionSeal:
    classifier_bank: ActionClassifierBank
    source_store: ActionPredictionStore
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        unhashed = {key: value for key, value in payload.items() if key != "source_prediction_seal_hash"}
        if (
            self.source_store.frame_role != "source"
            or payload.get("source_prediction_seal_hash") != canonical_hash(unhashed)
            or payload.get("status")
            != "SEALED_1458_SOURCE_ACTION_FITS_AND_PREDICTIONS"
            or payload.get("classifier_bank_seal_hash") != self.classifier_bank.seal_hash
            or payload.get("source_prediction_store_hash") != self.source_store.store_hash
            or payload.get("fit_count") != EXPECTED_CLASSIFIER_FIT_COUNT
            or payload.get("source_prediction_cell_count")
            != EXPECTED_CLASSIFIER_FIT_COUNT
            or payload.get("source_labels_opened") is not False
            or payload.get("test_cache_admitted") is not False
            or payload.get("target_labels_available") is not False
        ):
            raise ProtocolError("Prediction-only source prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["source_prediction_seal_hash"])

    @property
    def action_library_hash(self) -> str:
        return self.source_store.action_library_hash


@dataclass(frozen=True)
class GlobalTestPredictionSeal:
    classifier_bank: ActionClassifierBank
    test_store: ActionPredictionStore
    admission: TestInferenceAdmission
    seal_payload: Mapping[str, object]
    arrays_path: Path
    index_path: Path
    seal_path: Path

    def __post_init__(self) -> None:
        payload = dict(self.seal_payload)
        unhashed = {key: value for key, value in payload.items() if key != "test_prediction_seal_hash"}
        if (
            self.test_store.frame_role != "test"
            or payload.get("test_prediction_seal_hash") != canonical_hash(unhashed)
            or payload.get("status") != "SEALED_WHOLE_TEST_LABEL_FREE_INFERENCE"
            or payload.get("classifier_bank_seal_hash") != self.classifier_bank.seal_hash
            or payload.get("test_prediction_store_hash") != self.test_store.store_hash
            or payload.get("test_inference_admission_hash") != self.admission.admission_hash
            or payload.get("source_prediction_seal_hash")
            != self.admission.source_prediction_seal_hash
            or payload.get("regret_model_bank_seal_hash")
            != self.admission.regret_model_bank_seal_hash
            or payload.get("fit_count_during_test_phase") != 0
            or payload.get("target_labels_available") is not False
            or payload.get("target_scoring_permitted") is not False
        ):
            raise ProtocolError("Prediction-only test prediction seal drifted.")
        object.__setattr__(self, "seal_payload", MappingProxyType(payload))

    @property
    def seal_hash(self) -> str:
        return str(self.seal_payload["test_prediction_seal_hash"])

    @property
    def action_library_hash(self) -> str:
        return self.test_store.action_library_hash


def prediction_store_hash(
    frame_role: str,
    cells: Sequence[PredictionCell],
    *,
    rows_by_outer_target: Mapping[str, Sequence[str]],
    case_ids_by_outer_target: Mapping[str, Sequence[str]],
    query_ids_by_outer_target: Mapping[str, Sequence[str]],
    frame_cache_binding_hash: str,
    action_library_hash: str,
    action_classifier_bank_seal_hash: str,
) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_prediction_only_probability_store_v1",
            "frame_role": frame_role,
            "frame_cache_binding_hash": frame_cache_binding_hash,
            "action_library_hash": action_library_hash,
            "action_classifier_bank_seal_hash": action_classifier_bank_seal_hash,
            "rows_by_outer_target": {
                target: list(rows_by_outer_target[target]) for target in CENTERS
            },
            "case_ids_by_outer_target": {
                target: list(case_ids_by_outer_target[target]) for target in CENTERS
            },
            "query_ids_by_outer_target": {
                target: list(query_ids_by_outer_target[target]) for target in CENTERS
            },
            "cells": [
                cell.index_payload(array_member=f"cell_{ordinal:04d}")
                for ordinal, cell in enumerate(cells)
            ],
        }
    )


def expected_action_library_hash() -> str:
    return str(action_library_payload()["action_library_hash"])


__all__ = (
    "ActionClassifierBank",
    "ActionPredictionConfig",
    "ActionPredictionStore",
    "ClassifierBankCell",
    "GlobalSourcePredictionSeal",
    "GlobalTestPredictionSeal",
    "PredictionCell",
    "PredictionCellKey",
    "canonical_action_hashes",
    "canonical_cell_keys",
    "classifier_parameter_sha256",
    "expected_action_library_hash",
    "prediction_store_hash",
)
