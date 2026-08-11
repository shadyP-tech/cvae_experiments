"""Typed label-free source/test frames and the post-model test-admission token."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_TEST_ROWS,
    FEATURE_DIM,
    OPAQUE_SOURCE_ID_NAMESPACE,
)
from .hashing import canonical_hash


def _sha256(value: object, role: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return text


def opaque_source_row_id(raw_sample_id: object, *, cache_sha256: object) -> str:
    """Project a historical outcome-bearing ID into a neutral identity."""

    sample_id = str(raw_sample_id)
    if not sample_id:
        raise ProtocolError("Source cache sample identity is empty.")
    cache_hash = _sha256(cache_sha256, "source cache hash")
    digest = canonical_hash(
        {
            "namespace": OPAQUE_SOURCE_ID_NAMESPACE,
            "source_cache_sha256": cache_hash,
            "historical_sample_identity": sample_id,
        }
    )
    return f"src_{digest}"


@dataclass(frozen=True, order=True)
class SourceRowIdentity:
    row_ordinal: int
    cache_row_index: int
    source_row_id: str
    case_id: str
    center: str
    split: str = "train"

    def __post_init__(self) -> None:
        if (
            type(self.row_ordinal) is not int
            or self.row_ordinal < 0
            or type(self.cache_row_index) is not int
            or self.cache_row_index < 0
            or self.center not in CENTERS
            or self.split != "train"
            or not self.case_id
            or not self.source_row_id.startswith("src_")
            or len(self.source_row_id) != 68
            or any(character not in "0123456789abcdef" for character in self.source_row_id[4:])
        ):
            raise ProtocolError("Prediction-only source-row identity drifted.")

    @property
    def sample_id(self) -> str:
        return self.source_row_id

    def to_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "cache_row_index": self.cache_row_index,
            "source_row_id": self.source_row_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
        }


@dataclass(frozen=True, order=True)
class TestRowIdentity:
    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    case_id: str
    center: str
    split: str = "test"

    def __post_init__(self) -> None:
        if (
            type(self.row_ordinal) is not int
            or self.row_ordinal < 0
            or type(self.manifest_row_index) is not int
            or self.manifest_row_index < 0
            or self.center not in CENTERS
            or self.split != "test"
            or not self.case_id
            or not self.evaluation_row_id.startswith("eval_")
            or len(self.evaluation_row_id) != 69
            or any(character not in "0123456789abcdef" for character in self.evaluation_row_id[5:])
        ):
            raise ProtocolError("Prediction-only test-row identity drifted.")

    @property
    def sample_id(self) -> str:
        return self.evaluation_row_id

    def to_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "evaluation_row_id": self.evaluation_row_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
        }


def row_identity_hash(rows: Sequence[SourceRowIdentity | TestRowIdentity]) -> str:
    return canonical_hash([row.to_payload() for row in rows])


@dataclass(frozen=True)
class LabelFreeSourceFrame:
    embeddings: np.ndarray
    rows: tuple[SourceRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[SourceRowIdentity, ...]]
    cache_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {str(key): tuple(value) for key, value in self.rows_by_center.items()}
        if (
            values.shape != (len(rows), FEATURE_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or len(rows) != EXPECTED_SOURCE_ROWS
            or len({row.source_row_id for row in rows}) != len(rows)
            or len({row.cache_row_index for row in rows}) != len(rows)
            or any(hasattr(row, "label") for row in rows)
        ):
            raise ProtocolError("Prediction-only label-free source frame is malformed.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", MappingProxyType(dict(self.cache_binding)))

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[SourceRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].source_row_id for index in ordinals)
            != tuple(row.source_row_id for row in rows)
        ):
            raise ProtocolError("Prediction-only source-row slice drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


@dataclass(frozen=True)
class TestInferenceAdmission:
    source_prediction_seal_hash: str
    action_classifier_bank_seal_hash: str
    regret_model_bank_seal_hash: str
    regret_model_bank_status: str
    target_labels_available: bool
    test_scoring_permitted: bool
    admission_hash: str

    def __post_init__(self) -> None:
        source = _sha256(self.source_prediction_seal_hash, "source prediction seal")
        classifiers = _sha256(
            self.action_classifier_bank_seal_hash, "action classifier bank seal"
        )
        models = _sha256(self.regret_model_bank_seal_hash, "regret model bank seal")
        unhashed = {
            "schema_version": "midogpp_prediction_only_test_inference_admission_v1",
            "source_prediction_seal_hash": source,
            "action_classifier_bank_seal_hash": classifiers,
            "regret_model_bank_seal_hash": models,
            "regret_model_bank_status": self.regret_model_bank_status,
            "target_labels_available": self.target_labels_available,
            "test_scoring_permitted": self.test_scoring_permitted,
            "classifier_refit_permitted": False,
        }
        if (
            self.regret_model_bank_status != "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION"
            or self.target_labels_available is not False
            or self.test_scoring_permitted is not False
            or self.admission_hash != canonical_hash(unhashed)
        ):
            raise ProtocolError("Prediction-only test admission is invalid.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_prediction_only_test_inference_admission_v1",
            "source_prediction_seal_hash": self.source_prediction_seal_hash,
            "action_classifier_bank_seal_hash": self.action_classifier_bank_seal_hash,
            "regret_model_bank_seal_hash": self.regret_model_bank_seal_hash,
            "regret_model_bank_status": self.regret_model_bank_status,
            "target_labels_available": False,
            "test_scoring_permitted": False,
            "classifier_refit_permitted": False,
            "admission_hash": self.admission_hash,
        }


@dataclass(frozen=True)
class LabelFreeTestFrame:
    embeddings: np.ndarray
    rows: tuple[TestRowIdentity, ...]
    rows_by_center: Mapping[str, tuple[TestRowIdentity, ...]]
    cache_binding: Mapping[str, object]
    admission: TestInferenceAdmission

    def __post_init__(self) -> None:
        values = np.asarray(self.embeddings)
        rows = tuple(self.rows)
        by_center = {str(key): tuple(value) for key, value in self.rows_by_center.items()}
        if (
            values.shape != (len(rows), FEATURE_DIM)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or tuple(by_center) != CENTERS
            or tuple(row for center in CENTERS for row in by_center[center]) != rows
            or len(rows) != EXPECTED_TEST_ROWS
            or len({row.evaluation_row_id for row in rows}) != len(rows)
            or len({row.manifest_row_index for row in rows}) != len(rows)
            or any(hasattr(row, "label") for row in rows)
        ):
            raise ProtocolError("Prediction-only label-free test frame is malformed.")
        frozen = np.ascontiguousarray(values, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "embeddings", frozen)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "rows_by_center", MappingProxyType(by_center))
        object.__setattr__(self, "cache_binding", MappingProxyType(dict(self.cache_binding)))

    @property
    def cache_binding_hash(self) -> str:
        return canonical_hash(dict(self.cache_binding))

    def embeddings_for(self, rows: Sequence[TestRowIdentity]) -> np.ndarray:
        ordinals = np.asarray([row.row_ordinal for row in rows], dtype=np.int64)
        if (
            not len(ordinals)
            or np.any(ordinals < 0)
            or np.any(ordinals >= len(self.rows))
            or tuple(self.rows[int(index)].evaluation_row_id for index in ordinals)
            != tuple(row.evaluation_row_id for row in rows)
        ):
            raise ProtocolError("Prediction-only test-row slice drifted.")
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


__all__ = (
    "LabelFreeSourceFrame",
    "LabelFreeTestFrame",
    "SourceRowIdentity",
    "TestInferenceAdmission",
    "TestRowIdentity",
    "opaque_source_row_id",
    "row_identity_hash",
)
