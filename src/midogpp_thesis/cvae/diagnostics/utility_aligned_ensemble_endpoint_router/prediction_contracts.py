"""Immutable combined support/evaluation probability-store contracts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.utility_aligned.ensemble_endpoint_contracts import SeedProbabilityVector
from .input_contracts import row_identity_hash


PredictionKey = tuple[str, str, int, int]


def array_sha256(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CombinedPredictionCell:
    scope_id: str
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    support_row_identity_hash: str
    evaluation_row_identity_hash: str
    support_predictions: np.ndarray
    support_probabilities: np.ndarray
    evaluation_predictions: np.ndarray
    evaluation_probabilities: np.ndarray
    composition_hash: str
    scaler_state_hash: str
    fit_provenance_hash: str
    aliased_from_action_id: str | None = None

    def __post_init__(self) -> None:
        if not self.scope_id or not self.action_id or not self.action_hash:
            raise ProtocolError("Combined prediction identity is incomplete.")
        for role in ("support", "evaluation"):
            prediction = np.ascontiguousarray(
                getattr(self, f"{role}_predictions"), dtype=np.uint8
            )
            probability = np.ascontiguousarray(
                getattr(self, f"{role}_probabilities"), dtype=np.float32
            )
            if (
                prediction.ndim != 1
                or prediction.dtype != np.uint8
                or probability.shape != prediction.shape
                or probability.dtype != np.float32
                or not len(prediction)
                or not np.isin(prediction, (0, 1)).all()
                or not np.isfinite(probability).all()
                or np.any(probability < 0.0)
                or np.any(probability > 1.0)
            ):
                raise ProtocolError(f"Combined {role} prediction vector drifted.")
            prediction.setflags(write=False)
            probability.setflags(write=False)
            object.__setattr__(self, f"{role}_predictions", prediction)
            object.__setattr__(self, f"{role}_probabilities", probability)
        if self.aliased_from_action_id == self.action_id:
            raise ProtocolError("A combined prediction cannot alias itself.")

    @property
    def key(self) -> PredictionKey:
        return self.scope_id, self.action_id, self.training_seed, self.generation_seed

    def probability_vector(self, role: str) -> SeedProbabilityVector:
        if role not in {"support", "evaluation"}:
            raise ProtocolError("Combined probability-vector role is invalid.")
        return SeedProbabilityVector(
            training_seed=self.training_seed,
            generation_seed=self.generation_seed,
            row_identity_hash=str(getattr(self, f"{role}_row_identity_hash")),
            prediction_provenance_hash=self.fit_provenance_hash,
            positive_class_probabilities=getattr(self, f"{role}_probabilities"),
        )

    def hash_payload(self) -> dict[str, object]:
        return {
            "key": list(self.key),
            "action_hash": self.action_hash,
            "support_row_identity_hash": self.support_row_identity_hash,
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "support_prediction_sha256": array_sha256(self.support_predictions),
            "support_probability_sha256": array_sha256(self.support_probabilities),
            "evaluation_prediction_sha256": array_sha256(self.evaluation_predictions),
            "evaluation_probability_sha256": array_sha256(self.evaluation_probabilities),
            "composition_hash": self.composition_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "fit_provenance_hash": self.fit_provenance_hash,
            "aliased_from_action_id": self.aliased_from_action_id,
        }


@dataclass(frozen=True)
class CombinedPredictionStore:
    role: str
    cells: tuple[CombinedPredictionCell, ...]
    source_cache_lock_hash: str
    partition_lock_hash: str
    action_library_hash: str
    expected_cell_count: int
    unique_classifier_fit_count: int
    store_hash: str

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if (
            self.role not in {"development", "target_probe", "target_final"}
            or len(cells) != self.expected_cell_count
            or len({cell.key for cell in cells}) != len(cells)
            or not 0 < self.unique_classifier_fit_count <= len(cells)
        ):
            raise ProtocolError("Combined prediction store coverage drifted.")
        unhashed = self._unhashed_payload(cells)
        if self.store_hash != stable_hash(unhashed):
            raise ProtocolError("Combined prediction store semantic hash drifted.")
        object.__setattr__(self, "cells", cells)

    @cached_property
    def by_key(self) -> Mapping[PredictionKey, CombinedPredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def vectors(self, scope_id: str, action_id: str, role: str) -> tuple[SeedProbabilityVector, ...]:
        cells = tuple(
            cell
            for cell in self.cells
            if cell.scope_id == str(scope_id) and cell.action_id == str(action_id)
        )
        if len(cells) != 9:
            raise ProtocolError("Exact-nine combined probability coverage drifted.")
        return tuple(cell.probability_vector(role) for cell in cells)

    def _unhashed_payload(
        self, cells: Sequence[CombinedPredictionCell] | None = None
    ) -> dict[str, object]:
        values = tuple(self.cells if cells is None else cells)
        return {
            "schema_version": "midogpp_stage90_ensemble_endpoint_combined_store_v1",
            "role": self.role,
            "source_cache_lock_hash": self.source_cache_lock_hash,
            "partition_lock_hash": self.partition_lock_hash,
            "action_library_hash": self.action_library_hash,
            "expected_cell_count": self.expected_cell_count,
            "unique_classifier_fit_count": self.unique_classifier_fit_count,
            "cell_hash_payloads": [cell.hash_payload() for cell in values],
            "support_and_evaluation_predicted_by_same_fit": True,
            "labels_available_during_fit_or_predict": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "store_hash": self.store_hash}


def build_store(
    *,
    role: str,
    cells: Sequence[CombinedPredictionCell],
    source_cache_lock_hash: str,
    partition_lock_hash: str,
    action_library_hash: str,
    expected_cell_count: int,
    unique_classifier_fit_count: int,
) -> CombinedPredictionStore:
    values = tuple(cells)
    prototype = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_combined_store_v1",
        "role": role,
        "source_cache_lock_hash": source_cache_lock_hash,
        "partition_lock_hash": partition_lock_hash,
        "action_library_hash": action_library_hash,
        "expected_cell_count": expected_cell_count,
        "unique_classifier_fit_count": unique_classifier_fit_count,
        "cell_hash_payloads": [cell.hash_payload() for cell in values],
        "support_and_evaluation_predicted_by_same_fit": True,
        "labels_available_during_fit_or_predict": False,
    }
    return CombinedPredictionStore(
        role=role,
        cells=values,
        source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=partition_lock_hash,
        action_library_hash=action_library_hash,
        expected_cell_count=expected_cell_count,
        unique_classifier_fit_count=unique_classifier_fit_count,
        store_hash=stable_hash(prototype),
    )


def assert_partition_rows(partitions: object) -> None:
    support = getattr(partitions, "support_rows_by_center", None)
    evaluation = getattr(partitions, "evaluation_rows_by_center", None)
    if not isinstance(support, Mapping) or not isinstance(evaluation, Mapping):
        raise ProtocolError("Combined prediction partitions are unavailable.")
    for center in support:
        support_cases = {row.case_id for row in support[center]}
        evaluation_cases = {row.case_id for row in evaluation[center]}
        if (
            len(support_cases) != 2
            or support_cases & evaluation_cases
            or {row.partition_role for row in support[center]} != {"support"}
            or {row.partition_role for row in evaluation[center]} != {"evaluation"}
            or not row_identity_hash(support[center])
            or not row_identity_hash(evaluation[center])
        ):
            raise ProtocolError("Combined support/evaluation partition firewall failed.")


__all__ = (
    "CombinedPredictionCell",
    "CombinedPredictionStore",
    "PredictionKey",
    "array_sha256",
    "assert_partition_rows",
    "build_store",
)
