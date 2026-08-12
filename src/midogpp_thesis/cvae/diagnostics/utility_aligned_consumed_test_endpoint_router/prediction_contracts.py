"""Immutable prediction plans, cells, and exact-nine store contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from ...routing.utility_aligned.ensemble_endpoint_contracts import SeedProbabilityVector
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    SEED_PAIRS,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_development_action_ids,
    h_x_e_action_id,
)


DEVELOPMENT_ROLE = "development"
TARGET_ROLE = "target"
DEVELOPMENT_TASK_COUNT = 648
DEVELOPMENT_CELL_COUNT = 5_184
TARGET_TASK_COUNT = 81
TARGET_CELL_COUNT = 810

DEVELOPMENT_ARRAY_MEMBER = "arrays/development_probabilities.npz"
DEVELOPMENT_INDEX_MEMBER = "manifests/development_prediction_index.json"
DEVELOPMENT_SEAL_MEMBER = "manifests/development_prediction_seal.json"
TARGET_ARRAY_MEMBER = "arrays/target_action_probabilities.npz"
TARGET_INDEX_MEMBER = "manifests/target_prediction_index.json"
TARGET_SEAL_MEMBER = "manifests/target_prediction_seal.json"

PredictionCellKey = tuple[str, str, str, int, int]


def physical_target_action_ids(target: object) -> tuple[str, ...]:
    rendered = str(target)
    if rendered not in CENTERS:
        raise ProtocolError("Target physical action has an invalid center.")
    return (
        BASE_ACTION_ID,
        UNIFORM_ACTION_ID,
        *(h_x_e_action_id(source) for source in candidate_sources(rendered)),
    )


@dataclass(frozen=True)
class PlannedPhysicalAction:
    phase: str
    outer_target: str
    query_center: str
    action_id: str
    sources: tuple[str, ...]
    rows_per_class_by_source: Mapping[str, int]
    action_hash: str

    def __post_init__(self) -> None:
        sources = tuple(map(str, self.sources))
        counts = {str(key): int(value) for key, value in self.rows_per_class_by_source.items()}
        if (
            self.phase not in {DEVELOPMENT_ROLE, TARGET_ROLE}
            or self.outer_target not in CENTERS
            or self.query_center not in CENTERS
            or (self.phase == DEVELOPMENT_ROLE and self.outer_target == self.query_center)
            or tuple(counts) != sources
            or len(sources) != len(set(sources))
            or any(source not in CENTERS for source in sources)
            or any(value <= 0 or value > 270 for value in counts.values())
        ):
            raise ProtocolError("Endpoint-router planned action geometry drifted.")
        expected_ids = (
            expected_development_action_ids(self.outer_target, self.query_center)
            if self.phase == DEVELOPMENT_ROLE
            else physical_target_action_ids(self.outer_target)
        )
        if self.action_id not in expected_ids:
            raise ProtocolError("Endpoint-router planned action identity drifted.")
        unhashed = self.unhashed_payload(sources=sources, counts=counts)
        if self.action_hash != canonical_sha256(unhashed):
            raise ProtocolError("Endpoint-router planned action hash drifted.")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "rows_per_class_by_source", MappingProxyType(counts))

    def unhashed_payload(
        self,
        *,
        sources: Sequence[str] | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> dict[str, object]:
        source_values = tuple(self.sources if sources is None else sources)
        count_values = dict(self.rows_per_class_by_source if counts is None else counts)
        return {
            "schema_version": "midogpp_endpoint_router_physical_action_v1",
            "phase": self.phase,
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_id": self.action_id,
            "sources": list(source_values),
            "rows_per_class_by_source": count_values,
            "labels_used": False,
            "source_prefix_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.unhashed_payload(), "action_hash": self.action_hash}

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Cross a spawned worker boundary without serializing mappingproxy."""

        return (
            type(self),
            (
                self.phase,
                self.outer_target,
                self.query_center,
                self.action_id,
                self.sources,
                dict(self.rows_per_class_by_source),
                self.action_hash,
            ),
        )


@dataclass(frozen=True)
class PredictionTask:
    phase: str
    task_ordinal: int
    outer_target: str
    query_center: str
    training_seed: int
    generation_seed: int
    actions: tuple[PlannedPhysicalAction, ...]
    source_array_path: str
    target_array_path: str
    target_array_sha256: str
    support_row_ordinals: tuple[int, ...]
    evaluation_row_ordinals: tuple[int, ...]
    support_row_ids: tuple[str, ...]
    evaluation_row_ids: tuple[str, ...]
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    support_row_identity_hash: str
    evaluation_row_identity_hash: str
    config_contract_hash: str
    source_stream_lock_hash: str
    partition_lock_hash: str
    cache_binding_hash: str
    classifier_payload: Mapping[str, object]
    checkpoint_npz_path: str
    checkpoint_json_path: str
    task_hash: str

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        classifier = MappingProxyType(dict(self.classifier_payload))
        expected_actions = (
            expected_development_action_ids(self.outer_target, self.query_center)
            if self.phase == DEVELOPMENT_ROLE
            else physical_target_action_ids(self.outer_target)
        )
        if (
            self.phase not in {DEVELOPMENT_ROLE, TARGET_ROLE}
            or self.task_ordinal < 0
            or self.outer_target not in CENTERS
            or self.query_center not in CENTERS
            or (self.phase == DEVELOPMENT_ROLE and self.outer_target == self.query_center)
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or tuple(action.action_id for action in actions) != expected_actions
            or any(
                action.phase != self.phase
                or action.outer_target != self.outer_target
                or action.query_center != self.query_center
                for action in actions
            )
            or not self.support_row_ordinals
            or not self.evaluation_row_ordinals
            or len(self.support_row_ordinals) != len(self.support_row_ids)
            or len(self.support_row_ids) != len(self.support_case_ids)
            or len(self.evaluation_row_ordinals) != len(self.evaluation_row_ids)
            or len(self.evaluation_row_ids) != len(self.evaluation_case_ids)
            or set(self.support_row_ordinals).intersection(self.evaluation_row_ordinals)
        ):
            raise ProtocolError("Endpoint-router prediction task drifted.")
        if self.task_hash != canonical_sha256(self.unhashed_payload(classifier=classifier)):
            raise ProtocolError("Endpoint-router prediction task hash drifted.")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "classifier_payload", classifier)

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.outer_target,
            self.query_center,
            self.training_seed,
            self.generation_seed,
        )

    def unhashed_payload(
        self, *, classifier: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        return {
            "schema_version": "midogpp_endpoint_router_prediction_task_v1",
            "phase": self.phase,
            "task_ordinal": self.task_ordinal,
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "actions": [action.to_payload() for action in self.actions],
            "source_array_path": self.source_array_path,
            "target_array_path": self.target_array_path,
            "target_array_sha256": self.target_array_sha256,
            "support_row_ordinals": list(self.support_row_ordinals),
            "evaluation_row_ordinals": list(self.evaluation_row_ordinals),
            "support_row_ids": list(self.support_row_ids),
            "evaluation_row_ids": list(self.evaluation_row_ids),
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "support_row_identity_hash": self.support_row_identity_hash,
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "config_contract_hash": self.config_contract_hash,
            "source_stream_lock_hash": self.source_stream_lock_hash,
            "partition_lock_hash": self.partition_lock_hash,
            "cache_binding_hash": self.cache_binding_hash,
            "classifier_payload": dict(
                self.classifier_payload if classifier is None else classifier
            ),
            "checkpoint_npz_path": self.checkpoint_npz_path,
            "checkpoint_json_path": self.checkpoint_json_path,
            "labels_available": False,
        }

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize the complete validated task graph using plain mappings."""

        return (
            type(self),
            (
                self.phase,
                self.task_ordinal,
                self.outer_target,
                self.query_center,
                self.training_seed,
                self.generation_seed,
                self.actions,
                self.source_array_path,
                self.target_array_path,
                self.target_array_sha256,
                self.support_row_ordinals,
                self.evaluation_row_ordinals,
                self.support_row_ids,
                self.evaluation_row_ids,
                self.support_case_ids,
                self.evaluation_case_ids,
                self.support_row_identity_hash,
                self.evaluation_row_identity_hash,
                self.config_contract_hash,
                self.source_stream_lock_hash,
                self.partition_lock_hash,
                self.cache_binding_hash,
                dict(self.classifier_payload),
                self.checkpoint_npz_path,
                self.checkpoint_json_path,
                self.task_hash,
            ),
        )


@dataclass(frozen=True)
class PredictionCell:
    phase: str
    outer_target: str
    query_center: str
    action_id: str
    action_hash: str
    training_seed: int
    generation_seed: int
    support_row_identity_hash: str
    evaluation_row_identity_hash: str
    support_probabilities: np.ndarray
    evaluation_probabilities: np.ndarray
    composition_hash: str
    scaler_state_hash: str
    fit_provenance_hash: str
    support_probability_sha256: str = field(init=False)
    evaluation_probability_sha256: str = field(init=False)
    cell_hash: str = field(init=False)

    def __post_init__(self) -> None:
        support = np.ascontiguousarray(self.support_probabilities, dtype=np.float32)
        evaluation = np.ascontiguousarray(self.evaluation_probabilities, dtype=np.float32)
        if (
            self.phase not in {DEVELOPMENT_ROLE, TARGET_ROLE}
            or self.outer_target not in CENTERS
            or self.query_center not in CENTERS
            or self.training_seed not in TRAINING_SEEDS
            or self.generation_seed not in GENERATION_SEEDS
            or support.ndim != 1
            or evaluation.ndim != 1
            or not len(support)
            or not len(evaluation)
            or not np.isfinite(support).all()
            or not np.isfinite(evaluation).all()
            or np.any((support < 0.0) | (support > 1.0))
            or np.any((evaluation < 0.0) | (evaluation > 1.0))
        ):
            raise ProtocolError("Endpoint-router prediction cell drifted.")
        support_hash = array_sha256(support)
        evaluation_hash = array_sha256(evaluation)
        unhashed = self._unhashed_payload(support_hash, evaluation_hash)
        support.setflags(write=False)
        evaluation.setflags(write=False)
        object.__setattr__(self, "support_probabilities", support)
        object.__setattr__(self, "evaluation_probabilities", evaluation)
        object.__setattr__(self, "support_probability_sha256", support_hash)
        object.__setattr__(self, "evaluation_probability_sha256", evaluation_hash)
        object.__setattr__(self, "cell_hash", canonical_sha256(unhashed))

    @property
    def key(self) -> PredictionCellKey:
        return (
            self.outer_target,
            self.query_center,
            self.action_id,
            self.training_seed,
            self.generation_seed,
        )

    def _unhashed_payload(
        self, support_hash: str, evaluation_hash: str
    ) -> dict[str, object]:
        return {
            "schema_version": "midogpp_endpoint_router_prediction_cell_v1",
            "phase": self.phase,
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_id": self.action_id,
            "action_hash": self.action_hash,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "support_row_identity_hash": self.support_row_identity_hash,
            "evaluation_row_identity_hash": self.evaluation_row_identity_hash,
            "support_row_count": len(self.support_probabilities),
            "evaluation_row_count": len(self.evaluation_probabilities),
            "support_probability_sha256": support_hash,
            "evaluation_probability_sha256": evaluation_hash,
            "composition_hash": self.composition_hash,
            "scaler_state_hash": self.scaler_state_hash,
            "fit_provenance_hash": self.fit_provenance_hash,
            "labels_stored": False,
        }

    def index_payload(self) -> dict[str, object]:
        return {
            **self._unhashed_payload(
                self.support_probability_sha256,
                self.evaluation_probability_sha256,
            ),
            "cell_hash": self.cell_hash,
        }


@dataclass(frozen=True)
class PredictionStore:
    phase: str
    cells: tuple[PredictionCell, ...]
    support_row_ids_by_scope: Mapping[str, tuple[str, ...]]
    evaluation_row_ids_by_scope: Mapping[str, tuple[str, ...]]
    support_case_ids_by_scope: Mapping[str, tuple[str, ...]]
    evaluation_case_ids_by_scope: Mapping[str, tuple[str, ...]]
    source_stream_lock_hash: str
    partition_lock_hash: str
    cache_binding_hash: str
    action_library_hash: str
    store_hash: str

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        mappings = [
            {
                str(key): tuple(str(value) for value in values)
                for key, values in mapping.items()
            }
            for mapping in (
                self.support_row_ids_by_scope,
                self.evaluation_row_ids_by_scope,
                self.support_case_ids_by_scope,
                self.evaluation_case_ids_by_scope,
            )
        ]
        expected = canonical_cell_keys(self.phase)
        expected_scopes = canonical_scopes(self.phase)
        if (
            tuple(cell.key for cell in cells) != expected
            or len({cell.key for cell in cells}) != len(expected)
            or any(tuple(mapping) != expected_scopes for mapping in mappings)
            or any(cell.phase != self.phase for cell in cells)
            or self.store_hash
            != prediction_store_hash(
                self.phase,
                cells,
                support_row_ids_by_scope=mappings[0],
                evaluation_row_ids_by_scope=mappings[1],
                support_case_ids_by_scope=mappings[2],
                evaluation_case_ids_by_scope=mappings[3],
                source_stream_lock_hash=self.source_stream_lock_hash,
                partition_lock_hash=self.partition_lock_hash,
                cache_binding_hash=self.cache_binding_hash,
                action_library_hash=self.action_library_hash,
            )
        ):
            raise ProtocolError("Endpoint-router prediction store topology drifted.")
        for cell in cells:
            scope = cell_scope(self.phase, cell.outer_target, cell.query_center)
            if (
                len(cell.support_probabilities) != len(mappings[0][scope])
                or len(cell.evaluation_probabilities) != len(mappings[1][scope])
                or len(mappings[0][scope]) != len(mappings[2][scope])
                or len(mappings[1][scope]) != len(mappings[3][scope])
            ):
                raise ProtocolError("Endpoint-router prediction row binding drifted.")
        object.__setattr__(self, "cells", cells)
        for name, value in zip(
            (
                "support_row_ids_by_scope",
                "evaluation_row_ids_by_scope",
                "support_case_ids_by_scope",
                "evaluation_case_ids_by_scope",
            ),
            mappings,
            strict=True,
        ):
            object.__setattr__(self, name, MappingProxyType(value))

    @cached_property
    def by_key(self) -> Mapping[PredictionCellKey, PredictionCell]:
        return MappingProxyType({cell.key: cell for cell in self.cells})

    def vectors(
        self,
        *,
        outer_target: str,
        query_center: str,
        action_id: str,
        role: str,
    ) -> tuple[SeedProbabilityVector, ...]:
        if role not in {"support", "evaluation"}:
            raise ProtocolError("Endpoint-router probability role is invalid.")
        vectors: list[SeedProbabilityVector] = []
        for training_seed, generation_seed in SEED_PAIRS:
            try:
                cell = self.by_key[
                    (
                        str(outer_target),
                        str(query_center),
                        str(action_id),
                        training_seed,
                        generation_seed,
                    )
                ]
            except KeyError as exc:
                raise ProtocolError("Endpoint-router prediction cell is absent.") from exc
            probabilities = (
                cell.support_probabilities
                if role == "support"
                else cell.evaluation_probabilities
            )
            row_hash = (
                cell.support_row_identity_hash
                if role == "support"
                else cell.evaluation_row_identity_hash
            )
            vectors.append(
                SeedProbabilityVector(
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    row_identity_hash=row_hash,
                    prediction_provenance_hash=cell.cell_hash,
                    positive_class_probabilities=probabilities,
                )
            )
        return tuple(vectors)

    def exact_nine_mean(
        self, *, outer_target: str, query_center: str, action_id: str, role: str
    ) -> np.ndarray:
        values = np.stack(
            [vector.positive_class_probabilities for vector in self.vectors(
                outer_target=outer_target,
                query_center=query_center,
                action_id=action_id,
                role=role,
            )]
        )
        return np.mean(values, axis=0, dtype=np.float64)


def cell_scope(phase: str, outer_target: str, query_center: str) -> str:
    return f"{outer_target}::{query_center}" if phase == DEVELOPMENT_ROLE else outer_target


def canonical_scopes(phase: str) -> tuple[str, ...]:
    if phase == DEVELOPMENT_ROLE:
        return tuple(f"{outer}::{query}" for outer in CENTERS for query in CENTERS if query != outer)
    if phase == TARGET_ROLE:
        return CENTERS
    raise ProtocolError("Endpoint-router prediction-store phase is invalid.")


def canonical_cell_keys(phase: str) -> tuple[PredictionCellKey, ...]:
    if phase == DEVELOPMENT_ROLE:
        return tuple(
            (outer, query, action, training, generation)
            for outer in CENTERS
            for query in CENTERS
            if query != outer
            for training, generation in SEED_PAIRS
            for action in expected_development_action_ids(outer, query)
        )
    if phase == TARGET_ROLE:
        return tuple(
            (target, target, action, training, generation)
            for target in CENTERS
            for training, generation in SEED_PAIRS
            for action in physical_target_action_ids(target)
        )
    raise ProtocolError("Endpoint-router prediction phase is invalid.")


def prediction_store_hash(
    phase: str,
    cells: Sequence[PredictionCell],
    *,
    support_row_ids_by_scope: Mapping[str, Sequence[str]],
    evaluation_row_ids_by_scope: Mapping[str, Sequence[str]],
    support_case_ids_by_scope: Mapping[str, Sequence[str]],
    evaluation_case_ids_by_scope: Mapping[str, Sequence[str]],
    source_stream_lock_hash: str,
    partition_lock_hash: str,
    cache_binding_hash: str,
    action_library_hash: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "midogpp_endpoint_router_prediction_store_v1",
            "phase": phase,
            "cells": [cell.index_payload() for cell in cells],
            "support_row_ids_by_scope": {
                key: list(value) for key, value in support_row_ids_by_scope.items()
            },
            "evaluation_row_ids_by_scope": {
                key: list(value) for key, value in evaluation_row_ids_by_scope.items()
            },
            "support_case_ids_by_scope": {
                key: list(value) for key, value in support_case_ids_by_scope.items()
            },
            "evaluation_case_ids_by_scope": {
                key: list(value) for key, value in evaluation_case_ids_by_scope.items()
            },
            "source_stream_lock_hash": source_stream_lock_hash,
            "partition_lock_hash": partition_lock_hash,
            "cache_binding_hash": cache_binding_hash,
            "action_library_hash": action_library_hash,
            "labels_stored": False,
            "storage_dtype": "float32",
            "reductions_dtype": "float64",
        }
    )


__all__ = (
    "DEVELOPMENT_ARRAY_MEMBER",
    "DEVELOPMENT_CELL_COUNT",
    "DEVELOPMENT_INDEX_MEMBER",
    "DEVELOPMENT_ROLE",
    "DEVELOPMENT_SEAL_MEMBER",
    "DEVELOPMENT_TASK_COUNT",
    "PlannedPhysicalAction",
    "PredictionCell",
    "PredictionCellKey",
    "PredictionStore",
    "PredictionTask",
    "TARGET_ARRAY_MEMBER",
    "TARGET_CELL_COUNT",
    "TARGET_INDEX_MEMBER",
    "TARGET_ROLE",
    "TARGET_SEAL_MEMBER",
    "TARGET_TASK_COUNT",
    "canonical_cell_keys",
    "canonical_scopes",
    "cell_scope",
    "physical_target_action_ids",
    "prediction_store_hash",
)
