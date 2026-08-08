"""Typed contracts for the consumed exact-tail development prediction pass."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import build_inner_exact_tail_actions
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    H_X_E_ACTION_PREFIX,
    INNER_BASE_PER_SOURCE_PER_CLASS,
    INNER_CANDIDATE_COUNT,
    INNER_SELECTED_SOURCE_CAPACITY_PER_CLASS,
    INNER_TOPUP_TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    h_x_e_action_id,
    h_x_e_source,
    inner_candidate_sources,
)


INNER_BASE_ACTION_ID = BASE_ACTION_ID
INNER_TAIL_ACTION_PREFIX = H_X_E_ACTION_PREFIX
EXPECTED_COARSE_TASK_COUNT = (
    len(CENTERS) * (len(CENTERS) - 1) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)
EXPECTED_PREDICTION_CELL_COUNT = EXPECTED_COARSE_TASK_COUNT * (
    1 + INNER_CANDIDATE_COUNT
)
EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT = (
    EXPECTED_COARSE_TASK_COUNT * INNER_CANDIDATE_COUNT
)
PREDICTION_WORKERS = 4
BLAS_THREADS_PER_WORKER = 3


def inner_tail_action_id(source_center: object) -> str:
    return h_x_e_action_id(source_center)


def inner_tail_source(action_id: object) -> str | None:
    return h_x_e_source(action_id)


@dataclass(frozen=True)
class ExactTailDevelopmentAction:
    outer_target: str
    query_center: str
    action_id: str
    selected_source: str | None
    source_order: tuple[str, ...]
    counts_per_class: Mapping[str, int]
    action_hash: str

    def __post_init__(self) -> None:
        expected = inner_candidate_sources(self.outer_target, self.query_center)
        counts = {str(key): int(value) for key, value in self.counts_per_class.items()}
        if self.source_order != expected or tuple(counts) != expected:
            raise ProtocolError("Stage-90 exact-tail action source order drifted.")
        if self.action_id == INNER_BASE_ACTION_ID:
            if self.selected_source is not None or set(counts.values()) != {
                INNER_BASE_PER_SOURCE_PER_CLASS
            }:
                raise ProtocolError("Stage-90 exact-tail base geometry drifted.")
        else:
            selected = inner_tail_source(self.action_id)
            if selected != self.selected_source or selected not in expected:
                raise ProtocolError("Stage-90 exact-tail selected source drifted.")
            for source, count in counts.items():
                expected_count = (
                    INNER_SELECTED_SOURCE_CAPACITY_PER_CLASS
                    if source == selected
                    else INNER_BASE_PER_SOURCE_PER_CLASS
                )
                if count != expected_count:
                    raise ProtocolError("Stage-90 exact additive-tail geometry drifted.")
        canonical_by_id = {
            action.action_id: action
            for action in build_inner_exact_tail_actions(
                self.outer_target, self.query_center
            )
        }
        canonical = canonical_by_id.get(self.action_id)
        if (
            canonical is None
            or canonical.selected_source != self.selected_source
            or canonical.source_order != self.source_order
            or dict(canonical.final_counts_by_class[0]) != counts
            or dict(canonical.final_counts_by_class[1]) != counts
            or canonical.action_hash != self.action_hash
        ):
            raise ProtocolError(
                "Stage-90 execution action drifted from the canonical action library."
            )
        object.__setattr__(self, "counts_per_class", MappingProxyType(counts))

    @property
    def total_per_class(self) -> int:
        return sum(self.counts_per_class.values())

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage90_utility_aligned_inner_action_execution_v1",
            "scientific_identity_owner": "actions.FrozenExactTailAction",
            "outer_target": self.outer_target,
            "query_center": self.query_center,
            "action_id": self.action_id,
            "selected_source": self.selected_source,
            "source_order": list(self.source_order),
            "counts_per_class": dict(self.counts_per_class),
            "base_per_source_per_class": INNER_BASE_PER_SOURCE_PER_CLASS,
            "additive_tail_per_class": (
                0 if self.selected_source is None else INNER_TOPUP_TOTAL_PER_CLASS
            ),
            "total_per_class": self.total_per_class,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "action_hash": self.action_hash}


def action_library_for(
    *, outer_target: object, query_center: object
) -> tuple[ExactTailDevelopmentAction, ...]:
    outer, query = str(outer_target), str(query_center)
    sources = inner_candidate_sources(outer, query)
    output: list[ExactTailDevelopmentAction] = []
    for canonical in build_inner_exact_tail_actions(outer, query):
        counts = dict(canonical.final_counts_by_class[0])
        output.append(
            ExactTailDevelopmentAction(
                outer_target=outer,
                query_center=query,
                action_id=canonical.action_id,
                selected_source=canonical.selected_source,
                source_order=sources,
                counts_per_class=counts,
                action_hash=canonical.action_hash,
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class CoarseDevelopmentTask:
    task_ordinal: int
    outer_target: str
    query_center: str
    training_seed: int
    generation_seed: int
    candidate_sources: tuple[str, ...]
    action_ids: tuple[str, ...]
    task_hash: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.outer_target,
            self.query_center,
            self.training_seed,
            self.generation_seed,
        )


@dataclass(frozen=True)
class SourceSlice:
    source_center: str
    block_ordinal: int
    stream_id: str
    expert_lock_hash: str
    output_sha256: str


@dataclass(frozen=True)
class PredictionWorkerInput:
    task: CoarseDevelopmentTask
    source_array_path: str
    source_slices: tuple[SourceSlice, ...]
    source_cache_lock_hash: str
    evaluation_array_path: str
    evaluation_array_sha256: str
    evaluation_row_ids: tuple[str, ...]
    evaluation_row_identity_hash: str
    support_partition_hash: str
    partition_lock_hash: str
    generation_lock_hash: str
    config_contract_hash: str
    classifier_payload: Mapping[str, object]
    checkpoint_json_path: str
    checkpoint_npz_path: str
    threads_per_fit: int = BLAS_THREADS_PER_WORKER

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "classifier_payload", MappingProxyType(dict(self.classifier_payload))
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.task,
                self.source_array_path,
                self.source_slices,
                self.source_cache_lock_hash,
                self.evaluation_array_path,
                self.evaluation_array_sha256,
                self.evaluation_row_ids,
                self.evaluation_row_identity_hash,
                self.support_partition_hash,
                self.partition_lock_hash,
                self.generation_lock_hash,
                self.config_contract_hash,
                dict(self.classifier_payload),
                self.checkpoint_json_path,
                self.checkpoint_npz_path,
                self.threads_per_fit,
            ),
        )


@dataclass(frozen=True)
class PredictionCheckpointRecord:
    task: CoarseDevelopmentTask
    checkpoint_json_path: str
    checkpoint_npz_path: str
    checkpoint_file_sha256: str
    checkpoint_hash: str
    evaluation_row_count: int
    action_prediction_sha256: Mapping[str, str]
    action_probability_sha256: Mapping[str, str]
    action_composition_sha256: Mapping[str, str]
    action_scaler_state_hash: Mapping[str, str]

    def __post_init__(self) -> None:
        for field in (
            "action_prediction_sha256",
            "action_probability_sha256",
            "action_composition_sha256",
            "action_scaler_state_hash",
        ):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self),
            (
                self.task,
                self.checkpoint_json_path,
                self.checkpoint_npz_path,
                self.checkpoint_file_sha256,
                self.checkpoint_hash,
                self.evaluation_row_count,
                dict(self.action_prediction_sha256),
                dict(self.action_probability_sha256),
                dict(self.action_composition_sha256),
                dict(self.action_scaler_state_hash),
            ),
        )


def expected_coarse_task_keys() -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (outer, query, training_seed, generation_seed)
        for outer in CENTERS
        for query in CENTERS
        if query != outer
        for training_seed, generation_seed in product(TRAINING_SEEDS, GENERATION_SEEDS)
    )


def expected_prediction_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, action.action_id, training_seed, generation_seed)
        for outer, query, training_seed, generation_seed in expected_coarse_task_keys()
        for action in action_library_for(outer_target=outer, query_center=query)
    )


def expected_utility_keys() -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, source, training_seed, generation_seed)
        for outer, query, training_seed, generation_seed in expected_coarse_task_keys()
        for source in inner_candidate_sources(outer, query)
    )


__all__ = (
    "BLAS_THREADS_PER_WORKER",
    "EXPECTED_COARSE_TASK_COUNT",
    "EXPECTED_EXACT_TAIL_UTILITY_ROW_COUNT",
    "EXPECTED_PREDICTION_CELL_COUNT",
    "INNER_BASE_ACTION_ID",
    "INNER_TAIL_ACTION_PREFIX",
    "PREDICTION_WORKERS",
    "CoarseDevelopmentTask",
    "ExactTailDevelopmentAction",
    "PredictionCheckpointRecord",
    "PredictionWorkerInput",
    "SourceSlice",
    "action_library_for",
    "expected_coarse_task_keys",
    "expected_prediction_keys",
    "expected_utility_keys",
    "inner_tail_action_id",
    "inner_tail_source",
)
