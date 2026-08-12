"""Label-free physical prediction menus and deterministic task planning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .artifact_io import atomic_json, read_json, sha256_file
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
    h_x_e_source,
    inner_candidate_sources,
)
from .input_contracts import LabelFreeTestFrame, row_identity_hash
from .partitions import ConsumedTestPartitionSurface, LabelFreeCaseRow
from .prediction_contracts import (
    DEVELOPMENT_ROLE,
    DEVELOPMENT_TASK_COUNT,
    TARGET_ROLE,
    TARGET_TASK_COUNT,
    PlannedPhysicalAction,
    PredictionTask,
    canonical_cell_keys,
    physical_target_action_ids,
)


TARGET_EMBEDDING_MEMBER = "target_embeddings.npy"
TARGET_EMBEDDING_INDEX_MEMBER = "target_embeddings.json"


@dataclass(frozen=True)
class PredictionPlan:
    phase: str
    tasks: tuple[PredictionTask, ...]
    action_library_payload: Mapping[str, object]
    action_library_hash: str
    plan_hash: str

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        payload = MappingProxyType(dict(self.action_library_payload))
        expected_task_count = (
            DEVELOPMENT_TASK_COUNT if self.phase == DEVELOPMENT_ROLE else TARGET_TASK_COUNT
        )
        flattened_keys = tuple(
            (
                task.outer_target,
                task.query_center,
                action.action_id,
                task.training_seed,
                task.generation_seed,
            )
            for task in tasks
            for action in task.actions
        )
        if (
            self.phase not in {DEVELOPMENT_ROLE, TARGET_ROLE}
            or len(tasks) != expected_task_count
            or tuple(task.task_ordinal for task in tasks) != tuple(range(len(tasks)))
            or any(task.phase != self.phase for task in tasks)
            or flattened_keys != canonical_cell_keys(self.phase)
            or self.action_library_hash != canonical_sha256(dict(payload))
            or self.plan_hash != canonical_sha256(self._unhashed_payload(tasks, payload))
        ):
            raise ProtocolError("Endpoint-router prediction plan drifted.")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "action_library_payload", payload)

    def _unhashed_payload(
        self,
        tasks: Sequence[PredictionTask] | None = None,
        action_library: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        values = self.tasks if tasks is None else tasks
        library = self.action_library_payload if action_library is None else action_library
        return {
            "schema_version": "midogpp_endpoint_router_prediction_plan_v1",
            "phase": self.phase,
            "task_hashes": [task.task_hash for task in values],
            "task_count": len(values),
            "prediction_cell_count": sum(len(task.actions) for task in values),
            "action_library_hash": canonical_sha256(dict(library)),
            "labels_available": False,
        }


@dataclass(frozen=True)
class StagedTargetEmbeddings:
    array_path: Path
    array_sha256: str
    cache_binding_hash: str


def cleanup_staged_target_embeddings(staged: StagedTargetEmbeddings) -> None:
    """Remove only the two task-derived, label-free scratch members."""

    array_path = Path(staged.array_path)
    index_path = array_path.with_name(TARGET_EMBEDDING_INDEX_MEMBER)
    if (
        array_path.name != TARGET_EMBEDDING_MEMBER
        or index_path.name != TARGET_EMBEDDING_INDEX_MEMBER
        or array_path.parent != index_path.parent
        or not array_path.is_absolute()
    ):
        raise ProtocolError("Endpoint-router target scratch paths drifted.")
    for path in (array_path, index_path):
        if path.is_symlink():
            raise ProtocolError("Endpoint-router target scratch member is unsafe.")
        path.unlink(missing_ok=True)


def stage_target_embeddings(
    frame: LabelFreeTestFrame, *, scratch_root: Path
) -> StagedTargetEmbeddings:
    """Create or validate one float32 memmap shared by spawned CPU workers."""

    scratch_root.mkdir(parents=True, exist_ok=True)
    array_path = scratch_root / TARGET_EMBEDDING_MEMBER
    index_path = scratch_root / TARGET_EMBEDDING_INDEX_MEMBER
    if array_path.is_file() and index_path.is_file():
        index = read_json(index_path)
        if (
            index.get("status") != "COMPLETE"
            or index.get("cache_binding_hash") != frame.cache_binding_hash
            or index.get("array_sha256") != sha256_file(array_path)
            or index.get("shape") != list(frame.embeddings.shape)
            or index.get("dtype") != "float32"
        ):
            raise ProtocolError("Endpoint-router staged target embeddings drifted.")
        return StagedTargetEmbeddings(
            array_path=array_path,
            array_sha256=str(index["array_sha256"]),
            cache_binding_hash=frame.cache_binding_hash,
        )
    temporary = array_path.with_suffix(f"{array_path.suffix}.{os.getpid()}.tmp")
    values = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=frame.embeddings.shape,
    )
    values[:] = frame.embeddings
    values.flush()
    del values
    os.replace(temporary, array_path)
    digest = sha256_file(array_path)
    atomic_json(
        index_path,
        {
            "schema_version": "midogpp_endpoint_router_target_embedding_stage_v1",
            "status": "COMPLETE",
            "cache_binding_hash": frame.cache_binding_hash,
            "array_sha256": digest,
            "shape": list(frame.embeddings.shape),
            "dtype": "float32",
            "labels_stored": False,
        },
    )
    return StagedTargetEmbeddings(
        array_path=array_path,
        array_sha256=digest,
        cache_binding_hash=frame.cache_binding_hash,
    )


def build_development_prediction_plan(
    config: object,
    *,
    frame: LabelFreeTestFrame,
    partitions: ConsumedTestPartitionSurface,
    source_cache: FrozenSourceStreamCache,
    target_embeddings: StagedTargetEmbeddings,
    checkpoint_root: Path,
) -> PredictionPlan:
    tasks: list[PredictionTask] = []
    library: dict[str, object] = {}
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            scope = f"{outer}::{query}"
            actions = _development_actions(outer, query)
            library[scope] = [action.to_payload() for action in actions]
            support = partitions.support_rows_by_center[query]
            evaluation = partitions.evaluation_rows_by_center[query]
            for training_seed, generation_seed in SEED_PAIRS:
                tasks.append(
                    _prediction_task(
                        config,
                        phase=DEVELOPMENT_ROLE,
                        task_ordinal=len(tasks),
                        outer=outer,
                        query=query,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        actions=actions,
                        support=support,
                        evaluation=evaluation,
                        source_cache=source_cache,
                        target_embeddings=target_embeddings,
                        partitions=partitions,
                        checkpoint_root=checkpoint_root,
                    )
                )
    return _plan(DEVELOPMENT_ROLE, tasks, library)


def build_target_prediction_plan(
    config: object,
    *,
    frame: LabelFreeTestFrame,
    partitions: ConsumedTestPartitionSurface,
    source_cache: FrozenSourceStreamCache,
    target_embeddings: StagedTargetEmbeddings,
    checkpoint_root: Path,
) -> PredictionPlan:
    tasks: list[PredictionTask] = []
    library: dict[str, object] = {}
    for target in CENTERS:
        actions = _target_actions(target)
        library[target] = [action.to_payload() for action in actions]
        support = partitions.support_rows_by_center[target]
        evaluation = partitions.evaluation_rows_by_center[target]
        for training_seed, generation_seed in SEED_PAIRS:
            tasks.append(
                _prediction_task(
                    config,
                    phase=TARGET_ROLE,
                    task_ordinal=len(tasks),
                    outer=target,
                    query=target,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    actions=actions,
                    support=support,
                    evaluation=evaluation,
                    source_cache=source_cache,
                    target_embeddings=target_embeddings,
                    partitions=partitions,
                    checkpoint_root=checkpoint_root,
                )
            )
    return _plan(TARGET_ROLE, tasks, library)


def _development_actions(outer: str, query: str) -> tuple[PlannedPhysicalAction, ...]:
    sources = inner_candidate_sources(outer, query)
    actions: list[PlannedPhysicalAction] = []
    for action_id in expected_development_action_ids(outer, query):
        selected = h_x_e_source(action_id)
        counts = {
            source: (270 if source == selected else 144) for source in sources
        }
        actions.append(_action(DEVELOPMENT_ROLE, outer, query, action_id, sources, counts))
    return tuple(actions)


def _target_actions(target: str) -> tuple[PlannedPhysicalAction, ...]:
    sources = candidate_sources(target)
    actions: list[PlannedPhysicalAction] = []
    for action_id in physical_target_action_ids(target):
        selected = h_x_e_source(action_id)
        if action_id == BASE_ACTION_ID:
            counts = {source: 128 for source in sources}
        elif action_id == UNIFORM_ACTION_ID:
            counts = {source: 144 for source in sources}
        else:
            counts = {
                source: (256 if source == selected else 128) for source in sources
            }
        actions.append(_action(TARGET_ROLE, target, target, action_id, sources, counts))
    return tuple(actions)


def _action(
    phase: str,
    outer: str,
    query: str,
    action_id: str,
    sources: tuple[str, ...],
    counts: Mapping[str, int],
) -> PlannedPhysicalAction:
    unhashed = {
        "schema_version": "midogpp_endpoint_router_physical_action_v1",
        "phase": phase,
        "outer_target": outer,
        "query_center": query,
        "action_id": action_id,
        "sources": list(sources),
        "rows_per_class_by_source": dict(counts),
        "labels_used": False,
        "source_prefix_only": True,
    }
    return PlannedPhysicalAction(
        phase=phase,
        outer_target=outer,
        query_center=query,
        action_id=action_id,
        sources=sources,
        rows_per_class_by_source=counts,
        action_hash=canonical_sha256(unhashed),
    )


def _prediction_task(
    config: object,
    *,
    phase: str,
    task_ordinal: int,
    outer: str,
    query: str,
    training_seed: int,
    generation_seed: int,
    actions: tuple[PlannedPhysicalAction, ...],
    support: Sequence[LabelFreeCaseRow],
    evaluation: Sequence[LabelFreeCaseRow],
    source_cache: FrozenSourceStreamCache,
    target_embeddings: StagedTargetEmbeddings,
    partitions: ConsumedTestPartitionSurface,
    checkpoint_root: Path,
) -> PredictionTask:
    classifier = dict(getattr(config, "classifier").to_payload())
    stem = (
        f"{phase}_H{outer}_q{query}_train{training_seed}_gen{generation_seed}"
    )
    values = {
        "phase": phase,
        "task_ordinal": task_ordinal,
        "outer_target": outer,
        "query_center": query,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "actions": actions,
        "source_array_path": str(source_cache.source_array_path),
        "target_array_path": str(target_embeddings.array_path),
        "target_array_sha256": target_embeddings.array_sha256,
        "support_row_ordinals": tuple(row.row_ordinal for row in support),
        "evaluation_row_ordinals": tuple(row.row_ordinal for row in evaluation),
        "support_row_ids": tuple(row.evaluation_row_id for row in support),
        "evaluation_row_ids": tuple(row.evaluation_row_id for row in evaluation),
        "support_case_ids": tuple(row.case_id for row in support),
        "evaluation_case_ids": tuple(row.case_id for row in evaluation),
        "support_row_identity_hash": row_identity_hash(support),
        "evaluation_row_identity_hash": row_identity_hash(evaluation),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "source_stream_lock_hash": source_cache.lock_hash,
        "partition_lock_hash": partitions.lock_hash,
        "cache_binding_hash": target_embeddings.cache_binding_hash,
        "classifier_payload": classifier,
        "checkpoint_npz_path": str(checkpoint_root / f"{stem}.npz"),
        "checkpoint_json_path": str(checkpoint_root / f"{stem}.json"),
    }
    unhashed = {
        "schema_version": "midogpp_endpoint_router_prediction_task_v1",
        **{
            key: (
                [action.to_payload() for action in value]
                if key == "actions"
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in values.items()
        },
        "labels_available": False,
    }
    return PredictionTask(**values, task_hash=canonical_sha256(unhashed))


def _plan(
    phase: str,
    tasks: Sequence[PredictionTask],
    library: Mapping[str, object],
) -> PredictionPlan:
    library_payload = {
        "schema_version": "midogpp_endpoint_router_physical_action_library_v1",
        "phase": phase,
        "by_scope": dict(library),
        "labels_used": False,
    }
    action_hash = canonical_sha256(library_payload)
    unhashed = {
        "schema_version": "midogpp_endpoint_router_prediction_plan_v1",
        "phase": phase,
        "task_hashes": [task.task_hash for task in tasks],
        "task_count": len(tasks),
        "prediction_cell_count": sum(len(task.actions) for task in tasks),
        "action_library_hash": action_hash,
        "labels_available": False,
    }
    return PredictionPlan(
        phase=phase,
        tasks=tuple(tasks),
        action_library_payload=library_payload,
        action_library_hash=action_hash,
        plan_hash=canonical_sha256(unhashed),
    )


__all__ = (
    "cleanup_staged_target_embeddings",
    "PredictionPlan",
    "StagedTargetEmbeddings",
    "TARGET_EMBEDDING_INDEX_MEMBER",
    "TARGET_EMBEDDING_MEMBER",
    "build_development_prediction_plan",
    "build_target_prediction_plan",
    "stage_target_embeddings",
)
