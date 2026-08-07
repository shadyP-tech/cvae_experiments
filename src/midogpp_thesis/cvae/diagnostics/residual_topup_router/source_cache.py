"""Public facade and coordinator for the residual top-up source cache.

The cache is target independent. Every promoted expert replica is loaded once
inside one of two persistent spawned GPU workers, used to score all nine
label-free support sets, and then used to generate all three frozen source
streams. Downstream composition only reads the resulting float32 memmap.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable, Mapping, Protocol, Sequence

from ...generation.contracts import GenerationLock, SourceGenerationKey
from ...generation.generation import source_generation_plan
from ...protocol import ProtocolError
from ._source_worker import (
    _atomic_json,
    generate_source_task as _generate_source_task,
    load_generation_checkpoint as _load_generation_checkpoint,
)
from .contracts import (
    CENTERS,
    COMMON_FEATURE_DIM,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
)
from .partitions import LabelFreeValidationFrame, PartitionSurface
from .source_cache_scheduler import (
    build_source_tasks as _scheduler_build_source_tasks,
    execute_pending_tasks as _scheduler_execute_pending_tasks,
    resume_source_tasks as _scheduler_resume_source_tasks,
    source_task_key as _source_task_key,
)
from .source_cache_store import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_CASE_MEMBER,
    SOURCE_BLOCK_ARRAY_MEMBER,
    SOURCE_BLOCK_INDEX_COLUMNS,
    SOURCE_BLOCK_INDEX_MEMBER,
    SOURCE_CACHE_LOCK_MEMBER,
    CachedSourceBlock,
    CachedSourceKey,
    SourceCache,
    atomic_write_csv as _atomic_write_csv,
    build_compatibility_case_rows as _build_compatibility_case_rows,
    build_source_cache_lock as _store_build_source_cache_lock,
    load_source_cache as _store_load_source_cache,
    materialize_source_array as _materialize_source_array,
    read_csv as _read_csv,
    validate_source_cache_lock as _store_validate_source_cache_lock,
    write_support_scratch as _write_support_scratch,
)


class _Config(Protocol):
    expert_bank_root: Path
    contract_hash: str


def materialize_source_cache(
    config: _Config,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    *,
    root: Path,
) -> SourceCache:
    """Materialize or validate the target-independent residual top-up cache."""

    final_members = (
        root / SOURCE_BLOCK_ARRAY_MEMBER,
        root / SOURCE_BLOCK_INDEX_MEMBER,
        root / COMPATIBILITY_CASE_MEMBER,
    )
    lock_path = root / SOURCE_CACHE_LOCK_MEMBER
    if all(path.is_file() for path in final_members) and lock_path.is_file():
        cache = load_source_cache(root)
        validate_source_cache_lock(
            root,
            config=config,
            generation_lock=generation_lock,
            frame=frame,
            partitions=partitions,
            source_cache=cache,
        )
        shutil.rmtree(root / "checkpoints/source_cache", ignore_errors=True)
        return cache

    checkpoint_root = root / "checkpoints/source_cache"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    support_array_path = checkpoint_root / "support_embeddings.npy"
    support_index_path = checkpoint_root / "support_index.json"
    support_scratch = _write_support_scratch(
        support_array_path,
        support_index_path,
        frame=frame,
        partitions=partitions,
    )
    tasks, key_map = _build_source_tasks(
        config,
        generation_lock,
        checkpoint_root=checkpoint_root,
        support_array_path=support_array_path,
        support_index_path=support_index_path,
        support_scratch_hash=str(support_scratch["support_scratch_hash"]),
    )
    completed, pending = _resume_source_tasks(tasks)

    for finished_count, payload in enumerate(
        _execute_pending_tasks(pending), start=1
    ):
        key = (str(payload["source_center"]), int(payload["training_seed"]))
        task = next(task for task in pending if _source_task_key(task) == key)
        verified = _load_generation_checkpoint(
            Path(str(task["checkpoint_path"])), task=task
        )
        if payload.get("checkpoint_hash") != verified.get("checkpoint_hash"):
            raise ProtocolError("Residual top-up worker checkpoint return drifted.")
        completed[key] = verified
        print(
            f"[residual-topup] source jobs {len(completed)}/"
            f"{EXPECTED_SOURCE_TASK_COUNT} (new {finished_count}/{len(pending)})",
            flush=True,
        )
    if len(completed) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("Residual top-up source checkpoint coverage is incomplete.")

    array_path = root / SOURCE_BLOCK_ARRAY_MEMBER
    index_rows = _materialize_source_array(
        array_path,
        completed=completed,
        key_map=key_map,
    )
    compatibility_rows = _build_compatibility_case_rows(completed)
    _atomic_write_csv(
        root / SOURCE_BLOCK_INDEX_MEMBER,
        index_rows,
        columns=SOURCE_BLOCK_INDEX_COLUMNS,
    )
    _atomic_write_csv(
        root / COMPATIBILITY_CASE_MEMBER,
        compatibility_rows,
        columns=COMPATIBILITY_CASE_COLUMNS,
    )
    cache = SourceCache(
        array_path=array_path,
        index_rows=tuple(index_rows),
        compatibility_case_rows=tuple(compatibility_rows),
    )
    _atomic_json(
        lock_path,
        build_source_cache_lock(
            root,
            config=config,
            generation_lock=generation_lock,
            frame=frame,
            partitions=partitions,
            source_cache=cache,
        ),
    )
    # The phase lock, not loose checkpoints, is the durable resume boundary.
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return cache


def load_source_cache(root: Path) -> SourceCache:
    return _store_load_source_cache(root)


def build_source_cache_lock(
    root: Path,
    *,
    config: _Config,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    source_cache: SourceCache,
) -> dict[str, object]:
    return _store_build_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_cache=source_cache,
    )


def validate_source_cache_lock(
    root: Path,
    *,
    config: _Config,
    generation_lock: GenerationLock,
    frame: LabelFreeValidationFrame,
    partitions: PartitionSurface,
    source_cache: SourceCache,
) -> Mapping[str, object]:
    return _store_validate_source_cache_lock(
        root,
        config=config,
        generation_lock=generation_lock,
        frame=frame,
        partitions=partitions,
        source_cache=source_cache,
    )


def _build_source_tasks(
    config: _Config,
    generation_lock: GenerationLock,
    *,
    checkpoint_root: Path,
    support_array_path: Path,
    support_index_path: Path,
    support_scratch_hash: str,
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[tuple[str, int, int], SourceGenerationKey],
]:
    return _scheduler_build_source_tasks(
        config,
        generation_lock,
        checkpoint_root=checkpoint_root,
        support_array_path=support_array_path,
        support_index_path=support_index_path,
        support_scratch_hash=support_scratch_hash,
        centers=CENTERS,
        training_seeds=TRAINING_SEEDS,
        generation_seeds=GENERATION_SEEDS,
        generation_devices=GENERATION_DEVICES,
        expected_source_block_count=EXPECTED_SOURCE_BLOCK_COUNT,
        expected_source_task_count=EXPECTED_SOURCE_TASK_COUNT,
        generation_plan=source_generation_plan,
    )


def _execute_pending_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> Iterable[Mapping[str, object]]:
    return _scheduler_execute_pending_tasks(
        tasks,
        generation_devices=GENERATION_DEVICES,
        worker=_generate_source_task,
    )


def _resume_source_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, int], Mapping[str, object]],
    list[Mapping[str, object]],
]:
    return _scheduler_resume_source_tasks(
        tasks,
        checkpoint_loader=_load_generation_checkpoint,
    )


__all__ = (
    "COMPATIBILITY_CASE_COLUMNS",
    "COMPATIBILITY_CASE_MEMBER",
    "CachedSourceBlock",
    "CachedSourceKey",
    "SOURCE_BLOCK_ARRAY_MEMBER",
    "SOURCE_BLOCK_INDEX_COLUMNS",
    "SOURCE_BLOCK_INDEX_MEMBER",
    "SOURCE_CACHE_LOCK_MEMBER",
    "SourceCache",
    "build_source_cache_lock",
    "load_source_cache",
    "materialize_source_cache",
    "validate_source_cache_lock",
)
