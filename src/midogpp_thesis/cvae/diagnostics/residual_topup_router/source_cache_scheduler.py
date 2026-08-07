"""Spawn-safe task planning and scheduling for the residual top-up source cache."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from ...generation.contracts import GenerationLock, SourceGenerationKey
from ...protocol import ProtocolError


class SourceTaskConfig(Protocol):
    expert_bank_root: Path
    contract_hash: str


GenerationPlan = Callable[[GenerationLock], Sequence[SourceGenerationKey]]
SourceWorker = Callable[[Mapping[str, object]], dict[str, object]]
CheckpointLoader = Callable[..., Mapping[str, object]]


def build_source_tasks(
    config: SourceTaskConfig,
    generation_lock: GenerationLock,
    *,
    checkpoint_root: Path,
    support_array_path: Path,
    support_index_path: Path,
    support_scratch_hash: str,
    centers: Sequence[str],
    training_seeds: Sequence[int],
    generation_seeds: Sequence[int],
    generation_devices: Sequence[str],
    expected_source_block_count: int,
    expected_source_task_count: int,
    generation_plan: GenerationPlan,
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[tuple[str, int, int], SourceGenerationKey],
]:
    """Build the canonical one-expert-per-task, round-robin device schedule."""

    keys = tuple(generation_plan(generation_lock))
    key_map = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in keys
    }
    expected_keys = set(product(centers, training_seeds, generation_seeds))
    if len(keys) != expected_source_block_count or set(key_map) != expected_keys:
        raise ProtocolError("Residual top-up GenerationLock source grid drifted.")
    tasks: list[Mapping[str, object]] = []
    for task_ordinal, (source, training_seed) in enumerate(
        product(centers, training_seeds)
    ):
        tasks.append(
            {
                "task_ordinal": task_ordinal,
                "source_center": source,
                "training_seed": training_seed,
                "generation_keys": tuple(
                    key_map[(source, training_seed, seed)]
                    for seed in generation_seeds
                ),
                "device": generation_devices[
                    task_ordinal % len(generation_devices)
                ],
                "expert_bank_root": str(config.expert_bank_root),
                "support_array_path": str(support_array_path),
                "support_index_path": str(support_index_path),
                "checkpoint_path": str(
                    checkpoint_root / f"source_{source}_train_{training_seed}.json"
                ),
                "array_path": str(
                    checkpoint_root / f"source_{source}_train_{training_seed}.npy"
                ),
                "config_contract_hash": config.contract_hash,
                "generation_lock_hash": generation_lock.generation_lock_hash,
                "support_scratch_hash": support_scratch_hash,
            }
        )
    if len(tasks) != expected_source_task_count:
        raise ProtocolError("Residual top-up source-task scheduler drifted.")
    return tuple(tasks), key_map


def execute_pending_tasks(
    tasks: Sequence[Mapping[str, object]],
    *,
    generation_devices: Sequence[str],
    worker: SourceWorker,
) -> Iterable[Mapping[str, object]]:
    """Run one persistent spawned process per device without changing task order."""

    if not tasks:
        return ()
    context = mp.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context)
        for _ in generation_devices
    ]
    future_to_task: dict[Future[dict[str, object]], Mapping[str, object]] = {}
    try:
        for task in tasks:
            device_index = generation_devices.index(str(task["device"]))
            future = executors[device_index].submit(worker, task)
            future_to_task[future] = task
        results: list[Mapping[str, object]] = []
        for future in as_completed(future_to_task):
            results.append(future.result())
        return tuple(results)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def resume_source_tasks(
    tasks: Sequence[Mapping[str, object]],
    *,
    checkpoint_loader: CheckpointLoader,
) -> tuple[
    dict[tuple[str, int], Mapping[str, object]],
    list[Mapping[str, object]],
]:
    """Restore every valid task checkpoint and return only absent work."""

    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint_path = Path(str(task["checkpoint_path"]))
        if not checkpoint_path.is_file():
            pending.append(task)
            continue
        payload = checkpoint_loader(checkpoint_path, task=task)
        key = source_task_key(task)
        if key in completed:
            raise ProtocolError("Residual top-up source task was duplicated.")
        completed[key] = payload
    return completed, pending


def source_task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


__all__ = (
    "SourceTaskConfig",
    "build_source_tasks",
    "execute_pending_tasks",
    "resume_source_tasks",
    "source_task_key",
)
