"""Surface orchestration and persistent two-device spawned execution."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from ...protocol import ProtocolError
from .proxy_surface_checkpoints import load_fresh_proxy_score_checkpoint
from .proxy_surface_contracts import (
    ArrayLoader,
    DEFAULT_DEVICES,
    EXPECTED_EXPERT_TASK_COUNT,
    SCORE_CHUNK_ROWS,
    FreshProxyScoreSurface,
    FreshProxyScoreTask,
    FreshProxyTaskResult,
    FreshQueryShard,
    default_array_loader,
)
from .proxy_surface_planning import build_fresh_proxy_score_tasks
from .proxy_surface_validation import validate_fresh_proxy_score_surface
from .proxy_surface_worker import (
    CompatibilityScorer,
    ExpertLoader,
    default_compatibility_scorer,
    default_expert_loader,
    execute_fresh_proxy_score_task,
)


TaskWorker = Callable[[FreshProxyScoreTask], FreshProxyTaskResult]
TaskExecutor = Callable[
    [Sequence[FreshProxyScoreTask], TaskWorker],
    Iterable[FreshProxyTaskResult],
]


def build_fresh_proxy_score_surface(
    shards: Iterable[FreshQueryShard],
    *,
    expert_bank_root: str | Path,
    expert_bank_binding_hash: str,
    checkpoint_root: str | Path,
    devices: Sequence[str] = DEFAULT_DEVICES,
    chunk_rows: int = SCORE_CHUNK_ROWS,
    array_loader: ArrayLoader = None,  # type: ignore[assignment]
    expert_loader: ExpertLoader = None,  # type: ignore[assignment]
    scorer: CompatibilityScorer = None,  # type: ignore[assignment]
    executor: TaskExecutor | None = None,
) -> FreshProxyScoreSurface:
    """Resume or execute all 27 jobs, then validate the canonical score grid."""

    active_array_loader = array_loader or default_array_loader
    active_expert_loader = expert_loader or default_expert_loader
    active_scorer = scorer or default_compatibility_scorer
    shard_tuple = tuple(shards)
    tasks = build_fresh_proxy_score_tasks(
        shard_tuple,
        expert_bank_root=expert_bank_root,
        expert_bank_binding_hash=expert_bank_binding_hash,
        checkpoint_root=checkpoint_root,
        devices=devices,
        chunk_rows=chunk_rows,
    )
    completed: dict[tuple[str, int], FreshProxyTaskResult] = {}
    pending: list[FreshProxyScoreTask] = []
    for task in tasks:
        if task.checkpoint_path.is_file():
            result = load_fresh_proxy_score_checkpoint(
                task.checkpoint_path, task=task
            )
            completed[task.key] = result
        else:
            pending.append(task)

    worker = partial(
        execute_fresh_proxy_score_task,
        array_loader=active_array_loader,
        expert_loader=active_expert_loader,
        scorer=active_scorer,
    )
    if pending:
        if executor is None:
            if (
                active_array_loader is not default_array_loader
                or active_expert_loader is not default_expert_loader
                or active_scorer is not default_compatibility_scorer
            ):
                raise ProtocolError(
                    "Injected proxy loaders/scorers require an injected executor."
                )
            if tuple(devices) != DEFAULT_DEVICES:
                raise ProtocolError(
                    "Default fresh proxy execution is locked to cuda:0/cuda:1."
                )
            new_results = execute_spawned_tasks(tuple(pending))
        else:
            new_results = tuple(executor(tuple(pending), worker))
        by_key = {result.key: result for result in new_results}
        if len(by_key) != len(new_results) or set(by_key) != {
            task.key for task in pending
        }:
            raise ProtocolError("Fresh proxy executor result coverage drifted.")
        for task in pending:
            returned = by_key[task.key]
            verified = load_fresh_proxy_score_checkpoint(
                task.checkpoint_path, task=task
            )
            if (
                returned.task_hash != task.task_hash
                or returned.checkpoint_hash != verified.checkpoint_hash
            ):
                raise ProtocolError("Fresh proxy worker checkpoint return drifted.")
            completed[task.key] = FreshProxyTaskResult(
                source_center=verified.source_center,
                training_seed=verified.training_seed,
                rows=verified.rows,
                checkpoint_hash=verified.checkpoint_hash,
                task_hash=verified.task_hash,
                expert_lock_hash=verified.expert_lock_hash,
                expert_checkpoint_hash=verified.expert_checkpoint_hash,
                resumed=False,
            )
    if len(completed) != EXPECTED_EXPERT_TASK_COUNT:
        raise ProtocolError("Fresh proxy expert checkpoint coverage is incomplete.")

    task_results = tuple(completed[task.key] for task in tasks)
    rows = tuple(row for result in task_results for row in result.rows)
    canonical_rows = validate_fresh_proxy_score_surface(
        rows, shards=shard_tuple
    )
    return FreshProxyScoreSurface(
        rows=canonical_rows,
        task_results=task_results,
        surface_hash=tasks[0].surface_hash,
        expert_bank_binding_hash=tasks[0].expert_bank_binding_hash,
        resumed_task_count=sum(result.resumed for result in task_results),
        executed_task_count=len(pending),
        labels_consumed=False,
        source_experts_updated=False,
    )


def execute_spawned_tasks(
    tasks: Sequence[FreshProxyScoreTask],
) -> tuple[FreshProxyTaskResult, ...]:
    if not tasks:
        return ()
    if {task.device for task in tasks}.difference(DEFAULT_DEVICES):
        raise ProtocolError("Fresh proxy spawned tasks escaped cuda:0/cuda:1.")
    context = mp.get_context("spawn")
    executors = {
        device: ProcessPoolExecutor(max_workers=1, mp_context=context)
        for device in DEFAULT_DEVICES
    }
    futures: dict[tuple[str, int], Future[FreshProxyTaskResult]] = {}
    try:
        for task in tasks:
            futures[task.key] = executors[task.device].submit(
                execute_fresh_proxy_score_task, task
            )
        return tuple(futures[task.key].result() for task in tasks)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)


__all__ = ("build_fresh_proxy_score_surface",)
