"""Deterministic coarse outer-H orchestration with callback-only science."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
from types import FunctionType
from typing import Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

from ..protocol import GovernanceError
from .dtos import CANONICAL_CENTERS, OuterCenterResult, OuterCenterTask
from .memmaps import open_memmap_bundle, validate_memmap_references
from .workstation import (
    assert_coordinator_process,
    assert_outer_worker_environment,
    build_workstation_plan,
    initialize_cpu_outer_worker,
)


class OuterCenterWorker(Protocol):
    """V2-local science callback; lifecycle never fits or selects anything."""

    def __call__(
        self, task: OuterCenterTask, arrays: Mapping[str, np.memmap]
    ) -> OuterCenterResult: ...


FoldValue = TypeVar("FoldValue")


def run_support_folds_sequentially(
    task: OuterCenterTask,
    fold_fn: Callable[[int], FoldValue],
) -> tuple[FoldValue, ...]:
    """The only lifecycle helper for support folds: ordered, in-process calls."""

    values: list[FoldValue] = []
    for fold_id in task.support_fold_ids:
        values.append(fold_fn(fold_id))
    if len(values) != len(task.support_fold_ids):  # defensive, never partial-return
        raise GovernanceError("SCALE-BP v2 support-fold execution was incomplete.")
    return tuple(values)


def execute_outer_center_task(
    task: OuterCenterTask,
    worker_fn: OuterCenterWorker,
) -> OuterCenterResult:
    """Open sealed inputs and delegate one complete H to a v2-local callback."""

    if not isinstance(task, OuterCenterTask) or not isinstance(worker_fn, FunctionType):
        raise GovernanceError("SCALE-BP v2 outer worker contract is malformed.")
    # Parent preflight is useful for early failure, but every child hashes the
    # exact slices again immediately before opening them.  A one-shot run must
    # not trust bytes that may have changed between coordinator and spawn.
    with open_memmap_bundle(task.memmaps, verify_content=True) as arrays:
        result = worker_fn(task, arrays)
    _validate_result(task, result)
    return result


def run_outer_center_tasks(
    tasks: Sequence[OuterCenterTask],
    worker_fn: OuterCenterWorker,
    use_processes: bool,
) -> tuple[OuterCenterResult, ...]:
    """Run stable one-H tasks serially or in a four-worker spawn pool.

    The callback must be a module-level pickleable callable for spawn mode.
    Each callback invocation owns one complete outer center and must return all
    support-fold IDs in their original sequential order.
    """

    rows = tuple(tasks)
    _validate_tasks(rows)
    if not isinstance(worker_fn, FunctionType) or worker_fn.__closure__ is not None:
        raise GovernanceError(
            "SCALE-BP v2 worker must be a stateless module function, not an estimator."
        )
    references = tuple(reference for task in rows for reference in task.memmaps)
    validate_memmap_references(references)
    if not use_processes:
        return tuple(
            execute_outer_center_task(task, worker_fn)
            for task in rows
        )
    assert_coordinator_process()
    try:
        pickle.dumps(worker_fn)
    except Exception as exc:  # pragma: no cover - exact exception is callable-specific
        raise GovernanceError(
            "SCALE-BP v2 spawn worker callback must be module-level and pickleable."
        ) from exc
    plan = build_workstation_plan()
    with ProcessPoolExecutor(
        max_workers=min(plan.cpu_outer_workers, len(rows)),
        mp_context=mp.get_context(plan.multiprocessing_start_method),
        initializer=initialize_cpu_outer_worker,
    ) as executor:
        futures = [executor.submit(_spawn_entry, task, worker_fn) for task in rows]
        unordered = tuple(future.result() for future in futures)
    by_center = {result.target_center: result for result in unordered}
    if set(by_center) != {task.target_center for task in rows}:
        raise GovernanceError("SCALE-BP v2 outer worker result inventory drifted.")
    ordered = tuple(by_center[task.target_center] for task in rows)
    for task, result in zip(rows, ordered, strict=True):
        _validate_result(task, result)
    return ordered


def _spawn_entry(
    task: OuterCenterTask, worker_fn: OuterCenterWorker
) -> OuterCenterResult:
    assert_outer_worker_environment()
    return execute_outer_center_task(task, worker_fn)


def _validate_tasks(tasks: tuple[OuterCenterTask, ...]) -> None:
    centers = tuple(task.target_center for task in tasks)
    if (
        not tasks
        or len(set(centers)) != len(centers)
        or centers != tuple(sorted(centers, key=CANONICAL_CENTERS.index))
        or any(not isinstance(task, OuterCenterTask) for task in tasks)
        or len({task.protocol_hash for task in tasks}) != 1
    ):
        raise GovernanceError("SCALE-BP v2 outer task inventory drifted.")


def _validate_result(task: OuterCenterTask, result: object) -> None:
    if (
        not isinstance(result, OuterCenterResult)
        or result.target_center != task.target_center
        or result.task_hash != task.task_hash
        or result.completed_support_fold_ids != task.support_fold_ids
        or len(result.route_hashes) != len(task.case_ids)
    ):
        raise GovernanceError("SCALE-BP v2 outer worker returned a foreign result.")


__all__ = (
    "OuterCenterWorker",
    "execute_outer_center_task",
    "run_outer_center_tasks",
    "run_support_folds_sequentially",
)
