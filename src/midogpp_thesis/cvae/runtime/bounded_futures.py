"""Small deterministic backpressure primitive for local process pools.

The helper deliberately owns no CUDA, routing, or label semantics.  Callers
provide already validated tasks and executors; this module only limits how
many futures may be resident at once and restores input order on return.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass
from typing import Generic, TypeVar, cast


_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class BoundedExecutionStats:
    """Auditable upper bounds observed by one bounded execution."""

    task_count: int
    completed_count: int
    max_total_inflight: int
    max_inflight_by_executor: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoundedExecutionResult(Generic[_Result]):
    values: tuple[_Result, ...]
    stats: BoundedExecutionStats


def execute_bounded(
    executors: Sequence[Executor],
    tasks: Sequence[_Task],
    function: Callable[[_Task], _Result],
    *,
    executor_index: Callable[[_Task], int],
    max_inflight_per_executor: int,
    on_complete: Callable[[int, _Task, _Result], None] | None = None,
) -> BoundedExecutionResult[_Result]:
    """Execute tasks with a hard per-executor submission bound.

    Results are returned in the original task order even when workers finish
    out of order.  ``on_complete`` runs in the parent immediately after a
    future resolves, which lets callers validate durable checkpoints before
    admitting more work without moving that validation into worker processes.
    """

    if not executors:
        raise ValueError("at least one executor is required")
    if type(max_inflight_per_executor) is not int or max_inflight_per_executor < 1:
        raise ValueError("max_inflight_per_executor must be a positive integer")

    queues: list[deque[tuple[int, _Task]]] = [deque() for _ in executors]
    for position, task in enumerate(tasks):
        slot = executor_index(task)
        if type(slot) is not int or slot < 0 or slot >= len(executors):
            raise ValueError("executor_index selected an unavailable executor")
        queues[slot].append((position, task))

    missing = object()
    values: list[object] = [missing] * len(tasks)
    pending: dict[Future[_Result], tuple[int, int, _Task]] = {}
    inflight = [0] * len(executors)
    maximum = [0] * len(executors)
    max_total = 0
    completed = 0

    def fill(slot: int) -> None:
        nonlocal max_total
        while queues[slot] and inflight[slot] < max_inflight_per_executor:
            position, task = queues[slot].popleft()
            future = executors[slot].submit(function, task)
            pending[future] = (position, slot, task)
            inflight[slot] += 1
            maximum[slot] = max(maximum[slot], inflight[slot])
            max_total = max(max_total, sum(inflight))

    for slot in range(len(executors)):
        fill(slot)

    try:
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            released: set[int] = set()
            for future in sorted(done, key=lambda item: pending[item][0]):
                position, slot, task = pending.pop(future)
                inflight[slot] -= 1
                released.add(slot)
                result = future.result()
                values[position] = result
                completed += 1
                if on_complete is not None:
                    on_complete(position, task, result)
            for slot in sorted(released):
                fill(slot)
    except BaseException:
        for future in pending:
            future.cancel()
        raise

    if completed != len(tasks) or any(value is missing for value in values):
        raise RuntimeError("bounded execution completed an inconsistent task set")
    return BoundedExecutionResult(
        values=tuple(cast(_Result, value) for value in values),
        stats=BoundedExecutionStats(
            task_count=len(tasks),
            completed_count=completed,
            max_total_inflight=max_total,
            max_inflight_by_executor=tuple(maximum),
        ),
    )
