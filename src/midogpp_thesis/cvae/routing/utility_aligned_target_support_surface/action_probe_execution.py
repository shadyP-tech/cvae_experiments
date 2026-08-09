"""Four-worker spawned execution and hash-valid action-probe resume."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Sequence

from ...protocol import ProtocolError
from .action_probe_checkpoint import load_action_probe_checkpoint
from .action_probe_contracts import (
    ActionProbeCheckpoint,
    ActionProbeRuntime,
    ActionProbeTask,
)
from .action_probe_planning import EXPECTED_ACTION_PROBE_TASK_COUNT
from .action_probe_worker import action_probe_worker


def execute_or_resume_action_probes(
    tasks: Sequence[ActionProbeTask],
    *,
    runtime: ActionProbeRuntime,
) -> tuple[ActionProbeCheckpoint, ...]:
    """Resume valid tasks and run only absent target/seed cells."""

    ordered = tuple(tasks)
    if (
        runtime.task_count != EXPECTED_ACTION_PROBE_TASK_COUNT
        or runtime.fit_count != EXPECTED_ACTION_PROBE_TASK_COUNT * 9
        or len(ordered) != runtime.task_count
        or tuple(task.task_ordinal for task in ordered)
        != tuple(range(runtime.task_count))
        or len({task.task_hash for task in ordered}) != len(ordered)
        or any(task.runtime != runtime for task in ordered)
    ):
        raise ProtocolError("Target-support action-probe execution grid drifted.")
    completed: dict[str, ActionProbeCheckpoint] = {}
    pending: list[ActionProbeTask] = []
    for task in ordered:
        existing = load_action_probe_checkpoint(task)
        if existing is None:
            pending.append(task)
        else:
            completed[task.task_hash] = existing
    if pending:
        context = mp.get_context(runtime.multiprocessing_start_method)
        with ProcessPoolExecutor(
            max_workers=runtime.classifier_workers,
            mp_context=context,
        ) as pool:
            futures = {pool.submit(action_probe_worker, task): task for task in pending}
            finished = len(completed)
            for future in as_completed(futures):
                task = futures[future]
                if task.task_hash in completed:
                    raise ProtocolError(
                        "Target-support action-probe task completed twice."
                    )
                result = future.result()
                if result.task_hash != task.task_hash:
                    raise ProtocolError(
                        "Target-support action-probe worker returned the wrong task."
                    )
                completed[task.task_hash] = result
                finished += 1
                print(
                    f"[utility-target-support] action tasks {finished}/"
                    f"{runtime.task_count}",
                    flush=True,
                )
    if set(completed) != {task.task_hash for task in ordered}:
        raise ProtocolError("Target-support action-probe checkpoint coverage drifted.")
    return tuple(completed[task.task_hash] for task in ordered)


__all__ = (
    "execute_or_resume_action_probes",
)
