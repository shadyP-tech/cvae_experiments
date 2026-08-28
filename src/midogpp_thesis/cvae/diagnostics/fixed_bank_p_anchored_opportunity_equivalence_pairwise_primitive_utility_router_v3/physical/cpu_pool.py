"""Four-spawn-worker, one-BLAS-thread fixed-bank prediction executor."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import pickle
from types import MappingProxyType
from typing import Mapping, Sequence

from ....protocol import ProtocolError
from ....runtime.fixed_bank_a1_prediction_worker import (
    execute_prediction_task,
    load_prediction_checkpoint,
)
from ..workstation import CPU_SPAWN_WORKER_COUNT, CPU_WORKER_ENVIRONMENT


def execute_prediction_tasks_one_thread(
    tasks: Sequence[Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
    """Resume or execute all plain tasks without the neutral 3-thread facade."""

    rows = tuple(_validated_plain_task(task) for task in tasks)
    if not rows:
        raise ProtocolError("OE-PPUR v3 prediction task inventory is empty.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[dict[str, object]] = []
    for task in rows:
        loaded = load_prediction_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(
            max_workers=CPU_SPAWN_WORKER_COUNT,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_one_thread_cpu_worker,
        ) as executor:
            futures = {
                executor.submit(execute_prediction_task, task): task
                for task in pending
            }
            for future in as_completed(futures):
                future.result()
                task = futures[future]
                loaded = load_prediction_checkpoint(task)
                if loaded is None:
                    raise ProtocolError(
                        "OE-PPUR v3 prediction worker omitted its checkpoint."
                    )
                completed[str(task["task_id"])] = loaded
                print(
                    f"[oe-ppur-v3:predictions] tasks {len(completed)}/{len(rows)}",
                    flush=True,
                )
    if len(completed) != len(rows):
        raise ProtocolError("OE-PPUR v3 prediction checkpoint coverage drifted.")
    return MappingProxyType(completed)


def _validated_plain_task(task: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(task, Mapping) or int(task.get("threads_per_fit", -1)) != 1:
        raise ProtocolError("OE-PPUR v3 prediction task thread topology drifted.")
    row = dict(task)
    try:
        rebuilt = pickle.loads(pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))
    except (pickle.PickleError, TypeError, ValueError, AttributeError) as exc:
        raise ProtocolError("OE-PPUR v3 prediction task is not spawn-safe.") from exc
    if type(rebuilt) is not dict or rebuilt != row:
        raise ProtocolError("OE-PPUR v3 prediction task pickle identity drifted.")
    return row


def _initialize_one_thread_cpu_worker() -> None:
    for name, value in CPU_WORKER_ENVIRONMENT.items():
        os.environ[name] = value
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(name) != "1"
        for name in CPU_WORKER_ENVIRONMENT
        if name != "CUDA_VISIBLE_DEVICES"
    ):
        raise ProtocolError("OE-PPUR v3 CPU worker isolation failed.")


__all__ = ("execute_prediction_tasks_one_thread",)
