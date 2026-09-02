from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from midogpp_thesis.cvae.runtime.harp_v8_execution import physical


def test_classifier_pool_plan_is_four_one_process_executors_with_local_bounds() -> None:
    pool = physical.classifier_pool_plan()
    whole = physical.build_physical_plan()

    assert pool == {
        "schema_version": "midogpp_harp_v8_classifier_pool_plan_v1",
        "executor_count": 4,
        "processes_per_executor": 1,
        "total_worker_processes": 4,
        "blas_threads_per_process": 3,
        "multiprocessing_start_method": "spawn",
        "max_inflight_per_executor": 2,
        "max_total_inflight": 8,
        "task_assignment": "ordinal_modulo_executor_count",
        "plan_hash": pool["plan_hash"],
    }
    assert whole["classifier_executor_count"] == 4
    assert whole["classifier_processes_per_executor"] == 1
    assert whole["classifier_executor_assignment"] == "ordinal_modulo_executor_count"
    assert whole["bounded_inflight_classifier_tasks_per_worker"] == 2
    assert whole["max_inflight_classifier_tasks"] == 8


def test_classifier_execution_constructs_and_maps_four_single_process_pools(
    monkeypatch: Any,
) -> None:
    instances: list[_ImmediateExecutor] = []

    class RecordingExecutor(_ImmediateExecutor):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(slot=len(instances), **kwargs)
            instances.append(self)

    loads: dict[int, int] = {}

    def checkpoint(task: dict[str, object]) -> dict[str, object] | None:
        ordinal = int(task["ordinal"])
        count = loads.get(ordinal, 0)
        loads[ordinal] = count + 1
        if count == 0:
            return None
        return {"checkpoint_hash": f"checkpoint-{ordinal}"}

    monkeypatch.setattr(physical, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(physical, "_load_task_checkpoint", checkpoint)
    monkeypatch.setattr(physical, "_classifier_task", lambda _task: None)

    tasks = tuple({"ordinal": ordinal} for ordinal in range(12))
    complete = physical._execute_tasks(
        tasks,
        workstation=physical._DEFAULT_WORKSTATION_PROFILE,
    )

    assert set(complete) == set(range(12))
    assert len(instances) == 4
    assert all(instance.max_workers == 1 for instance in instances)
    assert all(instance.mp_context.get_start_method() == "spawn" for instance in instances)
    assert all(
        instance.initializer is physical._initialize_classifier_worker
        for instance in instances
    )
    assert all(instance.initargs == (3,) for instance in instances)
    assert [instance.ordinals for instance in instances] == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]


class _ImmediateExecutor:
    def __init__(
        self,
        *,
        slot: int,
        max_workers: int,
        mp_context: object,
        initializer: object,
        initargs: tuple[object, ...],
    ) -> None:
        self.slot = slot
        self.max_workers = max_workers
        self.mp_context = mp_context
        self.initializer = initializer
        self.initargs = initargs
        self.ordinals: list[int] = []

    def __enter__(self) -> _ImmediateExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def submit(self, function: object, task: dict[str, object]) -> Future[None]:
        self.ordinals.append(int(task["ordinal"]))
        future: Future[None] = Future()
        future.set_result(None)
        return future
