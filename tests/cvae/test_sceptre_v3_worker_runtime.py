from __future__ import annotations

from concurrent.futures import Future
import os
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3 import (
    worker_runtime,
)
from midogpp_thesis.cvae.diagnostics.sceptre_runtime.worker_lifecycle import (
    single_worker_spawn_executor,
)
from midogpp_thesis.cvae.protocol import ProtocolError


_SPAWN_INITIALIZER_COUNT = 0


def _spawn_initializer() -> None:
    global _SPAWN_INITIALIZER_COUNT
    _SPAWN_INITIALIZER_COUNT += 1


def _spawn_probe(ordinal: int) -> tuple[int, int, int]:
    return os.getpid(), _SPAWN_INITIALIZER_COUNT, ordinal


class _FakeCuda:
    def __init__(self) -> None:
        self.index = -1

    def set_device(self, device: str) -> None:
        self.index = int(device.split(":", 1)[1])

    def current_device(self) -> int:
        return self.index


class _FakeTorch:
    def __init__(self) -> None:
        self.interop = 8
        self.intraop = 8
        self.cuda = _FakeCuda()
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True),
        )

    def set_num_interop_threads(self, value: int) -> None:
        self.interop = value

    def set_num_threads(self, value: int) -> None:
        self.intraop = value

    def set_float32_matmul_precision(self, value: str) -> None:
        assert value == "highest"

    def get_num_interop_threads(self) -> int:
        return self.interop

    def get_num_threads(self) -> int:
        return self.intraop


class _FakeExecutor:
    def __init__(self, device: str) -> None:
        self.device = device
        self.pid = 9000 + int(device.split(":", 1)[1])
        self.submitted: list[int] = []

    def submit(self, _function: object, device: str, ordinal: int) -> Future:
        assert device == self.device
        self.submitted.append(ordinal)
        future: Future[dict[str, object]] = Future()
        future.set_result(
            {
                "device": device,
                "device_index": int(device.split(":", 1)[1]),
                "process_id": self.pid,
                "initializer_invocation_count": 1,
                "torch_intraop_threads": 1,
                "torch_interop_threads": 1,
                "tf32_enabled": False,
                "amp_enabled": False,
                "task_ordinal": ordinal,
            }
        )
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        assert wait is True
        assert cancel_futures is True


def test_identity_neutral_spawn_executor_reuses_one_initialized_process() -> None:
    executor = single_worker_spawn_executor(initializer=_spawn_initializer)
    try:
        rows = tuple(executor.submit(_spawn_probe, ordinal).result() for ordinal in (0, 1))
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    assert rows[0][0] == rows[1][0]
    assert rows[0][1] == rows[1][1] == 1
    assert tuple(row[2] for row in rows) == (0, 1)


def test_gpu_initializer_configures_once_and_tasks_only_authenticate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeTorch()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    monkeypatch.setattr(worker_runtime, "_WORKER_BINDING", None)
    worker_runtime.initialize_gpu_worker("cuda:0")
    binding = worker_runtime.assert_gpu_worker_ready("cuda:0")
    assert binding["initializer_invocation_count"] == 1
    assert binding["torch_intraop_threads"] == 1
    assert binding["torch_interop_threads"] == 1
    with pytest.raises(ProtocolError, match="initialized twice"):
        worker_runtime.initialize_gpu_worker("cuda:0")


def test_prelease_smoke_submits_two_tasks_to_each_exact_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors: list[_FakeExecutor] = []

    def _factory(*, initializer: object, initargs: tuple[object, ...]):
        assert initializer is worker_runtime.initialize_gpu_worker
        executor = _FakeExecutor(str(initargs[0]))
        executors.append(executor)
        return executor

    monkeypatch.setattr(worker_runtime, "single_worker_spawn_executor", _factory)
    receipt = worker_runtime.run_gpu_worker_runtime_smoke()
    validated = worker_runtime.validate_worker_runtime_smoke(receipt)
    assert tuple(executor.submitted for executor in executors) == ([0, 1], [0, 1])
    assert validated["scientific_gpu_work_performed"] is False
    assert validated["target_labels_opened"] is False
    assert validated["filesystem_mutations"] == 0
    assert validated["parent_cuda_state_checked_before_and_after_smoke"] is True


def test_prelease_smoke_rejects_initialized_parent_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", parent_torch)
    with pytest.raises(ProtocolError, match="parent CUDA context"):
        worker_runtime.run_gpu_worker_runtime_smoke()


def test_smoke_rejects_initializer_pool_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(**_kwargs: object):
        raise RuntimeError("initializer unavailable")

    monkeypatch.setattr(worker_runtime, "single_worker_spawn_executor", _fail)
    with pytest.raises(RuntimeError, match="initializer unavailable"):
        worker_runtime.run_gpu_worker_runtime_smoke()
