from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v3.config import load_config
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime import frozen_source_streams
from midogpp_thesis.cvae.runtime.bounded_futures import execute_bounded
from midogpp_thesis.cvae.runtime.harp_v3_execution.physical import (
    build_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v3_execution import production
from midogpp_thesis.cvae.runtime.harp_v3_execution import workstation


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v3.yaml"
)


def test_bounded_execution_limits_submission_and_preserves_input_order() -> None:
    tasks = tuple(range(20))
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(
        max_workers=1
    ) as second:
        result = execute_bounded(
            (first, second),
            tasks,
            lambda value: value * value,
            executor_index=lambda value: value % 2,
            max_inflight_per_executor=2,
        )

    assert result.values == tuple(value * value for value in tasks)
    assert result.stats.completed_count == len(tasks)
    assert result.stats.max_inflight_by_executor == (2, 2)
    assert result.stats.max_total_inflight <= 4


def test_bounded_execution_rejects_invalid_capacity_and_executor_index() -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="positive integer"):
            execute_bounded(
                (pool,),
                (1,),
                lambda value: value,
                executor_index=lambda _value: 0,
                max_inflight_per_executor=0,
            )
        with pytest.raises(ValueError, match="unavailable executor"):
            execute_bounded(
                (pool,),
                (1,),
                lambda value: value,
                executor_index=lambda _value: 1,
                max_inflight_per_executor=1,
            )


def test_v3_workstation_plan_binds_both_queue_limits() -> None:
    config = load_config(CONFIG)
    plan = build_physical_plan()

    assert config.runtime["bounded_inflight_batches_per_gpu"] == 2
    assert config.runtime["bounded_inflight_tasks_per_cpu_worker"] == 2
    assert plan["bounded_inflight_batches_per_gpu"] == 2
    assert plan["max_inflight_source_tasks"] == 4
    assert plan["bounded_inflight_tasks_per_cpu_worker"] == 2
    assert plan["max_inflight_classifier_tasks"] == 8


def test_neutral_source_runtime_rejects_nonpositive_gpu_queue_bound() -> None:
    runtime = {
        "generation_devices": ["cuda:0", "cuda:1"],
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "generated_cache_format": "float32_npy_memmap",
        "source_prefix_rows_per_class": frozen_source_streams.SOURCE_ROWS_PER_CLASS,
        "bounded_inflight_batches_per_gpu": 2,
    }
    frozen_source_streams._assert_runtime(runtime)
    runtime["bounded_inflight_batches_per_gpu"] = 0
    with pytest.raises(ProtocolError, match="two exact float32 GPU streams"):
        frozen_source_streams._assert_runtime(runtime)


def test_v3_source_fit_worker_enforces_bound_blas_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    sentinel = object()

    @contextmanager
    def limited(*, limits: int):
        events.append(("enter", limits))
        yield
        events.append(("exit", limits))

    def fake_fit(rows: object, **kwargs: object) -> object:
        events.append(("fit", (rows, kwargs)))
        return sentinel

    import threadpoolctl

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", limited)
    monkeypatch.setattr(production, "fit_harp_v3", fake_fit)

    payload = ("9", (), (0.01, 0.1), 0.9, 0.95, 3)
    assert production._fit_worker(payload) is sentinel
    assert events[0] == ("enter", 3)
    assert events[-1] == ("exit", 3)
    assert events[1][0] == "fit"

    with pytest.raises(ProtocolError, match="BLAS limit"):
        production._fit_worker(("9", (), (0.01,), 0.9, 0.95, 0))


def test_v3_live_preflight_is_mutation_free_and_resource_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG)
    for key, value in workstation._EXPECTED_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(workstation, "_available_cpu_count", lambda: 24)
    monkeypatch.setattr(workstation, "_physical_ram_bytes", lambda: 125 * 1024**3)
    monkeypatch.setattr(
        workstation, "_safe_nearest_existing_parent", lambda _path: tmp_path
    )
    monkeypatch.setattr(workstation.os, "access", lambda *_args: True)
    monkeypatch.setattr(
        workstation.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=6 * 1024**4),
    )
    monkeypatch.setattr(
        workstation,
        "_nvidia_smi_rows",
        lambda: tuple(
            {
                "index": index,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24_576,
                "memory_free_mib": 24_000,
            }
            for index in (0, 1)
        ),
    )
    monkeypatch.setattr(workstation, "_package_versions", lambda: {"numpy": "test"})

    before = tuple(tmp_path.iterdir())
    report = workstation.inspect_harp_v3_workstation(config.runtime)
    assert report["status"] == "PASS"
    assert report["filesystem_mutations"] == 0
    assert report["available_cpu_affinity_count"] == 24
    assert tuple(tmp_path.iterdir()) == before

    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    with pytest.raises(ProtocolError, match="launch through workspace run"):
        workstation.inspect_harp_v3_workstation(config.runtime)

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setattr(
        workstation,
        "_nvidia_smi_rows",
        lambda: tuple(
            {
                "index": index,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24_576,
                "memory_free_mib": 24_000,
            }
            for index in (0, 0, 1)
        ),
    )
    with pytest.raises(ProtocolError, match="exactly CUDA devices"):
        workstation.inspect_harp_v3_workstation(config.runtime)
