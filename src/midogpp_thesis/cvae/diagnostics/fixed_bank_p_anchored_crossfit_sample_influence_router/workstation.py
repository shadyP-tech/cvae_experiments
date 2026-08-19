"""Deterministic workstation scheduling and phase isolation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Iterator, Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    BLAS_THREADS_PER_CPU_WORKER,
    CENTERS,
    CPU_WORKERS,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    GPU_DEVICES,
    PERSISTENT_GPU_WORKERS,
)


BLAS_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
_PARENT_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True, order=True)
class CenterWorkload:
    center: str
    case_count: int
    outer_state_count: int
    double_exclusion_state_count: int
    endpoint_model_fit_count: int

    @property
    def state_count(self) -> int:
        return self.outer_state_count + self.double_exclusion_state_count


def center_workloads(case_counts: Mapping[str, int]) -> tuple[CenterWorkload, ...]:
    if tuple(case_counts) != CENTERS or any(int(value) < 3 for value in case_counts.values()):
        raise ProtocolError("Workstation case counts must cover all centers in order.")
    return tuple(
        CenterWorkload(
            center,
            int(case_counts[center]),
            int(case_counts[center]),
            0,
            16 * int(case_counts[center]),
        )
        for center in CENTERS
    )


def schedule_center_batches(
    case_counts: Mapping[str, int], *, workers: int = CPU_WORKERS
) -> tuple[tuple[CenterWorkload, ...], ...]:
    """Greedy LPT batches balance outer-case fits without tensor IPC."""

    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= len(CENTERS):
        raise ProtocolError("Workstation worker count drifted.")
    batches: list[list[CenterWorkload]] = [[] for _ in range(int(workers))]
    loads = [0] * int(workers)
    for workload in sorted(
        center_workloads(case_counts),
        key=lambda row: (-row.endpoint_model_fit_count, CENTERS.index(row.center)),
    ):
        worker = min(range(int(workers)), key=lambda index: (loads[index], index))
        batches[worker].append(workload)
        loads[worker] += workload.endpoint_model_fit_count
    return tuple(tuple(batch) for batch in batches)


def assert_canonical_workload(case_counts: Mapping[str, int]) -> None:
    rows = center_workloads(case_counts)
    if (
        sum(row.outer_state_count for row in rows) != EXPECTED_OUTER_PLAN_COUNT
        or sum(row.endpoint_model_fit_count for row in rows)
        != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
    ):
        raise ProtocolError("Canonical PCSI outer-endpoint workload drifted.")


def assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        tuple(runtime.get("generation_devices", ())) != GPU_DEVICES
        or int(runtime.get("persistent_generation_worker_count", -1))
        != PERSISTENT_GPU_WORKERS
        or int(runtime.get("route_model_workers", -1)) != CPU_WORKERS
        or int(runtime.get("classifier_threads_per_worker", -1))
        != BLAS_THREADS_PER_CPU_WORKER
        or int(runtime.get("target_posterior_threads_per_worker", -1)) != 1
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("probability_storage_dtype") != "float32"
        or runtime.get("confusion_count_dtype") != "int64"
        or runtime.get("scientific_reductions_dtype") != "float64"
        or int(runtime.get("expected_outer_endpoint_model_fit_count", -1))
        != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        or int(runtime.get("expected_utility_model_fit_count", -1))
        != EXPECTED_UTILITY_MODEL_FIT_COUNT
        or int(runtime.get("expected_target_posterior_model_fit_count", -1))
        != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or int(runtime.get("double_exclusion_state_count", -1)) != 0
    ):
        raise ProtocolError("PCSI workstation runtime contract drifted.")


def enter_cuda_free_cpu_phase() -> None:
    """Permanently hide CUDA and bind the orchestration process to one BLAS thread."""

    global _PARENT_THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ProtocolError("PCSI runtime lacks threadpoolctl.") from exc
    _PARENT_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def assert_cuda_free_cpu_phase() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("PCSI CPU parent still exposes CUDA.")
    import sys

    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("PCSI parent initialized CUDA.")
    from threadpoolctl import threadpool_info

    pools = tuple(row for row in threadpool_info() if row.get("user_api") == "blas")
    if pools and any(int(row.get("num_threads", -1)) != 1 for row in pools):
        raise ProtocolError("PCSI parent BLAS topology is not one thread.")


@contextmanager
def cpu_phase_environment(
    threads_per_worker: int = BLAS_THREADS_PER_CPU_WORKER,
) -> Iterator[None]:
    """Hide CUDA and bind each spawned CPU worker to exactly three BLAS threads."""

    if isinstance(threads_per_worker, bool) or int(threads_per_worker) != threads_per_worker or int(threads_per_worker) <= 0:
        raise ProtocolError("CPU phase thread count drifted.")
    names = ("CUDA_VISIBLE_DEVICES", *BLAS_ENVIRONMENT_NAMES)
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for name in BLAS_ENVIRONMENT_NAMES:
            os.environ[name] = str(int(threads_per_worker))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = (
    "CenterWorkload",
    "assert_canonical_workload",
    "assert_cuda_free_cpu_phase",
    "assert_runtime",
    "center_workloads",
    "cpu_phase_environment",
    "enter_cuda_free_cpu_phase",
    "schedule_center_batches",
)
