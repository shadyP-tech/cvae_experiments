"""Frozen workstation topology for the planned SCALE-BP experiment."""

from __future__ import annotations

from dataclasses import dataclass
import os

from ..identity import PHYSICAL_CELL_COUNT, SUPPORT_FOLD_COUNT
from ..protocol import ProtocolError


BLAS_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
OUTER_WORKER_ENV = "MIDOGPP_SCALE_BP_OUTER_WORKER"
_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True, slots=True)
class WorkstationPlan:
    profile: str
    gpu_devices: tuple[str, str]
    persistent_gpu_workers: int
    physical_cell_count: int
    cpu_outer_workers: int
    blas_threads_per_worker: int
    support_fold_count: int
    storage_dtype: str
    reduction_dtype: str
    multiprocessing_start_method: str
    nested_pools_allowed: bool
    execution_authorized: bool


def build_workstation_plan() -> WorkstationPlan:
    """Return the immutable plan without probing or starting hardware."""

    return WorkstationPlan(
        profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        gpu_devices=("cuda:0", "cuda:1"),
        persistent_gpu_workers=2,
        physical_cell_count=PHYSICAL_CELL_COUNT,
        cpu_outer_workers=4,
        blas_threads_per_worker=1,
        support_fold_count=SUPPORT_FOLD_COUNT,
        storage_dtype="float32",
        reduction_dtype="float64",
        multiprocessing_start_method="spawn",
        nested_pools_allowed=False,
        execution_authorized=False,
    )


def initialize_cpu_outer_worker() -> None:
    """Hide CUDA and cap native libraries before scientific imports execute."""

    if os.environ.get(OUTER_WORKER_ENV) == "1":
        raise ProtocolError("SCALE-BP nested outer worker initialization refused.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    os.environ[OUTER_WORKER_ENV] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - workstation dependency
        raise ProtocolError("SCALE-BP requires threadpoolctl.") from exc
    global _THREADPOOL_LIMITER
    _THREADPOOL_LIMITER = threadpool_limits(limits=1)


def assert_coordinator_process() -> None:
    if os.environ.get(OUTER_WORKER_ENV) == "1":
        raise ProtocolError("SCALE-BP nested process pools are forbidden.")


__all__ = (
    "BLAS_ENVIRONMENT_NAMES",
    "OUTER_WORKER_ENV",
    "WorkstationPlan",
    "assert_coordinator_process",
    "build_workstation_plan",
    "initialize_cpu_outer_worker",
)
