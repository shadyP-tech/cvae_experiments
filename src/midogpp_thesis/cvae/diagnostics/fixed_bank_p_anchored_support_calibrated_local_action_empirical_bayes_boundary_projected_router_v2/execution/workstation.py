"""Execution-facing adapters to the frozen v2 workstation contract."""

from __future__ import annotations

import os

from ..protocol import GovernanceError
from ..workstation import (
    BLAS_ENVIRONMENT_NAMES,
    CPU_WORKER_ENV as OUTER_WORKER_ENV,
    WorkstationPlan,
    assert_coordinator_process,
    canonical_workstation_plan,
    initialize_cpu_outer_worker,
)


NO_REFIT_ENV = "MIDOGPP_SCALE_BP_V2_NO_REFIT"
THREAD_ENVIRONMENT = {name: "1" for name in BLAS_ENVIRONMENT_NAMES}


def build_workstation_plan() -> WorkstationPlan:
    return canonical_workstation_plan()


def assert_outer_worker_environment() -> None:
    if (
        os.environ.get(OUTER_WORKER_ENV) != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or any(os.environ.get(key) != value for key, value in THREAD_ENVIRONMENT.items())
    ):
        raise GovernanceError("SCALE-BP v2 outer-worker environment drifted.")


__all__ = (
    "NO_REFIT_ENV",
    "OUTER_WORKER_ENV",
    "THREAD_ENVIRONMENT",
    "WorkstationPlan",
    "assert_coordinator_process",
    "assert_outer_worker_environment",
    "build_workstation_plan",
    "initialize_cpu_outer_worker",
)
