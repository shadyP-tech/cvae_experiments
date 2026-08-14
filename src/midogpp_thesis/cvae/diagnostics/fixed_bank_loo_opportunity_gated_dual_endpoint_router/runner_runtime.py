"""Stable facade over route execution and fail-closed run admission."""

from __future__ import annotations

from .route_worker_runtime import (
    ROUTE_BLAS_THREADS,
    ROUTE_WORKERS,
    RouteJobResult,
    assert_cuda_free_cpu_phase,
    assert_exact_route_blas_topology,
    compute_route_job,
    enter_cuda_free_cpu_phase,
    exact_route_blas_scope,
    execute_route_jobs,
)
from .run_admission import (
    assert_launch_files,
    assert_no_foreign_or_partial_state,
    assert_workspace_resolved_paths,
    exclusive_run_lock,
    observe,
    reject_existing_run_state,
    write_state,
)


__all__ = (
    "ROUTE_BLAS_THREADS",
    "ROUTE_WORKERS",
    "RouteJobResult",
    "assert_cuda_free_cpu_phase",
    "assert_exact_route_blas_topology",
    "assert_launch_files",
    "assert_no_foreign_or_partial_state",
    "assert_workspace_resolved_paths",
    "compute_route_job",
    "enter_cuda_free_cpu_phase",
    "exact_route_blas_scope",
    "execute_route_jobs",
    "exclusive_run_lock",
    "observe",
    "reject_existing_run_state",
    "write_state",
)
