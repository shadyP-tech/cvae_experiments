"""Pure deterministic CPU runtime limits for disagreement-regret development.

This module describes a bounded workstation profile; it does not start workers,
inspect a machine, read environment variables, or expose a runnable entrypoint.
GPU generation and probability materialization are upstream adapter concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

from midogpp_thesis.cvae.protocol import ProtocolError


MAX_WORKERS = 4
MAX_THREADS_PER_WORKER = 3
MAX_TOTAL_THREADS = 12
MAX_DENSE_FIT_BYTES = 512 * 1024 * 1024
CPU_DEVICE = "cpu"


@dataclass(frozen=True, kw_only=True)
class WorkstationRuntime:
    """A deterministic, CPU-only execution budget for an external adapter."""

    workers: int = MAX_WORKERS
    threads_per_worker: int = MAX_THREADS_PER_WORKER
    device: str = CPU_DEVICE
    deterministic: bool = True
    serial_test_override: bool = False

    def __post_init__(self) -> None:
        validate_runtime(self)

    @property
    def total_threads(self) -> int:
        return self.workers * self.threads_per_worker

    def to_payload(self) -> dict[str, object]:
        validate_runtime(self)
        return {
            "schema_version": "midogpp_disagreement_regret_runtime_v1",
            "workers": self.workers,
            "threads_per_worker": self.threads_per_worker,
            "total_threads": self.total_threads,
            "maximum_workers": MAX_WORKERS,
            "maximum_threads_per_worker": MAX_THREADS_PER_WORKER,
            "maximum_total_threads": MAX_TOTAL_THREADS,
            "maximum_dense_fit_bytes": MAX_DENSE_FIT_BYTES,
            "device": CPU_DEVICE,
            "deterministic": True,
            "serial_test_override": self.serial_test_override,
            "gpu_surfaces": "upstream_out_of_scope",
            "starts_workers": False,
        }


def canonical_workstation_runtime(
    *, serial_test_override: bool = False
) -> WorkstationRuntime:
    """Build the locked workstation profile or its explicit serial test form."""

    if type(serial_test_override) is not bool:
        raise ProtocolError("serial_test_override must be an explicit boolean.")
    if serial_test_override:
        return WorkstationRuntime(
            workers=1,
            threads_per_worker=1,
            serial_test_override=True,
        )
    return WorkstationRuntime()


def validate_runtime(runtime: WorkstationRuntime) -> WorkstationRuntime:
    """Fail closed on oversubscription, GPU use, or nondeterministic profiles."""

    if not isinstance(runtime, WorkstationRuntime):
        raise ProtocolError("Runtime must use the locked workstation runtime type.")
    if (
        type(runtime.workers) is not int
        or not 1 <= runtime.workers <= MAX_WORKERS
        or type(runtime.threads_per_worker) is not int
        or not 1 <= runtime.threads_per_worker <= MAX_THREADS_PER_WORKER
        or runtime.workers * runtime.threads_per_worker > MAX_TOTAL_THREADS
    ):
        raise ProtocolError(
            "CPU runtime exceeds the frozen 4-worker x 3-thread, 12-thread budget."
        )
    if runtime.device != CPU_DEVICE:
        raise ProtocolError(
            "The disagreement-regret core is CPU-only; GPU surfaces are upstream."
        )
    if runtime.deterministic is not True:
        raise ProtocolError("The disagreement-regret runtime must be deterministic.")
    if type(runtime.serial_test_override) is not bool:
        raise ProtocolError("serial_test_override must be an explicit boolean.")
    if runtime.serial_test_override and (
        runtime.workers != 1 or runtime.threads_per_worker != 1
    ):
        raise ProtocolError("The serial test override requires an exact 1 x 1 budget.")
    return runtime


def estimate_dense_fit_bytes(
    *,
    pair_count: int,
    design_dimension: int,
    encoded_row_count: int = 0,
) -> int:
    """Conservatively estimate one vectorized pairwise fit's peak core arrays."""

    if (
        type(pair_count) is not int
        or pair_count <= 0
        or type(design_dimension) is not int
        or design_dimension <= 0
        or type(encoded_row_count) is not int
        or encoded_row_count < 0
    ):
        raise ProtocolError("Dense fit dimensions must be positive integers.")
    float_bytes = 8
    # The solver can simultaneously hold the design and a curvature-weighted
    # design temporary.  Candidate-wise encoding is retained once, while the
    # clustered sandwich/Newton path can hold several dense square workspaces.
    pair_designs = 2 * pair_count * design_dimension * float_bytes
    encoded_rows = encoded_row_count * design_dimension * float_bytes
    row_vectors = pair_count * (10 * float_bytes + 128)
    square_workspaces = 12 * design_dimension * design_dimension * float_bytes
    linear_workspaces = 32 * design_dimension * float_bytes
    return (
        pair_designs
        + encoded_rows
        + row_vectors
        + square_workspaces
        + linear_workspaces
    )


def assert_dense_fit_within_budget(
    *,
    pair_count: int,
    design_dimension: int,
    encoded_row_count: int = 0,
) -> int:
    estimate = estimate_dense_fit_bytes(
        pair_count=pair_count,
        design_dimension=design_dimension,
        encoded_row_count=encoded_row_count,
    )
    if estimate > MAX_DENSE_FIT_BYTES:
        raise ProtocolError(
            "Pairwise design exceeds the frozen 512 MiB per-fit workstation budget."
        )
    return estimate


__all__ = (
    "CPU_DEVICE",
    "MAX_THREADS_PER_WORKER",
    "MAX_TOTAL_THREADS",
    "MAX_DENSE_FIT_BYTES",
    "MAX_WORKERS",
    "WorkstationRuntime",
    "canonical_workstation_runtime",
    "estimate_dense_fit_bytes",
    "assert_dense_fit_within_budget",
    "validate_runtime",
)
