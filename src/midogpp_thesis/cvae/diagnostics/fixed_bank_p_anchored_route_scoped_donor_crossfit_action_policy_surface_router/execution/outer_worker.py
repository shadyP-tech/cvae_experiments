"""Top-level, spawn-picklable coarse-H worker entrypoints."""

from __future__ import annotations

import os

from ....protocol import ProtocolError
from ..identity import canonical_hash
from .contracts import WorkerRequest, WorkerResult


HASH_MANIFEST_OPERATION = "HASH_MANIFEST_V1"
WORKER_DEPTH_ENV = "MIDOGPP_PDCAPS_OUTER_WORKER_DEPTH"
BLAS_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)

_THREADPOOL_LIMITER: object | None = None


def initialize_outer_worker(threads_per_worker: int) -> None:
    """Hide CUDA and cap native math pools once in each spawned H worker."""

    global _THREADPOOL_LIMITER
    if (
        isinstance(threads_per_worker, bool)
        or int(threads_per_worker) <= 0
    ):
        raise ProtocolError("P-DCAPS worker thread cap drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ[WORKER_DEPTH_ENV] = "1"
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(int(threads_per_worker))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ProtocolError("P-DCAPS worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=int(threads_per_worker))


def execute_outer_worker(request: WorkerRequest) -> WorkerResult:
    """Reference compact worker used for spawn-contract and manifest jobs.

    Scientific engine workers may wrap this contract with their own top-level
    function, persist their large H-local arrays, and return ``WorkerResult``.
    They must not create another process pool.
    """

    if not isinstance(request, WorkerRequest):
        raise ProtocolError("P-DCAPS outer worker request type drifted.")
    if request.operation != HASH_MANIFEST_OPERATION:
        raise ProtocolError("P-DCAPS outer worker operation is unsupported.")
    payload_hash = canonical_hash(request.payload_entries)
    byte_count = sum(row.nbytes for row in request.arrays)
    return WorkerResult(
        request.outer_center,
        request.ordinal,
        request.request_hash,
        request.operation,
        (
            ("array_count", len(request.arrays)),
            ("byte_count", byte_count),
            ("operation_version", HASH_MANIFEST_OPERATION),
            ("payload_hash", payload_hash),
            ("thread_cap", request.threads_per_worker),
        ),
        tuple((row.name, row.array_hash) for row in request.arrays),
        (),
        len(request.arrays),
    )


__all__ = (
    "BLAS_ENVIRONMENT_NAMES",
    "HASH_MANIFEST_OPERATION",
    "WORKER_DEPTH_ENV",
    "execute_outer_worker",
    "initialize_outer_worker",
)
