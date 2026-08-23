"""Deterministic serial or one-level spawned execution over coarse H jobs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import multiprocessing as mp
import os

from ....protocol import ProtocolError
from ..identity import canonical_hash
from .contracts import WorkerRequest, WorkerResult
from .outer_worker import (
    BLAS_ENVIRONMENT_NAMES,
    WORKER_DEPTH_ENV,
    execute_outer_worker,
    initialize_outer_worker,
)


OuterWorker = Callable[[WorkerRequest], WorkerResult]


@dataclass(frozen=True)
class ExecutionManifest:
    execution_mode: str
    requested_worker_count: int
    threads_per_worker: int
    results: tuple[WorkerResult, ...]
    science_hash: str = field(init=False)
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.results)
        if (
            self.execution_mode not in {"serial", "spawn"}
            or self.requested_worker_count <= 0
            or self.threads_per_worker <= 0
            or not rows
            or tuple(row.ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.outer_center for row in rows}) != len(rows)
        ):
            raise ProtocolError("P-DCAPS execution manifest topology drifted.")
        science_hash = canonical_hash(
            {
                "schema_version": "pdcaps_execution_science_v1",
                "result_hashes": tuple(row.result_hash for row in rows),
                "deterministic_order": tuple(
                    (row.ordinal, row.outer_center) for row in rows
                ),
            }
        )
        object.__setattr__(self, "results", rows)
        object.__setattr__(self, "science_hash", science_hash)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_execution_runtime_v1",
                    "science_hash": science_hash,
                    "execution_mode": self.execution_mode,
                    "requested_worker_count": self.requested_worker_count,
                    "threads_per_worker": self.threads_per_worker,
                    "multiprocessing_start_method": (
                        "spawn" if self.execution_mode == "spawn" else None
                    ),
                    "nested_process_pools_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_execution_manifest_v1",
            "execution_mode": self.execution_mode,
            "requested_worker_count": self.requested_worker_count,
            "threads_per_worker": self.threads_per_worker,
            "multiprocessing_start_method": (
                "spawn" if self.execution_mode == "spawn" else None
            ),
            "nested_process_pools_used": False,
            "results": [row.to_payload() for row in self.results],
            "science_hash": self.science_hash,
            "runtime_hash": self.runtime_hash,
        }


def execute_outer_jobs(
    requests: Sequence[WorkerRequest],
    *,
    use_processes: bool = True,
    max_workers: int = 4,
    worker: OuterWorker = execute_outer_worker,
) -> ExecutionManifest:
    """Run exactly one coarse job per H with no nested pools."""

    rows = tuple(sorted(tuple(requests), key=lambda row: (row.ordinal, row.outer_center)))
    if (
        not rows
        or any(not isinstance(row, WorkerRequest) for row in rows)
        or tuple(row.ordinal for row in rows) != tuple(range(len(rows)))
        or len({row.outer_center for row in rows}) != len(rows)
        or len({row.request_hash for row in rows}) != len(rows)
        or len({row.threads_per_worker for row in rows}) != 1
        or isinstance(max_workers, bool)
        or int(max_workers) <= 0
    ):
        raise ProtocolError("P-DCAPS outer-H job topology drifted.")
    _validate_worker(worker, require_spawn_safe=bool(use_processes))
    threads = rows[0].threads_per_worker
    worker_count = min(int(max_workers), len(rows))
    if use_processes:
        if os.environ.get(WORKER_DEPTH_ENV) is not None:
            raise ProtocolError("P-DCAPS nested process pools are prohibited.")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_outer_worker,
            initargs=(threads,),
        ) as executor:
            raw = tuple(executor.map(worker, rows, chunksize=1))
        mode = "spawn"
    else:
        with _serial_worker_environment(threads):
            raw = tuple(worker(row) for row in rows)
        mode = "serial"
    _validate_results(rows, raw)
    return ExecutionManifest(mode, worker_count, threads, raw)


def _validate_worker(worker: OuterWorker, *, require_spawn_safe: bool) -> None:
    if not callable(worker) or getattr(worker, "__self__", None) is not None:
        raise ProtocolError("P-DCAPS outer worker must be a top-level function.")
    module = str(getattr(worker, "__module__", ""))
    qualname = str(getattr(worker, "__qualname__", ""))
    if (
        not module
        or not qualname
        or "<locals>" in qualname
        or (require_spawn_safe and module == "__main__")
    ):
        raise ProtocolError("P-DCAPS outer worker is not spawn-picklable.")


def _validate_results(
    requests: tuple[WorkerRequest, ...],
    results: tuple[WorkerResult, ...],
) -> None:
    if (
        len(results) != len(requests)
        or any(not isinstance(row, WorkerResult) for row in results)
        or any(
            result.outer_center != request.outer_center
            or result.ordinal != request.ordinal
            or result.request_hash != request.request_hash
            or result.operation != request.operation
            for request, result in zip(requests, results, strict=True)
        )
        or len({row.result_hash for row in results}) != len(results)
    ):
        raise ProtocolError("P-DCAPS outer worker result lineage drifted.")


@contextmanager
def _serial_worker_environment(threads: int) -> Iterator[None]:
    names = ("CUDA_VISIBLE_DEVICES", WORKER_DEPTH_ENV, *BLAS_ENVIRONMENT_NAMES)
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ[WORKER_DEPTH_ENV] = "1"
        for name in BLAS_ENVIRONMENT_NAMES:
            os.environ[name] = str(threads)
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ProtocolError("P-DCAPS runtime lacks threadpoolctl.") from exc
        with threadpool_limits(limits=threads):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = ("ExecutionManifest", "OuterWorker", "execute_outer_jobs")
