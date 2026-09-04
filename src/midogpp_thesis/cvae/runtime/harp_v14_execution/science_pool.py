"""CUDA-blind, one-BLAS-thread execution for HARP v14 source science."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Generic, TypeVar

from ...protocol import ProtocolError


T = TypeVar("T")
R = TypeVar("R")
_THREADPOOL_CONTROLLER: object | None = None


@dataclass(frozen=True, slots=True)
class SciencePoolReceipt(Generic[R]):
    values: tuple[R, ...]
    worker_count: int
    threads_per_worker: int
    batch_ordinals: tuple[tuple[int, ...], ...]
    cuda_visible_to_workers: bool = False
    nested_pools_used: bool = False


def lpt_batches(weights: Sequence[int], *, workers: int) -> tuple[tuple[int, ...], ...]:
    """Deterministic longest-processing-time batches with stable tie breaks."""

    if type(workers) is not int or workers <= 0:
        raise ProtocolError("HARP v14 science worker count is invalid.")
    normalized = tuple(int(value) for value in weights)
    if any(value <= 0 for value in normalized):
        raise ProtocolError("HARP v14 science job weights must be positive.")
    bins: list[list[int]] = [[] for _ in range(min(workers, max(1, len(normalized))))]
    totals = [0 for _ in bins]
    for ordinal in sorted(range(len(normalized)), key=lambda i: (-normalized[i], i)):
        destination = min(range(len(bins)), key=lambda i: (totals[i], i))
        bins[destination].append(ordinal)
        totals[destination] += normalized[ordinal]
    return tuple(tuple(sorted(batch)) for batch in bins)


def initialize_science_worker(threads: int = 1) -> None:
    """Initialize a spawned worker before NumPy/BLAS parallel work begins."""

    global _THREADPOOL_CONTROLLER
    if threads != 1:
        raise ProtocolError("HARP v14 science workers require one BLAS thread.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("HARP v14 science workers require threadpoolctl.") from exc
    _THREADPOOL_CONTROLLER = threadpool_limits(limits=1)


def _run_batch(payload: tuple[Callable[[T], R], tuple[tuple[int, T], ...]]) -> tuple[tuple[int, R], ...]:
    worker, indexed = payload
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("HARP v14 science worker can see CUDA devices.")
    return tuple((ordinal, worker(task)) for ordinal, task in indexed)


def execute_science_jobs(
    tasks: Sequence[T],
    worker: Callable[[T], R],
    *,
    weights: Sequence[int] | None = None,
    workers: int = 4,
    threads_per_worker: int = 1,
) -> SciencePoolReceipt[R]:
    """Run complete outer-H/direction jobs without nested executors."""

    typed = tuple(tasks)
    if not typed:
        return SciencePoolReceipt((), 0, threads_per_worker, ())
    job_weights = tuple(1 for _ in typed) if weights is None else tuple(weights)
    if len(job_weights) != len(typed) or threads_per_worker != 1:
        raise ProtocolError("HARP v14 science pool contract drifted.")
    batches = lpt_batches(job_weights, workers=workers)
    payloads = tuple(
        (worker, tuple((ordinal, typed[ordinal]) for ordinal in batch))
        for batch in batches
    )
    with ProcessPoolExecutor(
        max_workers=len(batches),
        mp_context=mp.get_context("spawn"),
        initializer=initialize_science_worker,
        initargs=(threads_per_worker,),
    ) as pool:
        completed = tuple(pool.map(_run_batch, payloads))
    by_ordinal = {ordinal: result for batch in completed for ordinal, result in batch}
    if set(by_ordinal) != set(range(len(typed))):
        raise ProtocolError("HARP v14 science pool returned incomplete coverage.")
    return SciencePoolReceipt(
        values=tuple(by_ordinal[index] for index in range(len(typed))),
        worker_count=len(batches),
        threads_per_worker=threads_per_worker,
        batch_ordinals=batches,
    )


def science_pool_plan(runtime: Mapping[str, object]) -> Mapping[str, object]:
    if (
        runtime.get("science_workers") != 4
        or runtime.get("science_blas_threads_per_worker") != 1
        or runtime.get("multiprocessing_start_method") != "spawn"
    ):
        raise ProtocolError("HARP v14 science pool runtime drifted.")
    return {
        "schema_version": "midogpp_harp_v14_science_pool_plan_v1",
        "workers": 4,
        "blas_threads_per_worker": 1,
        "cuda_visible_to_workers": False,
        "nested_pools_used": False,
        "scheduling": "deterministic_lpt_outer_target_direction_batches",
    }


__all__ = (
    "SciencePoolReceipt",
    "execute_science_jobs",
    "initialize_science_worker",
    "lpt_batches",
    "science_pool_plan",
)
