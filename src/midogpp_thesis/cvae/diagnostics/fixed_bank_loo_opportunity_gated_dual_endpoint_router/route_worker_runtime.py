"""Spawned route fitting and shared deterministic BLAS replay scope."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Iterator, Mapping, Sequence

from ...protocol import ProtocolError


ROUTE_BLAS_THREADS = 3
ROUTE_WORKERS = 4
_ROUTE_SURFACE: object | None = None
_ROUTE_THREADPOOL_LIMITER: object | None = None
_PARENT_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class RouteJobResult:
    plan: object
    case_action_confusions: tuple[object, ...]
    correctness_observations_primary: tuple[object, ...]
    correctness_observations_permuted: tuple[object, ...]
    model_fits_primary: tuple[object, ...]
    model_fits_permuted: tuple[object, ...]
    denominators: object
    directional_support_gains: tuple[object, ...]
    identification_decisions: tuple[object, ...]
    robust_arm_decisions: tuple[object, ...]


def execute_route_jobs(
    probability_surface: object,
    jobs: Sequence[Mapping[str, object]],
    *,
    workers: int,
    threads_per_worker: int,
) -> tuple[RouteJobResult, ...]:
    tasks = tuple(dict(job) for job in jobs)
    if (
        type(workers) is not int
        or workers != ROUTE_WORKERS
        or type(threads_per_worker) is not int
        or threads_per_worker != ROUTE_BLAS_THREADS
        or not tasks
    ):
        raise ProtocolError("Dual-endpoint route worker topology drifted.")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_route_worker,
        initargs=(probability_surface, threads_per_worker),
    ) as executor:
        results = tuple(executor.map(_execute_route_job, tasks, chunksize=1))
    if len(results) != len(tasks) or any(
        result.plan != tasks[index]["plan"] for index, result in enumerate(results)
    ):
        raise ProtocolError("Dual-endpoint route worker result order drifted.")
    return results


def _initialize_route_worker(surface: object, threads: int) -> None:
    global _ROUTE_SURFACE, _ROUTE_THREADPOOL_LIMITER
    if type(threads) is not int or threads != ROUTE_BLAS_THREADS:
        raise ProtocolError("Dual-endpoint route worker thread count drifted.")
    _install_thread_environment(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("Dual-endpoint worker lacks threadpoolctl.") from exc
    _ROUTE_THREADPOOL_LIMITER = threadpool_limits(limits=threads)
    _ROUTE_SURFACE = surface


def _execute_route_job(job: Mapping[str, object]) -> RouteJobResult:
    if _ROUTE_SURFACE is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Dual-endpoint route worker was not initialized.")
    assert_exact_route_blas_topology()
    return compute_route_job(_ROUTE_SURFACE, job)


def compute_route_job(
    probability_surface: object, job: Mapping[str, object]
) -> RouteJobResult:
    """Pure route computation shared by spawned production and replay."""

    from .candidate_feature_permutation import permute_route_candidate_feature_blocks
    from .correctness_model import fit_route_correctness_models
    from .correctness_observations import (
        score_route_correctness_observations,
        support_class_denominators,
    )
    from .identification import select_case_identification_decision
    from .response_scoring import (
        score_case_action_confusions,
        score_loo_directional_gains,
    )
    from .robust import select_robust_arm_decisions

    plan = job["plan"]
    labels = tuple(job["support_labels"])
    priors = tuple(job["donor_priors"])
    features = tuple(job["route_features"])
    observations = score_route_correctness_observations(
        probability_surface, labels, plan, features=features
    )
    denominators = support_class_denominators(
        labels, plan, probability_surface_or_rows=probability_surface
    )
    models = fit_route_correctness_models(observations, plan)
    identification = (
        select_case_identification_decision(
            plan, features, models, denominators, priors
        ),
    )
    permuted = permute_route_candidate_feature_blocks(features, plan)
    permuted_observations = score_route_correctness_observations(
        probability_surface, labels, plan, features=permuted
    )
    permuted_models = fit_route_correctness_models(permuted_observations, plan)
    identification += (
        select_case_identification_decision(
            plan,
            permuted,
            permuted_models,
            denominators,
            priors,
            method_id="I_FEATURE_BLOCK_PERMUTED",
        ),
    )
    counts = tuple(score_case_action_confusions(probability_surface, labels))
    gains = tuple(score_loo_directional_gains(counts, plan))
    robust = (
        *select_robust_arm_decisions(plan, gains, priors),
        *select_robust_arm_decisions(
            plan, gains, priors, method_id="G_DIRECTIONAL_MATCHED"
        ),
    )
    return RouteJobResult(
        plan,
        counts,
        observations,
        permuted_observations,
        models,
        permuted_models,
        denominators,
        gains,
        identification,
        robust,
    )


@contextmanager
def exact_route_blas_scope(threads: int = ROUTE_BLAS_THREADS) -> Iterator[None]:
    """Shared deterministic numeric scope used by production and replay."""

    if type(threads) is not int or threads != ROUTE_BLAS_THREADS:
        raise ProtocolError("Dual-endpoint fitted numeric thread count drifted.")
    previous = {name: os.environ.get(name) for name in _thread_environment_names()}
    _install_thread_environment(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("Dual-endpoint validation lacks threadpoolctl.") from exc
    try:
        with threadpool_limits(limits=threads):
            assert_exact_route_blas_topology()
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def assert_exact_route_blas_topology() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Dual-endpoint fitted numeric scope exposes CUDA.")
    from threadpoolctl import threadpool_info

    pools = tuple(row for row in threadpool_info() if row.get("user_api") == "blas")
    if pools and any(
        int(row.get("num_threads", -1)) != ROUTE_BLAS_THREADS for row in pools
    ):
        raise ProtocolError("Dual-endpoint fitted BLAS topology is not three threads.")


def enter_cuda_free_cpu_phase() -> None:
    global _PARENT_THREADPOOL_LIMITER
    _install_thread_environment(1)
    from threadpoolctl import threadpool_limits

    _PARENT_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def assert_cuda_free_cpu_phase() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Dual-endpoint CPU parent still exposes CUDA.")
    import sys

    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Dual-endpoint parent initialized CUDA.")
    from threadpoolctl import threadpool_info

    pools = tuple(row for row in threadpool_info() if row.get("user_api") == "blas")
    if pools and any(int(row.get("num_threads", -1)) != 1 for row in pools):
        raise ProtocolError("Dual-endpoint parent BLAS topology is not one thread.")


def _install_thread_environment(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in _thread_environment_names()[1:]:
        os.environ[name] = str(threads)


def _thread_environment_names() -> tuple[str, ...]:
    return (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )


__all__ = (
    "ROUTE_BLAS_THREADS",
    "ROUTE_WORKERS",
    "RouteJobResult",
    "assert_cuda_free_cpu_phase",
    "assert_exact_route_blas_topology",
    "compute_route_job",
    "enter_cuda_free_cpu_phase",
    "exact_route_blas_scope",
    "execute_route_jobs",
)
