"""Center-batched H-c endpoint reconstruction without unused nested voters."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    BLAS_THREADS_PER_CPU_WORKER,
    CENTERS,
    CPU_WORKERS,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
)
from .contracts import EndpointCasePrediction
from .endpoint_reconstruction import (
    CenterCaseOutcomes,
    EndpointState,
    PreparedCenter,
    fit_endpoint_state_from_outcomes,
    rebind_endpoint_state_priors,
    reconstruct_case_endpoints,
)
from .outer_plans import WholeCaseOuterPlan
from .workstation import BLAS_ENVIRONMENT_NAMES


_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class OuterEndpointJob:
    target_center: str
    prepared: PreparedCenter
    outcomes: CenterCaseOutcomes
    outer_plans: tuple[WholeCaseOuterPlan, ...]
    donor_priors: tuple[tuple[tuple[str, str], float], ...]


@dataclass(frozen=True)
class OuterEndpointProducts:
    target_center: str
    predictions: tuple[EndpointCasePrediction, ...]
    states: tuple[tuple[str, EndpointState], ...]
    state_hashes: tuple[tuple[str, str], ...]
    endpoint_model_fit_count: int


def compute_outer_endpoint_products(job: OuterEndpointJob) -> OuterEndpointProducts:
    priors = dict(job.donor_priors)
    states: list[tuple[str, EndpointState]] = []
    predictions: list[EndpointCasePrediction] = []
    fit_count = 0
    for plan in job.outer_plans:
        state = fit_endpoint_state_from_outcomes(
            job.prepared,
            support_case_ids=plan.support_case_ids,
            outcomes=job.outcomes.subset(plan.support_case_ids),
            donor_priors=priors,
        )
        states.append((plan.case_id, state))
        predictions.append(
            reconstruct_case_endpoints(
                job.prepared,
                state,
                evaluation_case_id=plan.case_id,
            )
        )
        fit_count += state.model_fit_count
    if fit_count != 16 * len(job.outer_plans):
        raise ProtocolError("PCSI-RACR outer endpoint workload drifted.")
    return OuterEndpointProducts(
        job.target_center,
        tuple(predictions),
        tuple(states),
        tuple((case, state.state_hash) for case, state in states),
        fit_count,
    )


def recompose_outer_endpoint_products(
    job: OuterEndpointJob,
    fitted: OuterEndpointProducts,
    *,
    donor_priors: Mapping[tuple[str, str], float],
) -> OuterEndpointProducts:
    """Rebind outer-excluded priors without repeating endpoint model fits."""

    if fitted.target_center != job.target_center:
        raise ProtocolError("PCSI-RACR prior rebind target drifted.")
    states = tuple(
        (case, rebind_endpoint_state_priors(state, donor_priors))
        for case, state in fitted.states
    )
    return OuterEndpointProducts(
        job.target_center,
        tuple(
            reconstruct_case_endpoints(
                job.prepared,
                state,
                evaluation_case_id=case,
            )
            for case, state in states
        ),
        states,
        tuple((case, state.state_hash) for case, state in states),
        0,
    )


def execute_outer_endpoint_jobs(
    jobs: Sequence[OuterEndpointJob],
    *,
    use_processes: bool = True,
) -> tuple[OuterEndpointProducts, ...]:
    rows = tuple(jobs)
    if tuple(job.target_center for job in rows) != CENTERS:
        raise ProtocolError("PCSI-RACR endpoint job order drifted.")
    if not use_processes:
        results = tuple(compute_outer_endpoint_products(job) for job in rows)
    else:
        with ProcessPoolExecutor(
            max_workers=CPU_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(BLAS_THREADS_PER_CPU_WORKER,),
        ) as executor:
            unordered = tuple(executor.map(compute_outer_endpoint_products, rows, chunksize=1))
        by_center = {row.target_center: row for row in unordered}
        results = tuple(by_center[center] for center in CENTERS)
    if sum(row.endpoint_model_fit_count for row in results) != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT:
        raise ProtocolError("PCSI-RACR global outer endpoint workload drifted.")
    return results


def _initialize_worker(threads: int) -> None:
    global _THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("PCSI-RACR endpoint worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=threads)


__all__ = (
    "OuterEndpointJob",
    "OuterEndpointProducts",
    "compute_outer_endpoint_products",
    "execute_outer_endpoint_jobs",
    "recompose_outer_endpoint_products",
)
