"""Spawned, center-batched construction of all nested endpoint states."""

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
    DIRECTION_IDS,
    EXPECTED_ENDPOINT_MODEL_FIT_COUNT,
    candidate_sources,
)
from .contracts import BinaryLabel, CandidateDescriptor, EndpointCasePrediction
from .endpoint_reconstruction import (
    CenterCaseOutcomes,
    EndpointState,
    PreparedCenter,
    fit_endpoint_state_from_outcomes,
    rebind_endpoint_state_priors,
    reconstruct_case_endpoints,
)
from .nested_endpoint_regret import build_candidate_descriptor
from .split_plans import UnorderedPairPlan, WholeCaseOuterPlan
from .workstation import BLAS_ENVIRONMENT_NAMES, schedule_center_batches


_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class CenterEndpointJob:
    target_center: str
    prepared: PreparedCenter
    outcomes: CenterCaseOutcomes
    labels_by_case: tuple[tuple[str, tuple[BinaryLabel, ...]], ...]
    outer_plans: tuple[WholeCaseOuterPlan, ...]
    pair_plans: tuple[UnorderedPairPlan, ...]
    donor_priors: tuple[tuple[tuple[str, str], float], ...]

    def __post_init__(self) -> None:
        cases = tuple(case for case, _labels in self.labels_by_case)
        expected_prior_keys = tuple(
            (source, direction)
            for source in candidate_sources(self.target_center)
            for direction in DIRECTION_IDS
        )
        if (
            self.target_center not in CENTERS
            or self.prepared.surface.center != self.target_center
            or self.outcomes.center != self.target_center
            or set(cases) != set(self.prepared.cases)
            or self.outcomes.case_ids != tuple(sorted(cases))
            or tuple(plan.case_id for plan in self.outer_plans) != cases
            or any(plan.target_center != self.target_center for plan in self.outer_plans)
            or any(plan.target_center != self.target_center for plan in self.pair_plans)
            or tuple(key for key, _value in self.donor_priors) != expected_prior_keys
        ):
            raise ProtocolError("Center endpoint worker job topology drifted.")


@dataclass(frozen=True)
class CenterEndpointProducts:
    target_center: str
    outer_predictions: tuple[EndpointCasePrediction, ...]
    descriptors: tuple[CandidateDescriptor, ...]
    outer_states: tuple[tuple[str, EndpointState], ...]
    pair_states: tuple[tuple[str, str, EndpointState], ...]
    outer_state_hashes: tuple[tuple[str, str], ...]
    pair_state_hashes: tuple[tuple[str, str, str], ...]
    endpoint_model_fit_count: int
    ordered_voter_count: int


def compute_center_endpoint_products(
    job: CenterEndpointJob,
) -> CenterEndpointProducts:
    """Pure center job shared by production workers and deterministic replay."""

    labels_by_case = dict(job.labels_by_case)
    priors = dict(job.donor_priors)
    outer_predictions: dict[str, EndpointCasePrediction] = {}
    outer_states: list[tuple[str, EndpointState]] = []
    outer_hashes: list[tuple[str, str]] = []
    fit_count = 0
    for plan in job.outer_plans:
        state = fit_endpoint_state_from_outcomes(
            job.prepared,
            support_case_ids=plan.support_case_ids,
            outcomes=job.outcomes.subset(plan.support_case_ids),
            donor_priors=priors,
        )
        outer_predictions[plan.case_id] = reconstruct_case_endpoints(
            job.prepared, state, evaluation_case_id=plan.case_id
        )
        outer_hashes.append((plan.case_id, state.state_hash))
        outer_states.append((plan.case_id, state))
        fit_count += state.model_fit_count

    pair_predictions: dict[
        tuple[str, str], tuple[EndpointCasePrediction, EndpointCasePrediction]
    ] = {}
    pair_hashes: list[tuple[str, str, str]] = []
    pair_states: list[tuple[str, str, EndpointState]] = []
    for plan in job.pair_plans:
        state = fit_endpoint_state_from_outcomes(
            job.prepared,
            support_case_ids=plan.support_case_ids,
            outcomes=job.outcomes.subset(plan.support_case_ids),
            donor_priors=priors,
        )
        first = reconstruct_case_endpoints(
            job.prepared, state, evaluation_case_id=plan.first_case_id
        )
        second = reconstruct_case_endpoints(
            job.prepared, state, evaluation_case_id=plan.second_case_id
        )
        pair_predictions[(plan.first_case_id, plan.second_case_id)] = (first, second)
        pair_hashes.append(
            (plan.first_case_id, plan.second_case_id, state.state_hash)
        )
        pair_states.append((plan.first_case_id, plan.second_case_id, state))
        fit_count += state.model_fit_count

    descriptors: list[CandidateDescriptor] = []
    for plan in job.outer_plans:
        nested: dict[str, EndpointCasePrediction] = {}
        for voter in plan.support_case_ids:
            first, second = sorted((plan.case_id, voter))
            pair = pair_predictions[(first, second)]
            nested[voter] = pair[0] if first == voter else pair[1]
        support_labels = tuple(
            label
            for voter in plan.support_case_ids
            for label in labels_by_case[voter]
        )
        descriptors.append(
            build_candidate_descriptor(
                target_center=job.target_center,
                outer_case_id=plan.case_id,
                outer_prediction=outer_predictions[plan.case_id],
                nested_voter_predictions=nested,
                support_labels=support_labels,
            )
        )
    expected_fit_count = 16 * (len(job.outer_plans) + len(job.pair_plans))
    if (
        fit_count != expected_fit_count
        or len(descriptors) != len(job.outer_plans)
        or len(pair_predictions) != len(job.pair_plans)
    ):
        raise ProtocolError("Center endpoint worker workload drifted.")
    return CenterEndpointProducts(
        job.target_center,
        tuple(outer_predictions[plan.case_id] for plan in job.outer_plans),
        tuple(descriptors),
        tuple(outer_states),
        tuple(pair_states),
        tuple(outer_hashes),
        tuple(pair_hashes),
        fit_count,
        2 * len(job.pair_plans),
    )


def recompose_center_endpoint_products(
    job: CenterEndpointJob,
    fitted: CenterEndpointProducts,
    *,
    donor_priors: Mapping[tuple[str, str], float],
) -> CenterEndpointProducts:
    """Rebind external priors without repeating any IRLS model fit."""

    if fitted.target_center != job.target_center:
        raise ProtocolError("Prior rebind center does not match its fitted basis.")
    labels_by_case = dict(job.labels_by_case)
    outer_states = tuple(
        (case, rebind_endpoint_state_priors(state, donor_priors))
        for case, state in fitted.outer_states
    )
    pair_states = tuple(
        (first, second, rebind_endpoint_state_priors(state, donor_priors))
        for first, second, state in fitted.pair_states
    )
    outer_predictions = {
        case: reconstruct_case_endpoints(
            job.prepared, state, evaluation_case_id=case
        )
        for case, state in outer_states
    }
    pair_predictions = {
        (first, second): (
            reconstruct_case_endpoints(
                job.prepared, state, evaluation_case_id=first
            ),
            reconstruct_case_endpoints(
                job.prepared, state, evaluation_case_id=second
            ),
        )
        for first, second, state in pair_states
    }
    descriptors: list[CandidateDescriptor] = []
    for plan in job.outer_plans:
        nested: dict[str, EndpointCasePrediction] = {}
        for voter in plan.support_case_ids:
            first, second = sorted((plan.case_id, voter))
            pair = pair_predictions[(first, second)]
            nested[voter] = pair[0] if voter == first else pair[1]
        descriptors.append(
            build_candidate_descriptor(
                target_center=job.target_center,
                outer_case_id=plan.case_id,
                outer_prediction=outer_predictions[plan.case_id],
                nested_voter_predictions=nested,
                support_labels=tuple(
                    label
                    for voter in plan.support_case_ids
                    for label in labels_by_case[voter]
                ),
            )
        )
    return CenterEndpointProducts(
        job.target_center,
        tuple(outer_predictions[plan.case_id] for plan in job.outer_plans),
        tuple(descriptors),
        outer_states,
        pair_states,
        tuple((case, state.state_hash) for case, state in outer_states),
        tuple(
            (first, second, state.state_hash)
            for first, second, state in pair_states
        ),
        0,
        2 * len(pair_states),
    )


def execute_center_endpoint_jobs(
    jobs: Sequence[CenterEndpointJob],
    *,
    workers: int = CPU_WORKERS,
    threads_per_worker: int = BLAS_THREADS_PER_CPU_WORKER,
    use_processes: bool = True,
) -> tuple[CenterEndpointProducts, ...]:
    """Run deterministic LPT center batches with no repeated tensor IPC."""

    rows = tuple(jobs)
    if (
        tuple(job.target_center for job in rows) != CENTERS
        or workers != CPU_WORKERS
        or threads_per_worker != BLAS_THREADS_PER_CPU_WORKER
    ):
        raise ProtocolError("Endpoint worker launch topology drifted.")
    if not use_processes:
        results = tuple(compute_center_endpoint_products(job) for job in rows)
    else:
        counts = {job.target_center: len(job.outer_plans) for job in rows}
        by_center = {job.target_center: job for job in rows}
        scheduled = schedule_center_batches(counts, workers=workers)
        batches = tuple(
            tuple(by_center[work.center] for work in batch) for batch in scheduled
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(threads_per_worker,),
        ) as executor:
            nested = tuple(executor.map(_execute_batch, batches, chunksize=1))
        by_result = {row.target_center: row for batch in nested for row in batch}
        results = tuple(by_result[center] for center in CENTERS)
    if (
        tuple(row.target_center for row in results) != CENTERS
        or sum(row.endpoint_model_fit_count for row in results)
        != EXPECTED_ENDPOINT_MODEL_FIT_COUNT
    ):
        raise ProtocolError("Global endpoint worker result topology drifted.")
    return results


def _initialize_worker(threads: int) -> None:
    global _THREADPOOL_LIMITER
    if threads != BLAS_THREADS_PER_CPU_WORKER:
        raise ProtocolError("Endpoint worker BLAS thread count drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ProtocolError("Endpoint worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=threads)


def _execute_batch(
    batch: Sequence[CenterEndpointJob],
) -> tuple[CenterEndpointProducts, ...]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Endpoint worker unexpectedly exposes CUDA.")
    return tuple(compute_center_endpoint_products(job) for job in batch)


__all__ = (
    "CenterEndpointJob",
    "CenterEndpointProducts",
    "compute_center_endpoint_products",
    "execute_center_endpoint_jobs",
    "recompose_center_endpoint_products",
)
