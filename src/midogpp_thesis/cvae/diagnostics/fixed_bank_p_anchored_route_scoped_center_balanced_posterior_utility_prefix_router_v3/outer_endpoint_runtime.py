"""Center-batched H-c endpoint reconstruction without unused nested voters."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    BLAS_THREADS_PER_CPU_WORKER,
    CENTERS,
    CPU_WORKERS,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
)
from .contracts import (
    CenterProbabilitySurface,
    EndpointCasePrediction,
    PhysicalProbabilitySurface,
)
from .endpoint_reconstruction import (
    CenterCaseOutcomes,
    EndpointState,
    PreparedCenter,
    fit_endpoint_state_from_outcomes,
    rebind_endpoint_state_priors,
    reconstruct_case_endpoints,
)
from .endpoint_preparation import prepare_center
from .hashing import require_sha256
from .outer_plans import WholeCaseOuterPlan
from .workstation import BLAS_ENVIRONMENT_NAMES


_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class OuterEndpointJob:
    target_center: str
    physical_surface_hash: str
    center_surface_hash: str
    prepared: PreparedCenter
    route_outcomes: tuple[tuple[str, CenterCaseOutcomes], ...]
    outer_plans: tuple[WholeCaseOuterPlan, ...]
    donor_priors: tuple[tuple[tuple[str, str], float], ...]

    def __post_init__(self) -> None:
        _validate_job_lineage(self)


@dataclass(frozen=True)
class OuterEndpointProducts:
    target_center: str
    physical_surface_hash: str
    center_surface_hash: str
    predictions: tuple[EndpointCasePrediction, ...]
    states: tuple[tuple[str, EndpointState], ...]
    state_hashes: tuple[tuple[str, str], ...]
    endpoint_model_fit_count: int

    def __post_init__(self) -> None:
        _validate_surface_hash_pair(
            self.physical_surface_hash,
            self.center_surface_hash,
            role="endpoint products",
        )
        if self.target_center not in CENTERS:
            raise ProtocolError("CBPUPR endpoint product target drifted.")


def build_outer_endpoint_job(
    surface: PhysicalProbabilitySurface,
    *,
    target_center: str,
    prepared: PreparedCenter,
    route_outcomes: Sequence[tuple[str, CenterCaseOutcomes]],
    outer_plans: Sequence[WholeCaseOuterPlan],
    donor_priors: Mapping[tuple[str, str], float],
) -> OuterEndpointJob:
    """Bind one center-local endpoint job to the global physical surface.

    This is the production construction boundary.  The global hash binds the
    sealed whole-case plans, while the center hash binds the arrays actually
    sent to the endpoint worker.  They are deliberately separate contracts.
    """

    target = str(target_center)
    try:
        center_surface = surface.centers[target]
    except KeyError as exc:
        raise ProtocolError("CBPUPR endpoint job factory target drifted.") from exc
    if (
        prepared.surface.center != target
        or prepared.surface.surface_hash != center_surface.surface_hash
        or prepared.surface.probability_store_hash != surface.probability_store_hash
    ):
        raise ProtocolError("CBPUPR endpoint job factory surface lineage drifted.")
    return OuterEndpointJob(
        target_center=target,
        physical_surface_hash=surface.surface_hash,
        center_surface_hash=center_surface.surface_hash,
        prepared=prepared,
        route_outcomes=tuple(route_outcomes),
        outer_plans=tuple(outer_plans),
        donor_priors=tuple(
            ((str(source), str(direction)), float(value))
            for (source, direction), value in donor_priors.items()
        ),
    )


def compute_outer_endpoint_products(job: OuterEndpointJob) -> OuterEndpointProducts:
    _validate_job_lineage(job)
    priors = dict(job.donor_priors)
    outcomes_by_case = dict(job.route_outcomes)
    if (
        tuple(outcomes_by_case) != tuple(plan.case_id for plan in job.outer_plans)
        or any(
            outcomes_by_case[plan.case_id].case_ids != plan.support_case_ids
            for plan in job.outer_plans
        )
    ):
        raise ProtocolError("CBPUPR route-scoped endpoint capabilities drifted.")
    states: list[tuple[str, EndpointState]] = []
    predictions: list[EndpointCasePrediction] = []
    fit_count = 0
    for plan in job.outer_plans:
        state = fit_endpoint_state_from_outcomes(
            job.prepared,
            support_case_ids=plan.support_case_ids,
            outcomes=outcomes_by_case[plan.case_id],
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
        raise ProtocolError("CBPUPR outer endpoint workload drifted.")
    return OuterEndpointProducts(
        target_center=job.target_center,
        physical_surface_hash=job.physical_surface_hash,
        center_surface_hash=job.center_surface_hash,
        predictions=tuple(predictions),
        states=tuple(states),
        state_hashes=tuple((case, state.state_hash) for case, state in states),
        endpoint_model_fit_count=fit_count,
    )


def recompose_outer_endpoint_products(
    job: OuterEndpointJob,
    fitted: OuterEndpointProducts,
    *,
    donor_priors: Mapping[tuple[str, str], float],
    excluded_source_centers: Sequence[str] = (),
) -> OuterEndpointProducts:
    """Rebind outer-excluded priors without repeating endpoint model fits."""

    if fitted.target_center != job.target_center:
        raise ProtocolError("CBPUPR prior rebind target drifted.")
    if (
        fitted.physical_surface_hash != job.physical_surface_hash
        or fitted.center_surface_hash != job.center_surface_hash
    ):
        raise ProtocolError("CBPUPR prior rebind surface lineage drifted.")
    states = tuple(
        (
            case,
            rebind_endpoint_state_priors(
                state,
                donor_priors,
                excluded_source_centers=excluded_source_centers,
            ),
        )
        for case, state in fitted.states
    )
    return OuterEndpointProducts(
        target_center=job.target_center,
        physical_surface_hash=job.physical_surface_hash,
        center_surface_hash=job.center_surface_hash,
        predictions=tuple(
            reconstruct_case_endpoints(
                job.prepared,
                state,
                evaluation_case_id=case,
            )
            for case, state in states
        ),
        states=states,
        state_hashes=tuple((case, state.state_hash) for case, state in states),
        endpoint_model_fit_count=0,
    )


def execute_outer_endpoint_jobs(
    jobs: Sequence[OuterEndpointJob],
    *,
    use_processes: bool = True,
) -> tuple[OuterEndpointProducts, ...]:
    rows = tuple(jobs)
    if tuple(job.target_center for job in rows) != CENTERS:
        raise ProtocolError("CBPUPR endpoint job order drifted.")
    for job in rows:
        _validate_job_lineage(job)
    if len({job.physical_surface_hash for job in rows}) != 1:
        raise ProtocolError("CBPUPR endpoint global surface lineage drifted.")
    if not use_processes:
        results = tuple(compute_outer_endpoint_products(job) for job in rows)
    else:
        with ProcessPoolExecutor(
            max_workers=CPU_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(BLAS_THREADS_PER_CPU_WORKER,),
        ) as executor:
            payloads = tuple(_job_payload(row) for row in rows)
            raw = tuple(
                executor.map(_compute_outer_endpoint_payload, payloads, chunksize=1)
            )
        unordered = tuple(_products_from_payload(row) for row in raw)
        by_center = {row.target_center: row for row in unordered}
        results = tuple(by_center[center] for center in CENTERS)
    if sum(row.endpoint_model_fit_count for row in results) != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT:
        raise ProtocolError("CBPUPR global outer endpoint workload drifted.")
    for job, products in zip(rows, results, strict=True):
        _validate_products_against_job(job, products)
    return results


def _validate_products_against_job(
    job: OuterEndpointJob,
    products: OuterEndpointProducts,
) -> None:
    expected_cases = tuple(plan.case_id for plan in job.outer_plans)
    prediction_cases = tuple(row.case_id for row in products.predictions)
    state_cases = tuple(case for case, _state in products.states)
    state_hash_cases = tuple(case for case, _digest in products.state_hashes)
    if (
        products.target_center != job.target_center
        or products.physical_surface_hash != job.physical_surface_hash
        or products.center_surface_hash != job.center_surface_hash
        or prediction_cases != expected_cases
        or state_cases != expected_cases
        or state_hash_cases != expected_cases
    ):
        raise ProtocolError("CBPUPR endpoint worker result topology drifted.")
    for plan, prediction, (case, state) in zip(
        job.outer_plans,
        products.predictions,
        products.states,
        strict=True,
    ):
        if (
            case != plan.case_id
            or state.target_center != job.target_center
            or state.support_case_ids != plan.support_case_ids
            or prediction.center != job.target_center
            or prediction.sample_ids != plan.evaluation_sample_ids
            or plan.probability_surface_hash != job.physical_surface_hash
        ):
            raise ProtocolError("CBPUPR endpoint worker plan lineage drifted.")
        replay = reconstruct_case_endpoints(
            job.prepared,
            state,
            evaluation_case_id=case,
        )
        if (
            replay.prediction_hash != prediction.prediction_hash
            or replay.sample_ids != prediction.sample_ids
            or any(
                np.asarray(replay.probabilities[method], dtype=np.float64).tobytes(
                    order="C"
                )
                != np.asarray(
                    prediction.probabilities[method], dtype=np.float64
                ).tobytes(order="C")
                for method in replay.probabilities
            )
        ):
            raise ProtocolError("CBPUPR endpoint worker replay drifted.")


def _job_payload(job: OuterEndpointJob) -> dict[str, object]:
    """Strip every MappingProxyType before crossing the spawn boundary."""

    _validate_job_lineage(job)
    surface = job.prepared.surface
    return {
        "target_center": job.target_center,
        "physical_surface_hash": job.physical_surface_hash,
        "center_surface_hash": job.center_surface_hash,
        "surface": {
            "center": surface.center,
            "sample_ids": surface.sample_ids,
            "case_ids": surface.case_ids,
            "seed_probabilities": tuple(
                (action, np.asarray(values, dtype=np.float32))
                for action, values in surface.seed_probabilities.items()
            ),
            "probability_store_hash": surface.probability_store_hash,
            "surface_hash": surface.surface_hash,
        },
        "route_outcomes": tuple(
            (
                case,
                {
                    "center": outcomes.center,
                    "case_ids": outcomes.case_ids,
                    "successes": np.asarray(outcomes.successes, dtype=np.int64),
                    "trials": np.asarray(outcomes.trials, dtype=np.int64),
                    "n_positive": np.asarray(outcomes.n_positive, dtype=np.int64),
                    "n_negative": np.asarray(outcomes.n_negative, dtype=np.int64),
                },
            )
            for case, outcomes in job.route_outcomes
        ),
        "outer_plans": tuple(row.to_payload() for row in job.outer_plans),
        "donor_priors": tuple(
            ((str(source), str(direction)), float(value))
            for (source, direction), value in job.donor_priors
        ),
    }


def _job_from_payload(raw: dict[str, object]) -> OuterEndpointJob:
    surface_raw = raw["surface"]
    if not isinstance(surface_raw, dict):
        raise ProtocolError("CBPUPR endpoint worker input payload drifted.")
    surface = CenterProbabilitySurface(
        str(surface_raw["center"]),
        tuple(surface_raw["sample_ids"]),
        tuple(surface_raw["case_ids"]),
        {str(action): np.asarray(values, dtype=np.float32) for action, values in surface_raw["seed_probabilities"]},
        str(surface_raw["probability_store_hash"]),
    )
    if surface.surface_hash != surface_raw.get("surface_hash"):
        raise ProtocolError("CBPUPR endpoint worker surface hash drifted.")
    plans: list[WholeCaseOuterPlan] = []
    for value in raw["outer_plans"]:
        if not isinstance(value, dict):
            raise ProtocolError("CBPUPR endpoint plan worker payload drifted.")
        plan = WholeCaseOuterPlan(
            str(value["target_center"]),
            str(value["case_id"]),
            str(value["group_id"]),
            tuple(value["support_case_ids"]),
            tuple(value["evaluation_sample_ids"]),
            str(value["probability_surface_hash"]),
        )
        if plan.plan_hash != value.get("plan_hash") or plan.to_payload() != value:
            raise ProtocolError("CBPUPR endpoint plan hash drifted in worker.")
        plans.append(plan)
    return OuterEndpointJob(
        target_center=str(raw["target_center"]),
        physical_surface_hash=str(raw["physical_surface_hash"]),
        center_surface_hash=str(raw["center_surface_hash"]),
        prepared=prepare_center(surface),
        route_outcomes=tuple(
            (
                str(case),
                CenterCaseOutcomes(
                    str(value["center"]),
                    tuple(value["case_ids"]),
                    np.asarray(value["successes"], dtype=np.int64),
                    np.asarray(value["trials"], dtype=np.int64),
                    np.asarray(value["n_positive"], dtype=np.int64),
                    np.asarray(value["n_negative"], dtype=np.int64),
                ),
            )
            for case, value in raw["route_outcomes"]
        ),
        outer_plans=tuple(plans),
        donor_priors=tuple(
            ((str(key[0]), str(key[1])), float(value))
            for key, value in raw["donor_priors"]
        ),
    )


def _compute_outer_endpoint_payload(raw: dict[str, object]) -> dict[str, object]:
    return _products_payload(compute_outer_endpoint_products(_job_from_payload(raw)))


def _products_payload(products: OuterEndpointProducts) -> dict[str, object]:
    return {
        "target_center": products.target_center,
        "physical_surface_hash": products.physical_surface_hash,
        "center_surface_hash": products.center_surface_hash,
        "predictions": tuple(
            {
                "center": row.center,
                "case_id": row.case_id,
                "sample_ids": row.sample_ids,
                "probabilities": tuple(
                    (method, tuple(values)) for method, values in row.probabilities.items()
                ),
                "state_hash": row.state_hash,
                "prediction_hash": row.prediction_hash,
            }
            for row in products.predictions
        ),
        "states": tuple((case, _state_payload(state)) for case, state in products.states),
        "state_hashes": products.state_hashes,
        "endpoint_model_fit_count": products.endpoint_model_fit_count,
    }


def _state_payload(state: EndpointState) -> dict[str, object]:
    return state.to_payload()


def _products_from_payload(raw: dict[str, object]) -> OuterEndpointProducts:
    predictions: list[EndpointCasePrediction] = []
    for value in raw["predictions"]:
        prediction = EndpointCasePrediction(
            str(value["center"]),
            str(value["case_id"]),
            tuple(value["sample_ids"]),
            {str(method): tuple(probabilities) for method, probabilities in value["probabilities"]},
            str(value["state_hash"]),
        )
        if prediction.prediction_hash != value.get("prediction_hash"):
            raise ProtocolError("CBPUPR endpoint worker prediction hash drifted.")
        predictions.append(prediction)
    states = tuple((str(case), _state_from_payload(value)) for case, value in raw["states"])
    state_hashes = tuple(
        (str(case), str(digest)) for case, digest in raw["state_hashes"]
    )
    result = OuterEndpointProducts(
        target_center=str(raw["target_center"]),
        physical_surface_hash=str(raw["physical_surface_hash"]),
        center_surface_hash=str(raw["center_surface_hash"]),
        predictions=tuple(predictions),
        states=states,
        state_hashes=state_hashes,
        endpoint_model_fit_count=int(raw["endpoint_model_fit_count"]),
    )
    prediction_index = {
        (row.center, row.case_id): row for row in result.predictions
    }
    state_index = {
        (state.target_center, case): state for case, state in result.states
    }
    if (
        not states
        or len(prediction_index) != len(predictions)
        or len(state_index) != len(states)
        or set(prediction_index) != set(state_index)
        or any(key[0] != result.target_center for key in state_index)
        or result.endpoint_model_fit_count
        != sum(state.model_fit_count for _case, state in states)
        or any(state.model_fit_count != 16 for _case, state in states)
    ):
        raise ProtocolError("CBPUPR endpoint worker product topology drifted.")
    if result.state_hashes != tuple((case, state.state_hash) for case, state in states):
        raise ProtocolError("CBPUPR endpoint worker state index drifted.")
    if any(
        prediction_index[key].state_hash != state.state_hash
        for key, state in state_index.items()
    ):
        raise ProtocolError("CBPUPR endpoint worker prediction/state lineage drifted.")
    return result


def _state_from_payload(raw: object) -> EndpointState:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR endpoint worker state payload drifted.")
    return EndpointState.from_payload(raw)


def _validate_surface_hash_pair(
    physical_surface_hash: object,
    center_surface_hash: object,
    *,
    role: str,
) -> None:
    physical = require_sha256(
        physical_surface_hash, f"CBPUPR {role} physical_surface_hash"
    )
    center = require_sha256(
        center_surface_hash, f"CBPUPR {role} center_surface_hash"
    )
    if physical == center:
        raise ProtocolError(f"CBPUPR {role} surface hash roles collapsed.")


def _validate_job_lineage(job: OuterEndpointJob) -> None:
    _validate_surface_hash_pair(
        job.physical_surface_hash,
        job.center_surface_hash,
        role="endpoint job",
    )
    plans = tuple(job.outer_plans)
    outcomes = tuple(job.route_outcomes)
    outcome_by_case = dict(outcomes)
    expected_cases = tuple(plan.case_id for plan in plans)
    try:
        drifted = (
            job.target_center not in CENTERS
            or job.prepared.surface.center != job.target_center
            or job.prepared.surface.surface_hash != job.center_surface_hash
            or not plans
            or len(outcome_by_case) != len(outcomes)
            or tuple(outcome_by_case) != expected_cases
            or any(
                plan.target_center != job.target_center
                or plan.probability_surface_hash != job.physical_surface_hash
                or outcome_by_case[plan.case_id].center != job.target_center
                or outcome_by_case[plan.case_id].case_ids != plan.support_case_ids
                or set(plan.support_case_ids) | {plan.case_id}
                != set(job.prepared.cases)
                or plan.evaluation_sample_ids
                != tuple(
                    job.prepared.surface.sample_ids[position]
                    for position in job.prepared.case_positions[plan.case_id]
                )
                for plan in plans
            )
        )
    except (AttributeError, KeyError, TypeError):
        drifted = True
    if drifted:
        raise ProtocolError("CBPUPR endpoint job surface or plan lineage drifted.")


def _initialize_worker(threads: int) -> None:
    global _THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("CBPUPR endpoint worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=threads)


__all__ = (
    "OuterEndpointJob",
    "OuterEndpointProducts",
    "build_outer_endpoint_job",
    "compute_outer_endpoint_products",
    "execute_outer_endpoint_jobs",
    "recompose_outer_endpoint_products",
)
