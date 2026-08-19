"""Center-batched grouped posterior ensembles for the workstation CPU phase."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CPU_WORKERS,
    SUPPORT_CROSSFIT_FOLD_COUNT,
    TARGET_POSTERIOR_BLAS_THREADS_PER_WORKER,
)
from .contracts import BinaryLabel
from .posterior_contracts import (
    CasePosteriorPrediction,
    PhysicalFingerprintSurface,
    RoutePosteriorEnsemble,
    TargetLocalPosteriorModel,
)
from .support_crossfit import fit_route_posterior_ensemble
from .workstation import BLAS_ENVIRONMENT_NAMES


_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class TargetCenterPosteriorJob:
    target_center: str
    primary_fingerprint: PhysicalFingerprintSurface
    blocked_fingerprint: PhysicalFingerprintSurface
    route_support_labels: tuple[tuple[str, tuple[BinaryLabel, ...]], ...]


@dataclass(frozen=True)
class TargetCenterPosteriorProducts:
    target_center: str
    primary_models: tuple[TargetLocalPosteriorModel, ...]
    primary_predictions: tuple[CasePosteriorPrediction, ...]
    primary_ensembles: tuple[RoutePosteriorEnsemble, ...]
    blocked_models: tuple[TargetLocalPosteriorModel, ...]
    blocked_predictions: tuple[CasePosteriorPrediction, ...]
    blocked_ensembles: tuple[RoutePosteriorEnsemble, ...]
    model_fit_count: int


def compute_target_center_posteriors(
    job: TargetCenterPosteriorJob,
) -> TargetCenterPosteriorProducts:
    """Fit both the identity and blocked five-fold surfaces for one center."""

    if (
        job.target_center != job.primary_fingerprint.center
        or job.target_center != job.blocked_fingerprint.center
        or job.primary_fingerprint.cases != job.blocked_fingerprint.cases
        or tuple(case for case, _labels in job.route_support_labels)
        != job.primary_fingerprint.cases
    ):
        raise ProtocolError("PUMR target posterior center job drifted.")
    primary_models: list[TargetLocalPosteriorModel] = []
    primary_predictions: list[CasePosteriorPrediction] = []
    primary_ensembles: list[RoutePosteriorEnsemble] = []
    blocked_models: list[TargetLocalPosteriorModel] = []
    blocked_predictions: list[CasePosteriorPrediction] = []
    blocked_ensembles: list[RoutePosteriorEnsemble] = []
    for case_id, labels in job.route_support_labels:
        models, predictions, ensemble = fit_route_posterior_ensemble(
            job.primary_fingerprint,
            held_case_id=case_id,
            support_labels=labels,
        )
        primary_models.extend(models)
        primary_predictions.extend(predictions)
        primary_ensembles.append(ensemble)
        models, predictions, ensemble = fit_route_posterior_ensemble(
            job.blocked_fingerprint,
            held_case_id=case_id,
            support_labels=labels,
        )
        blocked_models.extend(models)
        blocked_predictions.extend(predictions)
        blocked_ensembles.append(ensemble)
    expected = 2 * SUPPORT_CROSSFIT_FOLD_COUNT * len(job.route_support_labels)
    return TargetCenterPosteriorProducts(
        job.target_center,
        tuple(primary_models),
        tuple(primary_predictions),
        tuple(primary_ensembles),
        tuple(blocked_models),
        tuple(blocked_predictions),
        tuple(blocked_ensembles),
        expected,
    )


def execute_target_center_posterior_jobs(
    jobs: Sequence[TargetCenterPosteriorJob],
    *,
    use_processes: bool = True,
) -> tuple[TargetCenterPosteriorProducts, ...]:
    rows = tuple(jobs)
    if tuple(row.target_center for row in rows) != CENTERS:
        raise ProtocolError("PUMR target posterior job order drifted.")
    if use_processes:
        with ProcessPoolExecutor(
            max_workers=CPU_WORKERS,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(TARGET_POSTERIOR_BLAS_THREADS_PER_WORKER,),
        ) as executor:
            unordered = tuple(
                executor.map(compute_target_center_posteriors, rows, chunksize=1)
            )
        by_center = {row.target_center: row for row in unordered}
        results = tuple(by_center[center] for center in CENTERS)
    else:
        results = tuple(compute_target_center_posteriors(row) for row in rows)
    expected = 2 * SUPPORT_CROSSFIT_FOLD_COUNT * sum(
        len(row.route_support_labels) for row in rows
    )
    if sum(row.model_fit_count for row in results) != expected:
        raise ProtocolError("PUMR target posterior workload drifted.")
    return results


def _initialize_worker(threads: int) -> None:
    global _THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("PUMR posterior worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=threads)


__all__ = (
    "TargetCenterPosteriorJob",
    "TargetCenterPosteriorProducts",
    "compute_target_center_posteriors",
    "execute_target_center_posterior_jobs",
)
