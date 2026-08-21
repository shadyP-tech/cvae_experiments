"""Center-batched one-fit route posteriors with pickle-safe worker payloads."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CPU_WORKERS,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    TARGET_POSTERIOR_BLAS_THREADS_PER_WORKER,
)
from .contracts import BinaryLabel
from .posterior_contracts import (
    CasePosteriorPrediction,
    PhysicalFingerprintSurface,
    TargetLocalPosteriorModel,
)
from .posterior_fit import fit_route_posterior
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
    models: tuple[TargetLocalPosteriorModel, ...]
    predictions: tuple[CasePosteriorPrediction, ...]
    model_fit_count: int


def compute_target_center_posteriors(
    job: TargetCenterPosteriorJob,
) -> TargetCenterPosteriorProducts:
    if (
        job.target_center != job.primary_fingerprint.center
        or job.target_center != job.blocked_fingerprint.center
        or job.primary_fingerprint.cases != job.blocked_fingerprint.cases
        or tuple(case for case, _ in job.route_support_labels)
        != job.primary_fingerprint.cases
    ):
        raise ProtocolError("CBPUPR target posterior center job drifted.")
    models: list[TargetLocalPosteriorModel] = []
    predictions: list[CasePosteriorPrediction] = []
    for case_id, labels in job.route_support_labels:
        for fingerprint in (job.primary_fingerprint, job.blocked_fingerprint):
            model, prediction = fit_route_posterior(
                fingerprint,
                held_case_id=case_id,
                support_labels=labels,
            )
            models.append(model)
            predictions.append(prediction)
    return TargetCenterPosteriorProducts(
        job.target_center,
        tuple(models),
        tuple(predictions),
        len(models),
    )


def execute_target_posterior_jobs(
    jobs: Sequence[TargetCenterPosteriorJob],
    *,
    use_processes: bool = True,
) -> tuple[TargetCenterPosteriorProducts, ...]:
    rows = tuple(jobs)
    if tuple(job.target_center for job in rows) != CENTERS:
        raise ProtocolError("CBPUPR target posterior job order drifted.")
    if use_processes:
        context = mp.get_context("spawn")
        payloads = tuple(_job_payload(job) for job in rows)
        with ProcessPoolExecutor(
            max_workers=CPU_WORKERS,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(TARGET_POSTERIOR_BLAS_THREADS_PER_WORKER,),
        ) as executor:
            raw = tuple(executor.map(_compute_worker_payload, payloads, chunksize=1))
        products = tuple(_products_from_payload(value) for value in raw)
    else:
        products = tuple(compute_target_center_posteriors(job) for job in rows)
    ordered = tuple(sorted(products, key=lambda row: CENTERS.index(row.target_center)))
    if (
        tuple(row.target_center for row in ordered) != CENTERS
        or sum(row.model_fit_count for row in ordered)
        != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
    ):
        raise ProtocolError("CBPUPR target posterior workload drifted.")
    return ordered


def _job_payload(job: TargetCenterPosteriorJob) -> dict[str, object]:
    return {
        "target_center": job.target_center,
        "primary": _fingerprint_payload(job.primary_fingerprint),
        "blocked": _fingerprint_payload(job.blocked_fingerprint),
        "support": tuple(
            (
                case,
                tuple((row.center, row.case_id, row.sample_id, row.value, row.scope) for row in labels),
            )
            for case, labels in job.route_support_labels
        ),
    }


def _fingerprint_payload(surface: PhysicalFingerprintSurface) -> dict[str, object]:
    return {
        "center": surface.center,
        "sample_ids": surface.sample_ids,
        "case_ids": surface.case_ids,
        "feature_names": surface.feature_names,
        "feature_values": np.asarray(surface.feature_values, dtype=np.float64),
        "source_surface_hash": surface.source_surface_hash,
        "control_id": surface.control_id,
    }


def _fingerprint_from_payload(raw: object) -> PhysicalFingerprintSurface:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR fingerprint worker payload drifted.")
    return PhysicalFingerprintSurface(
        str(raw["center"]),
        tuple(raw["sample_ids"]),
        tuple(raw["case_ids"]),
        tuple(raw["feature_names"]),
        np.asarray(raw["feature_values"], dtype=np.float64),
        str(raw["source_surface_hash"]),
        str(raw["control_id"]),
    )


def _compute_worker_payload(raw: dict[str, object]) -> dict[str, object]:
    support = tuple(
        (
            str(case),
            tuple(BinaryLabel(str(c), str(d), str(s), int(y), str(scope)) for c, d, s, y, scope in labels),
        )
        for case, labels in raw["support"]
    )
    result = compute_target_center_posteriors(
        TargetCenterPosteriorJob(
            str(raw["target_center"]),
            _fingerprint_from_payload(raw["primary"]),
            _fingerprint_from_payload(raw["blocked"]),
            support,
        )
    )
    return {
        "target_center": result.target_center,
        "models": tuple(row.to_payload() for row in result.models),
        "predictions": tuple(row.to_payload() for row in result.predictions),
        "model_fit_count": result.model_fit_count,
    }


def _products_from_payload(raw: dict[str, object]) -> TargetCenterPosteriorProducts:
    models = tuple(_model_from_payload(row) for row in raw["models"])
    predictions = tuple(_prediction_from_payload(row) for row in raw["predictions"])
    return TargetCenterPosteriorProducts(
        str(raw["target_center"]), models, predictions, int(raw["model_fit_count"])
    )


def _model_from_payload(raw: object) -> TargetLocalPosteriorModel:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR posterior model worker payload drifted.")
    return TargetLocalPosteriorModel(
        str(raw["target_center"]), str(raw["held_case_id"]), str(raw["control_id"]),
        tuple(raw["training_case_ids"]), tuple(raw["feature_names"]),
        tuple(raw["feature_mean"]), tuple(raw["feature_scale"]),
        tuple(raw["coefficients"]), float(raw["intercept"]),
        int(raw["training_row_count"]), int(raw["training_n_positive"]),
        int(raw["training_n_negative"]), str(raw["fingerprint_hash"]),
        str(raw["training_identity_hash"]), int(raw["iterations"]),
        bool(raw["converged"]),
    )


def _prediction_from_payload(raw: object) -> CasePosteriorPrediction:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR posterior prediction worker payload drifted.")
    return CasePosteriorPrediction(
        str(raw["target_center"]), str(raw["held_case_id"]), str(raw["control_id"]),
        tuple(raw["sample_ids"]), tuple(raw["natural_probabilities"]),
        str(raw["model_hash"]), str(raw["fingerprint_hash"]),
    )


def _initialize_worker(threads: int) -> None:
    global _THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(int(threads))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("CBPUPR posterior worker lacks threadpoolctl.") from exc
    _THREADPOOL_LIMITER = threadpool_limits(limits=int(threads))


__all__ = (
    "TargetCenterPosteriorJob",
    "TargetCenterPosteriorProducts",
    "compute_target_center_posteriors",
    "execute_target_posterior_jobs",
)
