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
from .posterior_fit import fit_route_posterior, predict_route_posterior
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
    for job, products in zip(rows, ordered, strict=True):
        _validate_products_against_job(job, products)
    return ordered


def _validate_products_against_job(
    job: TargetCenterPosteriorJob,
    products: TargetCenterPosteriorProducts,
) -> None:
    fingerprints = {
        job.primary_fingerprint.control_id: job.primary_fingerprint,
        job.blocked_fingerprint.control_id: job.blocked_fingerprint,
    }
    expected_keys = tuple(
        (job.target_center, case, fingerprint.control_id)
        for case, _labels in job.route_support_labels
        for fingerprint in (job.primary_fingerprint, job.blocked_fingerprint)
    )
    model_keys = tuple(
        (row.target_center, row.held_case_id, row.control_id)
        for row in products.models
    )
    prediction_keys = tuple(
        (row.target_center, row.held_case_id, row.control_id)
        for row in products.predictions
    )
    if (
        products.target_center != job.target_center
        or model_keys != expected_keys
        or prediction_keys != expected_keys
    ):
        raise ProtocolError("CBPUPR posterior worker result topology drifted.")
    for model, prediction in zip(
        products.models, products.predictions, strict=True
    ):
        fingerprint = fingerprints[model.control_id]
        if (
            model.fingerprint_hash != fingerprint.fingerprint_hash
            or prediction.fingerprint_hash != fingerprint.fingerprint_hash
        ):
            raise ProtocolError("CBPUPR posterior worker source lineage drifted.")
        replay = predict_route_posterior(fingerprint, model)
        if (
            replay.to_payload() != prediction.to_payload()
            or np.asarray(
                replay.natural_probabilities, dtype=np.float32
            ).tobytes(order="C")
            != np.asarray(
                prediction.natural_probabilities, dtype=np.float32
            ).tobytes(order="C")
        ):
            raise ProtocolError("CBPUPR posterior worker replay drifted.")


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
        "fingerprint_hash": surface.fingerprint_hash,
    }


def _fingerprint_from_payload(raw: object) -> PhysicalFingerprintSurface:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR fingerprint worker payload drifted.")
    surface = PhysicalFingerprintSurface(
        str(raw["center"]),
        tuple(raw["sample_ids"]),
        tuple(raw["case_ids"]),
        tuple(raw["feature_names"]),
        np.asarray(raw["feature_values"], dtype=np.float64),
        str(raw["source_surface_hash"]),
        str(raw["control_id"]),
    )
    if surface.fingerprint_hash != raw.get("fingerprint_hash"):
        raise ProtocolError("CBPUPR fingerprint worker hash drifted.")
    return surface


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
    target_center = str(raw["target_center"])
    model_fit_count = int(raw["model_fit_count"])
    model_index = {
        (row.target_center, row.held_case_id, row.control_id): row for row in models
    }
    prediction_index = {
        (row.target_center, row.held_case_id, row.control_id): row
        for row in predictions
    }
    if (
        not models
        or model_fit_count != len(models)
        or len(models) != len(predictions)
        or len(model_index) != len(models)
        or len(prediction_index) != len(predictions)
    ):
        raise ProtocolError("CBPUPR posterior worker product count drifted.")
    if (
        set(model_index) != set(prediction_index)
        or any(key[0] != target_center for key in model_index)
        or any(
            prediction_index[key].model_hash != model.model_hash
            or prediction_index[key].fingerprint_hash != model.fingerprint_hash
            for key, model in model_index.items()
        )
    ):
        raise ProtocolError("CBPUPR posterior worker model/prediction lineage drifted.")
    return TargetCenterPosteriorProducts(
        target_center, models, predictions, model_fit_count
    )


def _model_from_payload(raw: object) -> TargetLocalPosteriorModel:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR posterior model worker payload drifted.")
    model = TargetLocalPosteriorModel(
        str(raw["target_center"]), str(raw["held_case_id"]), str(raw["control_id"]),
        tuple(raw["training_case_ids"]), tuple(raw["feature_names"]),
        tuple(raw["feature_mean"]), tuple(raw["feature_scale"]),
        tuple(raw["coefficients"]), float(raw["intercept"]),
        int(raw["training_row_count"]), int(raw["training_n_positive"]),
        int(raw["training_n_negative"]), str(raw["fingerprint_hash"]),
        str(raw["training_identity_hash"]), int(raw["iterations"]),
        bool(raw["converged"]),
    )
    if model.model_hash != raw.get("model_hash") or model.to_payload() != raw:
        raise ProtocolError("CBPUPR posterior model worker hash drifted.")
    return model


def _prediction_from_payload(raw: object) -> CasePosteriorPrediction:
    if not isinstance(raw, dict):
        raise ProtocolError("CBPUPR posterior prediction worker payload drifted.")
    prediction = CasePosteriorPrediction(
        str(raw["target_center"]), str(raw["held_case_id"]), str(raw["control_id"]),
        tuple(raw["sample_ids"]), tuple(raw["natural_probabilities"]),
        str(raw["model_hash"]), str(raw["fingerprint_hash"]),
    )
    if (
        prediction.prediction_hash != raw.get("prediction_hash")
        or prediction.to_payload() != raw
    ):
        raise ProtocolError("CBPUPR posterior prediction worker hash drifted.")
    return prediction


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
