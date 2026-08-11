"""Independent train-only response and model-bank replay for validation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
from typing import Mapping, Sequence

from threadpoolctl import threadpool_limits

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import (
    WorkstationRuntime,
    build_exact_regret_surface,
    canonical_workstation_runtime,
    fit_known_bank_pairwise_models,
    freeze_pairwise_model_bank,
    serialize_pairwise_model_bank,
    validate_runtime,
)
from .constants import CENTERS, GEOMETRY_IDS
from .experiment_contracts import MODEL_FAMILY_IDS
from .hashing import canonical_hash
from .inputs import load_label_free_source_frame
from .source_capability import SourceOOFLabelCapability
from .validation_common import (
    EXPECTED_MODEL_BANK_COUNT,
    EXPECTED_MODEL_COUNT,
)
from .validation_surfaces import validate_response_table


RefitJob = tuple[
    tuple[str, str, str],
    object,
    object,
    object,
    object | None,
]


def validate_source_training_replay(
    root: Path,
    *,
    config: object,
    composite_seal: object,
    source_surfaces: Mapping[tuple[str, str, str], object],
    source_contexts: Mapping[tuple[str, str], object],
    source_features: Sequence[Mapping[str, str]],
    source_sample_counts: Mapping[tuple[str, str], int],
    persisted_capability_report: Mapping[str, object],
    persisted_model_records: Sequence[object],
    persisted_model_collection_hash: str,
    prelabel_feature_seal_hash: str,
    runtime: WorkstationRuntime | None = None,
) -> tuple[Mapping[str, str], ...]:
    """Reopen only train labels, rebuild responses, and refit all G/R/P banks."""

    source_frame = load_label_free_source_frame(config)
    _validate_source_frame_against_composite(source_frame, composite_seal)

    capability = SourceOOFLabelCapability(
        source_frame,
        train_cache_root=Path(getattr(config, "train_cache_root")),
    )
    capability.open_after_source_prediction_seal(composite_seal)
    labels_by_target = {
        target: capability.labels_for_outer_target(target) for target in CENTERS
    }
    replayed_capability_report = dict(capability.access_report())
    if replayed_capability_report != dict(persisted_capability_report):
        raise ProtocolError(
            "Prediction-only source label capability differs from fresh replay."
        )

    response_surfaces = {
        (target, geometry): build_exact_regret_surface(
            source_surfaces[(target, geometry, "R")],
            labels_by_target[target],
            context=source_contexts[(target, geometry)],
        )
        for target in CENTERS
        for geometry in GEOMETRY_IDS
    }
    if len(response_surfaces) != len(CENTERS) * len(GEOMETRY_IDS):
        raise ProtocolError("Prediction-only source response replay topology drifted.")
    response_rows = validate_response_table(
        root / "tables/source_regret_responses.csv",
        source_features=source_features,
        source_sample_counts=source_sample_counts,
        replayed_response_surfaces=response_surfaces,
    )

    jobs = tuple(
        (
            (target, geometry, family),
            source_surfaces[(target, geometry, family)],
            response_surfaces[(target, geometry)],
            source_contexts[(target, geometry)],
            (
                source_surfaces[(target, geometry, "R")]
                if family != "R"
                else None
            ),
        )
        for target in CENTERS
        for geometry in GEOMETRY_IDS
        for family in MODEL_FAMILY_IDS
    )
    chosen_runtime = runtime or canonical_workstation_runtime()
    replayed_banks = _refit_jobs(jobs, runtime=chosen_runtime)
    _compare_refitted_banks(
        replayed_banks,
        persisted_records=persisted_model_records,
        persisted_collection_hash=persisted_model_collection_hash,
        prelabel_feature_seal_hash=prelabel_feature_seal_hash,
        source_prediction_seal_hash=str(getattr(composite_seal, "seal_hash")),
    )
    return response_rows


def _validate_source_frame_against_composite(
    source_frame: object, composite_seal: object
) -> None:
    store = getattr(composite_seal, "source_store", None)
    target_bank = getattr(composite_seal, "target_classifier_bank", None)
    if (
        store is None
        or source_frame.cache_binding_hash
        != getattr(store, "frame_cache_binding_hash", None)
        or source_frame.cache_binding_hash
        != getattr(target_bank, "source_cache_binding_hash", None)
    ):
        raise ProtocolError("Prediction-only fresh source frame binding drifted.")
    for query in CENTERS:
        frame_rows = tuple(source_frame.rows_by_center[query])
        if (
            tuple(row.source_row_id for row in frame_rows)
            != tuple(store.rows_by_query[query])
            or tuple(row.case_id for row in frame_rows)
            != tuple(store.case_ids_by_query[query])
            or any(row.center != query for row in frame_rows)
        ):
            raise ProtocolError(
                "Prediction-only fresh source identity differs from sealed predictions."
            )


def _refit_job(job: RefitJob, threads: int) -> tuple[tuple[str, str, str], object]:
    key, features, responses, context, parent = job
    with threadpool_limits(limits=threads):
        models = fit_known_bank_pairwise_models(
            features,
            responses,
            context=context,
            family=key[2],
            aligned_parent_features=parent,
        )
    return key, freeze_pairwise_model_bank(models)


def _refit_job_star(arguments: tuple[RefitJob, int]) -> tuple[tuple[str, str, str], object]:
    return _refit_job(*arguments)


def _refit_jobs(
    jobs: Sequence[RefitJob], *, runtime: WorkstationRuntime
) -> dict[tuple[str, str, str], object]:
    validate_runtime(runtime)
    if len(jobs) != EXPECTED_MODEL_BANK_COUNT:
        raise ProtocolError("Prediction-only model-refit job topology drifted.")
    if runtime.serial_test_override:
        rows = tuple(_refit_job(job, runtime.threads_per_worker) for job in jobs)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=runtime.workers,
            mp_context=context,
            initializer=_initialize_worker,
        ) as pool:
            rows = tuple(
                pool.map(
                    _refit_job_star,
                    ((job, runtime.threads_per_worker) for job in jobs),
                    chunksize=1,
                )
            )
    result = dict(rows)
    if len(result) != EXPECTED_MODEL_BANK_COUNT:
        raise ProtocolError("Prediction-only model-refit result topology drifted.")
    return result


def _compare_refitted_banks(
    replayed: Mapping[tuple[str, str, str], object],
    *,
    persisted_records: Sequence[object],
    persisted_collection_hash: str,
    prelabel_feature_seal_hash: str,
    source_prediction_seal_hash: str,
) -> None:
    persisted = {record.key: record.bank for record in persisted_records}
    if set(replayed) != set(persisted) or len(persisted) != EXPECTED_MODEL_BANK_COUNT:
        raise ProtocolError("Prediction-only persisted/refit bank topology drifted.")
    model_count = 0
    for key in sorted(replayed):
        fresh, frozen = replayed[key], persisted[key]
        if (
            getattr(fresh, "model_bank_hash", None)
            != getattr(frozen, "model_bank_hash", None)
            or serialize_pairwise_model_bank(fresh)
            != serialize_pairwise_model_bank(frozen)
        ):
            raise ProtocolError(
                "Prediction-only persisted model bank differs from source-only refit."
            )
        model_count += len(tuple(getattr(fresh, "models")))
    if model_count != EXPECTED_MODEL_COUNT:
        raise ProtocolError("Prediction-only refit model count drifted.")

    collection_hash = canonical_hash(
        {
            "schema_version": (
                "midogpp_disagreement_regret_model_bank_collection_v1"
            ),
            "prelabel_feature_seal_hash": prelabel_feature_seal_hash,
            "source_prediction_seal_hash": source_prediction_seal_hash,
            "banks": [
                {
                    "outer_target_id": target,
                    "geometry_id": geometry,
                    "family": family,
                    "model_bank_hash": replayed[
                        (target, geometry, family)
                    ].model_bank_hash,
                }
                for target, geometry, family in sorted(replayed)
            ],
            "source_labels_used_for_training_only": True,
            "raw_source_labels_persisted": False,
            "test_labels_used": False,
        }
    )
    if collection_hash != persisted_collection_hash:
        raise ProtocolError(
            "Prediction-only refit model-bank collection hash drifted."
        )


def _initialize_worker() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


__all__ = ("validate_source_training_replay",)
