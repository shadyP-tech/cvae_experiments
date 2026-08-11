"""Source-OOF prelabel features and fixed-capacity regret model fitting."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Mapping, Sequence

from threadpoolctl import threadpool_limits

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import (
    DevelopmentContext,
    DevelopmentScope,
    ProbabilityRow,
    SourceOOFLabelRow,
    build_source_oof_training_feature_surface,
    build_exact_regret_surface,
    feature_surface_for_family,
    fit_known_bank_pairwise_models,
)
from ...routing.disagreement_regret_core.runtime import WorkstationRuntime, validate_runtime
from .experiment_contracts import CENTERS, GEOMETRY_IDS, MODEL_FAMILY_IDS
from .hashing import canonical_hash
from .products import (
    DevelopmentProducts,
    FeatureSurfaceRecord,
    ModelBankRecord,
    PrelabelProducts,
    ResponseSurfaceRecord,
    model_bank_collection_hash,
)


def build_posthoc_source_contexts(
    source_frame: object,
    *,
    authorization_hash: str,
) -> dict[tuple[str, str], DevelopmentContext]:
    """Bind truthful previously-consumed source labels to exact donor keys."""

    rows = tuple(getattr(source_frame, "rows"))
    contexts: dict[tuple[str, str], DevelopmentContext] = {}
    for target in CENTERS:
        donors = tuple(center for center in CENTERS if center != target)
        sample_keys = sorted(
            (
                str(getattr(row, "center")),
                str(getattr(row, "case_id")),
                str(getattr(row, "source_row_id")),
            )
            for row in rows
            if str(getattr(row, "center")) != target
        )
        if {key[0] for key in sample_keys} != set(donors):
            raise ProtocolError("Posthoc source context donor coverage drifted.")
        sample_hash = canonical_hash(
            {"sample_keys": [list(value) for value in sample_keys]}
        )
        context = DevelopmentContext(
            scope=DevelopmentScope.AUTHORIZED_POSTHOC_SOURCE_OOF,
            dataset_family="MIDOGPP_SOURCE_TRAIN_OOF_POSTHOC",
            outer_target_id=target,
            authorization_hash=authorization_hash,
            authorization_unused=False,
            authorized_query_ids=donors,
            authorized_sample_keys_hash=sample_hash,
            source_evidence_previously_consumed=True,
            consumed_data=False,
            target_labels_available=False,
        )
        for geometry in GEOMETRY_IDS:
            contexts[(target, geometry)] = context
    return contexts


def build_source_prelabel_products(
    probability_rows_by_surface: Mapping[
        tuple[str, str], Sequence[ProbabilityRow]
    ],
    *,
    source_prediction_seal_hash: str,
    contexts: Mapping[tuple[str, str], DevelopmentContext],
) -> PrelabelProducts:
    """Build all G/R/P source features before any source label is opened."""

    expected_contexts = {
        (target, geometry) for target in CENTERS for geometry in GEOMETRY_IDS
    }
    if set(contexts) != expected_contexts:
        raise ProtocolError("Source development-context topology drifted.")
    if set(probability_rows_by_surface) != expected_contexts:
        raise ProtocolError("Source probability-surface topology drifted.")
    records: list[FeatureSurfaceRecord] = []
    for target in CENTERS:
        for geometry in GEOMETRY_IDS:
            context = contexts[(target, geometry)]
            core_rows = tuple(probability_rows_by_surface[(target, geometry)])
            if (
                not core_rows
                or {row.query_id for row in core_rows}
                != set(CENTERS).difference({target})
                or {row.prediction_seal_hash for row in core_rows}
                != {source_prediction_seal_hash}
            ):
                raise ProtocolError("Source probability view lineage drifted.")
            aligned = build_source_oof_training_feature_surface(
                core_rows,
                baseline_action_id="B",
                control_action_id="U",
                context=context,
            )
            for family in MODEL_FAMILY_IDS:
                surface = feature_surface_for_family(aligned, family=family)
                records.append(
                    FeatureSurfaceRecord(
                        outer_target_id=target,
                        geometry_id=geometry,
                        family=family,
                        surface=surface,
                    )
                )
    return PrelabelProducts(
        feature_surfaces=tuple(sorted(records, key=lambda row: row.key)),
        development_contexts=dict(contexts),
        source_prediction_seal_hash=source_prediction_seal_hash,
    )


def fit_source_development_products(
    prelabel: PrelabelProducts,
    *,
    labels_by_outer_target: Mapping[str, Sequence[SourceOOFLabelRow]],
    source_label_capability_report: Mapping[str, object],
    runtime: WorkstationRuntime,
) -> DevelopmentProducts:
    """Create exact donor responses and fit fixed G/R/P model families."""

    validate_runtime(runtime)
    if set(labels_by_outer_target) != set(CENTERS):
        raise ProtocolError("Every outer target requires its donor-label view.")
    responses: list[ResponseSurfaceRecord] = []
    jobs: list[tuple[object, ...]] = []
    for target in CENTERS:
        for geometry in GEOMETRY_IDS:
            context = prelabel.development_contexts[(target, geometry)]
            aligned = prelabel.surface(target, geometry, "R")
            response = build_exact_regret_surface(
                aligned,
                tuple(labels_by_outer_target[target]),
                context=context,
            )
            responses.append(
                ResponseSurfaceRecord(
                    outer_target_id=target,
                    geometry_id=geometry,
                    surface=response,
                )
            )
            for family in MODEL_FAMILY_IDS:
                jobs.append(
                    (
                        prelabel.surface(target, geometry, family),
                        response,
                        context,
                        family,
                        aligned if family != "R" else None,
                    )
                )

    raw_banks = _fit_jobs(jobs, runtime=runtime)
    job_keys = tuple(
        (target, geometry, family)
        for target in CENTERS
        for geometry in GEOMETRY_IDS
        for family in MODEL_FAMILY_IDS
    )
    bank_records = tuple(
        ModelBankRecord(
            outer_target_id=target,
            geometry_id=geometry,
            family=family,
            bank=bank,
        )
        for (target, geometry, family), bank in zip(
            job_keys, raw_banks, strict=True
        )
    )
    bank_hash = model_bank_collection_hash(prelabel, bank_records)
    return DevelopmentProducts(
        prelabel=prelabel,
        response_surfaces=tuple(sorted(responses, key=lambda row: row.key)),
        model_banks=bank_records,
        source_label_capability_report=dict(source_label_capability_report),
        model_bank_hash=bank_hash,
    )


def _fit_job(job: tuple[object, ...], threads: int) -> object:
    features, responses, context, family, parent = job
    with threadpool_limits(limits=threads):
        models = fit_known_bank_pairwise_models(
            features,
            responses,
            context=context,
            family=str(family),
            aligned_parent_features=parent,
        )
    # Imported lazily so the core can evolve its durable bank contract without
    # introducing a runner dependency into the pure mathematics.
    from ...routing.disagreement_regret_core import freeze_pairwise_model_bank

    return freeze_pairwise_model_bank(models)


def _fit_jobs(
    jobs: Sequence[tuple[object, ...]], *, runtime: WorkstationRuntime
) -> tuple[object, ...]:
    if runtime.serial_test_override:
        return tuple(_fit_job(job, runtime.threads_per_worker) for job in jobs)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=runtime.workers,
        mp_context=context,
        initializer=_initialize_worker,
    ) as pool:
        return tuple(
            pool.map(
                _fit_job_star,
                ((job, runtime.threads_per_worker) for job in jobs),
                chunksize=1,
            )
        )


def _fit_job_star(arguments: tuple[tuple[object, ...], int]) -> object:
    return _fit_job(*arguments)


def _initialize_worker() -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


__all__ = (
    "build_posthoc_source_contexts",
    "build_source_prelabel_products",
    "fit_source_development_products",
)
