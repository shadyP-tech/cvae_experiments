"""Fully nested source-inner recipe selection for prior recovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ...real_features.classifier_reference.matched_reference import (
    PredictOnlySelection,
    select_nested_predict_spec_source_only,
)
from ...real_features.classifier_reference.midogpp_real_feature_classifier import RealFeatureFrame
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..feature_frame import ExpertFeatureFrame
from ..generation_samplers import (
    AggregatePosteriorSampler,
    DIAGONAL_SAMPLER,
    FULL_SAMPLER,
    STANDARD_SAMPLER,
)
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from ..reporting import prepare_artifact_dirs
from ..task_fisher import fit_task_fisher_metric
from ..training import TrainedCVAERuntime
from .prior_recovery_artifacts import write_source_inner_bundle
from .prior_recovery_common import (
    NO_TASK_FISHER_STATE,
    PRIOR_RECOVERY_METHOD,
    canonical_rows_hash,
    fit_samplers,
    generation_and_evaluation_hashes,
    load_frame,
    mean,
    protocol_hash,
    safe_ratio,
    selection_evidence_hash,
    train_runtime,
)
from .prior_recovery_config import (
    SourceInnerPriorRecoveryConfig,
    recipe_contract_hash,
    recipe_contract_payload,
)
from .prior_recovery_classifier import (
    SOURCE_INNER_CLASSIFIER_GRID_HASH,
    source_inner_classifier_specs,
)
from .prior_recovery_provenance import ProvenanceRecorder
from .prior_recovery_runtime_cache import FeatureFrameCache
from .prior_recovery_timing import RuntimeTimingRecorder, mark_run_failed, write_run_state
from .prior_recovery_schema import (
    NESTED_REAL_REFERENCE_SCHEMA,
    SAMPLER_REALIZATION_SCHEMA,
    SOURCE_INNER_METRIC_SCHEMA,
)
from .representations import (
    decode_means,
    posterior_samples,
    sampler_decodes,
    source_budget_labels,
)
from .scoring import RepresentationScore, score_representation
from .source_inner_selection import InnerCenterMetric, RecipeLock, select_recipe_lock
from .splits import (
    assert_identity_overlap_pass,
    frame_arrays,
    identity_overlap_audit,
    indices_for_centers,
    inner_split,
    row_hash,
    source_only_frame,
)


@dataclass
class _InnerFoldContext:
    outer: str
    inner: str
    fit_centers: tuple[str, ...]
    spec: ClassifierSpec
    selection: PredictOnlySelection
    frame: ExpertFeatureFrame
    x_fit: object
    y_fit: tuple[int, ...]
    source_ids: tuple[str, ...]
    x_eval: object
    y_eval: tuple[int, ...]
    eval_ids: tuple[str, ...]
    real_bacc: float
    real_reference_protocol_hash: str
    runtime_a: TrainedCVAERuntime
    decode_a: RepresentationScore
    posterior_a: Mapping[int, RepresentationScore]


def run_source_inner_prior_recovery(
    config: SourceInnerPriorRecoveryConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    if not isinstance(config, SourceInnerPriorRecoveryConfig):
        raise ProtocolError("Source-inner runner requires SourceInnerPriorRecoveryConfig.")
    root = prepare_artifact_dirs(Path(artifact_root or config.artifact_root))
    try:
        return _run_source_inner_prior_recovery(config, root=root)
    except Exception:
        mark_run_failed(root, mode="source_inner")
        raise


def _run_source_inner_prior_recovery(
    config: SourceInnerPriorRecoveryConfig,
    *,
    root: Path,
) -> Path:
    recorder = ProvenanceRecorder(root)
    frame_cache = FeatureFrameCache(root)
    frame = load_frame(config)
    specs = source_inner_classifier_specs(classifier_seed=23)
    runtime_protocol_hash = protocol_hash(config, frame)
    timings = RuntimeTimingRecorder(root, protocol_hash=runtime_protocol_hash, mode="source_inner")
    write_run_state(root, protocol_hash=runtime_protocol_hash, mode="source_inner", status="RUNNING")
    contract_hash = recipe_contract_hash(config)
    detail_rows: list[dict[str, object]] = []
    nested_reference_rows: list[dict[str, object]] = []
    nested_tuning_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    summaries_by_outer: dict[str, list[InnerCenterMetric]] = {}
    preliminary_by_outer: dict[str, RecipeLock] = {}
    fit_hash_by_outer: dict[str, str] = {}
    for outer in config.heldout_centers:
        source_frame = source_only_frame(frame, outer_target_center=outer)
        expected_inner = source_frame.eligible_centers
        contexts: list[_InnerFoldContext] = []
        summaries: list[InnerCenterMetric] = []
        outer_start = len(detail_rows)
        for inner in expected_inner:
            context, rows, nested_row, sampler_detail, center_summaries, audit_row = _run_isotropic_fold(
                config,
                frame=source_frame,
                outer=outer,
                inner=inner,
                candidate_specs=specs,
                runtime_protocol_hash=runtime_protocol_hash,
                recorder=recorder,
                frame_cache=frame_cache,
                timings=timings,
            )
            contexts.append(context)
            detail_rows.extend(rows)
            nested_reference_rows.append(nested_row)
            nested_tuning_rows.extend(dict(row) for row in context.selection.candidate_rows)
            sampler_rows.extend(sampler_detail)
            summaries.extend(center_summaries)
            identity_rows.append(audit_row)
        fit_sets_hash = stable_hash({context.inner: list(context.fit_centers) for context in contexts})
        fit_hash_by_outer[outer] = fit_sets_hash
        preliminary = _select_lock(
            config,
            summaries,
            outer=outer,
            expected_inner=expected_inner,
            runtime_protocol_hash=runtime_protocol_hash,
            fit_sets_hash=fit_sets_hash,
            source_metric_hash=canonical_rows_hash(detail_rows[outer_start:]),
            selection_bundle_hash="preliminary",
            require_task_factorial=False,
        )
        preliminary_by_outer[outer] = preliminary
        if preliminary.primary_arm == "C":
            for context in contexts:
                rows, sampler_detail, task_summaries = _run_task_fold(
                    config,
                    context=context,
                    selected_family=preliminary.sampler_family,
                    runtime_protocol_hash=runtime_protocol_hash,
                    recorder=recorder,
                    timings=timings,
                )
                detail_rows.extend(rows)
                sampler_rows.extend(sampler_detail)
                summaries.extend(task_summaries)
        summaries_by_outer[outer] = summaries
    recorder.write_indices()
    frame_cache.write_index()
    timings.finalize()
    checkpoint_index = json.loads((root / "manifests/checkpoint_index.json").read_text(encoding="utf-8"))
    fisher_index = json.loads((root / "manifests/task_fisher_index.json").read_text(encoding="utf-8"))
    frame_index = json.loads((root / "manifests/feature_frame_index.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "midogpp_prior_recovery_source_inner_protocol_v1",
        "experiment_name": config.name,
        "method": PRIOR_RECOVERY_METHOD,
        "claim_scope": "cvae_recipe_lock_only",
        "claim_role": "cvae_recipe_lock",
        "heldout_centers": list(config.heldout_centers),
        "eligible_centers": list(frame.eligible_centers),
        "coverage_mode": (
            "complete"
            if config.heldout_centers == frame.eligible_centers == MIDOGPP_ELIGIBLE_CENTERS
            else "partial_test"
        ),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "classifier_grid_hash": SOURCE_INNER_CLASSIFIER_GRID_HASH,
        "classifier_grid": [spec.to_payload() for spec in specs],
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "protocol_hash": runtime_protocol_hash,
        "recipe_contract": recipe_contract_payload(config),
        "recipe_contract_hash": contract_hash,
        "source_inner_labels_used_for_selection": True,
        "outer_target_rows_passed_to_training_or_selection": False,
        "target_eval_labels_used_for_selection": False,
        "target_eval_labels_used_for_scoring_only": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": True,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
    }
    bundle_hash = selection_evidence_hash(
        metric_rows=detail_rows,
        nested_reference_rows=nested_reference_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=manifest,
        checkpoint_index=checkpoint_index,
        task_fisher_index=fisher_index,
        feature_frame_index=frame_index,
    )
    for row in detail_rows:
        row["selection_bundle_hash"] = bundle_hash
    locks: list[RecipeLock] = []
    for outer in config.heldout_centers:
        expected_inner = tuple(center for center in frame.eligible_centers if center != outer)
        outer_rows = [row for row in detail_rows if row["outer_target_center"] == outer]
        locks.append(
            _select_lock(
                config,
                summaries_by_outer[outer],
                outer=outer,
                expected_inner=expected_inner,
                runtime_protocol_hash=runtime_protocol_hash,
                fit_sets_hash=fit_hash_by_outer[outer],
                source_metric_hash=canonical_rows_hash(outer_rows),
                selection_bundle_hash=bundle_hash,
                require_task_factorial=preliminary_by_outer[outer].primary_arm == "C",
            )
        )
    return write_source_inner_bundle(
        root,
        metric_rows=detail_rows,
        nested_reference_rows=nested_reference_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_audit_rows=identity_rows,
        locks=locks,
        protocol_manifest=manifest,
        selection_bundle_hash=bundle_hash,
    )


def _run_isotropic_fold(
    config: SourceInnerPriorRecoveryConfig,
    *,
    frame: RealFeatureFrame,
    outer: str,
    inner: str,
    candidate_specs: Sequence[ClassifierSpec],
    runtime_protocol_hash: str,
    recorder: ProvenanceRecorder,
    frame_cache: FeatureFrameCache,
    timings: RuntimeTimingRecorder,
) -> tuple[
    _InnerFoldContext,
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    list[InnerCenterMetric],
    dict[str, object],
]:
    split = inner_split(outer, inner, centers=frame.eligible_centers)
    started = perf_counter()
    selection = select_nested_predict_spec_source_only(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        candidate_specs=candidate_specs,
    )
    timings.record(
        phase="nested_classifier_selection",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
    )
    fit_idx = indices_for_centers(frame, split.fit_centers)
    eval_idx = indices_for_centers(frame, (inner,))
    audit = identity_overlap_audit(
        frame,
        fit_indices=fit_idx,
        eval_indices=eval_idx,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
    )
    assert_identity_overlap_pass(audit)
    x_fit_full, y_fit, source_ids = frame_arrays(frame, fit_idx)
    x_eval_full, y_eval, eval_ids = frame_arrays(frame, eval_idx)
    real_score = score_representation(x_fit_full, y_fit, x_eval_full, y_eval, spec=selection.selected_spec)
    if not real_score.converged:
        raise ProtocolError(f"Nested real reference did not converge for H={outer}, I={inner}.")
    started = perf_counter()
    feature_frame, frame_cache_hit = frame_cache.fit_or_load(
        expert_id=f"source_inner_H{outer}_I{inner}",
        source_train_embeddings=x_fit_full,
        fit_centers=split.fit_centers,
        fit_row_hash=row_hash(source_ids),
        requested_dim=config.pca_dim,
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
        protocol_hash=runtime_protocol_hash,
        code_version=config.code_version,
    )
    timings.record(
        phase="pca_frame",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        cache_status="hit" if frame_cache_hit else "miss",
    )
    x_fit = feature_frame.transform(x_fit_full)
    x_eval = feature_frame.transform(x_eval_full)
    started = perf_counter()
    runtime = train_runtime(
        config,
        variant=config.isotropic_variant,
        frame=feature_frame,
        fit_centers=split.fit_centers,
        source_ids=source_ids,
        x_fit=x_fit,
        y_fit=y_fit,
        training_seed=config.selection_training_seed,
        runtime_protocol_hash=runtime_protocol_hash,
        feature_cache_hash=frame.feature_cache_hash,
        manifest_hash=frame.manifest_hash,
        task_metric=None,
        objective_context_hash=NO_TASK_FISHER_STATE,
        recorder=recorder,
        task_fisher_state_hash=NO_TASK_FISHER_STATE,
        classifier_spec_hash=selection.selected_spec.config_hash,
    )
    timings.record(
        phase="cvae_training",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        objective_id=runtime.variant.objective_id,
        training_key_hash=runtime.training_key.hash,
        cache_status="hit" if runtime.resumed_from_checkpoint else "miss",
    )
    evaluation_started = perf_counter()
    samplers = fit_samplers(
        config,
        runtime=runtime,
        x_fit=x_fit,
        y_fit=y_fit,
        source_ids=source_ids,
        families=(STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER),
    )
    decoded, _, _ = decode_means(runtime, x_fit, y_fit)
    decode_score = score_representation(decoded, y_fit, x_eval, y_eval, spec=selection.selected_spec)
    posterior_scores = {
        seed: score_representation(
            posterior_samples(runtime, x_fit, y_fit, seed=seed)[0],
            y_fit,
            x_eval,
            y_eval,
            spec=selection.selected_spec,
        )
        for seed in config.generation_seeds
    }
    real_reference_hash = stable_hash(
        {
            "outer": outer,
            "inner": inner,
            "fit_row_hash": row_hash(source_ids),
            "eval_row_hash": row_hash(eval_ids),
            "classifier_spec_hash": selection.selected_spec.config_hash,
            "grid_hash": selection.grid_hash,
        }
    )
    rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    summaries: list[InnerCenterMetric] = []
    budget_labels = source_budget_labels(y_fit)
    for family in (STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER):
        arm = "A" if family == STANDARD_SAMPLER else "C"
        prior_scores = {}
        for seed in config.generation_seeds:
            score = score_representation(
                sampler_decodes(runtime, samplers[family], budget_labels, seed=seed),
                budget_labels,
                x_eval,
                y_eval,
                spec=selection.selected_spec,
            )
            prior_scores[seed] = score
            rows.append(
                _metric_row(
                    config,
                    outer=outer,
                    inner=inner,
                    fit_centers=split.fit_centers,
                    arm=arm,
                    objective_id=ISOTROPIC_OBJECTIVE,
                    sampler=samplers[family],
                    runtime=runtime,
                    generation_seed=seed,
                    role="prior",
                    labels=budget_labels,
                    score=score,
                    real_bacc=real_score.bacc,
                    real_reference_protocol_hash=real_reference_hash,
                    spec=selection.selected_spec,
                    frame=feature_frame,
                    source_ids=source_ids,
                    eval_ids=eval_ids,
                    runtime_protocol_hash=runtime_protocol_hash,
                    task_fisher_state_hash=NO_TASK_FISHER_STATE,
                    task_fisher_valid=True,
                )
            )
        sampler_viable = samplers[family].requested_family_realized_for_both_classes
        summaries.append(
            InnerCenterMetric(
                outer_target_center=outer,
                inner_pseudo_target_center=inner,
                arm=arm,
                sampler_family=family,
                objective_id=ISOTROPIC_OBJECTIVE,
                prior_ratio=mean(
                    safe_ratio(score.bacc, real_score.bacc, minimum_real_bacc=config.minimum_real_bacc)
                    for score in prior_scores.values()
                ),
                decode_bacc=decode_score.bacc,
                posterior_bacc=mean(score.bacc for score in posterior_scores.values()),
                real_reference_bacc=real_score.bacc,
                valid=(
                    decode_score.converged
                    and all(score.converged for score in posterior_scores.values())
                    and all(score.converged for score in prior_scores.values())
                ),
                sampler_viable=sampler_viable,
                realized_sampler_by_class=samplers[family].realized_family_by_class(),
                fallback_reason_by_class=samplers[family].fallback_reason_by_class(),
            )
        )
        sampler_rows.extend(_sampler_rows(outer, inner, arm, runtime, samplers[family]))
    for seed, score in posterior_scores.items():
        rows.append(
            _metric_row(
                config,
                outer=outer,
                inner=inner,
                fit_centers=split.fit_centers,
                arm="A",
                objective_id=ISOTROPIC_OBJECTIVE,
                sampler=samplers[STANDARD_SAMPLER],
                runtime=runtime,
                generation_seed=seed,
                role="posterior",
                labels=y_fit,
                score=score,
                real_bacc=real_score.bacc,
                real_reference_protocol_hash=real_reference_hash,
                spec=selection.selected_spec,
                frame=feature_frame,
                source_ids=source_ids,
                eval_ids=eval_ids,
                runtime_protocol_hash=runtime_protocol_hash,
                task_fisher_state_hash=NO_TASK_FISHER_STATE,
                task_fisher_valid=True,
            )
        )
    rows.append(
        _metric_row(
            config,
            outer=outer,
            inner=inner,
            fit_centers=split.fit_centers,
            arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler=samplers[STANDARD_SAMPLER],
            runtime=runtime,
            generation_seed=-1,
            role="decode",
            labels=y_fit,
            score=decode_score,
            real_bacc=real_score.bacc,
            real_reference_protocol_hash=real_reference_hash,
            spec=selection.selected_spec,
            frame=feature_frame,
            source_ids=source_ids,
            eval_ids=eval_ids,
            runtime_protocol_hash=runtime_protocol_hash,
            task_fisher_state_hash=NO_TASK_FISHER_STATE,
            task_fisher_valid=True,
        )
    )
    nested_row = {
        "schema_version": NESTED_REAL_REFERENCE_SCHEMA,
        "outer_target_center": outer,
        "inner_pseudo_target_center": inner,
        "deeper_validation_centers": json.dumps(list(selection.center_scores)),
        "fit_centers": json.dumps(list(split.fit_centers)),
        "fit_row_hash": row_hash(source_ids),
        "eval_row_hash": row_hash(eval_ids),
        "classifier_grid_hash": selection.grid_hash,
        "selected_classifier_spec": json.dumps(selection.selected_spec.to_payload(), sort_keys=True),
        "selected_classifier_spec_hash": selection.selected_spec.config_hash,
        "real_reference_protocol_hash": real_reference_hash,
        "n_fit": len(fit_idx),
        "n_eval": len(eval_idx),
        "bacc": real_score.bacc,
        "macro_f1": real_score.macro_f1,
        "converged": True,
        "status": "ok",
        "target_eval_labels_used_for_scoring_only": False,
        "selection_used_outer_or_inner_labels": False,
    }
    context = _InnerFoldContext(
        outer=outer,
        inner=inner,
        fit_centers=split.fit_centers,
        spec=selection.selected_spec,
        selection=selection,
        frame=feature_frame,
        x_fit=x_fit,
        y_fit=y_fit,
        source_ids=source_ids,
        x_eval=x_eval,
        y_eval=y_eval,
        eval_ids=eval_ids,
        real_bacc=real_score.bacc,
        real_reference_protocol_hash=real_reference_hash,
        runtime_a=runtime,
        decode_a=decode_score,
        posterior_a=posterior_scores,
    )
    timings.record(
        phase="sampling_and_scoring",
        elapsed_seconds=perf_counter() - evaluation_started,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        objective_id=runtime.variant.objective_id,
        training_key_hash=runtime.training_key.hash,
    )
    return context, rows, nested_row, sampler_rows, summaries, audit


def _run_task_fold(
    config: SourceInnerPriorRecoveryConfig,
    *,
    context: _InnerFoldContext,
    selected_family: str,
    runtime_protocol_hash: str,
    recorder: ProvenanceRecorder,
    timings: RuntimeTimingRecorder,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[InnerCenterMetric]]:
    started = perf_counter()
    fisher = fit_task_fisher_metric(context.x_fit, context.y_fit, spec=context.spec, alpha=1.0)
    fisher_hash = recorder.record_fisher(fisher)
    timings.record(
        phase="task_fisher_fit",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=context.outer,
        inner_pseudo_target_center=context.inner,
        objective_id=TASK_FISHER_OBJECTIVE,
    )
    if not fisher.valid:
        return [], [], [
            InnerCenterMetric(
                outer_target_center=context.outer,
                inner_pseudo_target_center=context.inner,
                arm=arm,
                sampler_family=family,
                objective_id=TASK_FISHER_OBJECTIVE,
                prior_ratio=math.nan,
                decode_bacc=math.nan,
                posterior_bacc=math.nan,
                real_reference_bacc=context.real_bacc,
                valid=False,
                task_fisher_valid=False,
                sampler_viable=False,
            )
            for arm, family in (("B", STANDARD_SAMPLER), ("D", selected_family))
        ]
    started = perf_counter()
    runtime = train_runtime(
        config,
        variant=config.task_fisher_variant,
        frame=context.frame,
        fit_centers=context.fit_centers,
        source_ids=context.source_ids,
        x_fit=context.x_fit,
        y_fit=context.y_fit,
        training_seed=config.selection_training_seed,
        runtime_protocol_hash=runtime_protocol_hash,
        feature_cache_hash=context.runtime_a.training_key.feature_cache_hash,
        manifest_hash=context.runtime_a.training_key.dataset_contract_hash,
        task_metric=fisher.metric,
        objective_context_hash=fisher_hash,
        recorder=recorder,
        task_fisher_state_hash=fisher_hash,
        classifier_spec_hash=context.spec.config_hash,
    )
    timings.record(
        phase="cvae_training",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=context.outer,
        inner_pseudo_target_center=context.inner,
        objective_id=runtime.variant.objective_id,
        training_key_hash=runtime.training_key.hash,
        cache_status="hit" if runtime.resumed_from_checkpoint else "miss",
    )
    evaluation_started = perf_counter()
    samplers = fit_samplers(
        config,
        runtime=runtime,
        x_fit=context.x_fit,
        y_fit=context.y_fit,
        source_ids=context.source_ids,
        families=(STANDARD_SAMPLER, selected_family),
    )
    decoded, _, _ = decode_means(runtime, context.x_fit, context.y_fit)
    decode_score = score_representation(decoded, context.y_fit, context.x_eval, context.y_eval, spec=context.spec)
    posterior_scores = {
        seed: score_representation(
            posterior_samples(runtime, context.x_fit, context.y_fit, seed=seed)[0],
            context.y_fit,
            context.x_eval,
            context.y_eval,
            spec=context.spec,
        )
        for seed in config.generation_seeds
    }
    rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    summaries: list[InnerCenterMetric] = []
    budget_labels = source_budget_labels(context.y_fit)
    for arm, family in (("B", STANDARD_SAMPLER), ("D", selected_family)):
        scores = {}
        for seed in config.generation_seeds:
            score = score_representation(
                sampler_decodes(runtime, samplers[family], budget_labels, seed=seed),
                budget_labels,
                context.x_eval,
                context.y_eval,
                spec=context.spec,
            )
            scores[seed] = score
            rows.append(
                _metric_row(
                    config,
                    outer=context.outer,
                    inner=context.inner,
                    fit_centers=context.fit_centers,
                    arm=arm,
                    objective_id=TASK_FISHER_OBJECTIVE,
                    sampler=samplers[family],
                    runtime=runtime,
                    generation_seed=seed,
                    role="prior",
                    labels=budget_labels,
                    score=score,
                    real_bacc=context.real_bacc,
                    real_reference_protocol_hash=context.real_reference_protocol_hash,
                    spec=context.spec,
                    frame=context.frame,
                    source_ids=context.source_ids,
                    eval_ids=context.eval_ids,
                    runtime_protocol_hash=runtime_protocol_hash,
                    task_fisher_state_hash=fisher_hash,
                    task_fisher_valid=True,
                )
            )
        sampler_viable = samplers[family].requested_family_realized_for_both_classes
        summaries.append(
            InnerCenterMetric(
                outer_target_center=context.outer,
                inner_pseudo_target_center=context.inner,
                arm=arm,
                sampler_family=family,
                objective_id=TASK_FISHER_OBJECTIVE,
                prior_ratio=mean(
                    safe_ratio(score.bacc, context.real_bacc, minimum_real_bacc=config.minimum_real_bacc)
                    for score in scores.values()
                ),
                decode_bacc=decode_score.bacc,
                posterior_bacc=mean(score.bacc for score in posterior_scores.values()),
                real_reference_bacc=context.real_bacc,
                valid=(
                    decode_score.converged
                    and all(score.converged for score in posterior_scores.values())
                    and all(score.converged for score in scores.values())
                ),
                sampler_viable=sampler_viable,
                realized_sampler_by_class=samplers[family].realized_family_by_class(),
                fallback_reason_by_class=samplers[family].fallback_reason_by_class(),
            )
        )
        sampler_rows.extend(_sampler_rows(context.outer, context.inner, arm, runtime, samplers[family]))
    for seed, score in posterior_scores.items():
        rows.append(
            _metric_row(
                config,
                outer=context.outer,
                inner=context.inner,
                fit_centers=context.fit_centers,
                arm="B",
                objective_id=TASK_FISHER_OBJECTIVE,
                sampler=samplers[STANDARD_SAMPLER],
                runtime=runtime,
                generation_seed=seed,
                role="posterior",
                labels=context.y_fit,
                score=score,
                real_bacc=context.real_bacc,
                real_reference_protocol_hash=context.real_reference_protocol_hash,
                spec=context.spec,
                frame=context.frame,
                source_ids=context.source_ids,
                eval_ids=context.eval_ids,
                runtime_protocol_hash=runtime_protocol_hash,
                task_fisher_state_hash=fisher_hash,
                task_fisher_valid=True,
            )
        )
    rows.append(
        _metric_row(
            config,
            outer=context.outer,
            inner=context.inner,
            fit_centers=context.fit_centers,
            arm="B",
            objective_id=TASK_FISHER_OBJECTIVE,
            sampler=samplers[STANDARD_SAMPLER],
            runtime=runtime,
            generation_seed=-1,
            role="decode",
            labels=context.y_fit,
            score=decode_score,
            real_bacc=context.real_bacc,
            real_reference_protocol_hash=context.real_reference_protocol_hash,
            spec=context.spec,
            frame=context.frame,
            source_ids=context.source_ids,
            eval_ids=context.eval_ids,
            runtime_protocol_hash=runtime_protocol_hash,
            task_fisher_state_hash=fisher_hash,
            task_fisher_valid=True,
        )
    )
    timings.record(
        phase="sampling_and_scoring",
        elapsed_seconds=perf_counter() - evaluation_started,
        outer_target_center=context.outer,
        inner_pseudo_target_center=context.inner,
        objective_id=runtime.variant.objective_id,
        training_key_hash=runtime.training_key.hash,
    )
    return rows, sampler_rows, summaries


def _metric_row(
    config: SourceInnerPriorRecoveryConfig,
    *,
    outer: str,
    inner: str,
    fit_centers: Sequence[str],
    arm: str,
    objective_id: str,
    sampler: AggregatePosteriorSampler,
    runtime: TrainedCVAERuntime,
    generation_seed: int,
    role: str,
    labels: Sequence[int],
    score: RepresentationScore,
    real_bacc: float,
    real_reference_protocol_hash: str,
    spec: ClassifierSpec,
    frame: ExpertFeatureFrame,
    source_ids: Sequence[str],
    eval_ids: Sequence[str],
    runtime_protocol_hash: str,
    task_fisher_state_hash: str,
    task_fisher_valid: bool,
) -> dict[str, object]:
    generation_hash, evaluation_hash = generation_and_evaluation_hashes(
        runtime=runtime,
        sampler=sampler,
        generation_seed=generation_seed,
        labels=labels,
        representation_role=role,
        classifier_spec_hash=spec.config_hash,
        eval_center=inner,
        eval_ids=eval_ids,
        runtime_protocol_hash=runtime_protocol_hash,
    )
    return {
        "schema_version": SOURCE_INNER_METRIC_SCHEMA,
        "method": PRIOR_RECOVERY_METHOD,
        "protocol_hash": runtime_protocol_hash,
        "recipe_contract_hash": recipe_contract_hash(config),
        "selection_bundle_hash": "",
        "outer_target_center": outer,
        "inner_pseudo_target_center": inner,
        "fit_centers": json.dumps(list(fit_centers)),
        "arm": arm,
        "objective_id": objective_id,
        "sampler_family": sampler.requested_family,
        "requested_sampler_family": sampler.requested_family,
        "realized_sampler_by_class": json.dumps(sampler.realized_family_by_class(), sort_keys=True),
        "fallback_reason_by_class": json.dumps(sampler.fallback_reason_by_class(), sort_keys=True),
        "sampler_viable": str(sampler.requested_family_realized_for_both_classes).lower(),
        "training_seed": runtime.training_key.training_seed,
        "generation_seed": generation_seed,
        "generation_class_counts": json.dumps(
            [
                sum(int(value) == 0 for value in labels),
                sum(int(value) == 1 for value in labels),
            ]
        ),
        "representation_role": role,
        "bacc": score.bacc,
        "macro_f1": score.macro_f1,
        "real_reference_bacc": real_bacc,
        "preservation_ratio": safe_ratio(score.bacc, real_bacc, minimum_real_bacc=config.minimum_real_bacc),
        "classifier_spec_hash": spec.config_hash,
        "real_reference_protocol_hash": real_reference_protocol_hash,
        "frame_hash": frame.state_hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "training_key_hash": runtime.training_key.hash,
        "variant_hash": runtime.training_key.variant_hash,
        "stochastic_pairing_hash": runtime.training_key.stochastic_pairing_hash,
        "task_fisher_state_hash": task_fisher_state_hash,
        "sampler_state_hash": sampler.state_hash,
        "fit_row_hash": row_hash(source_ids),
        "eval_row_hash": row_hash(eval_ids),
        "generation_key_hash": generation_hash,
        "evaluation_key_hash": evaluation_hash,
        "task_fisher_valid": str(task_fisher_valid).lower(),
        "status": "ok" if score.converged else "classifier_nonconverged",
        "claim_role": "cvae_recipe_selection",
        "row_role": role,
        "leakage_status": "PASS",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "selection_source": "fully_nested_source_inner",
        "source_inner_labels_used_for_selection": "true",
        "target_eval_labels_used_for_scoring_only": "false",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "true",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
        "query_object": "none",
    }


def _sampler_rows(
    outer: str,
    inner: str,
    arm: str,
    runtime: TrainedCVAERuntime,
    sampler: AggregatePosteriorSampler,
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": SAMPLER_REALIZATION_SCHEMA,
            "outer_target_center": outer,
            "inner_pseudo_target_center": inner,
            "arm": arm,
            "checkpoint_hash": runtime.checkpoint_hash,
            "training_key_hash": runtime.training_key.hash,
            "sampler_state_hash": sampler.state_hash,
            "latent_dim": sampler.latent_dim,
            "source_row_hash": sampler.source_row_hash,
            **state.to_payload(),
        }
        for state in sampler.classes.values()
    ]


def _select_lock(
    config: SourceInnerPriorRecoveryConfig,
    summaries: Sequence[InnerCenterMetric],
    *,
    outer: str,
    expected_inner: Sequence[str],
    runtime_protocol_hash: str,
    fit_sets_hash: str,
    source_metric_hash: str,
    selection_bundle_hash: str,
    require_task_factorial: bool,
) -> RecipeLock:
    return select_recipe_lock(
        summaries,
        outer_target_center=outer,
        expected_inner_centers=expected_inner,
        generation_seeds=config.generation_seeds,
        beta_final=config.isotropic_variant.beta_final,
        classifier_grid_hash=SOURCE_INNER_CLASSIFIER_GRID_HASH,
        protocol_hash=runtime_protocol_hash,
        fit_center_sets_hash=fit_sets_hash,
        recipe_contract_hash=recipe_contract_hash(config),
        selection_bundle_hash=selection_bundle_hash,
        source_metric_table_hash=source_metric_hash,
        gate_min_ratio_improvement=config.gate_min_ratio_improvement,
        gate_min_inner_wins=min(config.gate_min_inner_wins, len(expected_inner)),
        sampler_tie_margin=config.sampler_tie_margin,
        task_increment_min_ratio=config.task_increment_min_ratio,
        safety_max_bacc_regression=config.safety_max_bacc_regression,
        minimum_real_bacc=config.minimum_real_bacc,
        require_task_factorial=require_task_factorial,
    )
