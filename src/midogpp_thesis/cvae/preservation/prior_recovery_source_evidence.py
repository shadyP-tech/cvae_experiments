"""Shared fold preparation and seed execution for source-inner prior recovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from time import perf_counter
from typing import Mapping, Sequence

from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ..feature_frame import ExpertFeatureFrame
from ..generation_samplers import (
    AggregatePosteriorSampler,
    DIAGONAL_SAMPLER,
    FULL_SAMPLER,
    STANDARD_SAMPLER,
)
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from ..task_fisher import TaskFisherMetric, fit_task_fisher_metric
from ..training import TrainedCVAERuntime
from .prior_recovery_common import (
    NO_TASK_FISHER_STATE,
    PRIOR_RECOVERY_METHOD,
    fit_samplers,
    generation_and_evaluation_hashes,
    mean,
    safe_ratio,
    train_runtime,
)
from .prior_recovery_config import (
    SourceInnerPriorRecoveryConfig,
    recipe_contract_hash,
)
from .prior_recovery_classifier import SOURCE_INNER_CLASSIFIER_GRID_HASH
from .prior_recovery_provenance import ProvenanceRecorder
from .prior_recovery_source_preparation import PreparedSourceInnerFold
from .prior_recovery_timing import RuntimeTimingRecorder
from .prior_recovery_schema import SAMPLER_REALIZATION_SCHEMA, SOURCE_INNER_METRIC_SCHEMA
from .representations import (
    decode_means,
    posterior_samples,
    sampler_decodes,
    source_budget_labels,
)
from .scoring import RepresentationScore, score_representation
from .source_inner_selection import InnerCenterMetric, RecipeLock, select_recipe_lock
from .splits import row_hash


@dataclass(frozen=True)
class _InnerFoldContext(PreparedSourceInnerFold):
    runtime_a: TrainedCVAERuntime
    decode_a: RepresentationScore
    posterior_a: Mapping[int, RepresentationScore]


def run_isotropic_seed(
    config: SourceInnerPriorRecoveryConfig,
    *,
    prepared: PreparedSourceInnerFold,
    runtime_protocol_hash: str,
    recorder: ProvenanceRecorder,
    timings: RuntimeTimingRecorder,
) -> tuple[
    _InnerFoldContext,
    list[dict[str, object]],
    list[dict[str, object]],
    list[InnerCenterMetric],
]:
    """Execute one training seed against seed-invariant H/I preparation."""

    started = perf_counter()
    runtime = train_runtime(
        config,
        variant=config.isotropic_variant,
        frame=prepared.frame,
        fit_centers=prepared.fit_centers,
        source_ids=prepared.source_ids,
        x_fit=prepared.x_fit,
        y_fit=prepared.y_fit,
        training_seed=config.selection_training_seed,
        runtime_protocol_hash=runtime_protocol_hash,
        feature_cache_hash=prepared.feature_cache_hash,
        manifest_hash=prepared.manifest_hash,
        task_metric=None,
        objective_context_hash=NO_TASK_FISHER_STATE,
        recorder=recorder,
        task_fisher_state_hash=NO_TASK_FISHER_STATE,
        classifier_spec_hash=prepared.spec.config_hash,
    )
    timings.record(
        phase="cvae_training",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=prepared.outer,
        inner_pseudo_target_center=prepared.inner,
        objective_id=runtime.variant.objective_id,
        training_key_hash=runtime.training_key.hash,
        cache_status="hit" if runtime.resumed_from_checkpoint else "miss",
    )
    evaluation_started = perf_counter()
    samplers = fit_samplers(
        config,
        runtime=runtime,
        x_fit=prepared.x_fit,
        y_fit=prepared.y_fit,
        source_ids=prepared.source_ids,
        families=(STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER),
    )
    decoded, _, _ = decode_means(runtime, prepared.x_fit, prepared.y_fit)
    decode_score = score_representation(
        decoded,
        prepared.y_fit,
        prepared.x_eval,
        prepared.y_eval,
        spec=prepared.spec,
    )
    posterior_scores = {
        seed: score_representation(
            posterior_samples(
                runtime,
                prepared.x_fit,
                prepared.y_fit,
                seed=seed,
            )[0],
            prepared.y_fit,
            prepared.x_eval,
            prepared.y_eval,
            spec=prepared.spec,
        )
        for seed in config.generation_seeds
    }
    rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    summaries: list[InnerCenterMetric] = []
    budget_labels = source_budget_labels(prepared.y_fit)
    for family in (STANDARD_SAMPLER, DIAGONAL_SAMPLER, FULL_SAMPLER):
        arm = "A" if family == STANDARD_SAMPLER else "C"
        prior_scores: dict[int, RepresentationScore] = {}
        for seed in config.generation_seeds:
            score = score_representation(
                sampler_decodes(
                    runtime,
                    samplers[family],
                    budget_labels,
                    seed=seed,
                ),
                budget_labels,
                prepared.x_eval,
                prepared.y_eval,
                spec=prepared.spec,
            )
            prior_scores[seed] = score
            rows.append(
                _metric_row(
                    config,
                    outer=prepared.outer,
                    inner=prepared.inner,
                    fit_centers=prepared.fit_centers,
                    arm=arm,
                    objective_id=ISOTROPIC_OBJECTIVE,
                    sampler=samplers[family],
                    runtime=runtime,
                    generation_seed=seed,
                    role="prior",
                    labels=budget_labels,
                    score=score,
                    real_bacc=prepared.real_bacc,
                    real_reference_protocol_hash=(
                        prepared.real_reference_protocol_hash
                    ),
                    spec=prepared.spec,
                    frame=prepared.frame,
                    source_ids=prepared.source_ids,
                    eval_ids=prepared.eval_ids,
                    runtime_protocol_hash=runtime_protocol_hash,
                    task_fisher_state_hash=NO_TASK_FISHER_STATE,
                    task_fisher_valid=True,
                )
            )
        sampler_viable = (
            samplers[family].requested_family_realized_for_both_classes
        )
        summaries.append(
            InnerCenterMetric(
                outer_target_center=prepared.outer,
                inner_pseudo_target_center=prepared.inner,
                arm=arm,
                sampler_family=family,
                objective_id=ISOTROPIC_OBJECTIVE,
                prior_ratio=mean(
                    safe_ratio(
                        score.bacc,
                        prepared.real_bacc,
                        minimum_real_bacc=config.minimum_real_bacc,
                    )
                    for score in prior_scores.values()
                ),
                decode_bacc=decode_score.bacc,
                posterior_bacc=mean(
                    score.bacc for score in posterior_scores.values()
                ),
                real_reference_bacc=prepared.real_bacc,
                valid=(
                    decode_score.converged
                    and all(score.converged for score in posterior_scores.values())
                    and all(score.converged for score in prior_scores.values())
                ),
                sampler_viable=sampler_viable,
                realized_sampler_by_class=(
                    samplers[family].realized_family_by_class()
                ),
                fallback_reason_by_class=(
                    samplers[family].fallback_reason_by_class()
                ),
            )
        )
        sampler_rows.extend(
            _sampler_rows(
                prepared.outer,
                prepared.inner,
                arm,
                runtime,
                samplers[family],
            )
        )
    for seed, score in posterior_scores.items():
        rows.append(
            _metric_row(
                config,
                outer=prepared.outer,
                inner=prepared.inner,
                fit_centers=prepared.fit_centers,
                arm="A",
                objective_id=ISOTROPIC_OBJECTIVE,
                sampler=samplers[STANDARD_SAMPLER],
                runtime=runtime,
                generation_seed=seed,
                role="posterior",
                labels=prepared.y_fit,
                score=score,
                real_bacc=prepared.real_bacc,
                real_reference_protocol_hash=(
                    prepared.real_reference_protocol_hash
                ),
                spec=prepared.spec,
                frame=prepared.frame,
                source_ids=prepared.source_ids,
                eval_ids=prepared.eval_ids,
                runtime_protocol_hash=runtime_protocol_hash,
                task_fisher_state_hash=NO_TASK_FISHER_STATE,
                task_fisher_valid=True,
            )
        )
    rows.append(
        _metric_row(
            config,
            outer=prepared.outer,
            inner=prepared.inner,
            fit_centers=prepared.fit_centers,
            arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler=samplers[STANDARD_SAMPLER],
            runtime=runtime,
            generation_seed=-1,
            role="decode",
            labels=prepared.y_fit,
            score=decode_score,
            real_bacc=prepared.real_bacc,
            real_reference_protocol_hash=prepared.real_reference_protocol_hash,
            spec=prepared.spec,
            frame=prepared.frame,
            source_ids=prepared.source_ids,
            eval_ids=prepared.eval_ids,
            runtime_protocol_hash=runtime_protocol_hash,
            task_fisher_state_hash=NO_TASK_FISHER_STATE,
            task_fisher_valid=True,
        )
    )
    context = _InnerFoldContext(
        outer=prepared.outer,
        inner=prepared.inner,
        fit_centers=prepared.fit_centers,
        spec=prepared.spec,
        selection=prepared.selection,
        frame=prepared.frame,
        x_fit=prepared.x_fit,
        y_fit=prepared.y_fit,
        source_ids=prepared.source_ids,
        x_eval=prepared.x_eval,
        y_eval=prepared.y_eval,
        eval_ids=prepared.eval_ids,
        real_bacc=prepared.real_bacc,
        real_reference_protocol_hash=prepared.real_reference_protocol_hash,
        manifest_hash=prepared.manifest_hash,
        feature_cache_hash=prepared.feature_cache_hash,
        runtime_a=runtime,
        decode_a=decode_score,
        posterior_a=posterior_scores,
    )
    timings.record(
        phase="sampling_and_scoring",
        elapsed_seconds=perf_counter() - evaluation_started,
        outer_target_center=prepared.outer,
        inner_pseudo_target_center=prepared.inner,
        objective_id=runtime.variant.objective_id,
        training_key_hash=runtime.training_key.hash,
    )
    return context, rows, sampler_rows, summaries


@dataclass(frozen=True)
class SharedTaskFisherState:
    metric: TaskFisherMetric
    state_hash: str


def fit_shared_task_fisher_state(
    context: PreparedSourceInnerFold,
    *,
    recorder: ProvenanceRecorder,
    timings: RuntimeTimingRecorder,
) -> SharedTaskFisherState:
    started = perf_counter()
    fisher = fit_task_fisher_metric(
        context.x_fit,
        context.y_fit,
        spec=context.spec,
        alpha=1.0,
    )
    fisher_hash = recorder.record_fisher(fisher)
    timings.record(
        phase="task_fisher_fit",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=context.outer,
        inner_pseudo_target_center=context.inner,
        objective_id=TASK_FISHER_OBJECTIVE,
    )
    return SharedTaskFisherState(metric=fisher, state_hash=fisher_hash)


def _run_task_fold(
    config: SourceInnerPriorRecoveryConfig,
    *,
    context: _InnerFoldContext,
    selected_family: str,
    runtime_protocol_hash: str,
    recorder: ProvenanceRecorder,
    timings: RuntimeTimingRecorder,
    shared_fisher: SharedTaskFisherState | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[InnerCenterMetric]]:
    if shared_fisher is None:
        shared_fisher = fit_shared_task_fisher_state(
            context,
            recorder=recorder,
            timings=timings,
        )
    fisher = shared_fisher.metric
    fisher_hash = shared_fisher.state_hash
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


# Public evidence-kernel names used by the scalar and multiseed orchestrators.
SourceInnerSeedContext = _InnerFoldContext
run_isotropic_seed_from_prepared = run_isotropic_seed
run_task_fold = _run_task_fold
select_source_inner_lock = _select_lock
