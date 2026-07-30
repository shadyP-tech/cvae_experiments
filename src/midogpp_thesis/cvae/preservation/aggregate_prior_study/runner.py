"""Run the independent-source aggregate-posterior mixture plus GECO study."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...reporting import prepare_artifact_dirs, write_csv_rows, write_json
from ..prior_recovery_runtime_cache import FeatureFrameCache
from ..scoring import chance_normalized_preservation, score_representation
from ..splits import row_hash
from .checkpoint_store import SourceExpertCheckpointStore
from .config import AggregatePriorStudyConfig
from .contracts import (
    ARMS,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    PRIMARY_ARM,
    STANDARD_FIXED,
    SourceExpertEvaluationKey,
    SourceExpertTrainingKey,
    arm_contract,
    objective_family,
    prior_family,
    rate_family,
)
from .preparation import (
    PreparedEvaluation,
    PreparedSourceExpert,
    isolation_audit,
    load_frame,
    prepare_evaluation,
    prepare_source_expert,
)
from .training import (
    SourceExpertRuntime,
    generate_projected,
    paired_generation_noise,
    posterior_projected,
    train_source_expert_panel,
)


METRIC_SCHEMA = "midogpp_independent_source_inner_metric_v3"
PROTOCOL_SCHEMA = "midogpp_aggregate_prior_protocol_v3"
SELECTION_SCHEMA = "midogpp_aggregate_prior_selection_evidence_v3"
COVERAGE_SCHEMA = "midogpp_aggregate_prior_coverage_v3"
PUBLICATION_SCHEMA = "midogpp_aggregate_prior_publication_state_v3"


def run_aggregate_prior_source_inner_study(
    config: AggregatePriorStudyConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(artifact_root or config.artifact_root)
    started = perf_counter()
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_aggregate_prior_run_state_v3",
            "status": "RUNNING",
            "mode": config.mode,
            "claim_scope": CLAIM_SCOPE,
        },
    )
    try:
        result = _run(config, root=root, started=started)
    except Exception as exc:
        write_json(
            root / "reports/run_state.json",
            {
                "schema_version": "midogpp_aggregate_prior_run_state_v3",
                "status": "FAILED",
                "mode": config.mode,
                "claim_scope": CLAIM_SCOPE,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    return result


def _run(
    config: AggregatePriorStudyConfig,
    *,
    root: Path,
    started: float,
) -> Path:
    frame = load_frame(config)
    protocol = _protocol_manifest(config, frame=frame)
    protocol_hash = str(protocol["protocol_hash"])
    write_json(root / "manifests/protocol_manifest.json", protocol)

    frame_cache = FeatureFrameCache(root)
    checkpoint_store = SourceExpertCheckpointStore(root, config)
    sources = {
        source_center: prepare_source_expert(
            config,
            frame=frame,
            source_center=source_center,
            frame_cache=frame_cache,
            protocol_hash=protocol_hash,
        )
        for source_center in config.heldout_centers
    }
    frame_cache.write_index()
    evaluations, isolation_rows, reference_rows, tuning_rows = _prepare_evaluations(
        config,
        frame=frame,
        sources=sources,
    )
    if any(row["status"] != "PASS" for row in isolation_rows):
        raise ProtocolError("Independent-source isolation audit failed.")

    metric_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    mixture_rows: list[dict[str, object]] = []
    geco_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    generation_budget_rows: list[dict[str, object]] = []
    rng_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    generation_labels = tuple(
        [0] * config.generation_per_class
        + [1] * config.generation_per_class
    )
    for source_center, source in sources.items():
        for training_seed in config.training_seeds:
            training_keys = {
                arm: _training_key(
                    config,
                    source=source,
                    training_seed=training_seed,
                    arm=arm,
                    protocol_hash=protocol_hash,
                )
                for arm in ARMS
            }
            train_started = perf_counter()
            loaded = {
                arm: checkpoint_store.load(training_keys[arm]) for arm in ARMS
            }
            if all(runtime is not None for runtime in loaded.values()):
                runtimes = {
                    arm: runtime
                    for arm, runtime in loaded.items()
                    if runtime is not None
                }
                cache_status = "hit"
            else:
                trained = train_source_expert_panel(
                    source.x_projected,
                    source.labels,
                    source.case_ids,
                    config=config,
                    training_keys=training_keys,
                )
                for arm in ARMS:
                    existing = loaded[arm]
                    runtime = trained[arm]
                    if (
                        existing is not None
                        and existing.checkpoint_hash != runtime.checkpoint_hash
                    ):
                        raise ProtocolError(
                            "Recomputed paired panel differs from an existing checkpoint."
                        )
                    checkpoint_store.save(runtime)
                runtimes = dict(trained)
                cache_status = "miss"
            timing_rows.append(
                {
                    "phase": "source_expert_panel_training",
                    "source_center": source_center,
                    "training_seed": training_seed,
                    "elapsed_seconds": perf_counter() - train_started,
                    "cache_status": cache_status,
                }
            )
            _append_training_artifacts(
                runtimes,
                source=source,
                training_rows=training_rows,
                mixture_rows=mixture_rows,
                geco_rows=geco_rows,
                epoch_rows=epoch_rows,
            )

            for evaluation in evaluations[source_center]:
                for generation_seed in config.generation_seeds:
                    neutral_evaluation_hash = stable_hash(
                        {
                            "protocol_hash": protocol_hash,
                            "outer_target_center": (
                                evaluation.outer_target_center
                            ),
                            "inner_pseudo_target_center": (
                                evaluation.inner_pseudo_target_center
                            ),
                            "source_center": source_center,
                            "generation_seed": generation_seed,
                            "generation_labels": generation_labels,
                        }
                    )
                    epsilon, component_uniform, noise_hash = (
                        paired_generation_noise(
                            neutral_evaluation_hash=neutral_evaluation_hash,
                            labels=generation_labels,
                            latent_dim=config.latent_dim,
                        )
                    )
                    posterior_indices, posterior_source_index_hash = (
                        _balanced_source_indices(
                            source.labels,
                            per_class=config.generation_per_class,
                            neutral_evaluation_hash=neutral_evaluation_hash,
                        )
                    )
                    generation_budget_rows.append(
                        {
                            "schema_version": (
                                "midogpp_fixed_source_generation_budget_v3"
                            ),
                            "outer_target_center": (
                                evaluation.outer_target_center
                            ),
                            "inner_pseudo_target_center": (
                                evaluation.inner_pseudo_target_center
                            ),
                            "source_center": source_center,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "per_class": config.generation_per_class,
                            "total": len(generation_labels),
                            "inner_prevalence_used": False,
                            "source_prevalence_used": False,
                            "same_across_arms": True,
                        }
                    )
                    rng_rows.append(
                        {
                            "schema_version": "midogpp_v3_rng_pairing_audit_v1",
                            "outer_target_center": (
                                evaluation.outer_target_center
                            ),
                            "inner_pseudo_target_center": (
                                evaluation.inner_pseudo_target_center
                            ),
                            "source_center": source_center,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "noise_hash": noise_hash,
                            "posterior_source_index_hash": (
                                posterior_source_index_hash
                            ),
                            "same_epsilon_across_arms": True,
                            "same_component_uniform_across_arms": True,
                        }
                    )
                    for arm in ARMS:
                        runtime = runtimes[arm]
                        projected = generate_projected(
                            runtime,
                            generation_labels,
                            epsilon=epsilon,
                            component_uniform=component_uniform,
                        )
                        generated_full = np.asarray(
                            source.frame.inverse_transform(projected),
                            dtype=np.float32,
                        )
                        if generated_full.shape != (
                            len(generation_labels),
                            config.expected_feature_dim,
                        ):
                            raise ProtocolError(
                                "Generated source-local PCA samples were not "
                                "mapped back to the common 2560-d frame."
                            )
                        score = score_representation(
                            generated_full,
                            generation_labels,
                            evaluation.x_eval_full,
                            evaluation.y_eval,
                            spec=evaluation.selection.selected_spec,
                        )
                        posterior_generated = np.asarray(
                            source.frame.inverse_transform(
                                posterior_projected(
                                    runtime,
                                    source.x_projected[
                                        list(posterior_indices)
                                    ],
                                    generation_labels,
                                    epsilon=epsilon,
                                )
                            ),
                            dtype=np.float32,
                        )
                        if posterior_generated.shape != generated_full.shape:
                            raise ProtocolError(
                                "Posterior reference did not return to the "
                                "common Virchow2 frame."
                            )
                        posterior_score = score_representation(
                            posterior_generated,
                            generation_labels,
                            evaluation.x_eval_full,
                            evaluation.y_eval,
                            spec=evaluation.selection.selected_spec,
                        )
                        evaluation_keys = {
                            representation_role: SourceExpertEvaluationKey(
                                outer_target_center=(
                                    evaluation.outer_target_center
                                ),
                                inner_pseudo_target_center=(
                                    evaluation.inner_pseudo_target_center
                                ),
                                source_center=source_center,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                arm=arm,
                                representation_role=representation_role,
                                generation_noise_hash=noise_hash,
                                posterior_source_index_hash=(
                                    posterior_source_index_hash
                                    if representation_role == "posterior"
                                    else None
                                ),
                                training_key_hash=runtime.training_key.hash,
                                inner_eval_row_hash=evaluation.eval_row_hash,
                                classifier_spec_hash=(
                                    evaluation.selection.selected_spec.config_hash
                                ),
                                protocol_hash=protocol_hash,
                            )
                            for representation_role in ("prior", "posterior")
                        }
                        metric_rows.append(
                            _metric_row(
                                config,
                                source=source,
                                evaluation=evaluation,
                                runtime=runtime,
                                evaluation_key=evaluation_keys["prior"],
                                generation_seed=generation_seed,
                                generation_labels=generation_labels,
                                noise_hash=noise_hash,
                                score=score,
                                representation_role="prior",
                                posterior_source_index_hash=(
                                    posterior_source_index_hash
                                ),
                            )
                        )
                        metric_rows.append(
                            _metric_row(
                                config,
                                source=source,
                                evaluation=evaluation,
                                runtime=runtime,
                                evaluation_key=evaluation_keys["posterior"],
                                generation_seed=generation_seed,
                                generation_labels=generation_labels,
                                noise_hash=noise_hash,
                                score=posterior_score,
                                representation_role="posterior",
                                posterior_source_index_hash=(
                                    posterior_source_index_hash
                                ),
                            )
                        )

    checkpoint_index = checkpoint_store.write_index()
    mixture_index, geco_index = _write_state_indexes(
        root,
        mixture_rows=mixture_rows,
        geco_rows=geco_rows,
        checkpoint_records=tuple(checkpoint_store.records.values()),
    )
    coverage = _coverage_manifest(
        config,
        metric_rows=metric_rows,
        training_rows=training_rows,
        isolation_rows=isolation_rows,
    )
    write_json(root / "manifests/coverage_manifest.json", coverage)
    budget_manifest = {
        "schema_version": "midogpp_fixed_source_generation_budget_manifest_v3",
        "policy": "fixed_balanced_per_source_per_class",
        "per_source_per_class": config.generation_per_class,
        "class_labels": [0, 1],
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "inner_or_outer_prevalence_used": False,
        "n_rows": len(generation_budget_rows),
    }
    write_json(
        root / "manifests/generation_budget_manifest.json",
        budget_manifest,
    )

    child_decisions, consensus_decisions, delta_rows = _decisions(
        config,
        metric_rows=metric_rows,
        protocol_hash=protocol_hash,
    )
    for (training_seed, outer), decision in child_decisions.items():
        write_json(
            root
            / f"reports/child_decisions/seed{training_seed}/{outer}.json",
            decision,
        )
    for outer, decision in consensus_decisions.items():
        write_json(
            root / f"reports/consensus_decisions/{outer}.json",
            decision,
        )

    selection_hash = stable_hash(
        {
            "protocol_hash": protocol_hash,
            "metric_rows": _canonical_rows(metric_rows),
            "delta_rows": _canonical_rows(delta_rows),
            "training_rows": _canonical_rows(training_rows),
            "isolation_rows": _canonical_rows(isolation_rows),
            "source_local_real_reference_rows": _canonical_rows(
                reference_rows
            ),
            "nested_classifier_tuning_rows": _canonical_rows(tuning_rows),
            "training_epoch_rows": _canonical_rows(epoch_rows),
            "generation_budget_rows": _canonical_rows(
                generation_budget_rows
            ),
            "rng_pairing_rows": _canonical_rows(rng_rows),
            "coverage": coverage,
            "checkpoint_index_path": checkpoint_index.relative_to(root).as_posix(),
            "checkpoint_index_hash": stable_hash(
                json.loads(checkpoint_index.read_text(encoding="utf-8"))
            ),
            "mixture_index_hash": stable_hash(mixture_index),
            "geco_index_hash": stable_hash(geco_index),
            "generation_budget_manifest": budget_manifest,
        }
    )
    selection_manifest = {
        "schema_version": SELECTION_SCHEMA,
        "selection_evidence_hash": selection_hash,
        "primary_arm": PRIMARY_ARM,
        "decisions_may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
        "source_local_axis_complete": coverage["status"] == "PASS",
        "runtime_rows_included": False,
    }
    write_json(
        root / "manifests/selection_evidence_manifest.json",
        selection_manifest,
    )
    study_decision = _study_decision(
        consensus_decisions,
        protocol_hash=protocol_hash,
        selection_evidence_hash=selection_hash,
    )
    write_json(root / "reports/study_decision.json", study_decision)
    write_json(
        root / "reports/expert_isolation_report.json",
        {
            "schema_version": "midogpp_expert_isolation_report_v3",
            "status": (
                "PASS"
                if all(row["status"] == "PASS" for row in isolation_rows)
                else "FAIL"
            ),
            "n_cells": len(isolation_rows),
            "fit_centers_per_checkpoint": 1,
            "training_key_is_outer_inner_neutral": True,
        },
    )
    write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_aggregate_prior_leakage_report_v3",
            "status": "PASS",
            "protocol_hash": protocol_hash,
            "outer_target_rows_used": False,
            "outer_target_labels_used": False,
            "inner_rows_used_for_fit": False,
            "inner_labels_used_for_scoring_only": True,
            "source_experts_independently_trained": True,
            "target_or_inner_data_used_for_geco_target": False,
            "target_or_inner_data_used_for_mixture_fit": False,
            "target_or_inner_prevalence_used_for_generation_budget": False,
            "identity_overlap_pass": True,
        },
    )
    write_json(
        root / "reports/publication_state.json",
        {
            "schema_version": PUBLICATION_SCHEMA,
            "status": "NON_CONSUMABLE_STUDY_COMPLETE",
            "claim_scope": CLAIM_SCOPE,
            "may_feed_model_recipe": False,
            "may_feed_expert_bank": False,
            "stage30_recipe_ready": False,
            "separate_promotion_artifact_required": True,
            "selection_evidence_hash": selection_hash,
        },
    )
    elapsed = perf_counter() - started
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_aggregate_prior_runtime_summary_v3",
            "elapsed_seconds": elapsed,
            "n_source_expert_checkpoints": len(training_rows),
            "n_metric_rows": len(metric_rows),
            "n_isolation_cells": len(isolation_rows),
        },
    )

    write_csv_rows(root / "tables/source_expert_metrics.csv", metric_rows)
    write_csv_rows(root / "tables/paired_deltas.csv", delta_rows)
    write_csv_rows(root / "tables/source_local_real_references.csv", reference_rows)
    write_csv_rows(root / "tables/nested_classifier_tuning.csv", tuning_rows)
    write_csv_rows(root / "tables/source_expert_training_audit.csv", training_rows)
    write_csv_rows(root / "tables/mixture_prior_diagnostics.csv", mixture_rows)
    write_csv_rows(root / "tables/geco_trajectory.csv", geco_rows)
    write_csv_rows(root / "tables/training_epochs.csv", epoch_rows)
    write_csv_rows(
        root / "tables/generation_budget_audit.csv",
        generation_budget_rows,
    )
    write_csv_rows(root / "tables/rng_pairing_audit.csv", rng_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", isolation_rows)
    write_csv_rows(root / "tables/runtime_timings.csv", timing_rows)

    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_aggregate_prior_run_state_v3",
            "status": "COMPLETE",
            "mode": config.mode,
            "claim_scope": CLAIM_SCOPE,
            "protocol_hash": protocol_hash,
            "selection_evidence_hash": selection_hash,
        },
    )
    return root


def _prepare_evaluations(
    config: AggregatePriorStudyConfig,
    *,
    frame: object,
    sources: Mapping[str, PreparedSourceExpert],
) -> tuple[
    Mapping[str, tuple[PreparedEvaluation, ...]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    evaluations: dict[str, list[PreparedEvaluation]] = defaultdict(list)
    isolation_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    recorded_tuning: set[tuple[str, str]] = set()
    for outer in config.heldout_centers:
        for inner in config.heldout_centers:
            if inner == outer:
                continue
            for source_center, source in sources.items():
                if source_center in {outer, inner}:
                    continue
                evaluation = prepare_evaluation(
                    frame=frame,  # type: ignore[arg-type]
                    source=source,
                    outer_target_center=outer,
                    inner_pseudo_target_center=inner,
                )
                evaluations[source_center].append(evaluation)
                audit = dict(isolation_audit(source=source, evaluation=evaluation))
                isolation_rows.append(audit)
                reference_rows.append(
                    {
                        "schema_version": "midogpp_source_local_real_reference_v3",
                        "outer_target_center": outer,
                        "inner_pseudo_target_center": inner,
                        "source_center": source_center,
                        "source_row_hash": source.source_row_hash,
                        "inner_eval_row_hash": evaluation.eval_row_hash,
                        "classifier_spec_hash": (
                            evaluation.selection.selected_spec.config_hash
                        ),
                        "bacc": evaluation.real_source_score.bacc,
                        "macro_f1": evaluation.real_source_score.macro_f1,
                        "converged": evaluation.real_source_score.converged,
                        "inner_labels_used_for_scoring_only": True,
                    }
                )
                tuning_key = (outer, inner)
                if tuning_key not in recorded_tuning:
                    tuning_rows.extend(
                        {
                            **dict(row),
                            "outer_target_center": outer,
                            "inner_pseudo_target_center": inner,
                            "outer_or_inner_labels_used_for_selection": False,
                        }
                        for row in evaluation.selection.candidate_rows
                    )
                    recorded_tuning.add(tuning_key)
    return (
        {key: tuple(value) for key, value in evaluations.items()},
        isolation_rows,
        reference_rows,
        tuning_rows,
    )


def _training_key(
    config: AggregatePriorStudyConfig,
    *,
    source: PreparedSourceExpert,
    training_seed: int,
    arm: str,
    protocol_hash: str,
) -> SourceExpertTrainingKey:
    return SourceExpertTrainingKey(
        source_center=source.source_center,
        training_seed=training_seed,
        arm=arm,
        source_row_hash=source.source_row_hash,
        source_case_hash=source.source_case_hash,
        source_frame_hash=source.frame.state_hash,
        manifest_hash=source.manifest_hash,
        feature_cache_hash=source.feature_cache_hash,
        protocol_hash=protocol_hash,
        config_hash=config.contract_hash,
    )


def _metric_row(
    config: AggregatePriorStudyConfig,
    *,
    source: PreparedSourceExpert,
    evaluation: PreparedEvaluation,
    runtime: SourceExpertRuntime,
    evaluation_key: SourceExpertEvaluationKey,
    generation_seed: int,
    generation_labels: Sequence[int],
    noise_hash: str,
    score: object,
    representation_role: str,
    posterior_source_index_hash: str,
) -> dict[str, object]:
    real_bacc = evaluation.real_source_score.bacc
    try:
        ratio = chance_normalized_preservation(
            score.bacc,  # type: ignore[attr-defined]
            real_bacc,
            minimum_real_bacc=0.55,
        )
    except ValueError:
        ratio = float("nan")
    valid = bool(
        score.converged  # type: ignore[attr-defined]
        and math.isfinite(float(score.bacc))  # type: ignore[attr-defined]
        and math.isfinite(float(score.macro_f1))  # type: ignore[attr-defined]
    )
    return {
        "schema_version": METRIC_SCHEMA,
        "method": "independent_source_aggregate_posterior_mixture_geco_v3",
        "outer_target_center": evaluation.outer_target_center,
        "inner_pseudo_target_center": evaluation.inner_pseudo_target_center,
        "source_center": source.source_center,
        "fit_centers": json.dumps([source.source_center]),
        "training_seed": runtime.training_key.training_seed,
        "generation_seed": generation_seed,
        "arm": runtime.arm,
        "prior_family": prior_family(runtime.arm),
        "objective_family": objective_family(runtime.arm),
        "rate_family": rate_family(runtime.arm),
        "rate_is_exact_nelbo": False,
        "representation_role": representation_role,
        "bacc": score.bacc,  # type: ignore[attr-defined]
        "macro_f1": score.macro_f1,  # type: ignore[attr-defined]
        "source_local_real_bacc": real_bacc,
        "preservation_ratio": ratio,
        "generation_class_counts": json.dumps(
            [
                sum(int(label) == class_label for label in generation_labels)
                for class_label in (0, 1)
            ]
        ),
        "generated_output_dim": config.expected_feature_dim,
        "inverse_transformed_to_common_frame": True,
        "classifier_spec_hash": (
            evaluation.selection.selected_spec.config_hash
        ),
        "source_frame_hash": source.frame.state_hash,
        "source_fit_row_hash": source.source_row_hash,
        "inner_eval_row_hash": evaluation.eval_row_hash,
        "training_key_hash": runtime.training_key.hash,
        "evaluation_key_hash": evaluation_key.hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "warmup_checkpoint_hash": runtime.warmup_checkpoint_hash,
        "training_stream_hash": runtime.training_stream_hash,
        "noise_hash": noise_hash,
        "posterior_source_index_hash": posterior_source_index_hash,
        "protocol_hash": evaluation_key.protocol_hash,
        "valid": str(valid).lower(),
        "eligible": str(valid).lower(),
        "status": "ok" if valid else "invalid_or_nonconverged",
        "claim_scope": CLAIM_SCOPE,
        "selection_source": "fully_nested_independent_source_inner",
        "outer_target_rows_used": "false",
        "inner_rows_used_for_fit": "false",
        "inner_labels_used_for_scoring_only": "true",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
    }


def _append_training_artifacts(
    runtimes: Mapping[str, SourceExpertRuntime],
    *,
    source: PreparedSourceExpert,
    training_rows: list[dict[str, object]],
    mixture_rows: list[dict[str, object]],
    geco_rows: list[dict[str, object]],
    epoch_rows: list[dict[str, object]],
) -> None:
    for arm in ARMS:
        runtime = runtimes[arm]
        training_rows.append(
            {
                "schema_version": "midogpp_source_expert_training_audit_v3",
                "source_center": source.source_center,
                "fit_centers": json.dumps([source.source_center]),
                "training_seed": runtime.training_key.training_seed,
                "arm": arm,
                "training_key_hash": runtime.training_key.hash,
                "checkpoint_hash": runtime.checkpoint_hash,
                "warmup_checkpoint_hash": runtime.warmup_checkpoint_hash,
                "shared_initialization_hash": runtime.shared_initialization_hash,
                "training_stream_hash": runtime.training_stream_hash,
                "source_row_hash": source.source_row_hash,
                "source_case_hash": source.source_case_hash,
                "source_frame_hash": source.frame.state_hash,
                "manifest_hash": runtime.training_key.manifest_hash,
                "feature_cache_hash": runtime.training_key.feature_cache_hash,
                "protocol_hash": runtime.training_key.protocol_hash,
                "config_hash": runtime.training_key.config_hash,
                "outer_or_inner_identity_in_key": False,
                "source_only": True,
            }
        )
        for record in runtime.mixture_refit_records:
            diagnostics = record["diagnostics"]
            initialization = record["initialization"]
            mixture_rows.append(
                {
                    "schema_version": (
                        "midogpp_mixture_prior_diagnostic_row_v3"
                    ),
                    "source_center": source.source_center,
                    "training_seed": runtime.training_key.training_seed,
                    "arm": arm,
                    "training_key_hash": runtime.training_key.hash,
                    "refit_index": record["refit_index"],
                    "after_continuation_epoch": (
                        record["after_continuation_epoch"]
                    ),
                    "state_hash": record["state_hash"],
                    "record_hash": record["record_hash"],
                    "coordinate_update": record["coordinate_update"],
                    "optimizer_updates_prior_parameters": record[
                        "optimizer_updates_prior_parameters"
                    ],
                    "component_row_counts": json.dumps(
                        initialization["component_row_counts"]  # type: ignore[index]
                    ),
                    "component_case_counts": json.dumps(
                        initialization["component_case_counts"]  # type: ignore[index]
                    ),
                    "assignment_fallbacks": json.dumps(
                        initialization["assignment_fallbacks"]  # type: ignore[index]
                    ),
                    "covariance_fallbacks": json.dumps(
                        initialization["covariance_fallbacks"]  # type: ignore[index]
                    ),
                    "minimum_weight": diagnostics["minimum_weight"],  # type: ignore[index]
                    "maximum_condition_number": diagnostics[
                        "maximum_condition_number"  # type: ignore[index]
                    ],
                    "minimum_eigenvalue": diagnostics["minimum_eigenvalue"],  # type: ignore[index]
                    "finite": diagnostics["finite"],  # type: ignore[index]
                    "weight_floor_respected": diagnostics[
                        "weight_floor_respected"  # type: ignore[index]
                    ],
                    "covariance_positive_definite": diagnostics[
                        "covariance_positive_definite"  # type: ignore[index]
                    ],
                    "fit_scope": "source_center_only_all_rows",
                }
            )
        geco_rows.extend(dict(row) for row in runtime.geco_trajectory)
        epoch_rows.extend(dict(row) for row in runtime.epoch_diagnostics)


def _write_state_indexes(
    root: Path,
    *,
    mixture_rows: Sequence[Mapping[str, object]],
    geco_rows: Sequence[Mapping[str, object]],
    checkpoint_records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    mixture_index = {
        "schema_version": "midogpp_mixture_prior_state_index_v3",
        "n_refit_records": len(mixture_rows),
        "records": list(mixture_rows),
    }
    write_json(
        root / "manifests/mixture_prior_state_index.json",
        mixture_index,
    )
    geco_index = _geco_state_index_payload(
        geco_rows=geco_rows,
        checkpoint_records=checkpoint_records,
    )
    write_json(root / "manifests/geco_state_index.json", geco_index)
    return mixture_index, geco_index


def _geco_state_index_payload(
    *,
    geco_rows: Sequence[Mapping[str, object]],
    checkpoint_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    final_geco: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in geco_rows:
        key = (
            str(row["training_key_hash"]),
            str(row["arm"]),
        )
        final_geco[key] = row
    controller_states = [
        {
            "training_key_hash": str(record["training_key_hash"]),
            "arm": str(record["arm"]),
            "checkpoint_hash": str(record["checkpoint_hash"]),
            "controller_state_hash": stable_hash(record["geco_state"]),
            "controller_state": record["geco_state"],
        }
        for record in checkpoint_records
        if record.get("geco_state") is not None
    ]
    controller_states.sort(
        key=lambda row: (row["training_key_hash"], row["arm"])
    )
    return {
        "schema_version": "midogpp_geco_state_index_v3",
        "n_trajectory_rows": len(geco_rows),
        "trajectory_hash": stable_hash(_canonical_rows(geco_rows)),
        "n_controller_states": len(controller_states),
        "controller_states": controller_states,
        "final_trajectory_rows": [
            _canonical_rows([final_geco[key]])[0]
            for key in sorted(final_geco)
        ],
    }


def _coverage_manifest(
    config: AggregatePriorStudyConfig,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    training_rows: Sequence[Mapping[str, object]],
    isolation_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    n_centers = len(config.heldout_centers)
    expected_training = n_centers * len(config.training_seeds) * len(ARMS)
    expected_isolation = n_centers * (n_centers - 1) * (n_centers - 2)
    expected_metrics = (
        expected_isolation
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * len(ARMS)
        * 2
    )
    passed = (
        len(training_rows) == expected_training
        and len(isolation_rows) == expected_isolation
        and len(metric_rows) == expected_metrics
        and all(str(row.get("valid")) == "true" for row in metric_rows)
        and all(row.get("status") == "PASS" for row in isolation_rows)
    )
    return {
        "schema_version": COVERAGE_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "expected_training_checkpoints": expected_training,
        "observed_training_checkpoints": len(training_rows),
        "expected_isolation_cells": expected_isolation,
        "observed_isolation_cells": len(isolation_rows),
        "expected_metric_rows": expected_metrics,
        "observed_metric_rows": len(metric_rows),
        "axes": {
            "outer_target_centers": list(config.heldout_centers),
            "inner_pseudo_target_centers": "all_except_outer",
            "source_centers": "all_except_outer_and_inner",
            "training_seeds": list(config.training_seeds),
            "generation_seeds": list(config.generation_seeds),
            "arms": list(ARMS),
        },
    }


def _decisions(
    config: AggregatePriorStudyConfig,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    protocol_hash: str,
) -> tuple[
    Mapping[tuple[int, str], Mapping[str, object]],
    Mapping[str, Mapping[str, object]],
    list[dict[str, object]],
]:
    if not config.equal_weight_sources_then_inner_centers:
        raise ProtocolError(
            "V3 decisions require equal source and inner-center weighting."
        )
    child: dict[tuple[int, str], Mapping[str, object]] = {}
    consensus: dict[str, Mapping[str, object]] = {}
    delta_rows: list[dict[str, object]] = []
    for outer in config.heldout_centers:
        per_seed_primary: dict[int, float] = {}
        for training_seed in config.training_seeds:
            inner_arm: dict[tuple[str, str], float] = {}
            for inner in config.heldout_centers:
                if inner == outer:
                    continue
                for arm in ARMS:
                    values = [
                        float(row["bacc"])
                        for row in metric_rows
                        if row["outer_target_center"] == outer
                        and row["inner_pseudo_target_center"] == inner
                        and int(row["training_seed"]) == training_seed
                        and row["arm"] == arm
                        and row["representation_role"] == "prior"
                        and row["valid"] == "true"
                    ]
                    expected = (
                        (len(config.heldout_centers) - 2)
                        * len(config.generation_seeds)
                    )
                    if len(values) != expected:
                        raise ProtocolError(
                            "Decision cell lacks equal source/generation coverage."
                        )
                    inner_arm[(inner, arm)] = sum(values) / len(values)
            posterior_inner_arm: dict[tuple[str, str], float] = {}
            for inner in config.heldout_centers:
                if inner == outer:
                    continue
                for arm in ARMS:
                    values = [
                        float(row["bacc"])
                        for row in metric_rows
                        if row["outer_target_center"] == outer
                        and row["inner_pseudo_target_center"] == inner
                        and int(row["training_seed"]) == training_seed
                        and row["arm"] == arm
                        and row["representation_role"] == "posterior"
                        and row["valid"] == "true"
                    ]
                    expected = (
                        (len(config.heldout_centers) - 2)
                        * len(config.generation_seeds)
                    )
                    if len(values) != expected:
                        raise ProtocolError(
                            "Posterior decision cell lacks equal coverage."
                        )
                    posterior_inner_arm[(inner, arm)] = (
                        sum(values) / len(values)
                    )
            primary_deltas = {
                inner: inner_arm[(inner, PRIMARY_ARM)]
                - inner_arm[(inner, STANDARD_FIXED)]
                for inner in config.heldout_centers
                if inner != outer
            }
            for inner, delta in primary_deltas.items():
                delta_rows.append(
                    {
                        "schema_version": "midogpp_v3_paired_inner_delta_v1",
                        "outer_target_center": outer,
                        "inner_pseudo_target_center": inner,
                        "training_seed": training_seed,
                        "candidate_arm": PRIMARY_ARM,
                        "comparator_arm": STANDARD_FIXED,
                        "delta_bacc": delta,
                        "equal_source_weighting": True,
                        "equal_inner_weighting": True,
                    }
                )
            mean_delta = _mean(primary_deltas.values())
            wins = sum(value > 0.0 for value in primary_deltas.values())
            worst = min(primary_deltas.values())
            kg_values = [
                inner_arm[(inner, PRIMARY_ARM)]
                for inner in config.heldout_centers
                if inner != outer
            ]
            per_seed_primary[training_seed] = _mean(kg_values)
            kg_vs_kf = _mean(
                inner_arm[(inner, PRIMARY_ARM)] - inner_arm[(inner, "KF")]
                for inner in config.heldout_centers
                if inner != outer
            )
            kg_vs_sg = _mean(
                inner_arm[(inner, PRIMARY_ARM)] - inner_arm[(inner, "SG")]
                for inner in config.heldout_centers
                if inner != outer
            )
            posterior_kg_vs_sf = _mean(
                posterior_inner_arm[(inner, PRIMARY_ARM)]
                - posterior_inner_arm[(inner, STANDARD_FIXED)]
                for inner in config.heldout_centers
                if inner != outer
            )
            kg_prior_posterior_gap = _mean(
                posterior_inner_arm[(inner, PRIMARY_ARM)]
                - inner_arm[(inner, PRIMARY_ARM)]
                for inner in config.heldout_centers
                if inner != outer
            )
            sf_prior_posterior_gap = _mean(
                posterior_inner_arm[(inner, STANDARD_FIXED)]
                - inner_arm[(inner, STANDARD_FIXED)]
                for inner in config.heldout_centers
                if inner != outer
            )
            gap_reduction = sf_prior_posterior_gap - kg_prior_posterior_gap
            passed = (
                mean_delta >= config.min_mean_bacc_delta_vs_sf
                and wins >= config.min_inner_wins
                and worst >= config.max_worst_inner_regression
                and (
                    not config.require_positive_delta_vs_kf
                    or kg_vs_kf > 0.0
                )
                and (
                    not config.require_positive_delta_vs_sg
                    or kg_vs_sg > 0.0
                )
                and posterior_kg_vs_sf
                >= config.max_posterior_bacc_regression
                and gap_reduction
                >= config.min_prior_posterior_gap_reduction
            )
            child[(training_seed, outer)] = {
                "schema_version": "midogpp_v3_child_decision_v1",
                "outer_target_center": outer,
                "training_seed": training_seed,
                "primary_arm": PRIMARY_ARM,
                "mean_bacc_delta_vs_sf": mean_delta,
                "inner_wins_vs_sf": wins,
                "worst_inner_delta_vs_sf": worst,
                "mean_bacc_delta_vs_kf": kg_vs_kf,
                "mean_bacc_delta_vs_sg": kg_vs_sg,
                "posterior_mean_bacc_delta_vs_sf": posterior_kg_vs_sf,
                "kg_prior_posterior_bacc_gap": kg_prior_posterior_gap,
                "sf_prior_posterior_bacc_gap": sf_prior_posterior_gap,
                "prior_posterior_gap_reduction": gap_reduction,
                "gate_pass": passed,
                "decision": (
                    "SOURCE_LOCAL_PANEL_SUPPORTS_KG"
                    if passed
                    else "SOURCE_LOCAL_PANEL_DOES_NOT_SUPPORT_KG"
                ),
                "may_feed_model_recipe": False,
                "protocol_hash": protocol_hash,
            }
        seed_range = max(per_seed_primary.values()) - min(
            per_seed_primary.values()
        )
        all_seed_pass = all(
            bool(child[(seed, outer)]["gate_pass"])
            for seed in config.training_seeds
        )
        stable = seed_range <= config.max_training_seed_range
        consensus[outer] = {
            "schema_version": "midogpp_v3_consensus_decision_v1",
            "outer_target_center": outer,
            "training_seeds": list(config.training_seeds),
            "primary_arm": PRIMARY_ARM,
            "all_training_seeds_pass": all_seed_pass,
            "primary_arm_training_seed_range": seed_range,
            "training_seed_stable": stable,
            "decision": (
                "CANDIDATE_FOR_SEPARATE_PROMOTION_REVIEW"
                if all_seed_pass and stable
                else "NO_PROMOTION"
            ),
            "may_feed_model_recipe": False,
            "separate_promotion_artifact_required": True,
            "protocol_hash": protocol_hash,
        }
    return child, consensus, delta_rows


def _study_decision(
    consensus: Mapping[str, Mapping[str, object]],
    *,
    protocol_hash: str,
    selection_evidence_hash: str,
) -> dict[str, object]:
    candidate_centers = [
        outer
        for outer, row in consensus.items()
        if row["decision"] == "CANDIDATE_FOR_SEPARATE_PROMOTION_REVIEW"
    ]
    return {
        "schema_version": "midogpp_v3_study_decision_v1",
        "status": "COMPLETE",
        "decision": (
            "MECHANISM_CANDIDATE_REQUIRES_SEPARATE_PROMOTION"
            if len(candidate_centers) == len(consensus)
            else "DO_NOT_PROMOTE"
        ),
        "candidate_centers": candidate_centers,
        "n_candidate_centers": len(candidate_centers),
        "n_centers": len(consensus),
        "claim_scope": CLAIM_SCOPE,
        "may_feed_model_recipe": False,
        "may_feed_expert_bank": False,
        "protocol_hash": protocol_hash,
        "selection_evidence_hash": selection_evidence_hash,
    }


def _protocol_manifest(
    config: AggregatePriorStudyConfig,
    *,
    frame: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": config.mode,
        "study_version": config.study_version,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "heldout_centers": list(config.heldout_centers),
        "arm_contract": arm_contract(),
        "source_expert_contract": {
            "fit_centers": "exactly_one_source_center_E",
            "outer_inner_constraints": "H!=I and E not in {H,I}",
            "training_key_is_H_I_neutral": True,
            "source_local_scaler_pca": True,
            "inverse_transform_before_classifier": True,
        },
        "mixture_contract": {
            "n_components": config.n_components,
            "rank": config.mixture_rank,
            "weight_floor": config.weight_floor,
            "variance_floor": config.variance_floor,
            "covariance_shrinkage": config.covariance_shrinkage,
            "refit_interval_epochs": config.refit_interval_epochs,
            "final_stabilization_epochs": (
                config.final_stabilization_epochs
            ),
            "optimizer_updates_prior_parameters": False,
            "rate_semantics": "mixture_KL_upper_bound_not_exact_NELBO",
        },
        "geco_contract": {
            "target_policy": config.geco_target_policy,
            "target_slack": config.geco_target_slack,
            "target_scope": "source_only_warmup_rows",
            "inner_or_outer_data_used": False,
        },
        "generation_contract": {
            "fixed_per_source_per_class": config.generation_per_class,
            "paired_across_arms": True,
            "prevalence_used": False,
        },
        "posterior_reference_contract": {
            "source_rows_only": True,
            "balanced_per_class": config.generation_per_class,
            "paired_gaussian_noise": True,
            "diagnostic_ceiling_only": True,
        },
        "decision_contract": {
            "primary_arm": config.primary_arm,
            "min_mean_bacc_delta_vs_sf": (
                config.min_mean_bacc_delta_vs_sf
            ),
            "min_inner_wins": config.min_inner_wins,
            "max_worst_inner_regression": (
                config.max_worst_inner_regression
            ),
            "max_training_seed_range": config.max_training_seed_range,
            "max_posterior_bacc_regression": (
                config.max_posterior_bacc_regression
            ),
            "min_prior_posterior_gap_reduction": (
                config.min_prior_posterior_gap_reduction
            ),
            "require_positive_delta_vs_kf": (
                config.require_positive_delta_vs_kf
            ),
            "require_positive_delta_vs_sg": (
                config.require_positive_delta_vs_sg
            ),
            "equal_weight_sources_then_inner_centers": (
                config.equal_weight_sources_then_inner_centers
            ),
        },
        "publication_contract": {
            "may_feed_model_recipe": False,
            "may_feed_expert_bank": False,
            "separate_promotion_artifact_required": True,
        },
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _canonical_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    normalized = [
        {
            str(key): (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else "" if value is None else str(value)
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    return sorted(
        normalized,
        key=lambda row: json.dumps(row, sort_keys=True),
    )


def _balanced_source_indices(
    labels: Sequence[int],
    *,
    per_class: int,
    neutral_evaluation_hash: str,
) -> tuple[tuple[int, ...], str]:
    y = np.asarray(labels, dtype=np.int64)
    selected: list[int] = []
    class_seeds: dict[str, int] = {}
    for class_label in (0, 1):
        candidates = np.flatnonzero(y == class_label)
        if len(candidates) == 0:
            raise ProtocolError("Posterior reference source lacks one class.")
        digest = hashlib.sha256(
            (
                f"{neutral_evaluation_hash}|posterior_source_rows|"
                f"{class_label}"
            ).encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
        class_seeds[str(class_label)] = seed
        draws = np.random.default_rng(seed).choice(
            candidates,
            size=int(per_class),
            replace=True,
        )
        selected.extend(int(value) for value in draws)
    selected_array = np.asarray(selected, dtype=np.int64)
    selection_hash = stable_hash(
        {
            "neutral_evaluation_hash": neutral_evaluation_hash,
            "class_seeds": class_seeds,
            "per_class": int(per_class),
            "selected_indices_sha256": hashlib.sha256(
                selected_array.tobytes()
            ).hexdigest(),
        }
    )
    return tuple(selected), selection_hash


def _mean(values: Sequence[float] | object) -> float:
    resolved = list(values)  # type: ignore[arg-type]
    if not resolved:
        raise ProtocolError("Cannot average an empty decision cell.")
    return sum(float(value) for value in resolved) / len(resolved)
