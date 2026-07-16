"""Fully nested Stage-20 v2 shrunk Task-Fisher source-inner study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from ...objectives import TASK_FISHER_OBJECTIVE, validate_trace_normalized_metric
from ...preservation.representations import source_budget_labels
from ...preservation.scoring import chance_normalized_preservation, score_representation
from ...preservation.splits import row_hash
from ...reporting import prepare_artifact_dirs
from ...task_fisher import fit_task_fisher_metric
from .checkpoint_store import StudyCheckpointStore
from .config import (
    TaskFisherShrinkageStudyConfig,
    decision_contract_hash,
    study_contract_hash,
    study_contract_payload,
)
from .contracts import (
    FISHER_SHRINKAGE_MODE,
    STANDARD_MODEL_FAMILY,
    STANDARD_NORMAL_PRIOR,
    FisherStudyMetricV2,
    StudyTrainingKey,
)
from .fisher_artifacts import FISHER_STATE_INDEX_SCHEMA, write_fisher_study_bundle
from .fisher_decision import select_fisher_study_decision
from .preparation import embedded_v1_preparation_lineage, load_study_frame, prepare_outer_study_folds
from .training import (
    StudyRuntime,
    decode_means,
    paired_epsilon,
    posterior_decodes,
    prior_decodes_from_epsilon,
    train_study_cvae,
)
from .validation_common import (
    COVERAGE_SCHEMA,
    GENERATION_BUDGET_SCHEMA,
    FISHER_SAMPLER_SCHEMA,
    METRIC_SCHEMA,
    PAIRING_AUDIT_SCHEMA,
    PROTOCOL_SCHEMA,
    SELECTION_EVIDENCE_SCHEMA,
    StudyTimingRecorder,
    canonical_rows,
    read_json,
    selection_evidence_hash,
    study_implementation_lineage,
    write_study_run_state,
)


EXPERIMENT_ID = "midogpp.cvae.task_fisher_shrinkage_source_inner.v2"
STUDY_ID = "task_fisher_shrinkage_source_inner_v2"
METHOD = "fully_nested_task_fisher_shrinkage_study_v2"


def run_task_fisher_shrinkage_source_inner_study(
    config: TaskFisherShrinkageStudyConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    if not isinstance(config, TaskFisherShrinkageStudyConfig):
        raise ProtocolError("Fisher-shrinkage runner requires its v2 config type.")
    root = prepare_artifact_dirs(Path(artifact_root or config.artifact_root))
    protocol_hash = "unavailable"
    try:
        frame = load_study_frame(config)
        lineage = embedded_v1_preparation_lineage()
        protocol = _protocol_manifest(config, frame=frame, lineage=lineage)
        protocol_hash = str(protocol["protocol_hash"])
        return _run(config, root=root, frame=frame, lineage=lineage, protocol=protocol)
    except Exception:
        if protocol_hash != "unavailable":
            write_study_run_state(
                root,
                protocol_hash=protocol_hash,
                mode=FISHER_SHRINKAGE_MODE,
                status="FAILED",
            )
        raise


def _run(
    config: TaskFisherShrinkageStudyConfig,
    *,
    root: Path,
    frame: object,
    lineage: Mapping[str, object],
    protocol: Mapping[str, object],
) -> Path:
    from ..prior_recovery_runtime_cache import FeatureFrameCache

    protocol_hash = str(protocol["protocol_hash"])
    timings = StudyTimingRecorder(root, protocol_hash=protocol_hash, mode=config.mode)
    write_study_run_state(root, protocol_hash=protocol_hash, mode=config.mode, status="RUNNING")
    frame_cache = FeatureFrameCache(root)
    checkpoints = StudyCheckpointStore(root)
    metric_rows: list[dict[str, object]] = []
    decision_metrics: list[FisherStudyMetricV2] = []
    nested_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    initialization_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    rng_rows: list[dict[str, object]] = []
    fisher_records: list[dict[str, object]] = []

    for outer in config.heldout_centers:
        prepared_outer = prepare_outer_study_folds(
            config,
            frame=frame,
            outer_target_center=outer,
            preparation_protocol_hash=protocol_hash,
            frame_cache=frame_cache,
            timings=timings,
        )
        nested_rows.extend(dict(row) for row in prepared_outer.nested_reference_rows)
        tuning_rows.extend(dict(row) for row in prepared_outer.nested_tuning_rows)
        identity_rows.extend(dict(row) for row in prepared_outer.identity_rows)
        for prepared in prepared_outer.folds:
            budget = source_budget_labels(prepared.y_fit)
            budget_hash = hashlib.sha256(np.asarray(budget, dtype=np.int64).tobytes()).hexdigest()
            counts = [sum(int(value) == label for value in budget) for label in (0, 1)]
            budget_rows.append(
                {
                    "schema_version": GENERATION_BUDGET_SCHEMA,
                    "outer_target_center": outer,
                    "inner_pseudo_target_center": prepared.inner,
                    "source_row_hash": row_hash(prepared.source_ids),
                    "ordered_label_vector_hash": budget_hash,
                    "class_counts": json.dumps(counts),
                    "budget_policy": config.generation_budget_policy,
                    "derived_from_y_fit_only": "true",
                    "used_inner_labels": "false",
                }
            )
            raw_record, derived_metrics = _fit_raw_and_derived_fisher(prepared, config=config)
            fisher_records.append(raw_record)
            raw_valid = bool(raw_record["valid"])
            for training_seed in config.training_seeds:
                runtimes: dict[float, StudyRuntime] = {}
                for alpha in config.alphas:
                    if alpha > 0.0 and not raw_valid:
                        continue
                    metric_state = derived_metrics[alpha]
                    runtimes[alpha] = _runtime(
                        config,
                        prepared=prepared,
                        training_seed=training_seed,
                        alpha=alpha,
                        raw_fisher_state_hash=(
                            str(raw_record["raw_fisher_state_hash"]) if alpha > 0 else "none"
                        ),
                        objective_context_hash=(
                            str(metric_state["metric_state_hash"]) if alpha > 0 else "none"
                        ),
                        task_metric=(metric_state["metric"] if alpha > 0 else None),
                        protocol_hash=protocol_hash,
                        checkpoints=checkpoints,
                        timings=timings,
                    )
                all_valid_runtimes = list(runtimes.values())
                initialization_rows.append(
                    _initialization_audit(
                        outer=outer,
                        inner=prepared.inner,
                        training_seed=training_seed,
                        runtimes=all_valid_runtimes,
                        raw_valid=raw_valid,
                    )
                )
                checkpoint_rows.extend(
                    {
                        "outer_target_center": outer,
                        "inner_pseudo_target_center": prepared.inner,
                        "training_seed": training_seed,
                        "alpha": alpha,
                        "training_key_hash": runtime.training_key.hash,
                        "checkpoint_hash": runtime.checkpoint_hash,
                        "literal_alpha_zero": str(alpha == 0.0).lower(),
                        "raw_fisher_state_hash": runtime.variant.raw_fisher_state_hash,
                        "status": "PASS",
                    }
                    for alpha, runtime in runtimes.items()
                )
                decode_scores = {
                    alpha: _score(decode_means(runtime, prepared.x_fit, prepared.y_fit), prepared)
                    for alpha, runtime in runtimes.items()
                }
                for alpha in config.alphas:
                    if alpha in runtimes:
                        metric_rows.append(
                            _metric_row(
                                config,
                                prepared=prepared,
                                runtime=runtimes[alpha],
                                alpha=alpha,
                                generation_seed=-1,
                                role="decode",
                                score=decode_scores[alpha],
                                valid=True,
                                budget=budget,
                                protocol_hash=protocol_hash,
                            )
                        )
                    else:
                        metric_rows.append(
                            _invalid_metric_row(
                                config,
                                prepared=prepared,
                                training_seed=training_seed,
                                alpha=alpha,
                                generation_seed=-1,
                                role="decode",
                                budget=budget,
                                protocol_hash=protocol_hash,
                                reason="raw_fisher_invalid",
                                raw_fisher_state_hash=str(
                                    raw_record["raw_fisher_state_hash"]
                                ),
                                objective_context_hash=str(
                                    derived_metrics[alpha]["metric_state_hash"]
                                ),
                            )
                        )
                for alpha, runtime in runtimes.items():
                    sampler_rows.extend(
                        _standard_sampler_rows(
                            outer=outer,
                            inner=prepared.inner,
                            training_seed=training_seed,
                            alpha=alpha,
                            runtime=runtime,
                        )
                    )
                for generation_seed in config.generation_seeds:
                    prior_epsilon, prior_hash = paired_epsilon(
                        study_id=STUDY_ID,
                        outer_target_center=outer,
                        inner_pseudo_target_center=prepared.inner,
                        generation_seed=generation_seed,
                        labels=budget,
                        latent_dim=config.latent_dim,
                        stream="prior_generation",
                    )
                    posterior_epsilon, posterior_hash = paired_epsilon(
                        study_id=STUDY_ID,
                        outer_target_center=outer,
                        inner_pseudo_target_center=prepared.inner,
                        generation_seed=generation_seed,
                        labels=prepared.y_fit,
                        latent_dim=config.latent_dim,
                        stream="posterior_evaluation",
                    )
                    for alpha in config.alphas:
                        runtime = runtimes.get(alpha)
                        if runtime is None:
                            for role in ("prior", "posterior"):
                                metric_rows.append(
                                    _invalid_metric_row(
                                        config,
                                        prepared=prepared,
                                        training_seed=training_seed,
                                        alpha=alpha,
                                        generation_seed=generation_seed,
                                        role=role,
                                        budget=budget if role == "prior" else prepared.y_fit,
                                        protocol_hash=protocol_hash,
                                        reason="raw_fisher_invalid",
                                        raw_fisher_state_hash=str(
                                            raw_record["raw_fisher_state_hash"]
                                        ),
                                        objective_context_hash=str(
                                            derived_metrics[alpha]["metric_state_hash"]
                                        ),
                                    )
                                )
                            decision_metrics.append(
                                FisherStudyMetricV2(
                                    outer_target_center=outer,
                                    inner_pseudo_target_center=prepared.inner,
                                    training_seed=training_seed,
                                    generation_seed=generation_seed,
                                    alpha=alpha,
                                    preservation_ratio=math.nan,
                                    decode_bacc=math.nan,
                                    posterior_bacc=math.nan,
                                    valid=False,
                                )
                            )
                            continue
                        prior_score = _score(
                            prior_decodes_from_epsilon(runtime, budget, epsilon=prior_epsilon),
                            prepared,
                            train_labels=budget,
                        )
                        posterior_score = _score(
                            posterior_decodes(
                                runtime,
                                prepared.x_fit,
                                prepared.y_fit,
                                epsilon=posterior_epsilon,
                            ),
                            prepared,
                        )
                        for role, score, labels in (
                            ("prior", prior_score, budget),
                            ("posterior", posterior_score, prepared.y_fit),
                        ):
                            metric_rows.append(
                                _metric_row(
                                    config,
                                    prepared=prepared,
                                    runtime=runtime,
                                    alpha=alpha,
                                    generation_seed=generation_seed,
                                    role=role,
                                    score=score,
                                    valid=True,
                                    budget=labels,
                                    protocol_hash=protocol_hash,
                                )
                            )
                        decision_metrics.append(
                            FisherStudyMetricV2(
                                outer_target_center=outer,
                                inner_pseudo_target_center=prepared.inner,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                alpha=alpha,
                                preservation_ratio=_safe_ratio(
                                    prior_score.bacc,
                                    prepared.real_bacc,
                                    config.minimum_real_bacc,
                                ),
                                decode_bacc=decode_scores[alpha].bacc,
                                posterior_bacc=posterior_score.bacc,
                                valid=(
                                    prior_score.converged
                                    and posterior_score.converged
                                    and decode_scores[alpha].converged
                                ),
                            )
                        )
                        for stream, epsilon_hash in (
                            ("prior_generation", prior_hash),
                            ("posterior_evaluation", posterior_hash),
                        ):
                            rng_rows.append(
                                {
                                    "schema_version": PAIRING_AUDIT_SCHEMA,
                                    "outer_target_center": outer,
                                    "inner_pseudo_target_center": prepared.inner,
                                    "training_seed": training_seed,
                                    "generation_seed": generation_seed,
                                    "alpha": alpha,
                                    "stream": stream,
                                    "epsilon_hash": epsilon_hash,
                                    "epsilon_depends_on_training_seed": "false",
                                    "status": "PASS",
                                }
                            )

    checkpoints.write_indices()
    frame_cache.write_index()
    timings.finalize()
    checkpoint_index = read_json(root / "manifests/checkpoint_index.json")
    initialization_index = read_json(root / "manifests/initialization_index.json")
    frame_index = read_json(root / "manifests/feature_frame_index.json")
    fisher_index = {
        "schema_version": FISHER_STATE_INDEX_SCHEMA,
        "records": sorted(fisher_records, key=lambda row: str(row["raw_fisher_state_hash"])),
    }
    budget_manifest = {
        "schema_version": GENERATION_BUDGET_SCHEMA,
        "policy": config.generation_budget_policy,
        "records_hash": stable_hash(canonical_rows(budget_rows)),
        "n_records": len(budget_rows),
        "derived_from_y_fit_only": True,
    }
    evidence_hash = selection_evidence_hash(
        metric_rows=metric_rows,
        paired_delta_rows=[],
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        checkpoint_reuse_rows=checkpoint_rows,
        initialization_pairing_rows=initialization_rows,
        generation_budget_rows=budget_rows,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        feature_frame_index=frame_index,
        generation_budget_manifest=budget_manifest,
        rng_rows=rng_rows,
        protocol_manifest=protocol,
        study_state_index=fisher_index,
    )
    consensus, children, deltas = _decisions(
        config,
        decision_metrics=decision_metrics,
        protocol_hash=protocol_hash,
        evidence_hash=evidence_hash,
    )
    evidence_hash = selection_evidence_hash(
        metric_rows=metric_rows,
        paired_delta_rows=deltas,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        checkpoint_reuse_rows=checkpoint_rows,
        initialization_pairing_rows=initialization_rows,
        generation_budget_rows=budget_rows,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        feature_frame_index=frame_index,
        generation_budget_manifest=budget_manifest,
        rng_rows=rng_rows,
        protocol_manifest=protocol,
        study_state_index=fisher_index,
    )
    consensus, children, deltas = _decisions(
        config,
        decision_metrics=decision_metrics,
        protocol_hash=protocol_hash,
        evidence_hash=evidence_hash,
    )
    coverage = _coverage_manifest(config, decision_metrics=decision_metrics, metric_rows=metric_rows, fisher_records=fisher_records)
    leakage = _leakage_report(protocol_hash=protocol_hash, identity_rows=identity_rows)
    selection_manifest = {
        "schema_version": SELECTION_EVIDENCE_SCHEMA,
        "selection_evidence_hash": evidence_hash,
        "runtime_rows_included": False,
        "decisions_may_feed_model_recipe": False,
    }
    write_fisher_study_bundle(
        root,
        task_fisher_state_index=fisher_index,
        metric_rows=metric_rows,
        paired_delta_rows=deltas,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        checkpoint_reuse_rows=checkpoint_rows,
        initialization_pairing_rows=initialization_rows,
        generation_budget_rows=budget_rows,
        rng_rows=rng_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol,
        coverage_manifest=coverage,
        selection_evidence_manifest=selection_manifest,
        embedded_preparation_lineage=lineage,
        generation_budget_manifest=budget_manifest,
        child_decisions=children,
        consensus_decisions=consensus,
        study_decision=_study_summary(consensus, protocol_hash=protocol_hash, evidence_hash=evidence_hash),
        leakage_report=leakage,
    )
    write_study_run_state(root, protocol_hash=protocol_hash, mode=config.mode, status="COMPLETE")
    from .fisher_validation import validate_fisher_study_bundle

    validate_fisher_study_bundle(root, expected_config=config)
    return root


def _fit_raw_and_derived_fisher(prepared: object, *, config: TaskFisherShrinkageStudyConfig) -> tuple[dict[str, object], dict[float, dict[str, object]]]:
    fitted = fit_task_fisher_metric(prepared.x_fit, prepared.y_fit, spec=prepared.spec, alpha=0.0)
    raw_payload = {
        "outer_target_center": prepared.outer,
        "inner_pseudo_target_center": prepared.inner,
        "probe_config_hash": fitted.probe_config_hash,
        "fit_centers": list(prepared.fit_centers),
        "source_row_hash": row_hash(prepared.source_ids),
        "frame_hash": prepared.frame.state_hash,
        "raw_fisher": np.asarray(fitted.raw_fisher).tolist(),
        "trace_raw": fitted.trace_raw,
        "rank": fitted.rank,
        "valid": fitted.valid,
        "reason": fitted.reason,
        "fit_scope": config.raw_fisher_fit_scope,
        "shared_across_training_seeds": True,
    }
    raw_hash = stable_hash(raw_payload)
    raw_payload["raw_fisher_state_hash"] = raw_hash
    dimension = int(np.asarray(prepared.x_fit).shape[1])
    derived: dict[float, dict[str, object]] = {}
    raw = np.asarray(fitted.raw_fisher, dtype=np.float64)
    for alpha in config.alphas:
        if alpha == 0.0:
            metric = None
            direction = 1.0
            orthogonal = 1.0
        elif fitted.valid:
            normalized = float(dimension) * raw / float(fitted.trace_raw)
            metric = (np.eye(dimension) + float(alpha) * normalized) / (1.0 + float(alpha))
            validate_trace_normalized_metric(metric, input_dim=dimension)
            direction = (1.0 + float(alpha) * float(dimension)) / (1.0 + float(alpha))
            orthogonal = 1.0 / (1.0 + float(alpha))
        else:
            metric = None
            direction = math.nan
            orthogonal = math.nan
        state = {
            "alpha": alpha,
            "raw_fisher_state_hash": raw_hash if alpha > 0.0 else "none",
            "metric": None if metric is None else metric.tolist(),
            "fisher_direction_eigenvalue": direction,
            "orthogonal_eigenvalue": orthogonal,
            "directional_ratio": 1.0 + float(alpha) * float(dimension),
            "literal_isotropic_metric_none": alpha == 0.0,
            "valid": True if alpha == 0.0 else fitted.valid,
        }
        state["metric_state_hash"] = stable_hash(state)
        if metric is not None:
            state["metric"] = metric
        derived[alpha] = state
    raw_payload["derived_metrics"] = {
        format(alpha, ".2f"): {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in state.items()
        }
        for alpha, state in derived.items()
    }
    return raw_payload, derived


def _runtime(config: TaskFisherShrinkageStudyConfig, *, prepared: object, training_seed: int, alpha: float, raw_fisher_state_hash: str, objective_context_hash: str, task_metric: object | None, protocol_hash: str, checkpoints: StudyCheckpointStore, timings: StudyTimingRecorder) -> StudyRuntime:
    variant = config.training_variant(
        model_family=STANDARD_MODEL_FAMILY,
        prior_family=STANDARD_NORMAL_PRIOR,
        alpha=alpha,
        raw_fisher_state_hash=raw_fisher_state_hash,
        objective_context_hash=objective_context_hash,
    )
    key = StudyTrainingKey(
        study_id=STUDY_ID,
        study_version=config.study_version,
        outer_target_center=prepared.outer,
        inner_pseudo_target_center=prepared.inner,
        fit_centers=prepared.fit_centers,
        fit_row_hash=row_hash(prepared.source_ids),
        frame_hash=prepared.frame.state_hash,
        feature_cache_hash=prepared.feature_cache_hash,
        manifest_hash=prepared.manifest_hash,
        protocol_hash=protocol_hash,
        training_seed=training_seed,
        variant=variant,
    )
    started = perf_counter()
    runtime = checkpoints.load(training_key=key, variant=variant, input_dim=np.asarray(prepared.x_fit).shape[1], device=config.device)
    cache_status = "hit" if runtime is not None else "miss"
    if runtime is None:
        runtime = train_study_cvae(
            prepared.x_fit,
            prepared.y_fit,
            variant=variant,
            training_key=key,
            model_family=STANDARD_MODEL_FAMILY,
            task_metric=task_metric,
            device=config.device,
        )
        checkpoints.save(runtime)
    timings.record(
        phase="cvae_training",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=prepared.outer,
        inner_pseudo_target_center=prepared.inner,
        objective_id=variant.objective_id,
        training_key_hash=key.hash,
        cache_status=cache_status,
    )
    return runtime


def _initialization_audit(*, outer: str, inner: str, training_seed: int, runtimes: Sequence[StudyRuntime], raw_valid: bool) -> dict[str, object]:
    shared = {runtime.shared_initialization_hash for runtime in runtimes}
    streams = {runtime.training_stream_hash for runtime in runtimes}
    status = "PASS" if len(shared) == 1 and len(streams) == 1 else "FAIL"
    return {
        "outer_target_center": outer,
        "inner_pseudo_target_center": inner,
        "training_seed": training_seed,
        "alphas_present": json.dumps([runtime.variant.alpha for runtime in runtimes]),
        "shared_initialization_hashes": json.dumps(sorted(shared)),
        "training_stream_hashes": json.dumps(sorted(streams)),
        "raw_fisher_valid": str(raw_valid).lower(),
        "status": status,
    }


def _score(generated: object, prepared: object, *, train_labels: Sequence[int] | None = None):
    return score_representation(generated, prepared.y_fit if train_labels is None else train_labels, prepared.x_eval, prepared.y_eval, spec=prepared.spec)


def _metric_row(config: TaskFisherShrinkageStudyConfig, *, prepared: object, runtime: StudyRuntime, alpha: float, generation_seed: int, role: str, score: object, valid: bool, budget: Sequence[int], protocol_hash: str) -> dict[str, object]:
    ratio = _safe_ratio(score.bacc, prepared.real_bacc, config.minimum_real_bacc)
    row_valid = bool(valid and score.converged and math.isfinite(ratio))
    return {
        "schema_version": METRIC_SCHEMA,
        "method": METHOD,
        "protocol_hash": protocol_hash,
        "outer_target_center": prepared.outer,
        "inner_pseudo_target_center": prepared.inner,
        "fit_centers": json.dumps(list(prepared.fit_centers)),
        "training_seed": runtime.training_key.training_seed,
        "generation_seed": generation_seed,
        "arm": f"alpha={alpha:.2f}",
        "alpha": alpha,
        "model_family": runtime.model_family,
        "prior_family": runtime.variant.prior_family,
        "objective_id": runtime.variant.objective_id,
        "raw_fisher_state_hash": runtime.variant.raw_fisher_state_hash,
        "objective_context_hash": runtime.variant.objective_context_hash,
        "representation_role": role,
        "bacc": score.bacc,
        "macro_f1": score.macro_f1,
        "real_reference_bacc": prepared.real_bacc,
        "preservation_ratio": ratio,
        "generation_class_counts": json.dumps([sum(int(value) == label for value in budget) for label in (0, 1)]),
        "classifier_spec_hash": prepared.spec.config_hash,
        "frame_hash": prepared.frame.state_hash,
        "fit_row_hash": row_hash(prepared.source_ids),
        "eval_row_hash": row_hash(prepared.eval_ids),
        "training_key_hash": runtime.training_key.hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "shared_initialization_hash": runtime.shared_initialization_hash,
        "training_stream_hash": runtime.training_stream_hash,
        "sampler_state_hash": _standard_sampler_state_hash(runtime),
        "valid": str(row_valid).lower(),
        "eligible": str(row_valid).lower(),
        "status": "ok" if row_valid else "invalid_or_nonconverged",
        "claim_scope": "cvae_source_inner_study_only",
        "selection_source": "fully_nested_source_inner",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
    }


def _invalid_metric_row(config: TaskFisherShrinkageStudyConfig, *, prepared: object, training_seed: int, alpha: float, generation_seed: int, role: str, budget: Sequence[int], protocol_hash: str, reason: str, raw_fisher_state_hash: str, objective_context_hash: str) -> dict[str, object]:
    return {
        "schema_version": METRIC_SCHEMA,
        "method": METHOD,
        "protocol_hash": protocol_hash,
        "outer_target_center": prepared.outer,
        "inner_pseudo_target_center": prepared.inner,
        "fit_centers": json.dumps(list(prepared.fit_centers)),
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "arm": f"alpha={alpha:.2f}",
        "alpha": alpha,
        "model_family": STANDARD_MODEL_FAMILY,
        "prior_family": STANDARD_NORMAL_PRIOR,
        "objective_id": TASK_FISHER_OBJECTIVE,
        "raw_fisher_state_hash": raw_fisher_state_hash,
        "objective_context_hash": objective_context_hash,
        "representation_role": role,
        "bacc": math.nan,
        "macro_f1": math.nan,
        "real_reference_bacc": prepared.real_bacc,
        "preservation_ratio": math.nan,
        "generation_class_counts": json.dumps([sum(int(value) == label for value in budget) for label in (0, 1)]),
        "classifier_spec_hash": prepared.spec.config_hash,
        "frame_hash": prepared.frame.state_hash,
        "fit_row_hash": row_hash(prepared.source_ids),
        "eval_row_hash": row_hash(prepared.eval_ids),
        "training_key_hash": "none",
        "checkpoint_hash": "none",
        "shared_initialization_hash": "none",
        "training_stream_hash": "none",
        "sampler_state_hash": "none",
        "valid": "false",
        "eligible": "false",
        "status": reason,
        "claim_scope": "cvae_source_inner_study_only",
        "selection_source": "fully_nested_source_inner",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
    }


def _standard_sampler_rows(*, outer: str, inner: str, training_seed: int, alpha: float, runtime: StudyRuntime) -> list[dict[str, object]]:
    sampler_state_hash = _standard_sampler_state_hash(runtime)
    return [
        {
            "schema_version": FISHER_SAMPLER_SCHEMA,
            "mechanism": "standard_normal",
            "outer_target_center": outer,
            "inner_pseudo_target_center": inner,
            "training_seed": training_seed,
            "alpha": alpha,
            "class_label": class_label,
            "latent_dim": runtime.variant.latent_dim,
            "source_row_hash": runtime.training_key.fit_row_hash,
            "requested_family": STANDARD_NORMAL_PRIOR,
            "realized_family": STANDARD_NORMAL_PRIOR,
            "mean": json.dumps([0.0] * runtime.variant.latent_dim),
            "variance": json.dumps([1.0] * runtime.variant.latent_dim),
            "training_key_hash": runtime.training_key.hash,
            "checkpoint_hash": runtime.checkpoint_hash,
            "sampler_state_hash": sampler_state_hash,
            "fallback_reason": "",
        }
        for class_label in (0, 1)
    ]


def _standard_sampler_state_hash(runtime: StudyRuntime) -> str:
    return stable_hash(
        {
            "family": STANDARD_NORMAL_PRIOR,
            "latent_dim": runtime.variant.latent_dim,
            "source_row_hash": runtime.training_key.fit_row_hash,
            "training_key_hash": runtime.training_key.hash,
            "checkpoint_hash": runtime.checkpoint_hash,
        }
    )


def _decisions(config: TaskFisherShrinkageStudyConfig, *, decision_metrics: Sequence[FisherStudyMetricV2], protocol_hash: str, evidence_hash: str):
    consensus: dict[str, Mapping[str, object]] = {}
    children: dict[tuple[int, str], Mapping[str, object]] = {}
    deltas: list[dict[str, object]] = []
    for outer in config.heldout_centers:
        inners = tuple(center for center in MIDOGPP_ELIGIBLE_CENTERS if center != outer)
        rows = [row for row in decision_metrics if row.outer_target_center == outer]
        decision = select_fisher_study_decision(
            rows,
            outer_target_center=outer,
            expected_inner_centers=inners,
            protocol_hash=protocol_hash,
            decision_contract_hash=decision_contract_hash(config),
            training_seeds=config.training_seeds,
            generation_seeds=config.generation_seeds,
            alphas=config.alphas,
            fisher_min_mean_delta=config.fisher_min_mean_delta,
            min_inner_wins=config.min_inner_wins,
            tie_margin=config.tie_margin,
            safety_max_bacc_regression=config.safety_max_bacc_regression,
        )
        payload = decision.to_payload()
        payload["selection_evidence_hash"] = evidence_hash
        consensus[outer] = payload
        for seed in config.training_seeds:
            summary = decision.per_training_seed.get(str(seed), {})
            child = {
                "schema_version": "midogpp_fisher_shrinkage_child_decision_v2",
                "outer_target_center": outer,
                "training_seed": seed,
                "status": (
                    "INVALID_DECISION"
                    if decision.status == "INVALID_DECISION"
                    else (
                        f"SELECTED_ALPHA_{float(summary.get('selected_alpha')):.2f}"
                        if summary.get("selected_alpha") is not None
                        else "NO_NONZERO_ALPHA"
                    )
                ),
                "summary": summary,
                "selection_evidence_hash": evidence_hash,
                "may_feed_model_recipe": False,
                "may_feed_deployable_selection": False,
            }
            child["study_decision_hash"] = stable_hash(child)
            children[(seed, outer)] = child
            candidates = summary.get("candidate_summaries", {}) if isinstance(summary, Mapping) else {}
            if isinstance(candidates, Mapping):
                for alpha, comp in candidates.items():
                    if not isinstance(comp, Mapping):
                        continue
                    for inner, value in dict(comp.get("preservation_ratio_delta_by_inner", {})).items():
                        deltas.append(
                            {
                                "outer_target_center": outer,
                                "inner_pseudo_target_center": inner,
                                "training_seed": seed,
                                "alpha": alpha,
                                "preservation_ratio_delta": value,
                            }
                        )
    return consensus, children, deltas


def _protocol_manifest(config: TaskFisherShrinkageStudyConfig, *, frame: object, lineage: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": config.mode,
        "study_version": config.study_version,
        "coverage_mode": "complete" if config.heldout_centers == MIDOGPP_ELIGIBLE_CENTERS else "fixture",
        "claim_scope": "cvae_source_inner_study_only",
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "heldout_centers": list(config.heldout_centers),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "study_contract": study_contract_payload(config),
        "study_contract_hash": study_contract_hash(config),
        "implementation_lineage": study_implementation_lineage(config.mode),
        "decision_contract_hash": decision_contract_hash(config),
        "embedded_preparation_lineage_hash": lineage["lineage_hash"],
        "preservation_ratio": "(generated_bacc-0.5)/(real_bacc-0.5)",
        "minimum_real_bacc": config.minimum_real_bacc,
        "alpha_one_matched_comparison_present": False,
        "outer_target_rows_used": False,
        "inner_labels_used_for_fit": False,
        "target_eval_artifacts_used": False,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _coverage_manifest(config: TaskFisherShrinkageStudyConfig, *, decision_metrics: Sequence[FisherStudyMetricV2], metric_rows: Sequence[Mapping[str, object]], fisher_records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    expected = len(config.heldout_centers) * (len(MIDOGPP_ELIGIBLE_CENTERS) - 1) * len(config.training_seeds) * len(config.generation_seeds) * len(config.alphas)
    status = "PASS" if len(decision_metrics) == expected else "FAIL"
    return {
        "schema_version": COVERAGE_SCHEMA,
        "status": status,
        "expected_decision_cells": expected,
        "observed_decision_cells": len(decision_metrics),
        "metric_rows": len(metric_rows),
        "raw_fisher_states": len(fisher_records),
        "complete_training_generation_seed_cross": status == "PASS",
    }


def _leakage_report(*, protocol_hash: str, identity_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    passed = bool(identity_rows) and all(row.get("status") == "PASS" for row in identity_rows)
    return {
        "schema_version": "midogpp_source_inner_study_leakage_report_v2",
        "status": "PASS" if passed else "FAIL",
        "protocol_hash": protocol_hash,
        "outer_target_rows_used": False,
        "inner_rows_used_for_fit": False,
        "target_eval_labels_used_for_selection": False,
        "selection_used_target_eval_artifacts": False,
        "identity_overlap_pass": passed,
        "claim_scope": "cvae_source_inner_study_only",
    }


def _study_summary(consensus: Mapping[str, Mapping[str, object]], *, protocol_hash: str, evidence_hash: str) -> dict[str, object]:
    statuses = {outer: row.get("status") for outer, row in consensus.items()}
    return {
        "schema_version": "midogpp_fisher_shrinkage_study_summary_v2",
        "status": "COMPLETE" if all(value != "INVALID_DECISION" for value in statuses.values()) else "INVALID_INCOMPLETE",
        "protocol_hash": protocol_hash,
        "selection_evidence_hash": evidence_hash,
        "per_outer_status": statuses,
        "descriptive_counts": {status: sum(value == status for value in statuses.values()) for status in sorted(set(statuses.values()))},
        "cross_outer_selection_performed": False,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
    }


def _safe_ratio(generated_bacc: float, real_bacc: float, minimum_real_bacc: float) -> float:
    try:
        return chance_normalized_preservation(generated_bacc, real_bacc, minimum_real_bacc=minimum_real_bacc)
    except ValueError:
        return math.nan
