"""Fully nested Stage-20 v2 learned conditional-prior source-inner study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import torch

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)
from ...generation_samplers import (
    DIAGONAL_SAMPLER,
    fit_aggregate_posterior_sampler,
)
from ...preservation.representations import source_budget_labels
from ...preservation.scoring import chance_normalized_preservation, score_representation
from ...preservation.splits import row_hash
from ...reporting import prepare_artifact_dirs, write_json
from .checkpoint_store import StudyCheckpointStore
from .config import (
    LearnedConditionalPriorStudyConfig,
    decision_contract_hash,
    study_contract_hash,
    study_contract_payload,
)
from .contracts import (
    LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
    LEARNED_PRIOR_MODE,
    LEARNED_PRIOR_MODEL_FAMILY,
    STANDARD_MODEL_FAMILY,
    STANDARD_NORMAL_PRIOR,
    PriorStudyMetricV2,
    StudyTrainingKey,
)
from .preparation import (
    embedded_v1_preparation_lineage,
    load_study_frame,
    prepare_outer_study_folds,
)
from .prior_artifacts import PRIOR_STATE_INDEX_SCHEMA, write_prior_study_bundle
from .prior_decision import select_prior_study_decision
from .training import (
    StudyRuntime,
    decode_means,
    encode_runtime,
    paired_epsilon,
    posterior_decodes,
    prior_decodes_from_epsilon,
    train_study_cvae,
)
from .validation_common import (
    COVERAGE_SCHEMA,
    GENERATION_BUDGET_SCHEMA,
    METRIC_SCHEMA,
    PAIRING_AUDIT_SCHEMA,
    PRIOR_SAMPLER_SCHEMA,
    PROTOCOL_SCHEMA,
    SELECTION_EVIDENCE_SCHEMA,
    StudyTimingRecorder,
    canonical_rows,
    read_json,
    selection_evidence_hash,
    study_implementation_lineage,
    write_study_run_state,
)


EXPERIMENT_ID = "midogpp.cvae.learned_conditional_prior_source_inner.v2"
STUDY_ID = "learned_conditional_prior_source_inner_v2"
METHOD = "fully_nested_learned_conditional_prior_study_v2"


def run_learned_conditional_prior_source_inner_study(
    config: LearnedConditionalPriorStudyConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    if not isinstance(config, LearnedConditionalPriorStudyConfig):
        raise ProtocolError("Learned-prior runner requires its v2 config type.")
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
                mode=LEARNED_PRIOR_MODE,
                status="FAILED",
            )
        raise


def _run(
    config: LearnedConditionalPriorStudyConfig,
    *,
    root: Path,
    frame: object,
    lineage: Mapping[str, object],
    protocol: Mapping[str, object],
) -> Path:
    from ..prior_recovery_runtime_cache import FeatureFrameCache

    protocol_hash = str(protocol["protocol_hash"])
    timings = StudyTimingRecorder(root, protocol_hash=protocol_hash, mode=config.mode)
    write_study_run_state(
        root, protocol_hash=protocol_hash, mode=config.mode, status="RUNNING"
    )
    frame_cache = FeatureFrameCache(root)
    checkpoints = StudyCheckpointStore(root)

    metric_rows: list[dict[str, object]] = []
    decision_metrics: list[PriorStudyMetricV2] = []
    nested_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    checkpoint_reuse_rows: list[dict[str, object]] = []
    initialization_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    rng_rows: list[dict[str, object]] = []
    prior_state_records: list[dict[str, object]] = []

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
            budget_hash = hashlib.sha256(
                np.asarray(budget, dtype=np.int64).tobytes()
            ).hexdigest()
            counts = [sum(value == label for value in budget) for label in (0, 1)]
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
            for training_seed in config.training_seeds:
                runtime_a = _runtime(
                    config,
                    prepared=prepared,
                    training_seed=training_seed,
                    model_family=STANDARD_MODEL_FAMILY,
                    prior_family=STANDARD_NORMAL_PRIOR,
                    protocol_hash=protocol_hash,
                    checkpoints=checkpoints,
                    timings=timings,
                )
                runtime_e = _runtime(
                    config,
                    prepared=prepared,
                    training_seed=training_seed,
                    model_family=LEARNED_PRIOR_MODEL_FAMILY,
                    prior_family=LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
                    protocol_hash=protocol_hash,
                    checkpoints=checkpoints,
                    timings=timings,
                )
                mu_a, logvar_a = encode_runtime(runtime_a, prepared.x_fit, prepared.y_fit)
                sampler_c = fit_aggregate_posterior_sampler(
                    mu_a,
                    logvar_a,
                    prepared.y_fit,
                    family=DIAGONAL_SAMPLER,
                    source_row_hash=row_hash(prepared.source_ids),
                    min_class_count=_sampler_min_class_count(config),
                    max_condition_number=1e6,
                )
                c_viable = sampler_c.requested_family_realized_for_both_classes
                sampler_rows.extend(
                    _sampler_rows(
                        outer=outer,
                        inner=prepared.inner,
                        training_seed=training_seed,
                        arm="C-diag",
                        checkpoint_hash=runtime_a.checkpoint_hash,
                        training_key_hash=runtime_a.training_key.hash,
                        sampler=sampler_c,
                    )
                )
                mu_e, logvar_e = encode_runtime(
                    runtime_e, prepared.x_fit, prepared.y_fit
                )
                prior_record, e_integrity_valid, e_eligible = _learned_prior_record(
                    runtime_e,
                    posterior_mu=mu_e,
                    posterior_logvar=logvar_e,
                    labels=prepared.y_fit,
                    sampler_c=sampler_c,
                    outer=outer,
                    inner=prepared.inner,
                    training_seed=training_seed,
                )
                prior_state_records.append(prior_record)
                sampler_rows.extend(_learned_prior_sampler_rows(prior_record))
                sampler_state_hashes = {
                    "A": stable_hash(
                        {
                            "family": STANDARD_NORMAL_PRIOR,
                            "latent_dim": config.latent_dim,
                        }
                    ),
                    "C-diag": sampler_c.state_hash,
                    "E": prior_record["state_hash"],
                }

                checkpoint_reuse_rows.append(
                    {
                        "outer_target_center": outer,
                        "inner_pseudo_target_center": prepared.inner,
                        "training_seed": training_seed,
                        "arm_pair": "A/C-diag",
                        "a_training_key_hash": runtime_a.training_key.hash,
                        "c_training_key_hash": runtime_a.training_key.hash,
                        "a_checkpoint_hash": runtime_a.checkpoint_hash,
                        "c_checkpoint_hash": runtime_a.checkpoint_hash,
                        "single_checkpoint_reused": "true",
                        "status": "PASS",
                    }
                )
                initialization_rows.append(
                    {
                        "outer_target_center": outer,
                        "inner_pseudo_target_center": prepared.inner,
                        "training_seed": training_seed,
                        "arm_pair": "A/E",
                        "shared_initialization_hash_a": runtime_a.shared_initialization_hash,
                        "shared_initialization_hash_e": runtime_e.shared_initialization_hash,
                        "full_initialization_hash_a": runtime_a.full_initialization_hash,
                        "full_initialization_hash_e": runtime_e.full_initialization_hash,
                        "training_stream_hash_a": runtime_a.training_stream_hash,
                        "training_stream_hash_e": runtime_e.training_stream_hash,
                        "shared_initialization_equal": str(
                            runtime_a.shared_initialization_hash
                            == runtime_e.shared_initialization_hash
                        ).lower(),
                        "training_stream_equal": str(
                            runtime_a.training_stream_hash == runtime_e.training_stream_hash
                        ).lower(),
                        "full_training_identity_distinct": str(
                            runtime_a.training_key.hash != runtime_e.training_key.hash
                        ).lower(),
                        "status": (
                            "PASS"
                            if runtime_a.shared_initialization_hash
                            == runtime_e.shared_initialization_hash
                            and runtime_a.training_stream_hash
                            == runtime_e.training_stream_hash
                            and runtime_a.training_key.hash != runtime_e.training_key.hash
                            else "FAIL"
                        ),
                    }
                )
                decode_scores = {
                    "A": _score(
                        decode_means(runtime_a, prepared.x_fit, prepared.y_fit),
                        prepared,
                    ),
                    "E": _score(
                        decode_means(runtime_e, prepared.x_fit, prepared.y_fit),
                        prepared,
                    ),
                }
                decode_scores["C-diag"] = decode_scores["A"]
                for arm, runtime, valid, eligible in (
                    ("A", runtime_a, True, True),
                    ("C-diag", runtime_a, c_viable, c_viable),
                    ("E", runtime_e, e_integrity_valid, e_eligible),
                ):
                    metric_rows.append(
                        _metric_row(
                            config,
                            prepared=prepared,
                            runtime=runtime,
                            arm=arm,
                            generation_seed=-1,
                            role="decode",
                            score=decode_scores[arm],
                            valid=valid,
                            eligible=eligible,
                            sampler_state_hash=str(sampler_state_hashes[arm]),
                            budget=budget,
                            protocol_hash=protocol_hash,
                        )
                    )

                for generation_seed in config.generation_seeds:
                    prior_epsilon, prior_epsilon_hash = paired_epsilon(
                        study_id=STUDY_ID,
                        outer_target_center=outer,
                        inner_pseudo_target_center=prepared.inner,
                        generation_seed=generation_seed,
                        labels=budget,
                        latent_dim=config.latent_dim,
                        stream="prior_generation",
                    )
                    posterior_epsilon, posterior_epsilon_hash = paired_epsilon(
                        study_id=STUDY_ID,
                        outer_target_center=outer,
                        inner_pseudo_target_center=prepared.inner,
                        generation_seed=generation_seed,
                        labels=prepared.y_fit,
                        latent_dim=config.latent_dim,
                        stream="posterior_evaluation",
                    )
                    prior_generated = {
                        "A": prior_decodes_from_epsilon(
                            runtime_a, budget, epsilon=prior_epsilon
                        ),
                        "E": prior_decodes_from_epsilon(
                            runtime_e, budget, epsilon=prior_epsilon
                        ),
                    }
                    prior_generated["C-diag"] = (
                        prior_decodes_from_epsilon(
                            runtime_a,
                            budget,
                            epsilon=prior_epsilon,
                            sampler=sampler_c,
                        )
                        if c_viable
                        else prior_generated["A"]
                    )
                    posterior_generated = {
                        "A": posterior_decodes(
                            runtime_a,
                            prepared.x_fit,
                            prepared.y_fit,
                            epsilon=posterior_epsilon,
                        ),
                        "E": posterior_decodes(
                            runtime_e,
                            prepared.x_fit,
                            prepared.y_fit,
                            epsilon=posterior_epsilon,
                        ),
                    }
                    posterior_generated["C-diag"] = posterior_generated["A"]
                    prior_scores = {
                        arm: _score(generated, prepared, train_labels=budget)
                        for arm, generated in prior_generated.items()
                    }
                    posterior_scores = {
                        arm: _score(generated, prepared)
                        for arm, generated in posterior_generated.items()
                    }
                    validity = {
                        "A": (True, True),
                        "C-diag": (c_viable, c_viable),
                        "E": (e_integrity_valid, e_eligible),
                    }
                    runtimes = {"A": runtime_a, "C-diag": runtime_a, "E": runtime_e}
                    for arm in ("A", "C-diag", "E"):
                        valid, eligible = validity[arm]
                        metric_rows.append(
                            _metric_row(
                                config,
                                prepared=prepared,
                                runtime=runtimes[arm],
                                arm=arm,
                                generation_seed=generation_seed,
                                role="prior",
                                score=prior_scores[arm],
                                valid=valid,
                                eligible=eligible,
                                sampler_state_hash=str(sampler_state_hashes[arm]),
                                budget=budget,
                                protocol_hash=protocol_hash,
                            )
                        )
                        metric_rows.append(
                            _metric_row(
                                config,
                                prepared=prepared,
                                runtime=runtimes[arm],
                                arm=arm,
                                generation_seed=generation_seed,
                                role="posterior",
                                score=posterior_scores[arm],
                                valid=valid,
                                eligible=eligible,
                                sampler_state_hash=str(sampler_state_hashes[arm]),
                                budget=prepared.y_fit,
                                protocol_hash=protocol_hash,
                            )
                        )
                        decision_metrics.append(
                            PriorStudyMetricV2(
                                outer_target_center=outer,
                                inner_pseudo_target_center=prepared.inner,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                arm=arm,
                                preservation_ratio=_safe_ratio(
                                    prior_scores[arm].bacc,
                                    prepared.real_bacc,
                                    config.minimum_real_bacc,
                                ),
                                decode_bacc=decode_scores[arm].bacc,
                                posterior_bacc=posterior_scores[arm].bacc,
                                valid=(
                                    valid
                                    and prior_scores[arm].converged
                                    and posterior_scores[arm].converged
                                    and decode_scores[arm].converged
                                ),
                                eligible=eligible,
                                ineligibility_reason=(
                                    ""
                                    if eligible
                                    else (
                                        "conditional_sampler_not_realized"
                                        if arm == "C-diag"
                                        else "learned_prior_mechanism_ineligible"
                                    )
                                ),
                            )
                        )
                        for stream, epsilon_hash in (
                            ("prior_generation", prior_epsilon_hash),
                            ("posterior_evaluation", posterior_epsilon_hash),
                        ):
                            rng_rows.append(
                                {
                                    "schema_version": PAIRING_AUDIT_SCHEMA,
                                    "outer_target_center": outer,
                                    "inner_pseudo_target_center": prepared.inner,
                                    "training_seed": training_seed,
                                    "generation_seed": generation_seed,
                                    "arm": arm,
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
    prior_state_index = {
        "schema_version": PRIOR_STATE_INDEX_SCHEMA,
        "records": sorted(prior_state_records, key=lambda row: str(row["state_hash"])),
    }
    generation_budget_manifest = {
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
        checkpoint_reuse_rows=checkpoint_reuse_rows,
        initialization_pairing_rows=initialization_rows,
        generation_budget_rows=budget_rows,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        feature_frame_index=frame_index,
        generation_budget_manifest=generation_budget_manifest,
        rng_rows=rng_rows,
        protocol_manifest=protocol,
        study_state_index=prior_state_index,
    )
    consensus, children, paired_deltas = _decisions(
        config,
        decision_metrics=decision_metrics,
        protocol_hash=protocol_hash,
        selection_evidence_hash_value=evidence_hash,
    )
    # Paired deltas are a rendered view of the already hash-bound decision
    # inputs. Recompute the final evidence identity including that view.
    evidence_hash = selection_evidence_hash(
        metric_rows=metric_rows,
        paired_delta_rows=paired_deltas,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        checkpoint_reuse_rows=checkpoint_reuse_rows,
        initialization_pairing_rows=initialization_rows,
        generation_budget_rows=budget_rows,
        checkpoint_index=checkpoint_index,
        initialization_index=initialization_index,
        feature_frame_index=frame_index,
        generation_budget_manifest=generation_budget_manifest,
        rng_rows=rng_rows,
        protocol_manifest=protocol,
        study_state_index=prior_state_index,
    )
    # Bind the rendered consensus to the final bundle hash.
    consensus, children, paired_deltas = _decisions(
        config,
        decision_metrics=decision_metrics,
        protocol_hash=protocol_hash,
        selection_evidence_hash_value=evidence_hash,
    )
    coverage = _coverage_manifest(
        config,
        decision_metrics=decision_metrics,
        metric_rows=metric_rows,
        prior_state_records=prior_state_records,
    )
    selection_manifest = {
        "schema_version": SELECTION_EVIDENCE_SCHEMA,
        "selection_evidence_hash": evidence_hash,
        "runtime_rows_included": False,
        "decisions_may_feed_model_recipe": False,
    }
    leakage = _leakage_report(config, protocol_hash=protocol_hash, identity_rows=identity_rows)
    study_decision = _study_summary(consensus, protocol_hash=protocol_hash, evidence_hash=evidence_hash)
    write_json(root / "manifests/learned_prior_state_index.json", prior_state_index)
    write_prior_study_bundle(
        root,
        learned_prior_state_index=prior_state_index,
        metric_rows=metric_rows,
        paired_delta_rows=paired_deltas,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        checkpoint_reuse_rows=checkpoint_reuse_rows,
        initialization_pairing_rows=initialization_rows,
        generation_budget_rows=budget_rows,
        rng_rows=rng_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol,
        coverage_manifest=coverage,
        selection_evidence_manifest=selection_manifest,
        embedded_preparation_lineage=lineage,
        generation_budget_manifest=generation_budget_manifest,
        child_decisions=children,
        consensus_decisions=consensus,
        study_decision=study_decision,
        leakage_report=leakage,
    )
    write_study_run_state(
        root, protocol_hash=protocol_hash, mode=config.mode, status="COMPLETE"
    )
    from .prior_validation import validate_prior_study_bundle

    validate_prior_study_bundle(root, expected_config=config)
    return root


def _runtime(
    config: LearnedConditionalPriorStudyConfig,
    *,
    prepared: object,
    training_seed: int,
    model_family: str,
    prior_family: str,
    protocol_hash: str,
    checkpoints: StudyCheckpointStore,
    timings: StudyTimingRecorder,
) -> StudyRuntime:
    variant = config.training_variant(
        model_family=model_family,
        prior_family=prior_family,
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
        training_seed=int(training_seed),
        variant=variant,
    )
    started = perf_counter()
    runtime = checkpoints.load(
        training_key=key,
        variant=variant,
        input_dim=np.asarray(prepared.x_fit).shape[1],
        device=config.device,
    )
    cache_status = "hit" if runtime is not None else "miss"
    if runtime is None:
        runtime = train_study_cvae(
            prepared.x_fit,
            prepared.y_fit,
            variant=variant,
            training_key=key,
            model_family=model_family,
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


def _score(generated: object, prepared: object, *, train_labels: Sequence[int] | None = None):
    return score_representation(
        generated,
        prepared.y_fit if train_labels is None else train_labels,
        prepared.x_eval,
        prepared.y_eval,
        spec=prepared.spec,
    )


def _metric_row(
    config: LearnedConditionalPriorStudyConfig,
    *,
    prepared: object,
    runtime: StudyRuntime,
    arm: str,
    generation_seed: int,
    role: str,
    score: object,
    valid: bool,
    eligible: bool,
    sampler_state_hash: str,
    budget: Sequence[int],
    protocol_hash: str,
) -> dict[str, object]:
    ratio = _safe_ratio(score.bacc, prepared.real_bacc, config.minimum_real_bacc)
    row_valid = bool(valid and score.converged and math.isfinite(ratio))
    ineligibility_reason = (
        ""
        if eligible
        else (
            "conditional_sampler_not_realized"
            if arm == "C-diag"
            else "learned_prior_mechanism_ineligible"
        )
    )
    return {
        "schema_version": METRIC_SCHEMA,
        "method": METHOD,
        "protocol_hash": protocol_hash,
        "outer_target_center": prepared.outer,
        "inner_pseudo_target_center": prepared.inner,
        "fit_centers": json.dumps(list(prepared.fit_centers)),
        "training_seed": runtime.training_key.training_seed,
        "generation_seed": generation_seed,
        "arm": arm,
        "model_family": runtime.model_family,
        "prior_family": (
            config.ex_post_prior_family
            if arm == "C-diag"
            else runtime.variant.prior_family
        ),
        "objective_id": runtime.variant.objective_id,
        "alpha": runtime.variant.alpha,
        "representation_role": role,
        "bacc": score.bacc,
        "macro_f1": score.macro_f1,
        "real_reference_bacc": prepared.real_bacc,
        "preservation_ratio": ratio,
        "generation_class_counts": json.dumps(
            [sum(int(value) == label for value in budget) for label in (0, 1)]
        ),
        "classifier_spec_hash": prepared.spec.config_hash,
        "frame_hash": prepared.frame.state_hash,
        "fit_row_hash": row_hash(prepared.source_ids),
        "eval_row_hash": row_hash(prepared.eval_ids),
        "training_key_hash": runtime.training_key.hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "shared_initialization_hash": runtime.shared_initialization_hash,
        "training_stream_hash": runtime.training_stream_hash,
        "sampler_state_hash": sampler_state_hash,
        "valid": str(row_valid).lower(),
        "eligible": str(bool(eligible)).lower(),
        "ineligibility_reason": ineligibility_reason,
        "status": "ok" if row_valid else "invalid_or_nonconverged",
        "claim_scope": "cvae_source_inner_study_only",
        "selection_source": "fully_nested_source_inner",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
    }


def _learned_prior_record(
    runtime: StudyRuntime,
    *,
    posterior_mu: object,
    posterior_logvar: object,
    labels: Sequence[int],
    sampler_c: object,
    outer: str,
    inner: str,
    training_seed: int,
) -> tuple[dict[str, object], bool, bool]:
    device = torch.device(runtime.device)
    mu_tensor = torch.as_tensor(np.asarray(posterior_mu), dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
    diagnostics = runtime.model.prior_state_diagnostics(mu_tensor, label_tensor)
    state = dict(runtime.model.latent_prior.state_payload())
    posterior_statistics, kl_by_class = _posterior_prior_audit(
        posterior_mu=posterior_mu,
        posterior_logvar=posterior_logvar,
        labels=labels,
        state=state,
    )
    trajectory_fields = (
        "epoch",
        "prior_mu_min",
        "prior_mu_max",
        "prior_mu_l2_by_class",
        "prior_rho_min",
        "prior_rho_max",
        "effective_logvar_min",
        "effective_logvar_max",
        "prior_std_min",
        "prior_std_max",
        "prior_saturation_count",
        "prior_saturated",
    )
    trajectory = [
        {field: row[field] for field in trajectory_fields}
        for row in runtime.diagnostics
    ]
    gap_by_class: dict[str, object] = {}
    effective_logvar = np.asarray(state["effective_logvar"], dtype=np.float64)
    learned_mu = np.asarray(state["prior_mu"], dtype=np.float64)
    for class_label in (0, 1):
        c_state = sampler_c.classes[class_label]
        c_variance = np.diag(np.asarray(c_state.covariance, dtype=np.float64))
        gap_by_class[str(class_label)] = {
            "mean_l2": float(np.linalg.norm(learned_mu[class_label] - c_state.mean)),
            "variance_l2": float(
                np.linalg.norm(np.exp(effective_logvar[class_label]) - c_variance)
            ),
        }
    payload: dict[str, object] = {
        "schema_version": "midogpp_learned_conditional_prior_state_record_v2",
        "outer_target_center": outer,
        "inner_pseudo_target_center": inner,
        "training_seed": training_seed,
        "training_key_hash": runtime.training_key.hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "source_row_hash": runtime.training_key.fit_row_hash,
        "frame_hash": runtime.training_key.frame_hash,
        "state": state,
        "final_prior_partition_hash": stable_hash(
            {
                "prior_mu": state["prior_mu"],
                "prior_rho": state["prior_rho"],
                "effective_logvar": state["effective_logvar"],
            }
        ),
        "diagnostics": diagnostics.to_payload(),
        "posterior_sufficient_statistics_by_class": posterior_statistics,
        "posterior_sufficient_statistics_hash": stable_hash(
            posterior_statistics
        ),
        "kl_to_learned_prior_by_class": kl_by_class,
        "prior_training_trajectory": trajectory,
        "prior_training_trajectory_hash": stable_hash(trajectory),
        "transient_saturation_observed": any(
            bool(row["prior_saturated"]) for row in trajectory
        ),
        "ex_post_diagonal_moment_gap_by_class": gap_by_class,
        "integrity_valid": diagnostics.finite,
        "primary_preservation_eligible": (
            diagnostics.finite
            and not diagnostics.saturated
            and diagnostics.active_unit_count > 0
        ),
        "class_separation_status": (
            "NO_REALIZED_CLASS_SEPARATION"
            if diagnostics.near_class_independent
            else "REALIZED_CLASS_SEPARATION"
        ),
    }
    payload["state_hash"] = stable_hash(payload)
    return (
        payload,
        bool(diagnostics.finite),
        bool(
            diagnostics.finite
            and not diagnostics.saturated
            and diagnostics.active_unit_count > 0
        ),
    )


def _posterior_prior_audit(
    *,
    posterior_mu: object,
    posterior_logvar: object,
    labels: Sequence[int],
    state: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    mu = np.asarray(posterior_mu, dtype=np.float64)
    logvar = np.asarray(posterior_logvar, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    prior_mu = np.asarray(state["prior_mu"], dtype=np.float64)
    prior_logvar = np.asarray(state["effective_logvar"], dtype=np.float64)
    if (
        mu.ndim != 2
        or logvar.shape != mu.shape
        or len(y) != len(mu)
        or prior_mu.shape != (2, mu.shape[1])
        or prior_logvar.shape != prior_mu.shape
        or not all(
            np.isfinite(value).all()
            for value in (mu, logvar, prior_mu, prior_logvar)
        )
    ):
        raise ProtocolError("Learned-prior posterior audit received invalid state.")
    statistics: dict[str, object] = {}
    kl_audit: dict[str, object] = {}
    for class_label in (0, 1):
        mask = y == class_label
        if not bool(mask.any()):
            raise ProtocolError("Learned-prior posterior audit requires both classes.")
        class_mu = mu[mask]
        class_logvar = logvar[mask]
        mu_mean = class_mu.mean(axis=0)
        mu_variance = class_mu.var(axis=0)
        logvar_mean = class_logvar.mean(axis=0)
        posterior_variance_mean = np.exp(class_logvar).mean(axis=0)
        statistics[str(class_label)] = {
            "n_rows": int(mask.sum()),
            "posterior_mu_mean": mu_mean.tolist(),
            "posterior_mu_variance": mu_variance.tolist(),
            "posterior_logvar_mean": logvar_mean.tolist(),
            "posterior_variance_mean": posterior_variance_mean.tolist(),
        }
        squared_mean = mu_variance + (mu_mean - prior_mu[class_label]) ** 2
        kl_per_dimension = 0.5 * (
            prior_logvar[class_label]
            - logvar_mean
            + posterior_variance_mean * np.exp(-prior_logvar[class_label])
            + squared_mean * np.exp(-prior_logvar[class_label])
            - 1.0
        )
        kl_audit[str(class_label)] = {
            "mean_kl_per_dimension": kl_per_dimension.tolist(),
            "latent_normalized_mean_kl": float(kl_per_dimension.mean()),
            "mean_kl_sum": float(kl_per_dimension.sum()),
        }
    return statistics, kl_audit


def _learned_prior_sampler_rows(record: Mapping[str, object]) -> list[dict[str, object]]:
    state = record["state"]
    if not isinstance(state, Mapping):
        raise ProtocolError("Learned-prior sampler record lacks state payload.")
    means = state["prior_mu"]
    logvars = state["effective_logvar"]
    return [
        {
            "schema_version": PRIOR_SAMPLER_SCHEMA,
            "mechanism": "jointly_learned_class_conditional_diagonal_prior",
            "outer_target_center": record["outer_target_center"],
            "inner_pseudo_target_center": record["inner_pseudo_target_center"],
            "training_seed": record["training_seed"],
            "arm": "E",
            "training_key_hash": record["training_key_hash"],
            "checkpoint_hash": record["checkpoint_hash"],
            "source_row_hash": record["source_row_hash"],
            "latent_dim": state["latent_dim"],
            "class_label": class_label,
            "requested_family": LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
            "realized_family": LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
            "mean": json.dumps(means[class_label]),
            "logvar": json.dumps(logvars[class_label]),
            "variance": json.dumps(
                np.exp(np.asarray(logvars[class_label], dtype=np.float64)).tolist()
            ),
            "state_hash": record["state_hash"],
            "sampler_state_hash": record["state_hash"],
            "fallback_reason": "",
        }
        for class_label in (0, 1)
    ]


def _sampler_rows(**kwargs: object) -> list[dict[str, object]]:
    sampler = kwargs.pop("sampler")
    return [
        {
            "schema_version": PRIOR_SAMPLER_SCHEMA,
            "mechanism": "ex_post_aggregate_posterior_diagonal",
            **kwargs,
            "latent_dim": sampler.latent_dim,
            "source_row_hash": sampler.source_row_hash,
            "class_label": class_label,
            "sampler_state_hash": sampler.state_hash,
            **sampler.classes[class_label].to_payload(),
        }
        for class_label in (0, 1)
    ]


def _decisions(
    config: LearnedConditionalPriorStudyConfig,
    *,
    decision_metrics: Sequence[PriorStudyMetricV2],
    protocol_hash: str,
    selection_evidence_hash_value: str,
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[tuple[int, str], Mapping[str, object]],
    list[dict[str, object]],
]:
    consensus: dict[str, Mapping[str, object]] = {}
    children: dict[tuple[int, str], Mapping[str, object]] = {}
    deltas: list[dict[str, object]] = []
    for outer in config.heldout_centers:
        inner_centers = tuple(center for center in MIDOGPP_ELIGIBLE_CENTERS if center != outer)
        outer_metrics = [row for row in decision_metrics if row.outer_target_center == outer]
        decision = select_prior_study_decision(
            outer_metrics,
            outer_target_center=outer,
            expected_inner_centers=inner_centers,
            protocol_hash=protocol_hash,
            decision_contract_hash=decision_contract_hash(config),
            training_seeds=config.training_seeds,
            generation_seeds=config.generation_seeds,
            e_vs_a_min_mean_delta=config.e_vs_a_min_mean_delta,
            e_vs_c_min_mean_delta=config.e_vs_c_min_mean_delta,
            min_inner_wins=config.min_inner_wins,
            safety_max_bacc_regression=config.safety_max_bacc_regression,
        )
        payload = decision.to_payload()
        payload["selection_evidence_hash"] = selection_evidence_hash_value
        consensus[outer] = payload
        per_seed = decision.per_training_seed
        for seed in config.training_seeds:
            summary = per_seed.get(str(seed), {}) if isinstance(per_seed, Mapping) else {}
            child = {
                "schema_version": "midogpp_learned_prior_child_decision_v2",
                "outer_target_center": outer,
                "training_seed": seed,
                "status": (
                    "INVALID_DECISION"
                    if decision.status == "INVALID_DECISION"
                    else (
                        "E_PASS_A"
                        if bool(summary.get("e_vs_a", {}).get("pass", False))
                        else "E_FAIL_A"
                    )
                ),
                "e_vs_c_status": (
                    "E_C_UNAVAILABLE"
                    if not bool(
                        summary.get("e_vs_c_diag", {}).get("available", True)
                    )
                    else (
                        "E_C_PASS"
                        if bool(summary.get("e_vs_c_diag", {}).get("pass", False))
                        else "E_C_FAIL"
                    )
                ),
                "summary": summary,
                "selection_evidence_hash": selection_evidence_hash_value,
                "may_feed_model_recipe": False,
                "may_feed_deployable_selection": False,
            }
            child["study_decision_hash"] = stable_hash(child)
            children[(seed, outer)] = child
            if isinstance(summary, Mapping):
                for comparison in ("e_vs_a", "e_vs_c_diag"):
                    comp = summary.get(comparison)
                    if not isinstance(comp, Mapping):
                        continue
                    for inner, value in dict(
                        comp.get("preservation_ratio_delta_by_inner", {})
                    ).items():
                        deltas.append(
                            {
                                "outer_target_center": outer,
                                "inner_pseudo_target_center": inner,
                                "training_seed": seed,
                                "comparison": comparison,
                                "preservation_ratio_delta": value,
                            }
                        )
    return consensus, children, deltas


def _protocol_manifest(config: LearnedConditionalPriorStudyConfig, *, frame: object, lineage: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": config.mode,
        "study_version": config.study_version,
        "coverage_mode": (
            "complete" if config.heldout_centers == MIDOGPP_ELIGIBLE_CENTERS else "fixture"
        ),
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


def _coverage_manifest(config: LearnedConditionalPriorStudyConfig, *, decision_metrics: Sequence[PriorStudyMetricV2], metric_rows: Sequence[Mapping[str, object]], prior_state_records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    expected = (
        len(config.heldout_centers)
        * max(len(MIDOGPP_ELIGIBLE_CENTERS) - 1, 0)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * 3
    )
    status = "PASS" if len(decision_metrics) == expected else "FAIL"
    return {
        "schema_version": COVERAGE_SCHEMA,
        "status": status,
        "expected_decision_cells": expected,
        "observed_decision_cells": len(decision_metrics),
        "metric_rows": len(metric_rows),
        "learned_prior_states": len(prior_state_records),
        "complete_training_generation_seed_cross": status == "PASS",
    }


def _leakage_report(config: LearnedConditionalPriorStudyConfig, *, protocol_hash: str, identity_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
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
    statuses = {outer: payload.get("status") for outer, payload in consensus.items()}
    return {
        "schema_version": "midogpp_learned_prior_study_summary_v2",
        "status": "COMPLETE" if all(value != "INVALID_DECISION" for value in statuses.values()) else "INVALID_INCOMPLETE",
        "protocol_hash": protocol_hash,
        "selection_evidence_hash": evidence_hash,
        "per_outer_status": statuses,
        "descriptive_counts": {
            status: sum(value == status for value in statuses.values())
            for status in sorted(set(statuses.values()))
        },
        "cross_outer_selection_performed": False,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
    }


def _safe_ratio(generated_bacc: float, real_bacc: float, minimum_real_bacc: float) -> float:
    try:
        return chance_normalized_preservation(
            generated_bacc,
            real_bacc,
            minimum_real_bacc=minimum_real_bacc,
        )
    except ValueError:
        return math.nan


def _sampler_min_class_count(config: LearnedConditionalPriorStudyConfig) -> int:
    return 64 if config.heldout_centers == MIDOGPP_ELIGIBLE_CENTERS else 2
