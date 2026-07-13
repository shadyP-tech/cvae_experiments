"""Locked outer-fold A/B/C/D preservation evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from ..feature_frame import ExpertFeatureFrame, fit_expert_frame
from ..generation_samplers import AggregatePosteriorSampler, STANDARD_SAMPLER
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE
from ..reporting import prepare_artifact_dirs
from ..task_fisher import fit_task_fisher_metric
from ..training import TrainedCVAERuntime
from .prior_recovery_artifacts import (
    validate_source_inner_bundle,
    write_outer_bundle,
)
from .prior_recovery_common import (
    NO_TASK_FISHER_STATE,
    PRIOR_RECOVERY_METHOD,
    classifier_spec,
    fit_samplers,
    generation_and_evaluation_hashes,
    load_frame,
    protocol_hash,
    safe_ratio,
    train_runtime,
)
from .prior_recovery_config import (
    OuterPriorRecoveryConfig,
    outer_decision_contract_hash,
    outer_decision_contract_payload,
    recipe_contract_hash,
    recipe_contract_payload,
)
from .prior_recovery_decision import aggregate_outer
from .prior_recovery_provenance import ProvenanceRecorder
from .prior_recovery_schema import OUTER_METRIC_SCHEMA, SAMPLER_REALIZATION_SCHEMA
from .representations import decode_means, posterior_samples, sampler_decodes, source_budget_labels
from .scoring import RepresentationScore, score_representation
from .source_inner_selection import RecipeLock
from .splits import (
    assert_identity_overlap_pass,
    frame_arrays,
    identity_overlap_audit,
    indices_for_centers,
    outer_split,
    row_hash,
)
from .tuned_reference import (
    MATCHED_REFERENCE_SCHEMA_VERSION,
    REFERENCE_METHOD_PREDICT,
    load_tuned_classifier_reference,
)


def run_outer_prior_recovery(
    config: OuterPriorRecoveryConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    if not isinstance(config, OuterPriorRecoveryConfig):
        raise ProtocolError("Outer runner requires OuterPriorRecoveryConfig.")
    root = prepare_artifact_dirs(Path(artifact_root or config.artifact_root))
    recorder = ProvenanceRecorder(root)
    frame = load_frame(config)
    reference = load_tuned_classifier_reference(
        config.reference_artifact_root,
        expected_manifest_hash=frame.manifest_hash,
        expected_feature_cache_hash=frame.feature_cache_hash,
        required_centers=config.heldout_centers,
    )
    if reference.protocol.get("schema_version") != MATCHED_REFERENCE_SCHEMA_VERSION:
        raise ProtocolError("Outer prior recovery requires the eligible-only matched Stage-10 reference v2.")
    if reference.protocol.get("method") != REFERENCE_METHOD_PREDICT:
        raise ProtocolError("Outer prior recovery requires the frozen predict-policy reference.")
    if tuple(reference.heldout_centers) != config.heldout_centers:
        raise ProtocolError("Outer reference center coverage differs from the configured folds.")
    locks = validate_source_inner_bundle(
        config.recipe_lock_artifact_root,
        expected_config=config,
        require_factorial=True,
    )
    bundle_hashes = {lock.selection_bundle_hash for lock in locks.values()}
    if len(bundle_hashes) != 1:
        raise ProtocolError("RecipeLocks do not share one source-inner evidence bundle identity.")
    selection_bundle_hash = bundle_hashes.pop()
    source_protocol_hashes = {lock.protocol_hash for lock in locks.values()}
    if len(source_protocol_hashes) != 1:
        raise ProtocolError("RecipeLocks do not share one source-inner protocol identity.")
    source_inner_protocol_hash = source_protocol_hashes.pop()
    reference_protocol_hash = str(reference.protocol["protocol_hash"])
    reference_identity = {
        "real_reference_protocol_hash": reference_protocol_hash,
        "real_reference_bundle_hash": reference.protocol["reference_bundle_hash"],
        "classifier_spec_hashes": {
            center: reference.rows_by_center[center].selected_classifier_spec.config_hash
            for center in config.heldout_centers
        },
        "real_reference_bacc_by_center": {
            center: reference.rows_by_center[center].bacc
            for center in config.heldout_centers
        },
        "real_reference_eval_row_hashes": {
            center: reference.rows_by_center[center].source_row["eval_row_hash"]
            for center in config.heldout_centers
        },
    }
    frozen_reference_identity_hash = stable_hash(reference_identity)
    runtime_protocol_hash = protocol_hash(
        config,
        frame,
        reference_protocol_hash=reference_protocol_hash,
        selection_bundle_hash=selection_bundle_hash,
        source_inner_protocol_hash=source_inner_protocol_hash,
        frozen_reference_identity_hash=frozen_reference_identity_hash,
    )
    metric_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    for outer in config.heldout_centers:
        lock = locks[outer]
        if lock.primary_arm not in {"C", "D"}:
            raise ProtocolError("Outer factorial requires a conditional source-inner lock for every center.")
        reference_row = reference.rows_by_center[outer]
        spec = classifier_spec(reference_row.selected_classifier_spec)
        real_bacc = float(reference_row.bacc)
        if not math.isfinite(real_bacc) or real_bacc < config.minimum_real_bacc:
            raise ProtocolError(f"Outer real-reference denominator for center {outer} is below the locked floor.")
        split = outer_split(outer, centers=frame.eligible_centers)
        fit_idx = indices_for_centers(frame, split.fit_centers)
        eval_idx = indices_for_centers(frame, (outer,))
        audit = identity_overlap_audit(
            frame,
            fit_indices=fit_idx,
            eval_indices=eval_idx,
            outer_target_center=outer,
        )
        assert_identity_overlap_pass(audit)
        identity_rows.append(audit)
        x_fit_full, y_fit, source_ids = frame_arrays(frame, fit_idx)
        x_eval_full, y_eval, eval_ids = frame_arrays(frame, eval_idx)
        feature_frame = fit_expert_frame(
            expert_id=f"outer_H{outer}",
            source_train_embeddings=x_fit_full,
            requested_dim=config.pca_dim,
        )
        x_fit = feature_frame.transform(x_fit_full)
        x_eval = feature_frame.transform(x_eval_full)
        for training_seed in config.training_seeds:
            runtime_a = train_runtime(
                config,
                variant=config.isotropic_variant,
                frame=feature_frame,
                fit_centers=split.fit_centers,
                source_ids=source_ids,
                x_fit=x_fit,
                y_fit=y_fit,
                training_seed=training_seed,
                runtime_protocol_hash=runtime_protocol_hash,
                feature_cache_hash=frame.feature_cache_hash,
                manifest_hash=frame.manifest_hash,
                task_metric=None,
                objective_context_hash=NO_TASK_FISHER_STATE,
            )
            recorder.record_runtime(
                runtime_a,
                task_fisher_state_hash=NO_TASK_FISHER_STATE,
                classifier_spec_hash=spec.config_hash,
            )
            fisher = fit_task_fisher_metric(x_fit, y_fit, spec=spec, alpha=1.0)
            fisher_hash = recorder.record_fisher(fisher)
            if not fisher.valid:
                raise ProtocolError(f"Outer Task-Fisher state invalid for center {outer}: {fisher.reason}")
            runtime_b = train_runtime(
                config,
                variant=config.task_fisher_variant,
                frame=feature_frame,
                fit_centers=split.fit_centers,
                source_ids=source_ids,
                x_fit=x_fit,
                y_fit=y_fit,
                training_seed=training_seed,
                runtime_protocol_hash=runtime_protocol_hash,
                feature_cache_hash=frame.feature_cache_hash,
                manifest_hash=frame.manifest_hash,
                task_metric=fisher.metric,
                objective_context_hash=fisher_hash,
            )
            recorder.record_runtime(
                runtime_b,
                task_fisher_state_hash=fisher_hash,
                classifier_spec_hash=spec.config_hash,
            )
            metric_rows.extend(
                _objective_rows(
                    config,
                    outer=outer,
                    lock=lock,
                    objective_arm="A",
                    sampler_arm="C",
                    runtime=runtime_a,
                    task_fisher_state_hash=NO_TASK_FISHER_STATE,
                    requested_sampler=lock.sampler_family,
                    feature_frame=feature_frame,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    source_ids=source_ids,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    eval_ids=eval_ids,
                    spec=spec,
                    real_bacc=real_bacc,
                    real_reference_protocol_hash=reference_protocol_hash,
                    runtime_protocol_hash=runtime_protocol_hash,
                    sampler_audit_rows=sampler_rows,
                )
            )
            metric_rows.extend(
                _objective_rows(
                    config,
                    outer=outer,
                    lock=lock,
                    objective_arm="B",
                    sampler_arm="D",
                    runtime=runtime_b,
                    task_fisher_state_hash=fisher_hash,
                    requested_sampler=lock.sampler_family,
                    feature_frame=feature_frame,
                    x_fit=x_fit,
                    y_fit=y_fit,
                    source_ids=source_ids,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    eval_ids=eval_ids,
                    spec=spec,
                    real_bacc=real_bacc,
                    real_reference_protocol_hash=reference_protocol_hash,
                    runtime_protocol_hash=runtime_protocol_hash,
                    sampler_audit_rows=sampler_rows,
                )
            )
            checkpoint_rows.append(
                {
                    "outer_target_center": outer,
                    "training_seed": training_seed,
                    "checkpoint_a_hash": runtime_a.checkpoint_hash,
                    "checkpoint_c_hash": runtime_a.checkpoint_hash,
                    "checkpoint_b_hash": runtime_b.checkpoint_hash,
                    "checkpoint_d_hash": runtime_b.checkpoint_hash,
                    "a_c_identity": True,
                    "b_d_identity": True,
                    "a_b_initialization_paired": runtime_a.initialization_hash == runtime_b.initialization_hash,
                    "a_b_stochastic_stream_paired": runtime_a.stochastic_stream_hash == runtime_b.stochastic_stream_hash,
                    "stochastic_pairing_hash": runtime_a.training_key.stochastic_pairing_hash,
                    "task_fisher_state_hash": fisher_hash,
                    "classifier_spec_hash": spec.config_hash,
                    "frame_hash": feature_frame.state_hash,
                    "fit_row_hash": row_hash(source_ids),
                    "eval_row_hash": row_hash(eval_ids),
                    "status": "PASS",
                }
            )
    recorder.write_indices()
    aggregation_rows, paired_rows, decision, coverage = aggregate_outer(config, metric_rows, locks)
    protocol_manifest = {
        "schema_version": "midogpp_prior_recovery_outer_protocol_v1",
        "experiment_name": config.name,
        "method": PRIOR_RECOVERY_METHOD,
        "claim_scope": "cvae_preservation_only",
        "claim_role": "cvae_preservation",
        "protocol_hash": runtime_protocol_hash,
        "recipe_contract": recipe_contract_payload(config),
        "recipe_contract_hash": recipe_contract_hash(config),
        "selection_bundle_hash": selection_bundle_hash,
        "source_inner_protocol_hash": source_inner_protocol_hash,
        "real_reference_protocol_hash": reference_protocol_hash,
        **reference_identity,
        "frozen_reference_identity_hash": frozen_reference_identity_hash,
        "real_reference_protocol_file_sha256": _file_sha256(
            config.reference_artifact_root / "manifests/protocol_manifest.json"
        ),
        "source_inner_protocol_file_sha256": _file_sha256(
            config.recipe_lock_artifact_root / "manifests/protocol_manifest.json"
        ),
        "source_selection_evidence_file_sha256": _file_sha256(
            config.recipe_lock_artifact_root
            / "manifests/selection_evidence_manifest.json"
        ),
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "heldout_centers": list(config.heldout_centers),
        "eligible_centers": list(frame.eligible_centers),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "coverage_mode": (
            "complete"
            if config.heldout_centers == frame.eligible_centers == MIDOGPP_ELIGIBLE_CENTERS
            else "partial_test"
        ),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "outer_decision_contract": outer_decision_contract_payload(config),
        "outer_decision_contract_hash": outer_decision_contract_hash(config),
        "recipe_lock_hashes": {center: lock.hash for center, lock in locks.items()},
        "locked_recipes": {center: lock.to_payload() for center, lock in locks.items()},
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_selection": False,
        "support_labels_used": False,
        "oracle_eligible": False,
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
        "routing_performed": False,
        "composition_performed": False,
    }
    leakage = {
        "status": "PASS" if coverage["status"] == "PASS" else "FAIL",
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_selection": False,
        "outer_metrics_may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
        "center_4_excluded": True,
        "identity_overlap_status": "PASS",
        "routing_performed": False,
        "composition_performed": False,
        "query_object": "none",
        "forbidden_reuse": [
            "expert_bank_evidence",
            "routing_evidence",
            "expert_selection_evidence",
            "nelbo_compatibility_evidence",
        ],
    }
    return write_outer_bundle(
        root,
        metric_rows=metric_rows,
        sampler_rows=sampler_rows,
        paired_delta_rows=paired_rows,
        aggregation_rows=aggregation_rows,
        checkpoint_audit_rows=checkpoint_rows,
        identity_audit_rows=identity_rows,
        protocol_manifest=protocol_manifest,
        coverage_manifest=coverage,
        decision_report=decision,
        leakage_report=leakage,
    )


def _objective_rows(
    config: OuterPriorRecoveryConfig,
    *,
    outer: str,
    lock: RecipeLock,
    objective_arm: str,
    sampler_arm: str,
    runtime: TrainedCVAERuntime,
    task_fisher_state_hash: str,
    requested_sampler: str,
    feature_frame: ExpertFeatureFrame,
    x_fit: object,
    y_fit: tuple[int, ...],
    source_ids: tuple[str, ...],
    x_eval: object,
    y_eval: tuple[int, ...],
    eval_ids: tuple[str, ...],
    spec: object,
    real_bacc: float,
    real_reference_protocol_hash: str,
    runtime_protocol_hash: str,
    sampler_audit_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    samplers = fit_samplers(
        config,
        runtime=runtime,
        x_fit=x_fit,
        y_fit=y_fit,
        source_ids=source_ids,
        families=(STANDARD_SAMPLER, requested_sampler),
    )
    if not samplers[requested_sampler].requested_family_realized_for_both_classes:
        raise ProtocolError(
            f"Outer conditional sampler changed meaning through fallback for center {outer}."
        )
    for arm, family in ((objective_arm, STANDARD_SAMPLER), (sampler_arm, requested_sampler)):
        sampler_audit_rows.extend(
            _outer_sampler_rows(outer, arm, runtime, samplers[family])
        )
    budget_labels = source_budget_labels(y_fit)
    rows: list[dict[str, object]] = []
    for arm, family in ((objective_arm, STANDARD_SAMPLER), (sampler_arm, requested_sampler)):
        for seed in config.generation_seeds:
            score = score_representation(
                sampler_decodes(runtime, samplers[family], budget_labels, seed=seed),
                budget_labels,
                x_eval,
                y_eval,
                spec=spec,
            )
            rows.append(
                _outer_row(
                    config,
                    outer=outer,
                    lock=lock,
                    arm=arm,
                    runtime=runtime,
                    task_fisher_state_hash=task_fisher_state_hash,
                    sampler=samplers[family],
                    generation_seed=seed,
                    role="prior",
                    labels=budget_labels,
                    score=score,
                    real_bacc=real_bacc,
                    real_reference_protocol_hash=real_reference_protocol_hash,
                    spec=spec,
                    frame=feature_frame,
                    source_ids=source_ids,
                    eval_ids=eval_ids,
                    runtime_protocol_hash=runtime_protocol_hash,
                )
            )
    decoded, _, _ = decode_means(runtime, x_fit, y_fit)
    decode_score = score_representation(decoded, y_fit, x_eval, y_eval, spec=spec)
    for arm, family in ((objective_arm, STANDARD_SAMPLER), (sampler_arm, requested_sampler)):
        rows.append(
            _outer_row(
                config,
                outer=outer,
                lock=lock,
                arm=arm,
                runtime=runtime,
                task_fisher_state_hash=task_fisher_state_hash,
                sampler=samplers[family],
                generation_seed=-1,
                role="decode",
                labels=y_fit,
                score=decode_score,
                real_bacc=real_bacc,
                real_reference_protocol_hash=real_reference_protocol_hash,
                spec=spec,
                frame=feature_frame,
                source_ids=source_ids,
                eval_ids=eval_ids,
                runtime_protocol_hash=runtime_protocol_hash,
            )
        )
    for seed in config.generation_seeds:
        posterior_score = score_representation(
            posterior_samples(runtime, x_fit, y_fit, seed=seed)[0],
            y_fit,
            x_eval,
            y_eval,
            spec=spec,
        )
        for arm, family in ((objective_arm, STANDARD_SAMPLER), (sampler_arm, requested_sampler)):
            rows.append(
                _outer_row(
                    config,
                    outer=outer,
                    lock=lock,
                    arm=arm,
                    runtime=runtime,
                    task_fisher_state_hash=task_fisher_state_hash,
                    sampler=samplers[family],
                    generation_seed=seed,
                    role="posterior",
                    labels=y_fit,
                    score=posterior_score,
                    real_bacc=real_bacc,
                    real_reference_protocol_hash=real_reference_protocol_hash,
                    spec=spec,
                    frame=feature_frame,
                    source_ids=source_ids,
                    eval_ids=eval_ids,
                    runtime_protocol_hash=runtime_protocol_hash,
                )
            )
    return rows


def _outer_sampler_rows(
    outer: str,
    arm: str,
    runtime: TrainedCVAERuntime,
    sampler: AggregatePosteriorSampler,
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": SAMPLER_REALIZATION_SCHEMA,
            "outer_target_center": outer,
            "inner_pseudo_target_center": "",
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


def _outer_row(
    config: OuterPriorRecoveryConfig,
    *,
    outer: str,
    lock: RecipeLock,
    arm: str,
    runtime: TrainedCVAERuntime,
    task_fisher_state_hash: str,
    sampler: AggregatePosteriorSampler,
    generation_seed: int,
    role: str,
    labels: Sequence[int],
    score: RepresentationScore,
    real_bacc: float,
    real_reference_protocol_hash: str,
    spec: object,
    frame: ExpertFeatureFrame,
    source_ids: Sequence[str],
    eval_ids: Sequence[str],
    runtime_protocol_hash: str,
) -> dict[str, object]:
    generation_hash, evaluation_hash = generation_and_evaluation_hashes(
        runtime=runtime,
        sampler=sampler,
        generation_seed=generation_seed,
        labels=labels,
        representation_role=role,
        classifier_spec_hash=spec.config_hash,
        eval_center=outer,
        eval_ids=eval_ids,
        runtime_protocol_hash=runtime_protocol_hash,
    )
    status = "ok" if score.converged else "classifier_nonconverged"
    return {
        "schema_version": OUTER_METRIC_SCHEMA,
        "method": PRIOR_RECOVERY_METHOD,
        "protocol_hash": runtime_protocol_hash,
        "recipe_contract_hash": recipe_contract_hash(config),
        "selection_bundle_hash": lock.selection_bundle_hash,
        "real_reference_protocol_hash": real_reference_protocol_hash,
        "outer_target_center": outer,
        "fit_centers": json.dumps(list(runtime.training_key.fit_centers)),
        "arm": arm,
        "objective_id": runtime.variant.objective_id,
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
        "recipe_lock_hash": lock.hash,
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
        "is_prelocked_primary": str(lock.primary_arm == arm).lower(),
        "status": status,
        "claim_scope": "cvae_preservation_only" if status == "ok" else "diagnostic_only",
        "claim_role": "cvae_preservation",
        "row_role": role,
        "leakage_status": "PASS",
        "support_labels_used": "false",
        "oracle_eligible": "false",
        "selection_source": "source_inner_recipe_lock",
        "target_eval_labels_used_for_scoring_only": "true",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
        "query_object": "none",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
