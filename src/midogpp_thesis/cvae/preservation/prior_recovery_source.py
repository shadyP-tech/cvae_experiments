"""Scalar fully nested source-inner recipe selection for prior recovery."""

from __future__ import annotations

import json
from pathlib import Path

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ...real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
    MIDOGPP_EXCLUDED_CENTERS,
)
from .prior_recovery_artifacts import write_source_inner_bundle
from .prior_recovery_common import (
    PRIOR_RECOVERY_METHOD,
    canonical_rows_hash,
    load_frame,
    protocol_hash,
    selection_evidence_hash,
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
from .prior_recovery_source_evidence import (
    SourceInnerSeedContext,
    run_isotropic_seed,
    run_task_fold,
    select_source_inner_lock,
)
from .prior_recovery_source_preparation import prepare_source_inner_fold
from .prior_recovery_timing import (
    RuntimeTimingRecorder,
    mark_run_failed,
    write_run_state,
)
from .source_inner_selection import InnerCenterMetric, RecipeLock
from .splits import source_only_frame


def run_source_inner_prior_recovery(
    config: SourceInnerPriorRecoveryConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    if not isinstance(config, SourceInnerPriorRecoveryConfig):
        raise ProtocolError(
            "Source-inner runner requires SourceInnerPriorRecoveryConfig."
        )
    root = Path(artifact_root or config.artifact_root)
    from ..reporting import prepare_artifact_dirs

    root = prepare_artifact_dirs(root)
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
    timings = RuntimeTimingRecorder(
        root,
        protocol_hash=runtime_protocol_hash,
        mode="source_inner",
    )
    write_run_state(
        root,
        protocol_hash=runtime_protocol_hash,
        mode="source_inner",
        status="RUNNING",
    )
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
        contexts: list[SourceInnerSeedContext] = []
        summaries: list[InnerCenterMetric] = []
        outer_start = len(detail_rows)
        for inner in expected_inner:
            prepared, nested_row, audit_row = prepare_source_inner_fold(
                pca_dim=config.pca_dim,
                frame=source_frame,
                outer=outer,
                inner=inner,
                candidate_specs=specs,
                preparation_protocol_hash=runtime_protocol_hash,
                preparation_code_version=config.code_version,
                frame_cache=frame_cache,
                timings=timings,
            )
            context, rows, sampler_detail, center_summaries = run_isotropic_seed(
                config,
                prepared=prepared,
                runtime_protocol_hash=runtime_protocol_hash,
                recorder=recorder,
                timings=timings,
            )
            contexts.append(context)
            detail_rows.extend(rows)
            nested_reference_rows.append(nested_row)
            nested_tuning_rows.extend(
                dict(row) for row in context.selection.candidate_rows
            )
            sampler_rows.extend(sampler_detail)
            summaries.extend(center_summaries)
            identity_rows.append(audit_row)
        fit_sets_hash = stable_hash(
            {
                context.inner: list(context.fit_centers)
                for context in contexts
            }
        )
        fit_hash_by_outer[outer] = fit_sets_hash
        preliminary = select_source_inner_lock(
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
                rows, sampler_detail, task_summaries = run_task_fold(
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
    checkpoint_index = json.loads(
        (root / "manifests/checkpoint_index.json").read_text(encoding="utf-8")
    )
    fisher_index = json.loads(
        (root / "manifests/task_fisher_index.json").read_text(encoding="utf-8")
    )
    frame_index = json.loads(
        (root / "manifests/feature_frame_index.json").read_text(encoding="utf-8")
    )
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
            if config.heldout_centers
            == frame.eligible_centers
            == MIDOGPP_ELIGIBLE_CENTERS
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
        expected_inner = tuple(
            center for center in frame.eligible_centers if center != outer
        )
        outer_rows = [
            row for row in detail_rows if row["outer_target_center"] == outer
        ]
        locks.append(
            select_source_inner_lock(
                config,
                summaries_by_outer[outer],
                outer=outer,
                expected_inner=expected_inner,
                runtime_protocol_hash=runtime_protocol_hash,
                fit_sets_hash=fit_hash_by_outer[outer],
                source_metric_hash=canonical_rows_hash(outer_rows),
                selection_bundle_hash=bundle_hash,
                require_task_factorial=(
                    preliminary_by_outer[outer].primary_arm == "C"
                ),
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
