"""Bounded fully crossed training-seed stability for source-inner RecipeLocks."""

from __future__ import annotations

import json
from pathlib import Path

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from .prior_recovery_common import (
    canonical_rows_hash,
    load_frame,
)
from .prior_recovery_config import (
    SourceInnerStabilityConfig,
    common_source_inner_design_hash,
    scalar_source_inner_config,
)
from .prior_recovery_classifier import source_inner_classifier_specs
from .prior_recovery_provenance import ProvenanceRecorder
from .prior_recovery_runtime_cache import FeatureFrameCache
from .prior_recovery_source_evidence import (
    SharedTaskFisherState,
    SourceInnerSeedContext,
    fit_shared_task_fisher_state,
    run_isotropic_seed,
    run_task_fold,
    select_source_inner_lock,
)
from .prior_recovery_source_preparation import (
    PreparedSourceInnerFold,
    prepare_source_inner_fold,
)
from .prior_recovery_stability_artifacts import (
    STABILITY_MODE,
    write_stability_publication_state,
    write_stability_bundle,
)
from .prior_recovery_stability_common import (
    child_source_inner_protocol,
    derive_rng_pairing_audit,
    parent_stability_protocol,
    seed_selection_evidence_hash,
    stability_selection_evidence_hash,
)
from .prior_recovery_stability_consensus import (
    TrainingSeedRecipeLock,
    select_training_seed_consensus,
)
from .prior_recovery_timing import (
    RuntimeTimingRecorder,
    mark_run_failed,
    write_run_state,
)
from .source_inner_selection import InnerCenterMetric, RecipeLock
from .splits import source_only_frame


def run_source_inner_training_seed_stability(
    config: SourceInnerStabilityConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    if not isinstance(config, SourceInnerStabilityConfig):
        raise ProtocolError("Stability runner requires SourceInnerStabilityConfig.")
    from ..reporting import prepare_artifact_dirs

    root = prepare_artifact_dirs(Path(artifact_root or config.artifact_root))
    try:
        return _run_source_inner_training_seed_stability(config, root=root)
    except Exception:
        mark_run_failed(root, mode=STABILITY_MODE)
        publication_path = root / "reports/publication_state.json"
        publication = (
            _read_json(publication_path)
            if publication_path.is_file()
            else {
                "protocol_hash": "unavailable",
                "selection_bundle_hash": "unavailable",
            }
        )
        write_stability_publication_state(
            root,
            status="FAILED",
            protocol_hash=str(publication.get("protocol_hash", "unavailable")),
            selection_bundle_hash=str(
                publication.get("selection_bundle_hash", "unavailable")
            ),
        )
        raise


def _run_source_inner_training_seed_stability(
    config: SourceInnerStabilityConfig,
    *,
    root: Path,
) -> Path:
    frame = load_frame(config)
    scalar_configs = {
        seed: scalar_source_inner_config(config, training_seed=seed)
        for seed in config.training_seeds
    }
    child_protocols = {
        str(seed): child_source_inner_protocol(scalar_configs[seed], frame)
        for seed in config.training_seeds
    }
    protocol_manifest = parent_stability_protocol(
        config,
        frame,
        child_protocols=child_protocols,
    )
    parent_protocol_hash = str(protocol_manifest["protocol_hash"])
    recorder = ProvenanceRecorder(root, allow_shared_checkpoint_hashes=True)
    frame_cache = FeatureFrameCache(root)
    timings = RuntimeTimingRecorder(
        root,
        protocol_hash=parent_protocol_hash,
        mode=STABILITY_MODE,
    )
    write_run_state(
        root,
        protocol_hash=parent_protocol_hash,
        mode=STABILITY_MODE,
        status="RUNNING",
    )
    write_stability_publication_state(
        root,
        status="PENDING",
        protocol_hash=parent_protocol_hash,
        selection_bundle_hash="pending",
    )
    specs = source_inner_classifier_specs(classifier_seed=23)
    metric_rows: list[dict[str, object]] = []
    nested_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    contexts: dict[tuple[int, str], list[SourceInnerSeedContext]] = {}
    summaries: dict[tuple[int, str], list[InnerCenterMetric]] = {}
    preliminary: dict[tuple[int, str], RecipeLock] = {}
    fit_hashes: dict[str, str] = {}
    shared_fishers: dict[tuple[str, str], SharedTaskFisherState] = {}
    preparation_hash = common_source_inner_design_hash(config)

    for outer in config.heldout_centers:
        source_frame = source_only_frame(frame, outer_target_center=outer)
        expected_inner = source_frame.eligible_centers
        prepared_folds: list[PreparedSourceInnerFold] = []
        for inner in expected_inner:
            prepared, nested_row, audit_row = prepare_source_inner_fold(
                pca_dim=config.pca_dim,
                frame=source_frame,
                outer=outer,
                inner=inner,
                candidate_specs=specs,
                preparation_protocol_hash=preparation_hash,
                preparation_code_version=config.child_code_version,
                frame_cache=frame_cache,
                timings=timings,
            )
            prepared_folds.append(prepared)
            nested_rows.append(nested_row)
            tuning_rows.extend(dict(row) for row in prepared.selection.candidate_rows)
            identity_rows.append(audit_row)
        fit_hashes[outer] = stable_hash(
            {prepared.inner: list(prepared.fit_centers) for prepared in prepared_folds}
        )

        for seed in config.training_seeds:
            seed_contexts: list[SourceInnerSeedContext] = []
            seed_summaries: list[InnerCenterMetric] = []
            for prepared in prepared_folds:
                context, rows, sampler_detail, center_summaries = run_isotropic_seed(
                    scalar_configs[seed],
                    prepared=prepared,
                    runtime_protocol_hash=str(
                        child_protocols[str(seed)]["protocol_hash"]
                    ),
                    recorder=recorder,
                    timings=timings,
                )
                seed_contexts.append(context)
                seed_summaries.extend(center_summaries)
                metric_rows.extend(rows)
                sampler_rows.extend(sampler_detail)
            contexts[(seed, outer)] = seed_contexts
            summaries[(seed, outer)] = seed_summaries

        for seed in config.training_seeds:
            seed_outer_rows = [
                row
                for row in metric_rows
                if row["outer_target_center"] == outer
                and int(row["training_seed"]) == seed
            ]
            preliminary[(seed, outer)] = select_source_inner_lock(
                scalar_configs[seed],
                summaries[(seed, outer)],
                outer=outer,
                expected_inner=expected_inner,
                runtime_protocol_hash=str(
                    child_protocols[str(seed)]["protocol_hash"]
                ),
                fit_sets_hash=fit_hashes[outer],
                source_metric_hash=canonical_rows_hash(seed_outer_rows),
                selection_bundle_hash="preliminary",
                require_task_factorial=False,
            )

        for seed in config.training_seeds:
            selected = preliminary[(seed, outer)]
            if selected.primary_arm != "C":
                continue
            for context in contexts[(seed, outer)]:
                key = (outer, context.inner)
                if key not in shared_fishers:
                    prepared = next(
                        item for item in prepared_folds if item.inner == context.inner
                    )
                    shared_fishers[key] = fit_shared_task_fisher_state(
                        prepared,
                        recorder=recorder,
                        timings=timings,
                    )
                rows, sampler_detail, task_summaries = run_task_fold(
                    scalar_configs[seed],
                    context=context,
                    selected_family=selected.sampler_family,
                    runtime_protocol_hash=str(
                        child_protocols[str(seed)]["protocol_hash"]
                    ),
                    recorder=recorder,
                    timings=timings,
                    shared_fisher=shared_fishers[key],
                )
                metric_rows.extend(rows)
                sampler_rows.extend(sampler_detail)
                summaries[(seed, outer)].extend(task_summaries)

    recorder.write_indices()
    frame_cache.write_index()
    timings.finalize()
    checkpoint_index = _read_json(root / "manifests/checkpoint_index.json")
    fisher_index = _read_json(root / "manifests/task_fisher_index.json")
    frame_index = _read_json(root / "manifests/feature_frame_index.json")
    rng_audit_rows = derive_rng_pairing_audit(
        metric_rows,
        checkpoint_index=checkpoint_index,
    )
    parent_bundle_hash = stability_selection_evidence_hash(
        metric_rows=metric_rows,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol_manifest,
        checkpoint_index=checkpoint_index,
        task_fisher_index=fisher_index,
        feature_frame_index=frame_index,
        rng_audit_rows=rng_audit_rows,
    )
    seed_evidence_hashes: dict[str, str] = {}
    for seed in config.training_seeds:
        seed_rows = [row for row in metric_rows if int(row["training_seed"]) == seed]
        keys = {str(row["training_key_hash"]) for row in seed_rows}
        seed_sampler_rows = [
            row for row in sampler_rows if str(row["training_key_hash"]) in keys
        ]
        seed_evidence_hashes[str(seed)] = seed_selection_evidence_hash(
            metric_rows=seed_rows,
            nested_reference_rows=nested_rows,
            nested_tuning_rows=tuning_rows,
            sampler_rows=seed_sampler_rows,
            identity_rows=identity_rows,
            child_protocol=child_protocols[str(seed)],
            checkpoint_index=checkpoint_index,
            task_fisher_index=fisher_index,
            feature_frame_index=frame_index,
            rng_audit_rows=[
                row for row in rng_audit_rows if int(row["training_seed"]) == seed
            ],
        )
    for row in metric_rows:
        row["selection_bundle_hash"] = parent_bundle_hash

    wrapped_locks: list[TrainingSeedRecipeLock] = []
    for seed in config.training_seeds:
        seed_hash = seed_evidence_hashes[str(seed)]
        child = child_protocols[str(seed)]
        for outer in config.heldout_centers:
            outer_rows = [
                row
                for row in metric_rows
                if int(row["training_seed"]) == seed
                and row["outer_target_center"] == outer
            ]
            inner_centers = tuple(
                center for center in frame.eligible_centers if center != outer
            )
            lock = select_source_inner_lock(
                scalar_configs[seed],
                summaries[(seed, outer)],
                outer=outer,
                expected_inner=inner_centers,
                runtime_protocol_hash=str(child["protocol_hash"]),
                fit_sets_hash=fit_hashes[outer],
                source_metric_hash=canonical_rows_hash(outer_rows),
                selection_bundle_hash=seed_hash,
                require_task_factorial=(
                    preliminary[(seed, outer)].primary_arm == "C"
                ),
            )
            wrapped_locks.append(
                TrainingSeedRecipeLock(
                    training_seed=seed,
                    outer_target_center=outer,
                    recipe_lock=lock,
                    seed_evidence_hash=seed_hash,
                    per_seed_contract_hash=str(child["recipe_contract_hash"]),
                    parent_protocol_hash=parent_protocol_hash,
                    checkpoint_hashes=tuple(
                        sorted({str(row["checkpoint_hash"]) for row in outer_rows})
                    ),
                    sampler_state_hashes=tuple(
                        sorted({str(row["sampler_state_hash"]) for row in outer_rows})
                    ),
                )
            )
    consensus_locks = [
        select_training_seed_consensus(
            [
                lock
                for lock in wrapped_locks
                if lock.outer_target_center == outer
            ],
            outer_target_center=outer,
            training_seeds=config.training_seeds,
            parent_protocol_hash=parent_protocol_hash,
            parent_selection_bundle_hash=parent_bundle_hash,
            consensus_rule_id=config.consensus_rule_id,
        )
        for outer in config.heldout_centers
    ]
    return write_stability_bundle(
        root,
        metric_rows=metric_rows,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=tuning_rows,
        sampler_rows=sampler_rows,
        identity_audit_rows=identity_rows,
        rng_audit_rows=rng_audit_rows,
        seed_locks=wrapped_locks,
        consensus_locks=consensus_locks,
        protocol_manifest=protocol_manifest,
        child_protocols=child_protocols,
        selection_bundle_hash=parent_bundle_hash,
        seed_evidence_hashes=seed_evidence_hashes,
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload
