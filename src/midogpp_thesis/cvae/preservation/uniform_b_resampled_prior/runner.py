"""Orchestration for fresh BG training and outer-invariant P0/Pq scoring."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ....real_features.classifier_reference.real_feature_frame import (
    load_midogpp_real_feature_frame,
)
from ...keyed_training import derived_seed, model_state_hash
from ...protocol import ProtocolError
from ..independent_source import assert_source_evaluation_isolation, extract_source_data
from ..splits import frame_arrays, indices_for_centers, row_hash
from ..uniform_b_task_geometry.composition import compose_generated_blocks
from ..uniform_b_task_geometry.frame import fit_source_block_frame
from ..uniform_b_task_geometry.scoring_runtime import DeterministicScoringPool
from .artifacts import prepare_bundle, write_final_artifacts, write_resolved_config
from .checkpoint_store import ResampledPriorCheckpointStore
from .config import UniformBResampledPriorConfig
from .contracts import (
    COMPOSITION_MODE,
    P0,
    PQ,
    PRIORS,
    SourceTrainingKey,
    UniqueScoreKey,
    valid_outer_centers,
)
from .decisions import paired_deltas, study_decision
from .execution import resolve_runtime_plan
from .generation import generate_paired_prior_blocks
from .protocol import protocol_manifest
from .training_runtime import train_panel_grid
from .validation import validate_uniform_b_resampled_prior_bundle


def run_uniform_b_resampled_prior_source_inner_study(
    config: UniformBResampledPriorConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    root = prepare_bundle(Path(artifact_root or config.artifact_root))
    state_path = root / "reports/run_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE":
            validate_uniform_b_resampled_prior_bundle(root)
            return root
    write_resolved_config(root, config)
    _write_run_state(root, "RUNNING")
    try:
        _run(config, root)
        validate_uniform_b_resampled_prior_bundle(root)
        _write_run_state(root, "COMPLETE")
    except Exception:
        _write_run_state(root, "FAILED")
        raise
    return root


def _run(config: UniformBResampledPriorConfig, root: Path) -> None:
    started = perf_counter()
    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(config.manifest_path),
        feature_cache_path=Path(config.feature_cache_path),
        expected_feature_dim=3840,
        allow_excluded_center_omission=True,
    )
    if frame.eligible_centers != config.heldout_centers:
        raise ProtocolError("P0/Pq study requires exact eligible-center coverage.")
    centers = tuple(config.heldout_centers)
    runtime_plan = resolve_runtime_plan(config)
    protocol = protocol_manifest(
        config,
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
    )
    sources = {center: extract_source_data(frame, center) for center in centers}
    source_frames = {center: fit_source_block_frame(sources[center]) for center in centers}
    projected = {
        center: np.asarray(source_frames[center].frame.transform(sources[center].embeddings), dtype=np.float32)
        for center in centers
    }
    training_keys = {
        (center, seed): SourceTrainingKey(
            source_center=center,
            training_seed=seed,
            source_row_hash=sources[center].row_hash,
            source_case_hash=sources[center].case_hash,
            frame_hash=source_frames[center].state_hash,
            manifest_hash=frame.manifest_hash,
            feature_cache_hash=frame.feature_cache_hash,
            config_hash=config.contract_hash,
        )
        for center in centers
        for seed in config.training_seeds
    }
    training_results = train_panel_grid(
        root=root,
        config=config,
        sources=sources,
        projected=projected,
        training_keys=training_keys,
        runtime_plan=runtime_plan,
    )
    checkpoint_store = ResampledPriorCheckpointStore(root, config)
    ratio_states = {}
    timing_rows: list[dict[str, object]] = []
    for result in training_results:
        checkpoint_store.register_record(result.checkpoint_record)
        ratio_states[result.key] = result.ratio_state
        timing_rows.append(
            {
                "phase": "fresh_bg_training_and_ratio_fit",
                "source_center": result.source_center,
                "training_seed": result.training_seed,
                "runtime_device": result.runtime_device,
                "resumed_checkpoint": result.resumed_checkpoint,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
    checkpoint_store.write_index()
    ratio_index_records = [
        {
            "source_center": key[0],
            "training_seed": key[1],
            "state_hash": ratio_states[key].state_hash,
            "state": ratio_states[key].to_payload(),
        }
        for key in sorted(ratio_states)
    ]
    ratio_index = {
        "schema_version": "midogpp_posterior_ratio_state_index_v1",
        "n_records": len(ratio_index_records),
        "records": ratio_index_records,
    }
    ratio_rows = [
        {
            "schema_version": "midogpp_posterior_ratio_diagnostic_v1",
            "source_center": state.source_center,
            "training_seed": state.training_seed,
            "class_label": class_label,
            "ratio_state_hash": state.state_hash,
            "class_state_hash": class_state.state_hash,
            "crossfit_auc": class_state.crossfit_auc,
            "crossfit_log_loss": class_state.crossfit_log_loss,
            "baseline_log_loss": class_state.baseline_log_loss,
            "log_loss_gain": class_state.log_loss_gain,
            "converged": class_state.converged,
            "reliable": class_state.reliable,
            "n_source_rows": class_state.n_source_rows,
            "n_source_cases": class_state.n_source_cases,
            "outer_or_inner_rows_used": False,
        }
        for state in ratio_states.values()
        for class_label, class_state in sorted(state.classes.items())
    ]

    eval_data = {}
    identity_rows = []
    for inner in centers:
        indices = indices_for_centers(frame, (inner,))
        x_eval, y_eval, sample_ids = frame_arrays(frame, indices)
        cases = tuple(str(frame.rows[index].case_id) for index in indices)
        images = tuple(str(frame.rows[index].image_path) for index in indices)
        eval_data[inner] = (
            x_eval, y_eval, sample_ids, cases, images, row_hash(sample_ids)
        )
        for source in centers:
            if source == inner:
                continue
            outers = valid_outer_centers(centers, source_center=source, inner_center=inner)
            assert_source_evaluation_isolation(
                sources[source],
                outer_center=outers[0],
                inner_center=inner,
                eval_sample_ids=sample_ids,
                eval_case_ids=cases,
                eval_image_ids=images,
            )
            identity_rows.append(
                {
                    "schema_version": "midogpp_resampled_prior_identity_audit_v1",
                    "source_center": source,
                    "inner_center": inner,
                    "mapped_outer_centers": list(outers),
                    "source_row_hash": sources[source].row_hash,
                    "inner_row_hash": row_hash(sample_ids),
                    "sample_overlap_count": 0,
                    "case_overlap_count": 0,
                    "image_overlap_count": 0,
                    "status": "PASS",
                }
            )

    classifier_spec = ClassifierSpec(
        C=config.classifier_c,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
        random_state=config.classifier_seed,
        threshold_policy="predict",
        scaler_fit="synthetic_train_only",
    )
    unique_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    score_mapping: list[dict[str, object]] = []
    generation_rows: list[dict[str, object]] = []
    generation_manifest: list[dict[str, object]] = []
    with DeterministicScoringPool(runtime_plan.scoring_workers) as scoring_pool:
        for training_seed in config.training_seeds:
            states = {
                center: checkpoint_store.load(
                    training_keys[(center, training_seed)].hash,
                    device=config.device,
                )
                for center in centers
            }
            if any(state is None for state in states.values()):
                raise ProtocolError("Fresh BG checkpoint disappeared before generation.")
            for generation_seed in config.generation_seeds:
                generation_started = perf_counter()
                blocks = {}
                for source in centers:
                    state = states[source]
                    assert state is not None
                    checkpoint_hash = model_state_hash(state.model)
                    paired, audits = generate_paired_prior_blocks(
                        state.model,
                        source_frames[source],
                        ratio_states[(source, training_seed)],
                        source_center=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        per_class=config.base_generation_per_class,
                        checkpoint_hash=checkpoint_hash,
                        config=config,
                        device=state.device,
                    )
                    for prior, block in paired.items():
                        blocks[(source, prior)] = block
                        generation_manifest.append(
                            {
                                "schema_version": "midogpp_resampled_prior_generation_block_v1",
                                "source_center": source,
                                "training_seed": training_seed,
                                "generation_seed": generation_seed,
                                "prior": prior,
                                "block_hash": block.block_hash,
                                "checkpoint_hash": checkpoint_hash,
                                "ratio_state_hash": ratio_states[(source, training_seed)].state_hash,
                                "per_class": block.per_class,
                                "outer_or_inner_identity_present": False,
                                "class_balanced": True,
                            }
                        )
                    generation_rows.extend(audit.to_payload() for audit in audits)
                timing_rows.append(
                    {
                        "phase": "paired_p0_pq_generation",
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "runtime_device": config.device,
                        "elapsed_seconds": perf_counter() - generation_started,
                    }
                )
                for inner in centers:
                    score_started = perf_counter()
                    x_eval, y_eval, _, _, _, inner_row_hash = eval_data[inner]
                    sealed = []
                    for source in centers:
                        if source == inner:
                            continue
                        for prior in PRIORS:
                            synthetic = compose_generated_blocks(
                                {source: blocks[(source, prior)]},
                                mode=COMPOSITION_MODE,
                                base_per_class=config.base_generation_per_class,
                                shuffle_seed=derived_seed(
                                    source, inner, training_seed, generation_seed, prior, "single_base_shuffle"
                                ),
                                selected_source=source,
                            )
                            sealed.append((COMPOSITION_MODE, f"{source}|{prior}", synthetic))
                    scored = scoring_pool.score(
                        sealed,
                        x_eval,
                        y_eval,
                        classifier_spec=classifier_spec,
                    )
                    for item in scored:
                        source, prior = item.selected_source.split("|", 1)
                        key = UniqueScoreKey(
                            source_center=source,
                            inner_center=inner,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            prior=prior,
                            block_hash=item.synthetic.composition_hash,
                            inner_row_hash=inner_row_hash,
                            classifier_spec_hash=classifier_spec.config_hash,
                        )
                        outers = valid_outer_centers(
                            centers,
                            source_center=source,
                            inner_center=inner,
                        )
                        unique = {
                            "schema_version": "midogpp_resampled_prior_unique_tstr_v1",
                            "score_key_hash": key.hash,
                            "source_center": source,
                            "inner_center": inner,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "prior": prior,
                            "composition_mode": COMPOSITION_MODE,
                            "composition_hash": item.synthetic.composition_hash,
                            "bacc": item.diagnostic.bacc,
                            "macro_f1": item.diagnostic.macro_f1,
                            "classifier_converged": item.diagnostic.converged,
                            "classifier_spec_hash": item.diagnostic.classifier_spec_hash,
                            "outer_center_used_for_scoring": False,
                            "inner_labels_used_for_scoring_only": True,
                            "outer_rows_used": False,
                            "routing_or_compatibility": False,
                            **item.diagnostic.diversity,
                        }
                        unique_rows.append(unique)
                        score_mapping.append(
                            {
                                "score_key_hash": key.hash,
                                "source_center": source,
                                "inner_center": inner,
                                "training_seed": training_seed,
                                "generation_seed": generation_seed,
                                "prior": prior,
                                "mapped_outer_centers": list(outers),
                                "mapping_count": len(outers),
                            }
                        )
                        for outer in outers:
                            metric_rows.append(
                                {
                                    **unique,
                                    "schema_version": "midogpp_resampled_prior_source_inner_metric_v1",
                                    "outer_center": outer,
                                    "mapped_from_unique_score": True,
                                }
                            )
                    timing_rows.append(
                        {
                            "phase": "unique_source_inner_scoring",
                            "inner_center": inner,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "classifier_fits": len(sealed),
                            "mapped_outer_rows": len(sealed) * (len(centers) - 2),
                            "elapsed_seconds": perf_counter() - score_started,
                        }
                    )
            del states

    delta_rows = paired_deltas(metric_rows)
    decision = study_decision(
        unique_rows,
        delta_rows,
        generation_rows,
        config=config,
    )
    coverage = {
        "schema_version": "midogpp_resampled_prior_coverage_v1",
        "centers": list(centers),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "priors": list(PRIORS),
        "checkpoint_records": len(checkpoint_store.records),
        "ratio_state_records": len(ratio_states),
        "generation_blocks": len(generation_manifest),
        "unique_score_rows": len(unique_rows),
        "mapped_metric_rows": len(metric_rows),
        "score_reuse_factor": len(centers) - 2,
    }
    frame_index = {
        "schema_version": "midogpp_resampled_prior_frame_index_v1",
        "records": [
            {
                "source_center": center,
                "state_hash": source_frames[center].state_hash,
                "state": source_frames[center].to_payload(),
            }
            for center in centers
        ],
    }
    provenance = {
        "schema_version": "midogpp_resampled_prior_provenance_v1",
        "input_artifact_ids": [
            "midogpp_dataset_contract_annotation_patch_v1",
            config.feature_cache_artifact_id,
        ],
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "completed_uniform_b_task_geometry_artifact_used": False,
        "existing_checkpoint_input_used": False,
        "fresh_bg_training": True,
    }
    timing_rows.append(
        {
            "phase": "total",
            "elapsed_seconds": perf_counter() - started,
        }
    )
    write_final_artifacts(
        root,
        protocol=protocol,
        provenance=provenance,
        coverage=coverage,
        frame_index=frame_index,
        ratio_index=ratio_index,
        score_mapping=score_mapping,
        generation_manifest=generation_manifest,
        unique_rows=unique_rows,
        metric_rows=metric_rows,
        delta_rows=delta_rows,
        ratio_rows=ratio_rows,
        generation_rows=generation_rows,
        identity_rows=identity_rows,
        timing_rows=timing_rows,
        decision=decision,
        runtime_plan=runtime_plan.to_payload(),
    )


def _write_run_state(root: Path, status: str) -> None:
    from ...reporting import write_json

    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_resampled_prior_run_state_v1",
            "status": status,
            "claim_scope": "cvae_source_inner_study_only",
        },
    )


__all__ = ("run_uniform_b_resampled_prior_source_inner_study",)
