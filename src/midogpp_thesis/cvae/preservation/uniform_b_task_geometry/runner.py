"""Thin orchestration for the Uniform-B GECO/task-geometry source-inner study."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ....real_features.classifier_reference.real_feature_frame import (
    load_midogpp_real_feature_frame,
)
from ...keyed_training import derived_seed, model_state_hash
from ...protocol import ProtocolError
from ...reporting import write_json
from ..independent_source import (
    assert_source_evaluation_isolation,
    extract_source_data,
)
from ..splits import frame_arrays, indices_for_centers, row_hash
from .artifacts import prepare_bundle, write_final_artifacts, write_resolved_config
from .checkpoint_store import TaskGeometryCheckpointStore
from .composition import compose_generated_blocks
from .config import UniformBTaskGeometryConfig
from .contracts import (
    ARMS,
    COMPOSITION_MODES,
    SINGLE_BASE,
    SINGLE_BUDGET_MATCHED,
)
from .decisions import paired_deltas, prior_posterior_gaps, study_decision
from .execution import resolve_runtime_plan
from .frame import fit_source_block_frame
from .generation import (
    GeneratedBlock,
    generate_posterior_block,
    generate_prior_block,
)
from .protocol import candidate_pool_manifest, protocol_manifest
from .scoring_runtime import DeterministicScoringPool
from .training_runtime import train_panel_grid
from .validation import validate_uniform_b_task_geometry_bundle


def run_uniform_b_task_geometry_source_inner_study(
    config: UniformBTaskGeometryConfig,
    *,
    artifact_root: Path | None = None,
) -> Path:
    root = prepare_bundle(Path(artifact_root or config.artifact_root))
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_uniform_b_run_state_v1",
            "status": "RUNNING",
            "claim_scope": config.claim_scope,
        },
    )
    try:
        _run(config, root)
        write_json(
            root / "reports/run_state.json",
            {
                "schema_version": "midogpp_uniform_b_run_state_v1",
                "status": "COMPLETE",
                "claim_scope": config.claim_scope,
            },
        )
        validate_uniform_b_task_geometry_bundle(root)
    except Exception as exc:
        write_json(
            root / "reports/run_state.json",
            {
                "schema_version": "midogpp_uniform_b_run_state_v1",
                "status": "FAILED",
                "claim_scope": config.claim_scope,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    return root


def _run(config: UniformBTaskGeometryConfig, root: Path) -> None:
    started = perf_counter()
    write_resolved_config(root, config)
    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(config.manifest_path),
        feature_cache_path=Path(config.feature_cache_path),
        expected_feature_dim=3840,
        allow_excluded_center_omission=True,
    )
    if frame.eligible_centers != config.heldout_centers:
        raise ProtocolError("Uniform-B study requires exact eligible-center coverage.")
    runtime_plan = resolve_runtime_plan(config)
    protocol = protocol_manifest(
        config,
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
    )
    centers = tuple(config.heldout_centers)
    legal_count = len(centers) - 2
    maximum_per_class = legal_count * config.base_generation_per_class
    sources = {center: extract_source_data(frame, center) for center in centers}
    source_frames = {
        center: fit_source_block_frame(source)
        for center, source in sources.items()
    }
    projected = {
        center: np.asarray(
            source_frames[center].frame.transform(source.embeddings),
            dtype=np.float32,
        )
        for center, source in sources.items()
    }
    checkpoint_store = TaskGeometryCheckpointStore(root, config)
    geometries: dict[tuple[str, int], object] = {}
    training_keys: dict[tuple[str, int, str], str] = {}
    geometry_rows: list[dict[str, object]] = []
    rng_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    training_results = train_panel_grid(
        root=root,
        config=config,
        sources=sources,
        projected=projected,
        frame_hashes={
            center: source_frames[center].state_hash for center in centers
        },
        runtime_plan=runtime_plan,
    )
    for result in training_results:
        geometries[result.key] = result.geometry
        geometry_rows.extend(result.geometry_rows)
        rng_rows.extend(result.rng_rows)
        timing_rows.append(
            {
                "phase": "source_panel_training",
                "source_center": result.source_center,
                "training_seed": result.training_seed,
                "runtime_device": result.runtime_device,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        for record in result.checkpoint_records:
            checkpoint_store.register_record(record)
            arm = str(record["arm"])
            training_keys[(result.source_center, result.training_seed, arm)] = (
                str(record["training_key_hash"])
            )
    checkpoint_store.write_index()

    generation_rows: list[dict[str, object]] = []
    generation_manifest: list[dict[str, object]] = []
    candidate_pools = [
        candidate_pool_manifest(
            centers,
            outer_center=outer,
            inner_center=inner,
            base_per_class=config.base_generation_per_class,
        )
        for outer in centers
        for inner in centers
        if inner != outer
    ]
    pools_by_cell = {
        (str(row["outer_center"]), str(row["inner_center"])): row
        for row in candidate_pools
    }
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
    metric_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    diversity_pass = True
    eval_data = {}
    for outer in centers:
        for inner in centers:
            if inner == outer:
                continue
            pool = pools_by_cell[(outer, inner)]
            legal_sources = tuple(str(value) for value in pool["legal_sources"])
            eval_indices = indices_for_centers(frame, (inner,))
            x_eval, y_eval, eval_sample_ids = frame_arrays(frame, eval_indices)
            eval_case_ids = tuple(
                frame.rows[index].case_id for index in eval_indices
            )
            eval_image_ids = tuple(
                frame.rows[index].image_path for index in eval_indices
            )
            eval_data[(outer, inner)] = (x_eval, y_eval)
            for source in legal_sources:
                assert_source_evaluation_isolation(
                    sources[source],
                    outer_center=outer,
                    inner_center=inner,
                    eval_sample_ids=eval_sample_ids,
                    eval_case_ids=eval_case_ids,
                    eval_image_ids=eval_image_ids,
                )
                identity_rows.append(
                    {
                        "schema_version": "midogpp_uniform_b_identity_audit_v1",
                        "outer_center": outer,
                        "inner_center": inner,
                        "source_center": source,
                        "source_row_hash": sources[source].row_hash,
                        "inner_row_hash": row_hash(eval_sample_ids),
                        "sample_overlap_count": 0,
                        "case_overlap_count": 0,
                        "image_overlap_count": 0,
                        "status": "PASS",
                    }
                )

    # Generate one source panel at a time and release it after all H/I cells.
    # Peak generated-array memory is therefore O(number of sources), not
    # O(sources x arms x training seeds x generation seeds).
    scoring_pool = DeterministicScoringPool(runtime_plan.scoring_workers)
    for training_seed in config.training_seeds:
        for arm in ARMS:
            arm_states = {}
            for center in centers:
                state = checkpoint_store.load(
                    training_keys[(center, training_seed, arm)],
                    device=config.device,
                )
                if state is None:
                    raise ProtocolError(
                        "Uniform-B panel checkpoint disappeared before generation."
                    )
                arm_states[center] = state
            for generation_seed in config.generation_seeds:
                for generation_kind in ("prior", "posterior"):
                    blocks_all: dict[str, GeneratedBlock] = {}
                    for center in centers:
                        state = arm_states[center]
                        checkpoint_hash = model_state_hash(state.model)
                        if generation_kind == "prior":
                            block = generate_prior_block(
                                state.model,
                                source_frames[center],
                                source_center=center,
                                arm=arm,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                per_class=maximum_per_class,
                                checkpoint_hash=checkpoint_hash,
                                device=state.device,
                            )
                        else:
                            block = generate_posterior_block(
                                state.model,
                                source_frames[center],
                                projected[center],
                                np.asarray(
                                    sources[center].labels,
                                    dtype=np.int64,
                                ),
                                source_center=center,
                                arm=arm,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                per_class=maximum_per_class,
                                checkpoint_hash=checkpoint_hash,
                                device=state.device,
                            )
                        blocks_all[center] = block
                        record = {
                            "schema_version": "midogpp_uniform_b_generation_budget_v1",
                            "source_center": center,
                            "training_seed": training_seed,
                            "arm": arm,
                            "generation_seed": generation_seed,
                            "generation_kind": block.kind,
                            "per_class": maximum_per_class,
                            "block_hash": block.block_hash,
                            "stream_hash": block.stream_hash,
                            "outer_or_inner_identity_present": False,
                            "class_balanced": True,
                            "target_or_source_prevalence_used": False,
                        }
                        generation_rows.append(record)
                        generation_manifest.append(record)
                    for outer in centers:
                        for inner in centers:
                            if inner == outer:
                                continue
                            pool = pools_by_cell[(outer, inner)]
                            legal_sources = tuple(
                                str(value) for value in pool["legal_sources"]
                            )
                            blocks = {
                                source: blocks_all[source]
                                for source in legal_sources
                            }
                            sealed: list[tuple[str, str, object]] = []
                            for mode in COMPOSITION_MODES:
                                selected_sources = (
                                    legal_sources
                                    if mode
                                    in {SINGLE_BASE, SINGLE_BUDGET_MATCHED}
                                    else (None,)
                                )
                                for selected in selected_sources:
                                    synthetic = compose_generated_blocks(
                                        blocks,
                                        mode=mode,
                                        base_per_class=config.base_generation_per_class,
                                        shuffle_seed=derived_seed(
                                            pool["candidate_pool_hash"],
                                            training_seed,
                                            arm,
                                            generation_seed,
                                            generation_kind,
                                            mode,
                                            selected,
                                        ),
                                        selected_source=selected,
                                    )
                                    sealed.append(
                                        (mode, selected or "", synthetic)
                                    )
                                    composition_rows.append(
                                        {
                                            "schema_version": "midogpp_uniform_b_sealed_composition_v1",
                                            "outer_center": outer,
                                            "inner_center": inner,
                                            "training_seed": training_seed,
                                            "arm": arm,
                                            "generation_seed": generation_seed,
                                            "generation_kind": generation_kind,
                                            "composition_mode": mode,
                                            "selected_source": selected or "",
                                            "candidate_pool_hash": pool[
                                                "candidate_pool_hash"
                                            ],
                                            "composition_hash": synthetic.composition_hash,
                                            "source_counts": {
                                                key: dict(value)
                                                for key, value in synthetic.source_counts.items()
                                            },
                                            "sealed_before_inner_rows_loaded": True,
                                            "routing_or_selection": False,
                                        }
                                    )
                            x_eval, y_eval = eval_data[(outer, inner)]
                            scored = scoring_pool.score(
                                sealed,
                                x_eval,
                                y_eval,
                                classifier_spec=classifier_spec,
                            )
                            for item in scored:
                                mode = item.mode
                                selected = item.selected_source
                                synthetic = item.synthetic
                                result = item.diagnostic
                                for key, value in result.diversity.items():
                                    if "effective_rank_ratio" in key:
                                        diversity_pass &= (
                                            value
                                            >= config.min_effective_rank_ratio
                                        )
                                    elif "pairwise_" in key:
                                        diversity_pass &= (
                                            config.min_pairwise_distance_ratio
                                            <= value
                                            <= config.max_pairwise_distance_ratio
                                        )
                                metric_rows.append(
                                    {
                                        "schema_version": "midogpp_uniform_b_source_inner_metric_v1",
                                        "outer_center": outer,
                                        "inner_center": inner,
                                        "source_center": selected,
                                        "training_seed": training_seed,
                                        "arm": arm,
                                        "generation_seed": generation_seed,
                                        "composition_mode": mode,
                                        "generation_kind": synthetic.generation_kind,
                                        "candidate_pool_hash": pool[
                                            "candidate_pool_hash"
                                        ],
                                        "composition_hash": synthetic.composition_hash,
                                        "bacc": result.bacc,
                                        "macro_f1": result.macro_f1,
                                        "classifier_converged": result.converged,
                                        "classifier_spec_hash": result.classifier_spec_hash,
                                        "train_role": (
                                            "synthetic_requested_class_prior"
                                            if synthetic.generation_kind
                                            == "prior"
                                            else "source_posterior_decode_diagnostic"
                                        ),
                                        "eval_role": "held_out_inner_score_only",
                                        "selection_source": "predeclared_fixed_classifier",
                                        "claim_role": "held_out_inner_discriminative_prior_tstr_diagnostic",
                                        "method": "fixed_source_data_composition_diagnostic",
                                        "inner_labels_used_for_scoring_only": True,
                                        "outer_rows_used": False,
                                        "routing_or_compatibility": False,
                                        **result.diversity,
                                    }
                                )
            del arm_states
    scoring_pool.close()
    delta_rows = paired_deltas(metric_rows)
    delta_rows.extend(prior_posterior_gaps(metric_rows))
    decision = study_decision(
        metric_rows,
        delta_rows,
        diversity_pass=diversity_pass,
    )
    coverage = {
        "schema_version": "midogpp_uniform_b_coverage_v1",
        "centers": list(centers),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "arms": list(ARMS),
        "composition_modes": list(COMPOSITION_MODES),
        "candidate_pools": len(candidate_pools),
        "checkpoint_records": len(checkpoint_store.records),
        "metric_rows": len(metric_rows),
    }
    frame_index = {
        "schema_version": "midogpp_uniform_b_frame_index_v1",
        "records": [
            {
                "source_center": center,
                "state_hash": source_frames[center].state_hash,
                "state": source_frames[center].to_payload(),
            }
            for center in centers
        ],
    }
    geometry_index = {
        "schema_version": "midogpp_uniform_b_task_geometry_index_v1",
        "records": [
            {
                "source_center": key[0],
                "training_seed": key[1],
                "state_hash": geometries[key].state_hash,  # type: ignore[union-attr]
                "state": geometries[key].to_payload(),  # type: ignore[union-attr]
            }
            for key in sorted(geometries)
        ],
    }
    provenance = {
        "schema_version": "midogpp_uniform_b_input_provenance_v1",
        "dataset_contract": "midogpp_dataset_contract_annotation_patch_v1",
        "feature_cache_artifact_id": config.feature_cache_artifact_id,
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "expected_feature_dim": 3840,
        "stage90_input_used": False,
        "aggregate_prior_artifact_used": False,
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
        geometry_index=geometry_index,
        candidate_pools=candidate_pools,
        generation_manifest=generation_manifest,
        composition_manifest=composition_rows,
        metric_rows=metric_rows,
        delta_rows=delta_rows,
        geometry_rows=geometry_rows,
        generation_rows=generation_rows,
        identity_rows=identity_rows,
        rng_rows=rng_rows,
        timing_rows=timing_rows,
        decision=decision,
        runtime_plan=runtime_plan.to_payload(),
    )


__all__ = ("run_uniform_b_task_geometry_source_inner_study",)
