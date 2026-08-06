"""End-to-end v2 Uniform-B prior-capacity and legal-union study."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ....real_features.classifier_reference.real_feature_frame import load_midogpp_real_feature_frame
from ...keyed_training import derived_seed, model_state_hash
from ...protocol import ProtocolError
from ...reporting import prepare_artifact_dirs, write_csv_rows, write_json
from ..independent_source import assert_source_evaluation_isolation, extract_source_data
from ..splits import frame_arrays, indices_for_centers, row_hash
from ..uniform_b_task_geometry.composition import compose_generated_blocks
from ..uniform_b_task_geometry.scoring_runtime import DeterministicScoringPool
from .config import OptimizedPriorConfig
from .contracts import (
    ARMS, CLAIM_SCOPE, COMPOSITION_MODE, EXPERIMENT_ID, P0, PS, Q, QM, R,
    PUBLICATION_STATE, OptimizedTrainingKey, legal_sources,
)
from .core import (
    fit_optimized_source_frame, generate_paired_blocks, load_checkpoint,
    resolve_runtime_plan, train_panel,
)


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/coverage_manifest.json",
    "manifests/frame_index.json",
    "manifests/checkpoint_index.json",
    "manifests/generation_manifest.json",
    "manifests/content_index.json",
    "tables/task_metrics.csv",
    "tables/paired_deltas.csv",
    "tables/sampler_audit.csv",
    "tables/identity_overlap_audit.csv",
    "tables/runtime_timings.csv",
    "reports/study_decision.json",
    "reports/leakage_report.json",
    "reports/runtime_summary.json",
    "reports/publication_state.json",
    "reports/run_state.json",
)


def run_optimized_prior_source_inner_study(
    config: OptimizedPriorConfig, *, artifact_root: Path | None = None
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root or config.artifact_root))
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    state_path = root / "reports/run_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE":
            validate_optimized_prior_bundle(root)
            return root
    _write_config(root, config)
    _write_state(root, "RUNNING")
    try:
        _run(config, root)
        _write_state(root, "COMPLETE")
        validate_optimized_prior_bundle(root)
    except Exception:
        _write_state(root, "FAILED")
        raise
    return root


def _run(config: OptimizedPriorConfig, root: Path) -> None:
    started = perf_counter()
    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(config.manifest_path),
        feature_cache_path=Path(config.feature_cache_path),
        expected_feature_dim=3840,
        allow_excluded_center_omission=True,
    )
    if frame.eligible_centers != config.heldout_centers:
        raise ProtocolError("Optimized-prior study requires exact center coverage.")
    if frame.feature_cache_hash != config.expected_feature_cache_hash:
        raise ProtocolError("Canonical Uniform-B feature-cache hash mismatch.")
    centers = tuple(config.heldout_centers)
    runtime = resolve_runtime_plan(config)
    sources = {center: extract_source_data(frame, center) for center in centers}
    frame_workers = min(4, runtime.scoring_workers, len(centers))
    with ThreadPoolExecutor(
        max_workers=frame_workers, thread_name_prefix="uniform-b-frame"
    ) as frame_pool:
        fitted_frames = tuple(
            frame_pool.map(
                fit_optimized_source_frame,
                (sources[center] for center in centers),
            )
        )
    source_frames = dict(zip(centers, fitted_frames))
    projected = {
        center: np.asarray(source_frames[center].frame.transform(sources[center].embeddings), dtype=np.float32)
        for center in centers
    }
    keys = {
        (center, seed): OptimizedTrainingKey(
            source_center=center, training_seed=seed,
            source_row_hash=sources[center].row_hash,
            source_case_hash=sources[center].case_hash,
            frame_hash=source_frames[center].state_hash,
            manifest_hash=frame.manifest_hash,
            feature_cache_hash=frame.feature_cache_hash,
            config_hash=config.contract_hash,
        )
        for center in centers for seed in config.training_seeds
    }
    training = train_panel(
        root=root, config=config, sources=sources, projected=projected,
        keys=keys, runtime=runtime,
    )
    checkpoint_records = [dict(item.checkpoint_record) for item in training]
    write_json(
        root / "manifests/checkpoint_index.json",
        {
            "schema_version": "midogpp_uniform_b_optimized_checkpoint_index_v2",
            "n_records": len(checkpoint_records),
            "records": checkpoint_records,
        },
    )
    timing_rows: list[dict[str, object]] = [
        {
            "phase": "fresh_source_training", "source_center": item.source_center,
            "training_seed": item.training_seed, "runtime_device": item.device,
            "resumed": item.resumed, "elapsed_seconds": item.elapsed_seconds,
        }
        for item in training
    ]
    eval_data: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = {}
    for inner in centers:
        indices = indices_for_centers(frame, (inner,))
        x_eval, y_eval, sample_ids = frame_arrays(frame, indices)
        cases = tuple(str(frame.rows[index].case_id) for index in indices)
        images = tuple(str(frame.rows[index].image_path) for index in indices)
        eval_data[inner] = (x_eval, y_eval, sample_ids, cases, images, row_hash(sample_ids))
    classifier = ClassifierSpec(
        C=config.classifier_c, penalty="l2", solver="lbfgs", max_iter=3000,
        class_weight=None, random_state=config.classifier_seed,
        threshold_policy="predict", scaler_fit="synthetic_train_only",
    )
    metric_rows: list[dict[str, object]] = []
    sampler_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    generation_rows: list[dict[str, object]] = []
    with DeterministicScoringPool(runtime.scoring_workers) as scoring_pool:
        for training_seed in config.training_seeds:
            blocks: dict[tuple[int, str, str], object] = {}
            for source_index, source in enumerate(centers):
                device = runtime.training_devices[source_index % len(runtime.training_devices)]
                loaded = load_checkpoint(root, keys[(source, training_seed)].hash, config, device=device)
                if loaded is None:
                    raise ProtocolError("Optimized checkpoint disappeared before generation.")
                state, record = loaded
                checkpoint_hash = model_state_hash(state.model)
                for generation_seed in config.generation_seeds:
                    generation_started = perf_counter()
                    paired, audit = generate_paired_blocks(
                        state.model, source_frames[source], projected[source],
                        np.asarray(sources[source].labels, dtype=np.int64),
                        source_center=source, training_seed=training_seed,
                        generation_seed=generation_seed,
                        checkpoint_hash=checkpoint_hash, config=config,
                        device=state.device,
                    )
                    sampler_rows.append(audit)
                    for arm, block in paired.items():
                        blocks[(generation_seed, source, arm)] = block
                        generation_rows.append(
                            {
                                "schema_version": "midogpp_uniform_b_optimized_generation_block_v2",
                                "source_center": source,
                                "training_seed": training_seed,
                                "generation_seed": generation_seed,
                                "arm": arm,
                                "block_hash": block.block_hash,
                                "checkpoint_hash": record["checkpoint_hash"],
                                "per_class": block.per_class,
                                "source_only_generation": True,
                            }
                        )
                    timing_rows.append(
                        {
                            "phase": "paired_generation", "source_center": source,
                            "training_seed": training_seed, "generation_seed": generation_seed,
                            "runtime_device": state.device,
                            "elapsed_seconds": perf_counter() - generation_started,
                        }
                    )
                del state
            for generation_seed in config.generation_seeds:
                for inner in centers:
                    score_started = perf_counter()
                    x_eval, y_eval, sample_ids, cases, images, inner_hash = eval_data[inner]
                    sealed = []
                    for outer in centers:
                        if outer == inner:
                            continue
                        task_sources = legal_sources(
                            centers, outer_center=outer, inner_center=inner
                        )
                        for source in task_sources:
                            assert_source_evaluation_isolation(
                                sources[source], outer_center=outer, inner_center=inner,
                                eval_sample_ids=sample_ids, eval_case_ids=cases,
                                eval_image_ids=images,
                            )
                        identity_rows.append(
                            {
                                "schema_version": "midogpp_uniform_b_optimized_identity_audit_v2",
                                "outer_center": outer, "inner_center": inner,
                                "legal_sources": "|".join(task_sources),
                                "source_count": len(task_sources),
                                "sample_overlap_count": 0, "case_overlap_count": 0,
                                "image_overlap_count": 0, "status": "PASS",
                            }
                        )
                        paired_shuffle_seed = derived_seed(
                            outer, inner, training_seed, generation_seed,
                            "paired_union_equal_total_shuffle",
                        )
                        compositions = {}
                        for arm in ARMS:
                            composition = compose_generated_blocks(
                                {
                                    source: blocks[(generation_seed, source, arm)]
                                    for source in task_sources
                                },
                                mode=COMPOSITION_MODE,
                                base_per_class=config.total_generation_per_class,
                                shuffle_seed=paired_shuffle_seed,
                            )
                            compositions[arm] = composition
                            sealed.append(
                                (COMPOSITION_MODE, f"{outer}|{arm}", composition)
                            )
                        if len({item.shuffle_hash for item in compositions.values()}) != 1:
                            raise ProtocolError(
                                "Paired prior arms do not share one shuffle."
                            )
                    scored = scoring_pool.score(
                        sealed, x_eval, y_eval, classifier_spec=classifier
                    )
                    for item in scored:
                        outer, arm = item.selected_source.split("|", 1)
                        task_sources = legal_sources(
                            centers, outer_center=outer, inner_center=inner
                        )
                        metric_rows.append(
                            {
                                "schema_version": "midogpp_uniform_b_optimized_task_metric_v2",
                                "outer_center": outer, "inner_center": inner,
                                "training_seed": training_seed,
                                "generation_seed": generation_seed,
                                "arm": arm,
                                "composition_mode": COMPOSITION_MODE,
                                "legal_source_count": len(task_sources),
                                "legal_sources": "|".join(task_sources),
                                "total_per_class": config.total_generation_per_class,
                                "composition_hash": item.synthetic.composition_hash,
                                "shuffle_hash": item.synthetic.shuffle_hash,
                                "inner_row_hash": inner_hash,
                                "bacc": item.diagnostic.bacc,
                                "macro_f1": item.diagnostic.macro_f1,
                                "classifier_converged": item.diagnostic.converged,
                                "classifier_spec_hash": item.diagnostic.classifier_spec_hash,
                                "inner_labels_used_for_scoring_only": True,
                                "outer_or_inner_rows_used_for_fit": False,
                                **item.diagnostic.diversity,
                            }
                        )
                    timing_rows.append(
                        {
                            "phase": "batched_inner_scoring",
                            "inner_center": inner,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "classifier_fits": len(sealed),
                            "runtime_workers": runtime.scoring_workers,
                            "elapsed_seconds": perf_counter() - score_started,
                        }
                    )
            del blocks
    delta_rows = _paired_deltas(metric_rows)
    decision = _decision(metric_rows, config)
    protocol = _protocol(config, frame.manifest_hash, frame.feature_cache_hash)
    coverage = {
        "schema_version": "midogpp_uniform_b_optimized_coverage_v2",
        "centers": list(centers), "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds), "arms": list(ARMS),
        "checkpoint_records": len(checkpoint_records),
        "generation_blocks": len(generation_rows),
        "task_metric_rows": len(metric_rows),
        "expected_task_metric_rows": len(centers) * (len(centers) - 1) * len(config.training_seeds) * len(config.generation_seeds) * len(ARMS),
        "legal_sources_per_task": len(centers) - 2,
    }
    write_json(root / "manifests/protocol_manifest.json", protocol)
    write_json(root / "manifests/coverage_manifest.json", coverage)
    write_json(
        root / "manifests/frame_index.json",
        {
            "schema_version": "midogpp_uniform_b_optimized_frame_index_v2",
            "records": [
                {"source_center": center, "state_hash": source_frames[center].state_hash, "state": source_frames[center].to_payload()}
                for center in centers
            ],
        },
    )
    write_json(
        root / "manifests/generation_manifest.json",
        {"schema_version": "midogpp_uniform_b_optimized_generation_index_v2", "records": generation_rows},
    )
    write_json(
        root / "provenance/input_artifacts.json",
        {
            "schema_version": "midogpp_uniform_b_optimized_provenance_v2",
            "input_artifact_ids": ["midogpp_dataset_contract_annotation_patch_v1", config.feature_cache_artifact_id],
            "manifest_hash": frame.manifest_hash,
            "feature_cache_hash": frame.feature_cache_hash,
            "existing_checkpoint_input_used": False,
            "completed_experiment_artifact_used": False,
            "fresh_source_only_training": True,
        },
    )
    write_csv_rows(root / "tables/task_metrics.csv", metric_rows)
    write_csv_rows(root / "tables/paired_deltas.csv", delta_rows)
    write_csv_rows(root / "tables/sampler_audit.csv", sampler_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_rows)
    timing_rows.append({"phase": "total", "elapsed_seconds": perf_counter() - started})
    write_csv_rows(root / "tables/runtime_timings.csv", timing_rows)
    write_json(root / "reports/study_decision.json", decision)
    write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_uniform_b_optimized_leakage_report_v2",
            "status": "PASS", "outer_rows_used_for_fit": False,
            "inner_rows_used_for_fit": False,
            "inner_labels_used_for_scoring_only": True,
            "target_support_labels_used": False,
            "identity_audit_failures": sum(row["status"] != "PASS" for row in identity_rows),
        },
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_uniform_b_optimized_runtime_summary_v2",
            "runtime_plan": runtime.to_payload(),
            "checkpoint_records": len(checkpoint_records),
            "classifier_fit_count": len(metric_rows),
            "peak_generated_block_working_set_policy": "one_training_seed_at_a_time",
            "gpu_parallel_phase": "source_training",
            "cpu_parallel_phase": "tstr_scoring",
        },
    )
    write_json(
        root / "reports/publication_state.json",
        {
            "schema_version": "midogpp_uniform_b_optimized_publication_state_v2",
            "publication_state": PUBLICATION_STATE,
            "decision": decision["decision"],
            "may_feed_deployable_selection": False,
            "separate_promotion_artifact_required": True,
        },
    )
    _write_content_index(root)


def _protocol(config: OptimizedPriorConfig, manifest_hash: str, cache_hash: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_optimized_protocol_v2",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_hash": config.contract_hash,
        "manifest_hash": manifest_hash,
        "feature_cache_hash": cache_hash,
        "model": {
            "frame": config.block_frame, "pca_dim": config.pca_output_dim,
            "hidden_dim": config.hidden_dim, "latent_dim": config.latent_dim,
            "hidden_layers": config.num_hidden_layers,
            "warmup_steps": config.warmup_steps, "total_steps": config.total_steps,
        },
        "arms": {
            P0: "standard_normal",
            PS: "source_class_conditional_full_shrinkage_total_moment_with_all_or_none_fallback",
            Q: "source_posterior_samples_ceiling_control",
            QM: "source_posterior_mean_decode_ceiling_control",
            R: "source_rows_pca_transform_inverse_frame_ceiling_control",
        },
        "composition": "seven_legal_sources_equal_total_fixed_budget",
        "same_shuffle_across_arms": True,
        "classifier_c_fixed_before_inner_scoring": config.classifier_c,
        "outer_or_inner_rows_used_for_fit": False,
        "inner_labels_used_for_scoring_only": True,
        "routing_or_target_conditioned_selection": False,
        "fresh_source_only_training": True,
        "existing_checkpoint_input_allowed": False,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("outer_center", "inner_center", "training_seed", "generation_seed")
    grouped: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), {})[str(row["arm"])] = row
    result = []
    for key, arms in sorted(grouped.items()):
        if set(arms) != set(ARMS):
            raise ProtocolError("Paired task is missing a generation arm.")
        base = float(arms[P0]["bacc"])
        for arm in (PS, Q, QM, R):
            result.append(
                {
                    "schema_version": "midogpp_uniform_b_optimized_paired_delta_v2",
                    **dict(zip(keys, key)), "arm": arm,
                    "bacc": arms[arm]["bacc"], "p0_bacc": base,
                    "delta_bacc_vs_p0": float(arms[arm]["bacc"]) - base,
                    "paired_shuffle_hash_equal": arms[arm]["shuffle_hash"] == arms[P0]["shuffle_hash"],
                }
            )
    return result


def _decision(rows: list[dict[str, object]], config: OptimizedPriorConfig) -> dict[str, object]:
    by_arm = {
        arm: np.asarray([float(row["bacc"]) for row in rows if row["arm"] == arm], dtype=np.float64)
        for arm in ARMS
    }
    means = {arm: float(values.mean()) for arm, values in by_arm.items()}
    ps_seed_means = {
        str(seed): float(np.mean([float(row["bacc"]) for row in rows if row["arm"] == PS and row["training_seed"] == seed]))
        for seed in config.training_seeds
    }
    frame_ceiling = means[R]
    posterior_ceiling = max(means[Q], means[QM])
    reaches_target = means[PS] >= config.target_prior_bacc
    frame_ceiling_sufficient = frame_ceiling >= config.required_posterior_ceiling
    ceiling_sufficient = posterior_ceiling >= config.required_posterior_ceiling
    stable = min(ps_seed_means.values()) >= config.target_prior_bacc - 0.02
    if reaches_target and stable:
        decision = "TARGET_METRIC_REACHED_REQUIRES_SEPARATE_PROMOTION"
    elif not frame_ceiling_sufficient:
        decision = "FEATURE_FRAME_OR_COMPOSITION_CEILING_INSUFFICIENT"
    elif not ceiling_sufficient:
        decision = "DECODER_POSTERIOR_CEILING_INSUFFICIENT"
    else:
        decision = "PRIOR_GAP_REMAINS_WITH_SUFFICIENT_POSTERIOR_CEILING"
    return {
        "schema_version": "midogpp_uniform_b_optimized_decision_v2",
        "decision": decision,
        "target_prior_bacc": config.target_prior_bacc,
        "required_posterior_ceiling": config.required_posterior_ceiling,
        "mean_bacc_by_arm": means,
        "ps_training_seed_mean_bacc": ps_seed_means,
        "frame_ceiling_bacc": frame_ceiling,
        "posterior_ceiling_bacc": posterior_ceiling,
        "ps_reaches_target": reaches_target,
        "ps_seed_stability_gate": stable,
        "frame_ceiling_sufficient": frame_ceiling_sufficient,
        "posterior_ceiling_sufficient": ceiling_sufficient,
        "ps_minus_p0": means[PS] - means[P0],
        "posterior_ceiling_minus_ps": posterior_ceiling - means[PS],
        "may_feed_deployable_selection": False,
    }


def _write_config(root: Path, config: OptimizedPriorConfig) -> None:
    import yaml
    payload = asdict(config)
    for key, value in tuple(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _write_state(root: Path, status: str) -> None:
    write_json(
        root / "reports/run_state.json",
        {"schema_version": "midogpp_uniform_b_optimized_run_state_v2", "status": status, "claim_scope": CLAIM_SCOPE},
    )


def _write_content_index(root: Path) -> None:
    records = []
    excluded = {"manifests/content_index.json", "reports/run_state.json", "reports/validation_report.json"}
    for path in sorted(item for item in root.rglob("*") if item.is_file() and "runtime_cache" not in item.parts):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"relative_path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
    write_json(
        root / "manifests/content_index.json",
        {"schema_version": "midogpp_uniform_b_optimized_content_index_v2", "records": records},
    )


def validate_optimized_prior_bundle(root: str | Path) -> dict[str, object]:
    path = Path(root)
    missing = [relative for relative in REQUIRED_FILES if not (path / relative).is_file()]
    errors = list(missing)
    if not missing:
        coverage = json.loads((path / "manifests/coverage_manifest.json").read_text(encoding="utf-8"))
        if coverage.get("task_metric_rows") != coverage.get("expected_task_metric_rows"):
            errors.append("task metric coverage mismatch")
        leakage = json.loads((path / "reports/leakage_report.json").read_text(encoding="utf-8"))
        if leakage.get("status") != "PASS":
            errors.append("leakage report did not pass")
        protocol = json.loads((path / "manifests/protocol_manifest.json").read_text(encoding="utf-8"))
        if protocol.get("experiment_id") != EXPERIMENT_ID:
            errors.append("protocol experiment identity mismatch")
    report = {
        "schema_version": "midogpp_uniform_b_optimized_validation_v2",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "required_file_count": len(REQUIRED_FILES),
    }
    write_json(path / "reports/validation_report.json", report)
    if errors:
        raise ProtocolError("Optimized-prior bundle validation failed: " + "; ".join(errors))
    return report


__all__ = ("run_optimized_prior_source_inner_study", "validate_optimized_prior_bundle")
