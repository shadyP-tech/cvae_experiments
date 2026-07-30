"""Run the bounded sensitivity/specificity-constrained Nyström diagnostic."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.common.hashing import stable_hash

from ..artifacts import prepare_artifact_dirs, write_csv_rows, write_json
from ..protocol import ProtocolError
from ..real_feature_frame import load_midogpp_real_feature_frame
from ..uniform_b_nonlinear_probe.statistics import (
    binary_metrics,
    paired_case_bootstrap,
)
from ..uniform_b_robust_interaction_probe.estimator import (
    load_selected_nystroem_candidates,
)
from .config import ConstrainedNystroemConfig, load_constrained_nystroem_config
from .estimator import (
    fit_locked_outer_models,
    load_canonical_specs,
    replay_source_inner_capacity_path,
    run_source_inner_capacity_path,
    select_constrained_candidates,
)
from .workspace_binding import validate_production_workspace_binding


def run_constrained_nystroem_probe(config: ConstrainedNystroemConfig) -> Path:
    validate_production_workspace_binding(config)
    started = time.perf_counter()
    root = prepare_artifact_dirs(config.artifact_root)
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=config.expected_dim,
        allow_excluded_center_omission=True,
    )
    if len(frame.rows) != config.expected_rows:
        raise ProtocolError("Constrained-Nyström train surface drifted.")
    x = np.asarray(frame.embeddings, dtype=np.float32)
    y = np.asarray([row.label for row in frame.rows], dtype=np.int8)
    centers = np.asarray([row.center for row in frame.rows], dtype=str)
    sample_ids = np.asarray([row.sample_id for row in frame.rows], dtype=str)
    case_ids = np.asarray([row.case_id for row in frame.rows], dtype=str)
    kernels = load_selected_nystroem_candidates(config.nonlinear_root)
    linear_specs = load_canonical_specs(config.canonical_root)
    if set(kernels) != set(config.heldout_centers) or set(linear_specs) != set(
        config.heldout_centers
    ):
        raise ProtocolError("Inherited classifier locks are incomplete.")

    input_hashes = _input_hashes(config)
    frozen = _frozen_protocol(config, input_hashes, kernels, linear_specs)
    write_json(root / "manifests/frozen_protocol_snapshot.json", frozen)
    write_json(
        root / "manifests/blend_capacity_grid_lock.json",
        {
            "schema_version": "midogpp_constrained_blend_grid_lock_v1",
            "protocol_hash": frozen["protocol_hash"],
            "objectives": list(config.objectives),
            "alphas": list(config.alphas),
            "fallback_alpha": config.fallback_alpha,
            "fallback_role": config.fallback_role,
            "blend_definition": (
                "linear_logit+alpha*(nystroem_logit-linear_logit)"
            ),
            "threshold": config.threshold,
            "threshold_selected": False,
        },
    )
    write_json(
        root / "manifests/inherited_candidate_lock_index.json",
        {
            "schema_version": "midogpp_constrained_inherited_locks_v1",
            "protocol_hash": frozen["protocol_hash"],
            "kernels": {
                center: {
                    "candidate_id": kernels[center].candidate_id,
                    **kernels[center].to_payload(),
                }
                for center in config.heldout_centers
            },
            "linear_specs": linear_specs,
            "robust_lineage_only": True,
            "robust_outer_tables_read_for_selection": False,
        },
    )
    write_json(
        root / "provenance/input_artifacts.json",
        {
            "schema_version": "midogpp_constrained_inputs_v1",
            "input_hashes": input_hashes,
            "robust_interaction_used_for_lineage_only": True,
            "robust_outer_tables_read_for_selection": False,
            "validation_or_test_inputs": False,
            "source_inner_base_scores_replayed": (
                config.source_inner_replay_root is not None
            ),
        },
    )
    write_json(
        root / "manifests/protocol_manifest.json",
        {
            "schema_version": "midogpp_constrained_protocol_manifest_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "protocol_hash": frozen["protocol_hash"],
            "claim_scope": "diagnostic_only",
            "study_design_informed_by_prior_outer_scores": True,
            "independent_outer_confirmation": False,
            "selection_used_current_outer_labels": False,
            "validation_scored": False,
            "test_scored": False,
            "may_replace_canonical_reference": False,
            "may_feed_deployable_selection": False,
        },
    )

    if config.source_inner_replay_root is not None:
        from .validation import validate_constrained_nystroem_bundle

        replay_config = load_constrained_nystroem_config(
            config.source_inner_replay_root / "config.resolved.yaml"
        )
        validate_constrained_nystroem_bundle(
            config.source_inner_replay_root, config=replay_config
        )
        baseline_cells, blend_cells, base_scores = (
            replay_source_inner_capacity_path(
                config.source_inner_replay_root, config=config
            )
        )
    else:
        baseline_cells, blend_cells, base_scores = run_source_inner_capacity_path(
            x,
            y,
            centers,
            sample_ids,
            config=config,
            kernels=kernels,
            linear_specs=linear_specs,
        )
    _validate_linear_replay(baseline_cells, linear_specs)
    _validate_nonlinear_endpoint_replay(blend_cells, config, kernels)
    summaries, selected = select_constrained_candidates(blend_cells, config)
    decision_rows = [selected[center] for center in config.heldout_centers]
    write_csv_rows(root / "tables/source_inner_linear_baseline_cells.csv", baseline_cells)
    write_csv_rows(root / "tables/source_inner_blend_cells.csv", blend_cells)
    write_csv_rows(root / "tables/source_inner_base_scores.csv", base_scores)
    write_csv_rows(root / "tables/source_inner_candidate_summary.csv", summaries)
    write_csv_rows(root / "tables/blend_capacity_decisions.csv", decision_rows)
    selection_lock = {
        "schema_version": "midogpp_constrained_source_only_locks_v1",
        "protocol_hash": frozen["protocol_hash"],
        "selected": selected,
        "selection_hash": stable_hash(selected),
        "outer_labels_read_before_lock": False,
        "threshold_selected": False,
    }
    write_json(root / "manifests/source_only_candidate_locks.json", selection_lock)

    outer = fit_locked_outer_models(
        x,
        y,
        centers,
        sample_ids,
        case_ids,
        config=config,
        kernels=kernels,
        linear_specs=linear_specs,
        selected=selected,
    )
    canonical_rows = _read_csv(
        config.canonical_root / "tables/classifier_tuned_predictions.csv"
    )
    materialized = _materialize_outer(outer, canonical_rows, config)
    write_csv_rows(root / "tables/outer_results.csv", materialized["results"])
    write_csv_rows(root / "tables/outer_predictions.csv", materialized["primary"])
    write_csv_rows(
        root / "tables/stability_predictions.csv", materialized["stability"]
    )
    write_csv_rows(root / "tables/center_comparison.csv", materialized["comparisons"])
    write_csv_rows(root / "tables/error_exchange.csv", materialized["exchange"])
    bootstrap = paired_case_bootstrap(
        materialized["primary"],
        materialized["baseline"],
        centers=config.heldout_centers,
        replicates=2000,
        seed=42,
    )
    write_json(root / "reports/paired_bootstrap.json", bootstrap)
    progression = _progression_decision(
        materialized["comparisons"], selected, config
    )
    progression["bootstrap_supportive_only"] = bootstrap
    write_json(root / "reports/progression_decision.json", progression)
    feasibility = _feasibility_report(summaries, selected, config)
    write_json(root / "reports/constraint_feasibility.json", feasibility)
    summary = {
        "schema_version": "midogpp_constrained_diagnostic_summary_v1",
        "status": "COMPLETE",
        "decision": progression,
        "feasibility": feasibility,
        "validation_scored": False,
        "test_scored": False,
        "diagnostic_only": True,
    }
    write_json(root / "reports/diagnostic_summary.json", summary)
    (root / "reports/diagnostic_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_constrained_runtime_v1",
            "status": "COMPLETE",
            "elapsed_seconds": time.perf_counter() - started,
            "pair_jobs": config.pair_jobs,
            "threads_per_job": config.threads_per_job,
            "gpu_devices_used": [],
            "source_inner_base_scores_replayed": (
                config.source_inner_replay_root is not None
            ),
            "unordered_pair_frames": (
                0 if config.source_inner_replay_root is not None else 36
            ),
            "ordered_linear_cells": 72,
            "ordered_objective_evaluations": 72 * len(config.objectives),
            "source_inner_model_fit_count": (
                0 if config.source_inner_replay_root is not None else 1122
            ),
            "blend_metric_cells": 72 * len(config.objectives) * len(config.alphas),
            "alpha_additional_model_fits": 0,
            "outer_seed_slots": 9 * 3,
        },
    )
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_constrained_leakage_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "source_inner_excludes_outer_and_inner": True,
            "outer_fit_excludes_outer": True,
            "selection_used_current_outer_labels": False,
            "robust_outer_tables_read_for_selection": False,
            "threshold": config.threshold,
            "threshold_selected": False,
            "center_specific_thresholds": False,
            "validation_scored": False,
            "test_scored": False,
        },
    )
    _write_content_index(root)
    from .validation import validate_constrained_nystroem_bundle

    checks = validate_constrained_nystroem_bundle(
        root, config=config, allow_pending=True
    )
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_constrained_validation_v1",
            "status": "PASS",
            "checks": checks,
        },
    )
    leakage = _read_json(root / "reports/leakage_provenance_report.json")
    leakage["status"] = "PASS"
    write_json(root / "reports/leakage_provenance_report.json", leakage)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    protocol["status"] = "PASS"
    write_json(root / "manifests/protocol_manifest.json", protocol)
    _write_content_index(root)
    validate_constrained_nystroem_bundle(root, config=config)
    return root


def _materialize_outer(
    outputs: Sequence[Mapping[str, object]],
    canonical_rows: Sequence[Mapping[str, str]],
    config: ConstrainedNystroemConfig,
) -> dict[str, list[dict[str, object]]]:
    canonical = {row["sample_id"]: row for row in canonical_rows}
    results: list[dict[str, object]] = []
    primary: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    exchange: list[dict[str, object]] = []
    for output in outputs:
        outer = str(output["outer_center"])
        for seed_row in output["seed_rows"]:
            seed = int(seed_row["seed"])
            rows = list(seed_row["predictions"])
            for row in rows:
                reference = canonical[str(row["sample_id"])]
                linear_probability = 1.0 / (
                    1.0 + np.exp(-float(row["linear_logit"]))
                )
                if (
                    int(reference["y_true"]) != int(row["y_true"])
                    or (float(row["linear_logit"]) >= 0.0)
                    != bool(int(reference["y_pred"]))
                    or not np.isclose(
                        linear_probability,
                        float(reference["prob_pos"]),
                        atol=1e-10,
                    )
                ):
                    raise ProtocolError("Outer linear-B endpoint replay failed.")
            target = primary if seed == config.primary_seed else stability
            target.extend(rows)
            metrics = seed_row["metrics"]
            truth = np.asarray([int(row["y_true"]) for row in rows])
            baseline_pred = np.asarray(
                [int(canonical[str(row["sample_id"])]["y_pred"]) for row in rows]
            )
            baseline_metrics = binary_metrics(truth, baseline_pred)
            comparisons.append(
                {
                    "schema_version": "midogpp_constrained_center_comparison_v1",
                    "outer_center": outer,
                    "seed": seed,
                    "objective": output["selection"]["objective"],
                    "alpha": output["selection"]["alpha"],
                    **metrics,
                    "delta_bacc": metrics["bacc"] - baseline_metrics["bacc"],
                    "delta_recall": metrics["positive_recall"]
                    - baseline_metrics["positive_recall"],
                    "delta_specificity": metrics["specificity"]
                    - baseline_metrics["specificity"],
                }
            )
            results.append(
                {
                    "schema_version": "midogpp_constrained_outer_result_v1",
                    "outer_center": outer,
                    "seed": seed,
                    "objective": output["selection"]["objective"],
                    "alpha": output["selection"]["alpha"],
                    **metrics,
                }
            )
        primary_rows = [
            row
            for row in output["seed_rows"][0]["predictions"]
            if int(row["landmark_seed"]) == config.primary_seed
        ]
        for row in primary_rows:
            reference = canonical[str(row["sample_id"])]
            base_row = {
                "outer_center": outer,
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "center": outer,
                "y_true": int(row["y_true"]),
                "y_pred": int(reference["y_pred"]),
            }
            baseline.append(base_row)
            candidate_correct = int(row["y_pred"]) == int(row["y_true"])
            baseline_correct = int(reference["y_pred"]) == int(row["y_true"])
            exchange.append(
                {
                    "schema_version": "midogpp_constrained_error_exchange_v1",
                    "outer_center": outer,
                    "sample_id": row["sample_id"],
                    "y_true": int(row["y_true"]),
                    "linear_correct": baseline_correct,
                    "constrained_correct": candidate_correct,
                    "outcome": (
                        "rescue"
                        if candidate_correct and not baseline_correct
                        else "regression"
                        if baseline_correct and not candidate_correct
                        else "both_correct"
                        if baseline_correct
                        else "both_wrong"
                    ),
                }
            )
    return {
        "results": results,
        "primary": primary,
        "stability": stability,
        "comparisons": comparisons,
        "baseline": baseline,
        "exchange": exchange,
    }


def _progression_decision(
    comparisons: Sequence[Mapping[str, object]],
    selected: Mapping[str, Mapping[str, object]],
    config: ConstrainedNystroemConfig,
) -> dict[str, object]:
    primary = [
        row for row in comparisons if int(row["seed"]) == config.primary_seed
    ]
    supplemental = {
        seed: [row for row in comparisons if int(row["seed"]) == seed]
        for seed in config.stability_seeds
    }
    nonlinear_locks = sum(float(row["alpha"]) > 0.0 for row in selected.values())
    checks = {
        "nonlinear_outer_locks": nonlinear_locks,
        "mean_bacc_delta": float(np.mean([float(row["delta_bacc"]) for row in primary])),
        "strict_center_wins": sum(float(row["delta_bacc"]) > 0.0 for row in primary),
        "worst_center_bacc_delta": min(float(row["delta_bacc"]) for row in primary),
        "worst_recall_delta": min(float(row["delta_recall"]) for row in primary),
        "worst_specificity_delta": min(
            float(row["delta_specificity"]) for row in primary
        ),
        "mean_recall_delta": float(
            np.mean([float(row["delta_recall"]) for row in primary])
        ),
        "mean_specificity_delta": float(
            np.mean([float(row["delta_specificity"]) for row in primary])
        ),
        "stability": {},
    }
    for seed, rows in supplemental.items():
        checks["stability"][str(seed)] = {
            "mean_bacc_delta": float(
                np.mean([float(row["delta_bacc"]) for row in rows])
            ),
            "worst_center_bacc_delta": min(
                float(row["delta_bacc"]) for row in rows
            ),
            "worst_direction_delta": min(
                min(float(row["delta_recall"]), float(row["delta_specificity"]))
                for row in rows
            ),
        }
    gate = config.gate
    passed = bool(
        nonlinear_locks >= gate.nonlinear_outer_locks_min
        and checks["mean_bacc_delta"] >= gate.mean_bacc_delta_min
        and checks["strict_center_wins"] >= gate.strict_center_wins_min
        and checks["worst_center_bacc_delta"] >= gate.worst_center_bacc_delta_min
        and checks["worst_recall_delta"] >= gate.every_center_recall_delta_min
        and checks["worst_specificity_delta"]
        >= gate.every_center_specificity_delta_min
        and checks["mean_recall_delta"] > gate.mean_recall_delta_exclusive
        and checks["mean_specificity_delta"] >= gate.mean_specificity_delta_min
        and all(
            values["mean_bacc_delta"]
            > gate.stability_mean_bacc_delta_exclusive
            and values["worst_center_bacc_delta"]
            >= gate.stability_worst_center_bacc_delta_min
            and values["worst_direction_delta"]
            >= gate.stability_every_direction_delta_min
            for values in checks["stability"].values()
        )
    )
    return {
        "schema_version": "midogpp_constrained_progression_decision_v1",
        "decision": (
            "LOCK_POSTHOC_CONSTRAINED_BPLUS_RESEARCH_CANDIDATE"
            if passed
            else "NO_CONSTRAINED_BPLUS_CANDIDATE_PASSES"
        ),
        "passed": passed,
        "checks": checks,
        "diagnostic_only": True,
        "does_not_authorize_validation_or_test_scoring": True,
    }


def _feasibility_report(
    summaries: Sequence[Mapping[str, object]],
    selected: Mapping[str, Mapping[str, object]],
    config: ConstrainedNystroemConfig,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_constrained_feasibility_v1",
        "constraints": asdict(config.constraints),
        "feasible_counts": {
            center: sum(
                bool(row["hard_feasible"])
                for row in summaries
                if row["outer_center"] == center
            )
            for center in config.heldout_centers
        },
        "fallback_centers": [
            center for center, row in selected.items() if bool(row["fallback"])
        ],
        "selected": selected,
    }


def _validate_linear_replay(
    cells: Sequence[Mapping[str, object]],
    specs: Mapping[str, Mapping[str, object]],
) -> None:
    for row in cells:
        expected = specs[str(row["outer_center"])]["source_inner_bacc_vector"]
        if not np.isclose(
            float(row["bacc"]),
            float(expected[str(row["inner_center"])]),
            atol=1e-12,
        ):
            raise ProtocolError("Canonical linear-B source-inner replay failed.")


def _validate_nonlinear_endpoint_replay(
    cells: Sequence[Mapping[str, object]],
    config: ConstrainedNystroemConfig,
    kernels: Mapping[str, object],
) -> None:
    prior = _read_csv(config.nonlinear_root / "tables/source_inner_selector_cells.csv")
    index = {
        (row["outer_center"], row["inner_center"], row["candidate_id"]): row
        for row in prior
    }
    endpoints = [
        row
        for row in cells
        if row["objective"] == "canonical_class_weight"
        and np.isclose(float(row["alpha"]), 1.0)
    ]
    for row in endpoints:
        key = (
            str(row["outer_center"]),
            str(row["inner_center"]),
            kernels[str(row["outer_center"])].candidate_id,
        )
        expected = index.get(key)
        if expected is None or any(
            not np.isclose(float(row[field]), float(expected[field]), atol=1e-12)
            for field in ("bacc", "positive_recall", "specificity")
        ):
            raise ProtocolError("Standard Nyström source-inner endpoint replay failed.")


def _frozen_protocol(
    config: ConstrainedNystroemConfig,
    input_hashes: Mapping[str, str],
    kernels: Mapping[str, object],
    linear_specs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_constrained_frozen_protocol_v1",
        "experiment": config.name,
        "config": _jsonable(asdict(config)),
        "input_hashes": dict(input_hashes),
        "kernel_locks": {
            center: {
                "candidate_id": kernels[center].candidate_id,
                **kernels[center].to_payload(),
            }
            for center in config.heldout_centers
        },
        "linear_specs": dict(linear_specs),
        "threshold": config.threshold,
        "threshold_selected": False,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _input_hashes(config: ConstrainedNystroemConfig) -> dict[str, str]:
    files = {
        "manifest": config.manifest_path,
        "feature_cache": config.feature_cache_path,
        "canonical_results": config.canonical_root
        / "tables/classifier_tuned_source_results.csv",
        "canonical_predictions": config.canonical_root
        / "tables/classifier_tuned_predictions.csv",
        "nonlinear_candidate_summary": config.nonlinear_root
        / "tables/source_inner_candidate_summary.csv",
        "nonlinear_selector_cells": config.nonlinear_root
        / "tables/source_inner_selector_cells.csv",
        "robust_lineage_decision": config.robust_lineage_root
        / "reports/family_decision.json",
        "robust_lineage_protocol": config.robust_lineage_root
        / "manifests/protocol_manifest.json",
        "robust_lineage_content_index": config.robust_lineage_root
        / "manifests/content_index.json",
    }
    if config.source_inner_replay_root is not None:
        files.update(
            {
                "source_inner_replay_baseline": config.source_inner_replay_root
                / "tables/source_inner_linear_baseline_cells.csv",
                "source_inner_replay_base_scores": config.source_inner_replay_root
                / "tables/source_inner_base_scores.csv",
                "source_inner_replay_protocol": config.source_inner_replay_root
                / "manifests/protocol_manifest.json",
                "source_inner_replay_content_index": config.source_inner_replay_root
                / "manifests/content_index.json",
            }
        )
    return {key: _sha256(path) for key, path in files.items()}


def _render_report(summary: Mapping[str, object]) -> str:
    decision = summary["decision"]
    feasibility = summary["feasibility"]
    return (
        "# Uniform-B Sensitivity/Specificity-Constrained Nyström Probe\n\n"
        f"Decision: `{decision['decision']}`.\n\n"
        "The threshold remained fixed at `0.5`. Selection used only ordered "
        "source-inner folds and the frozen nonlinear-capacity path.\n\n"
        f"Fallback centers: `{feasibility['fallback_centers']}`.\n\n"
        f"Primary checks: `{json.dumps(decision['checks'], sort_keys=True)}`.\n\n"
        "This is a post-hoc Stage-90 diagnostic informed by previously "
        "inspected outer outcomes. Validation and test were not scored.\n"
    )


def _write_content_index(root: Path) -> None:
    files = []
    for member in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = str(member.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        files.append(
            {
                "path": relative,
                "sha256": _sha256(member),
                "bytes": member.stat().st_size,
            }
        )
    payload = {
        "schema_version": "midogpp_constrained_content_index_v1",
        "files": files,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
