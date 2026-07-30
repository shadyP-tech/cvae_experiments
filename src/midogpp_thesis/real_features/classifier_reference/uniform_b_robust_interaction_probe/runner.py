"""Run the regression audit and robust-Nyström versus bilinear comparison."""

from __future__ import annotations

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
from ..uniform_b_nonlinear_probe.statistics import binary_metrics
from .audit import build_error_audit
from .config import RobustInteractionConfig
from .estimator import (
    load_selected_nystroem_candidates,
    run_bilinear_selection,
    run_outer_fits,
    run_robust_selection,
    select_family_candidates,
)
from .workspace_binding import validate_production_workspace_binding


def run_robust_interaction_probe(config: RobustInteractionConfig) -> Path:
    validate_production_workspace_binding(config)
    started = time.perf_counter()
    root = prepare_artifact_dirs(config.artifact_root)
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=3840,
        allow_excluded_center_omission=True,
    )
    x = np.asarray(frame.embeddings, dtype=np.float32)
    y = np.asarray([row.label for row in frame.rows], dtype=np.int8)
    centers = np.asarray([row.center for row in frame.rows], dtype=str)
    sample_ids = np.asarray([row.sample_id for row in frame.rows], dtype=str)
    case_ids = np.asarray([row.case_id for row in frame.rows], dtype=str)
    if len(x) != 9648:
        raise ProtocolError("Robust-interaction train surface drifted.")
    standard_all = _read_csv(config.nonlinear_root / "tables/outer_predictions.csv")
    standard = [
        row for row in standard_all if row["model_role"] == "canonical_b_nystroem_primary"
    ]
    linear = [
        row for row in standard_all if row["model_role"] == "canonical_b_linear_baseline"
    ]
    manifest = _manifest_train_by_id(config.manifest_path)
    c_predictions = _c_predictions(config.multiscale_root)
    audit_rows, audit_summary = build_error_audit(
        standard, linear, manifest, c_predictions
    )
    write_csv_rows(root / "tables/paired_error_audit.csv", audit_rows)
    write_csv_rows(root / "tables/error_group_summary.csv", audit_summary)
    write_json(root / "reports/regression_audit_summary.json", _audit_report(audit_rows))

    inputs = _input_hashes(config)
    frozen = _frozen_protocol(config, inputs)
    write_json(root / "manifests/frozen_protocol_snapshot.json", frozen)
    write_json(
        root / "provenance/input_artifacts.json",
        {
            "schema_version": "midogpp_uniform_bplus_robust_inputs_v1",
            "input_hashes": inputs,
            "validation_or_test_inputs": False,
            "multiscale_predictions_used_for_audit_only": True,
        },
    )
    selected_kernels = load_selected_nystroem_candidates(config.nonlinear_root)
    robust_cells = run_robust_selection(
        x,
        y,
        centers,
        sample_ids,
        selected_kernels=selected_kernels,
        config=config,
    )
    bilinear_cells = run_bilinear_selection(x, y, centers, config=config)
    robust_summary, selected_robust, bilinear_summary, selected_bilinear = (
        select_family_candidates(robust_cells, bilinear_cells, config)
    )
    write_csv_rows(root / "tables/robust_selector_cells.csv", robust_cells)
    write_csv_rows(root / "tables/bilinear_selector_cells.csv", bilinear_cells)
    write_csv_rows(
        root / "tables/family_candidate_summary.csv",
        robust_summary + bilinear_summary,
    )
    write_json(
        root / "manifests/source_only_candidate_locks.json",
        {
            "schema_version": "midogpp_uniform_bplus_candidate_locks_v1",
            "protocol_hash": frozen["protocol_hash"],
            "selected_robust_objectives": selected_robust,
            "selected_bilinear_ranks": selected_bilinear,
            "selection_priority": (
                "worst_inner_center_class_recall_then_mean_bacc_then_worst_bacc"
            ),
        },
    )
    robust_outer, bilinear_outer = run_outer_fits(
        x,
        y,
        centers,
        sample_ids,
        case_ids,
        config=config,
        selected_kernels=selected_kernels,
        selected_robust=selected_robust,
        selected_bilinear=selected_bilinear,
    )
    tables = _materialize(
        config, linear, standard, robust_outer, bilinear_outer
    )
    write_csv_rows(root / "tables/outer_results.csv", tables["results"])
    write_csv_rows(root / "tables/outer_predictions.csv", tables["predictions"])
    write_csv_rows(
        root / "tables/stability_predictions.csv", tables["stability_predictions"]
    )
    write_csv_rows(root / "tables/center_family_comparison.csv", tables["comparisons"])
    decision = _family_decision(tables["comparisons"], config)
    write_json(root / "reports/family_decision.json", decision)
    summary = {
        "schema_version": "midogpp_uniform_bplus_robust_interaction_summary_v1",
        "status": "COMPLETE",
        "decision": decision,
        "regression_audit": _audit_report(audit_rows),
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
            "schema_version": "midogpp_uniform_bplus_runtime_v1",
            "status": "COMPLETE",
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_pair_jobs": config.cpu_pair_jobs,
            "cpu_threads_per_job": config.cpu_threads_per_job,
            "gpu_devices": list(config.gpu_devices),
            "robust_selector_fits": 36 * 2 * len(config.robust_objectives),
            "bilinear_selector_fits": 36 * len(config.bilinear_ranks),
            "outer_fits": 9 * 2 * 3,
        },
    )
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_uniform_bplus_leakage_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "source_inner_excludes_outer_and_inner": True,
            "outer_fit_excludes_outer": True,
            "validation_scored": False,
            "test_scored": False,
            "center_specific_thresholds": False,
            "threshold": 0.5,
        },
    )
    write_json(
        root / "manifests/protocol_manifest.json",
        {
            "schema_version": "midogpp_uniform_bplus_protocol_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "protocol_hash": frozen["protocol_hash"],
            "claim_scope": "diagnostic_only",
            "already_inspected_train_surface": True,
            "non_adoptive": True,
            "validation_scored": False,
            "test_scored": False,
            "may_replace_canonical_reference": False,
            "may_feed_deployable_selection": False,
        },
    )
    _write_content_index(root)
    from .validation import validate_robust_interaction_bundle

    checks = validate_robust_interaction_bundle(root, config=config, allow_pending=True)
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_bplus_validation_v1",
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
    validate_robust_interaction_bundle(root, config=config)
    return root


def _materialize(
    config: RobustInteractionConfig,
    linear: Sequence[Mapping[str, str]],
    standard: Sequence[Mapping[str, str]],
    robust_outer: Sequence[Mapping[str, object]],
    bilinear_outer: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    imported = {
        "linear_b": linear,
        "standard_nystroem": standard,
    }
    generated = {
        "robust_nystroem": robust_outer,
        "bilinear": bilinear_outer,
    }
    results = []
    predictions = []
    stability_predictions = []
    comparisons = []
    metrics_by_family: dict[tuple[str, str, int], Mapping[str, float]] = {}
    for family, rows in imported.items():
        for center in config.heldout_centers:
            selected = [
                row
                for row in rows
                if str(row["outer_center"]) == center
            ]
            truth = np.asarray([int(row["y_true"]) for row in selected])
            pred = np.asarray([int(row["y_pred"]) for row in selected])
            metrics = binary_metrics(truth, pred)
            metrics_by_family[(center, family, config.primary_seed)] = metrics
            predictions.extend(
                {
                    "family": family,
                    "seed": config.primary_seed,
                    "candidate": "imported_frozen",
                    "outer_center": center,
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "center": center,
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "prob_pos": float(row["prob_pos"]),
                }
                for row in selected
            )
    for family, outputs in generated.items():
        for output in outputs:
            center = str(output["outer_center"])
            for seed_output in output["seed_rows"]:
                seed = int(seed_output["seed"])
                metrics_by_family[(center, family, seed)] = seed_output["metrics"]
                target = (
                    predictions
                    if seed == config.primary_seed
                    else stability_predictions
                )
                target.extend(seed_output["predictions"])
    for (center, family, seed), metrics in sorted(metrics_by_family.items()):
        results.append(
            {
                "schema_version": "midogpp_uniform_bplus_outer_result_v1",
                "outer_center": center,
                "family": family,
                "seed": seed,
                **metrics,
            }
        )
    for center in config.heldout_centers:
        linear_metrics = metrics_by_family[(center, "linear_b", config.primary_seed)]
        standard_metrics = metrics_by_family[
            (center, "standard_nystroem", config.primary_seed)
        ]
        for family in ("robust_nystroem", "bilinear"):
            for seed in (config.primary_seed, *config.stability_seeds):
                metrics = metrics_by_family[(center, family, seed)]
                comparisons.append(
                    {
                        "schema_version": "midogpp_uniform_bplus_center_comparison_v1",
                        "outer_center": center,
                        "family": family,
                        "seed": seed,
                        "bacc": metrics["bacc"],
                        "positive_recall": metrics["positive_recall"],
                        "specificity": metrics["specificity"],
                        "delta_bacc_vs_linear": metrics["bacc"]
                        - linear_metrics["bacc"],
                        "delta_bacc_vs_standard_nystroem": metrics["bacc"]
                        - standard_metrics["bacc"],
                        "delta_recall_vs_linear": metrics["positive_recall"]
                        - linear_metrics["positive_recall"],
                        "delta_specificity_vs_linear": metrics["specificity"]
                        - linear_metrics["specificity"],
                    }
                )
    return {
        "results": results,
        "predictions": predictions,
        "stability_predictions": stability_predictions,
        "comparisons": comparisons,
    }


def _family_decision(
    comparisons: Sequence[Mapping[str, object]], config: RobustInteractionConfig
) -> dict[str, object]:
    family_rows = []
    for family in ("robust_nystroem", "bilinear"):
        primary = [
            row
            for row in comparisons
            if row["family"] == family and int(row["seed"]) == config.primary_seed
        ]
        stability = [
            row
            for row in comparisons
            if row["family"] == family and int(row["seed"]) != config.primary_seed
        ]
        mean_linear = float(np.mean([float(row["delta_bacc_vs_linear"]) for row in primary]))
        mean_standard = float(
            np.mean([float(row["delta_bacc_vs_standard_nystroem"]) for row in primary])
        )
        worst_class = min(
            min(float(row["delta_recall_vs_linear"]), float(row["delta_specificity_vs_linear"]))
            for row in primary
        )
        seed_groups = {
            seed: [row for row in stability if int(row["seed"]) == seed]
            for seed in config.stability_seeds
        }
        passed = (
            mean_linear >= 0.015
            and mean_standard >= -0.005
            and sum(float(row["delta_bacc_vs_linear"]) > 0 for row in primary) >= 6
            and min(float(row["delta_bacc_vs_linear"]) for row in primary) >= -0.01
            and worst_class >= -0.05
            and all(
                np.mean([float(row["delta_bacc_vs_linear"]) for row in rows]) > 0
                and min(float(row["delta_bacc_vs_linear"]) for row in rows) >= -0.01
                for rows in seed_groups.values()
            )
        )
        family_rows.append(
            {
                "family": family,
                "passed": bool(passed),
                "equal_center_delta_vs_linear": mean_linear,
                "equal_center_delta_vs_standard_nystroem": mean_standard,
                "strict_wins_vs_linear": sum(
                    float(row["delta_bacc_vs_linear"]) > 0 for row in primary
                ),
                "worst_center_bacc_delta_vs_linear": min(
                    float(row["delta_bacc_vs_linear"]) for row in primary
                ),
                "worst_center_class_direction_delta_vs_linear": worst_class,
            }
        )
    eligible = [row for row in family_rows if row["passed"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["worst_center_class_direction_delta_vs_linear"],
                row["equal_center_delta_vs_linear"],
            ),
        )["family"]
        if eligible
        else None
    )
    return {
        "schema_version": "midogpp_uniform_bplus_family_decision_v1",
        "decision": (
            f"FREEZE_{selected.upper()}_AS_BPLUS_RESEARCH_CANDIDATE"
            if selected
            else "NO_FAMILY_PASSES_ROBUST_BPLUS_GATE"
        ),
        "selected_family": selected,
        "family_checks": family_rows,
        "gate": {
            "mean_delta_vs_linear_min": 0.015,
            "mean_delta_vs_standard_nystroem_min": -0.005,
            "strict_wins_vs_linear_min": 6,
            "worst_center_bacc_delta_vs_linear_min": -0.01,
            "worst_center_class_direction_delta_vs_linear_min": -0.05,
            "supplemental_mean_delta_vs_linear_exclusive": 0.0,
            "supplemental_worst_center_delta_vs_linear_min": -0.01,
        },
        "diagnostic_only": True,
        "does_not_authorize_confirmation_scoring": True,
    }


def _audit_report(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    regressions = [row for row in rows if row["category"] == "nonlinear_regression"]
    rescues = [row for row in rows if row["category"] == "nonlinear_rescue"]
    hard = [row for row in rows if row["category"] == "shared_hard_core"]
    return {
        "schema_version": "midogpp_uniform_bplus_regression_audit_summary_v1",
        "nonlinear_rescues": len(rescues),
        "nonlinear_regressions": len(regressions),
        "shared_hard_core": len(hard),
        "confident_regressions": sum(bool(row["nonlinear_confident"]) for row in regressions),
        "near_boundary_regressions": sum(
            float(row["nonlinear_margin"]) < 0.1 for row in regressions
        ),
        "center_2_false_positive_regressions": sum(
            row["center"] == "2" and int(row["label"]) == 0 for row in regressions
        ),
        "center_9_false_negative_regressions": sum(
            row["center"] == "9" and int(row["label"]) == 1 for row in regressions
        ),
        "c_overlap_available": sum(bool(row["c_prediction_available"]) for row in rows),
        "c_correct_among_regressions": sum(
            bool(row["c_prediction_available"]) and bool(row["c_correct"])
            for row in regressions
        ),
    }


def _render_report(summary: Mapping[str, object]) -> str:
    audit = summary["regression_audit"]
    decision = summary["decision"]
    return "\n".join(
        [
            "# Uniform-B Robust Nyström versus Bilinear Probe",
            "",
            f"Decision: `{decision['decision']}`.",
            "",
            "## Paired-error audit",
            "",
            f"- Rescues: `{audit['nonlinear_rescues']}`",
            f"- Regressions: `{audit['nonlinear_regressions']}`",
            f"- Shared hard core: `{audit['shared_hard_core']}`",
            f"- Confident regressions: `{audit['confident_regressions']}`",
            f"- Near-boundary regressions: `{audit['near_boundary_regressions']}`",
            "",
            "## Family checks",
            "",
            f"`{json.dumps(decision['family_checks'], sort_keys=True)}`",
            "",
            "This remains a post-hoc Stage-90 diagnostic. Validation and test were not scored.",
            "",
        ]
    )


def _frozen_protocol(
    config: RobustInteractionConfig, inputs: Mapping[str, str]
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_bplus_robust_frozen_protocol_v1",
        "experiment_name": config.name,
        "heldout_centers": list(config.heldout_centers),
        "robust_objectives": list(config.robust_objectives),
        "dro_iterations": config.dro_iterations,
        "bilinear_ranks": list(config.bilinear_ranks),
        "bilinear": {
            "global_dim": config.global_dim,
            "local_dim": config.local_dim,
            "epochs": config.bilinear_epochs,
            "learning_rate": config.bilinear_learning_rate,
            "weight_decay": config.bilinear_weight_decay,
            "batch_size": config.bilinear_batch_size,
        },
        "seeds": [config.primary_seed, *config.stability_seeds],
        "threshold": 0.5,
        "selection_metric": (
            "worst_inner_center_class_recall_then_mean_bacc_then_worst_bacc"
        ),
        "claim_scope": "diagnostic_only",
        "validation_scored": False,
        "test_scored": False,
        "input_hashes": dict(sorted(inputs.items())),
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _c_predictions(root: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(root / "tables/outer_locked_predictions.csv")
    return {
        row["sample_id"]: row
        for row in rows
        if row["representation_id"] == "physical_multiscale_clipped_bbox_annotation_local_c_v3"
    }


def _manifest_train_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row["split"].strip().lower() == "train"
            and row["center"] in {"0", "1", "2", "3", "5", "6", "7", "8", "9"}
        }


def _input_hashes(config: RobustInteractionConfig) -> dict[str, str]:
    return {
        "manifest": _sha256(config.manifest_path),
        "canonical_b_cache": _sha256(config.feature_cache_path),
        "canonical_reference": _sha256(
            config.canonical_root / "manifests/content_index.json"
        ),
        "nonlinear_probe": _sha256(
            config.nonlinear_root / "manifests/content_index.json"
        ),
        "multiscale_probe": _sha256(
            config.multiscale_root / "manifests/content_index.json"
        ),
    }


def _write_content_index(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        files.append({"path": relative, "sha256": _sha256(path)})
    payload = {
        "schema_version": "midogpp_uniform_bplus_content_index_v1",
        "files": files,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


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
