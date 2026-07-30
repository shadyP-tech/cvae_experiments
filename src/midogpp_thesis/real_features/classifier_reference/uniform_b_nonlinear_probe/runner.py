"""Run the bounded nonlinear decision-boundary probe on frozen Uniform-B."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.common.hashing import stable_hash

from ..artifacts import prepare_artifact_dirs, write_csv_rows, write_json
from ..protocol import ProtocolError
from ..real_feature_frame import load_midogpp_real_feature_frame
from ..uniform_b_reference import (
    load_uniform_b_canonical_reference_config,
    validate_uniform_b_canonical_reference_bundle,
)
from .config import (
    EXPECTED_NYSTROEM_TRANSFORMS,
    EXPECTED_PAIR_FRAMES,
    EXPECTED_SELECTOR_CELLS,
    REPRESENTATION_ID,
    NonlinearProbeConfig,
)
from .estimator import fit_outer_models, run_source_inner_selection
from .statistics import (
    binary_metrics,
    paired_case_bootstrap,
    progression_decision,
    summarize_and_select,
)
from .workspace_binding import validate_production_workspace_binding


def run_nonlinear_probe(config: NonlinearProbeConfig) -> Path:
    validate_production_workspace_binding(config)
    started = time.perf_counter()
    root = prepare_artifact_dirs(config.artifact_root)
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    _validate_canonical_reference(config)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=config.expected_feature_dim,
        allow_excluded_center_omission=True,
    )
    if len(frame.rows) != config.expected_train_rows:
        raise ProtocolError("Uniform-B nonlinear train-row count drifted.")
    x = np.asarray(frame.embeddings, dtype=np.float32)
    y = np.asarray([row.label for row in frame.rows], dtype=np.int8)
    centers = np.asarray([row.center for row in frame.rows], dtype=str)
    sample_ids = np.asarray([row.sample_id for row in frame.rows], dtype=str)
    case_ids = np.asarray([row.case_id for row in frame.rows], dtype=str)
    baseline_results = _read_csv(
        config.canonical_reference_root / "tables/classifier_tuned_source_results.csv"
    )
    baseline_rows = _read_csv(
        config.canonical_reference_root / "tables/classifier_tuned_predictions.csv"
    )
    baseline_predictions = {row["sample_id"]: row for row in baseline_rows}
    if set(baseline_predictions) != set(sample_ids.tolist()):
        raise ProtocolError("Canonical-B baseline prediction rows do not match the cache.")
    class_weights = _baseline_class_weights(baseline_results)
    input_hashes = _input_hashes(config, frame.feature_cache_hash, frame.manifest_hash)
    frozen = _frozen_protocol(config, input_hashes, class_weights)
    write_json(root / "manifests/frozen_protocol_snapshot.json", frozen)
    write_json(
        root / "manifests/nystroem_grid_lock.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_grid_lock_v1",
            "protocol_hash": frozen["protocol_hash"],
            "candidate_count": len(config.candidates),
            "candidates": [
                {"candidate_id": candidate.candidate_id, **candidate.to_payload()}
                for candidate in config.candidates
            ],
            "width_definition": (
                "sigma=width_multiplier*source_only_median_distance;"
                "effective_gamma=1/(2*sigma^2)"
            ),
            "tie_break": [
                "mean_inner_bacc_desc",
                "worst_inner_bacc_desc",
                "n_components_asc",
                "width_preference_1_then_2_then_0.5",
                "logistic_c_asc",
                "candidate_id_asc",
            ],
        },
    )
    write_json(
        root / "manifests/baseline_identity_lock.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_baseline_lock_v1",
            "representation_id": REPRESENTATION_ID,
            "canonical_reference_root": str(config.canonical_reference_root),
            "canonical_results_sha256": _sha256_file(
                config.canonical_reference_root
                / "tables/classifier_tuned_source_results.csv"
            ),
            "canonical_predictions_sha256": _sha256_file(
                config.canonical_reference_root
                / "tables/classifier_tuned_predictions.csv"
            ),
            "per_outer_inherited_class_weight": {
                center: _class_weight_name(value)
                for center, value in class_weights.items()
            },
            "baseline_refit": False,
            "threshold_policy": "predict",
        },
    )
    reservation = _validation_reservation(config.manifest_path, config.heldout_centers)
    write_json(root / "manifests/validation_split_reservation_ledger.json", reservation)
    write_json(
        root / "provenance/input_artifacts.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_inputs_v1",
            "input_hashes": input_hashes,
            "uses_only_dataset_contract_canonical_b_cache_and_canonical_b_reference": True,
            "phase_b_test_artifact_is_not_an_input": True,
            "multiscale_c_is_not_an_input": True,
            "validation_feature_cache_is_not_an_input": True,
        },
    )

    selector_cells, selector_kernel_audits = run_source_inner_selection(
        x,
        y,
        centers,
        sample_ids,
        config=config,
        class_weights=class_weights,
    )
    summaries, selected = summarize_and_select(
        selector_cells, config.candidates, config.heldout_centers
    )
    outer_outputs = fit_outer_models(
        x,
        y,
        centers,
        sample_ids,
        case_ids,
        config=config,
        selected=selected,
        class_weights=class_weights,
        baseline_predictions=baseline_predictions,
    )
    materialized = _materialize_outer_tables(
        outer_outputs,
        baseline_predictions=baseline_predictions,
        centers=config.heldout_centers,
        primary_seed=config.primary_landmark_seed,
        stability_seeds=config.stability_landmark_seeds,
    )
    bootstrap = paired_case_bootstrap(
        materialized["primary_predictions"],
        materialized["baseline_predictions"],
        centers=config.heldout_centers,
        replicates=config.gate.bootstrap_replicates,
        seed=config.gate.bootstrap_seed,
    )
    decision = progression_decision(
        materialized["comparisons"],
        materialized["stability"],
        bootstrap,
        config.gate,
    )
    error_exchange = _error_exchange(
        materialized["primary_predictions"],
        materialized["baseline_predictions"],
    )
    centroid_exchange = _centroid_conflict_exchange(
        materialized["primary_predictions"]
    )

    write_csv_rows(root / "tables/source_inner_selector_cells.csv", selector_cells)
    write_csv_rows(root / "tables/source_inner_candidate_summary.csv", summaries)
    write_csv_rows(
        root / "tables/kernel_fit_audit.csv",
        selector_kernel_audits + materialized["outer_audits"],
    )
    write_csv_rows(root / "tables/outer_results.csv", materialized["results"])
    write_csv_rows(
        root / "tables/outer_predictions.csv",
        materialized["baseline_predictions"] + materialized["primary_predictions"],
    )
    write_csv_rows(
        root / "tables/seed_stability_predictions.csv",
        materialized["stability_predictions"],
    )
    write_csv_rows(
        root / "tables/paired_center_comparison.csv", materialized["comparisons"]
    )
    write_csv_rows(root / "tables/seed_stability.csv", materialized["stability"])
    write_csv_rows(root / "tables/error_exchange.csv", error_exchange)
    write_csv_rows(root / "tables/centroid_conflict_exchange.csv", centroid_exchange)
    write_json(root / "reports/conditional_bootstrap.json", bootstrap)
    write_json(root / "reports/progression_decision.json", decision)
    summary = _diagnostic_summary(
        decision,
        bootstrap,
        materialized["comparisons"],
        error_exchange,
        centroid_exchange,
        selected,
        reservation,
    )
    write_json(root / "reports/diagnostic_summary.json", summary)
    (root / "reports/diagnostic_report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_leakage_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "outer_target_used_for_selection_fit_scaler_gamma_or_landmarks": False,
            "source_inner_fit_excludes_outer_and_inner_centers": True,
            "class_weight_inherited_from_canonical_b_per_outer": True,
            "validation_features_generated": False,
            "validation_predictions_generated": False,
            "validation_labels_used_for_coverage_audit_only": True,
            "test_split_used": False,
            "diagnostic_surface_previously_inspected": True,
            "claim_scope": "diagnostic_only",
        },
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_runtime_v1",
            "status": "COMPLETE",
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_worker_processes": config.runtime.pair_jobs,
            "threads_per_process": config.runtime.threads_per_job,
            "gpu_used": False,
            "cpu_only_reason": "exact sklearn implementation is faster and more auditable here",
            "pair_preprocessing_frames": EXPECTED_PAIR_FRAMES,
            "primary_selection_nystroem_transforms": EXPECTED_NYSTROEM_TRANSFORMS,
            "selector_cells": len(selector_cells),
            "outer_final_fits": len(config.heldout_centers)
            * (1 + len(config.stability_landmark_seeds)),
            "python": platform.python_version(),
        },
    )
    write_json(
        root / "manifests/protocol_manifest.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_protocol_manifest_v1",
            "status": "PENDING_INDEPENDENT_VALIDATION",
            "protocol_hash": frozen["protocol_hash"],
            "claim_scope": "diagnostic_only",
            "representation_id": REPRESENTATION_ID,
            "diagnostic_surface_previously_inspected": True,
            "non_adoptive": True,
            "may_replace_canonical_reference": False,
            "may_feed_recipe_selection": False,
            "may_feed_deployable_selection": False,
            "new_center_generalization_claimed": False,
            "validation_scored": False,
            "test_scored": False,
            "uses_cvae": False,
            "uses_router": False,
        },
    )
    _write_content_index(root)
    from .validation import validate_nonlinear_probe_bundle

    pending = validate_nonlinear_probe_bundle(root, config=config, allow_pending=True)
    write_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_nonlinear_validation_v1",
            "status": "PASS",
            "validator": "validate_nonlinear_probe_bundle",
            "checks": pending,
        },
    )
    leakage = _read_json(root / "reports/leakage_provenance_report.json")
    leakage["status"] = "PASS"
    write_json(root / "reports/leakage_provenance_report.json", leakage)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    protocol["status"] = "PASS"
    write_json(root / "manifests/protocol_manifest.json", protocol)
    _write_content_index(root)
    validate_nonlinear_probe_bundle(root, config=config)
    return root


def _validate_canonical_reference(config: NonlinearProbeConfig) -> None:
    canonical_config = load_uniform_b_canonical_reference_config(
        config.canonical_reference_root / "config.resolved.yaml"
    )
    validate_uniform_b_canonical_reference_bundle(
        config.canonical_reference_root, config=canonical_config
    )


def _baseline_class_weights(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    output = {}
    for row in rows:
        payload = json.loads(row["selected_classifier_spec"])
        value = payload.get("class_weight")
        if value not in {None, "balanced"}:
            raise ProtocolError("Canonical-B class weight is not supported.")
        output[str(row["heldout_center"])] = value
    return output


def _materialize_outer_tables(
    outputs: Sequence[Mapping[str, object]],
    *,
    baseline_predictions: Mapping[str, Mapping[str, object]],
    centers: Sequence[str],
    primary_seed: int,
    stability_seeds: Sequence[int],
) -> dict[str, list[dict[str, object]]]:
    results = []
    baseline_table = []
    primary_table = []
    stability_predictions = []
    comparisons = []
    stability = []
    audits = []
    for output in outputs:
        outer = str(output["outer_center"])
        seed_outputs = {
            int(row["landmark_seed"]): row for row in output["seed_outputs"]
        }
        primary = seed_outputs[primary_seed]
        outer_baseline_source = [
            row
            for row in baseline_predictions.values()
            if str(row["heldout_center"]) == outer
        ]
        baseline_truth = np.asarray(
            [int(row["y_true"]) for row in outer_baseline_source]
        )
        baseline_pred = np.asarray(
            [int(row["y_pred"]) for row in outer_baseline_source]
        )
        baseline_metrics = binary_metrics(baseline_truth, baseline_pred)
        candidate = output["candidate"]
        selection = output["selection"]
        common = {
            "outer_center": outer,
            "n_train": output["n_train"],
            "n_eval": output["n_eval"],
            "fit_row_hash": output["fit_row_hash"],
            "eval_row_hash": output["eval_row_hash"],
        }
        results.append(
            {
                "schema_version": "midogpp_uniform_b_nonlinear_outer_result_v1",
                **common,
                "model_role": "canonical_b_linear_baseline",
                "candidate_id": "canonical_b_linear",
                "landmark_seed": "",
                **baseline_metrics,
                "diagnostic_only": True,
            }
        )
        results.append(
            {
                "schema_version": "midogpp_uniform_b_nonlinear_outer_result_v1",
                **common,
                "model_role": "canonical_b_nystroem_primary",
                "candidate_id": candidate.candidate_id,
                "width_multiplier": candidate.width_multiplier,
                "n_components": candidate.n_components,
                "logistic_c": candidate.logistic_c,
                "inherited_class_weight": _class_weight_name(output["class_weight"]),
                "landmark_seed": primary_seed,
                "source_inner_mean_bacc": selection["mean_inner_bacc"],
                "source_inner_worst_bacc": selection["worst_inner_bacc"],
                **primary["metrics"],
                "n_iter": primary["n_iter"],
                "selection_used_target_labels": False,
                "fit_used_target_center": False,
                "diagnostic_only": True,
            }
        )
        for row in outer_baseline_source:
            baseline_table.append(
                {
                    "schema_version": "midogpp_uniform_b_nonlinear_prediction_v1",
                    "model_role": "canonical_b_linear_baseline",
                    "outer_center": outer,
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "center": row["center"],
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "prob_pos": float(row["prob_pos"]),
                    "candidate_id": "canonical_b_linear",
                    "eval_row_hash": row["eval_row_hash"],
                    "target_labels_used_for_scoring_only": True,
                    "selection_used_target_labels": False,
                    "fit_used_target_center": False,
                }
            )
        for row in primary["predictions"]:
            primary_table.append({"model_role": "canonical_b_nystroem_primary", **row})
        comparison = {
            "schema_version": "midogpp_uniform_b_nonlinear_center_comparison_v1",
            "outer_center": outer,
            "n_eval": output["n_eval"],
            "baseline_bacc": baseline_metrics["bacc"],
            "nonlinear_bacc": primary["metrics"]["bacc"],
            "delta_bacc": primary["metrics"]["bacc"] - baseline_metrics["bacc"],
            "baseline_positive_recall": baseline_metrics["positive_recall"],
            "nonlinear_positive_recall": primary["metrics"]["positive_recall"],
            "delta_positive_recall": primary["metrics"]["positive_recall"]
            - baseline_metrics["positive_recall"],
            "baseline_specificity": baseline_metrics["specificity"],
            "nonlinear_specificity": primary["metrics"]["specificity"],
            "delta_specificity": primary["metrics"]["specificity"]
            - baseline_metrics["specificity"],
            "strict_win": primary["metrics"]["bacc"] > baseline_metrics["bacc"],
            "candidate_id": candidate.candidate_id,
        }
        comparisons.append(comparison)
        audits.extend(row["audit"] for row in seed_outputs.values())
        for seed in stability_seeds:
            seed_output = seed_outputs[seed]
            stability_predictions.extend(
                {"model_role": "canonical_b_nystroem_stability", **row}
                for row in seed_output["predictions"]
            )
            stability.append(
                {
                    "schema_version": "midogpp_uniform_b_nonlinear_seed_stability_v1",
                    "outer_center": outer,
                    "landmark_seed": seed,
                    "candidate_id": candidate.candidate_id,
                    "baseline_bacc": baseline_metrics["bacc"],
                    "nonlinear_bacc": seed_output["metrics"]["bacc"],
                    "delta_bacc": seed_output["metrics"]["bacc"]
                    - baseline_metrics["bacc"],
                    "positive_recall": seed_output["metrics"]["positive_recall"],
                    "specificity": seed_output["metrics"]["specificity"],
                }
            )
    return {
        "results": results,
        "baseline_predictions": baseline_table,
        "primary_predictions": primary_table,
        "stability_predictions": stability_predictions,
        "comparisons": comparisons,
        "stability": stability,
        "outer_audits": audits,
    }


def _error_exchange(
    primary: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline_by_id = {str(row["sample_id"]): row for row in baseline}
    groups: list[tuple[str, str, list[Mapping[str, object]]]] = [
        ("overall", "all", list(primary))
    ]
    for center in sorted({str(row["center"]) for row in primary}, key=int):
        groups.append(
            ("center", center, [row for row in primary if str(row["center"]) == center])
        )
    for label in (0, 1):
        groups.append(
            ("class", str(label), [row for row in primary if int(row["y_true"]) == label])
        )
    output = []
    for scope, value, rows in groups:
        counts = defaultdict(int)
        for row in rows:
            baseline_row = baseline_by_id[str(row["sample_id"])]
            baseline_correct = int(baseline_row["y_pred"]) == int(row["y_true"])
            nonlinear_correct = int(row["y_pred"]) == int(row["y_true"])
            key = {
                (False, True): "linear_wrong_nonlinear_correct",
                (True, False): "linear_correct_nonlinear_wrong",
                (False, False): "both_wrong",
                (True, True): "both_correct",
            }[(baseline_correct, nonlinear_correct)]
            counts[key] += 1
        output.append(
            {
                "schema_version": "midogpp_uniform_b_nonlinear_error_exchange_v1",
                "scope": scope,
                "scope_value": value,
                "n": len(rows),
                **{
                    key: counts[key]
                    for key in (
                        "linear_wrong_nonlinear_correct",
                        "linear_correct_nonlinear_wrong",
                        "both_wrong",
                        "both_correct",
                    )
                },
                "net_rescue": counts["linear_wrong_nonlinear_correct"]
                - counts["linear_correct_nonlinear_wrong"],
            }
        )
    return output


def _centroid_conflict_exchange(
    primary: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    groups = [("overall", "all", list(primary))]
    groups.extend(
        (
            "center",
            center,
            [row for row in primary if str(row["center"]) == center],
        )
        for center in sorted({str(row["center"]) for row in primary}, key=int)
    )
    for scope, value, rows in groups:
        conflict = [
            row
            for row in rows
            if bool(row["baseline_wrong"]) and bool(row["centroid_true_closer"])
        ]
        confident = [
            row for row in conflict if float(row["baseline_confidence"]) >= 0.75
        ]
        rescued = sum(int(row["y_pred"]) == int(row["y_true"]) for row in conflict)
        confident_rescued = sum(
            int(row["y_pred"]) == int(row["y_true"]) for row in confident
        )
        output.append(
            {
                "schema_version": "midogpp_uniform_b_nonlinear_centroid_conflict_v1",
                "scope": scope,
                "scope_value": value,
                "baseline_wrong_centroid_true_closer": len(conflict),
                "nonlinear_rescued": rescued,
                "rescue_rate": rescued / len(conflict) if conflict else 0.0,
                "confident_conflict_count": len(confident),
                "confident_conflict_rescued": confident_rescued,
                "confident_rescue_rate": (
                    confident_rescued / len(confident) if confident else 0.0
                ),
                "confidence_threshold": 0.75,
                "gate_component": False,
            }
        )
    return output


def _diagnostic_summary(
    decision: Mapping[str, object],
    bootstrap: Mapping[str, object],
    comparisons: Sequence[Mapping[str, object]],
    error_exchange: Sequence[Mapping[str, object]],
    centroid_exchange: Sequence[Mapping[str, object]],
    selected: Mapping[str, Mapping[str, object]],
    reservation: Mapping[str, object],
) -> dict[str, object]:
    baseline_mean = float(np.mean([float(row["baseline_bacc"]) for row in comparisons]))
    nonlinear_mean = float(np.mean([float(row["nonlinear_bacc"]) for row in comparisons]))
    overall_exchange = next(row for row in error_exchange if row["scope"] == "overall")
    overall_centroid = next(row for row in centroid_exchange if row["scope"] == "overall")
    return {
        "schema_version": "midogpp_uniform_b_nonlinear_diagnostic_summary_v1",
        "status": "COMPLETE",
        "decision": decision["decision"],
        "progression_gate_passed": decision["passed"],
        "equal_center_linear_b_bacc": baseline_mean,
        "equal_center_nonlinear_b_bacc": nonlinear_mean,
        "equal_center_delta_bacc": nonlinear_mean - baseline_mean,
        "strict_center_wins": sum(bool(row["strict_win"]) for row in comparisons),
        "worst_center_delta": min(float(row["delta_bacc"]) for row in comparisons),
        "bootstrap": dict(bootstrap),
        "progression": dict(decision),
        "overall_error_exchange": dict(overall_exchange),
        "overall_centroid_conflict": dict(overall_centroid),
        "selected_candidates": {
            center: {
                key: value
                for key, value in row.items()
                if key
                in {
                    "candidate_id",
                    "width_multiplier",
                    "n_components",
                    "logistic_c",
                    "mean_inner_bacc",
                    "worst_inner_bacc",
                }
            }
            for center, row in selected.items()
        },
        "validation_reservation_status": reservation["status"],
        "validation_scored": False,
        "diagnostic_only": True,
        "scientific_interpretation": (
            "Passing supports a nonlinear-boundary limitation hypothesis but does not "
            "prove that B is sufficient. Failing rejects only this frozen Nyström grid "
            "and advances B-spatial as the next bounded representation diagnostic."
        ),
    }


def _render_report(summary: Mapping[str, object]) -> str:
    exchange = summary["overall_error_exchange"]
    centroid = summary["overall_centroid_conflict"]
    progression = summary["progression"]
    return "\n".join(
        [
            "# Canonical Uniform-B Nonlinear-Boundary Probe",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            "This is a Stage-90 diagnostic over an already inspected train surface. "
            "It cannot replace the canonical reference or support a new-center claim.",
            "",
            "## Primary result",
            "",
            f"- Linear B equal-center BACC: `{float(summary['equal_center_linear_b_bacc']):.6f}`",
            f"- Nonlinear B equal-center BACC: `{float(summary['equal_center_nonlinear_b_bacc']):.6f}`",
            f"- Delta: `{float(summary['equal_center_delta_bacc']):+.6f}`",
            f"- Strict center wins: `{summary['strict_center_wins']}/9`",
            f"- Worst-center delta: `{float(summary['worst_center_delta']):+.6f}`",
            f"- Progression gate passed: `{str(summary['progression_gate_passed']).lower()}`",
            "",
            "## Error exchange",
            "",
            f"- Linear wrong / nonlinear correct: `{exchange['linear_wrong_nonlinear_correct']}`",
            f"- Linear correct / nonlinear wrong: `{exchange['linear_correct_nonlinear_wrong']}`",
            f"- Both wrong: `{exchange['both_wrong']}`",
            f"- Both correct: `{exchange['both_correct']}`",
            "",
            "## Centroid-conflict test",
            "",
            f"- Baseline-wrong but true-centroid-closer cases: `{centroid['baseline_wrong_centroid_true_closer']}`",
            f"- Nonlinear rescues: `{centroid['nonlinear_rescued']}` "
            f"(`{float(centroid['rescue_rate']):.3f}`)",
            "",
            "## Gate",
            "",
            f"`{json.dumps(progression['checks'], sort_keys=True)}`",
            "",
            "The reserved validation split was not featurized or scored and remains "
            "below the project minimum of ten cases per center.",
            "",
        ]
    )


def _validation_reservation(
    manifest_path: Path, eligible_centers: Sequence[str]
) -> dict[str, object]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    by_split = defaultdict(list)
    for row in rows:
        by_split[str(row["split"]).strip().lower()].append(row)
    val = [
        row for row in by_split["val"] if str(row["center"]) in set(eligible_centers)
    ]
    center_rows = {}
    for center in eligible_centers:
        selected = [row for row in val if str(row["center"]) == center]
        center_rows[center] = {
            "rows": len(selected),
            "cases": len({row["case_id"] for row in selected}),
            "class_counts": {
                label: sum(int(row["label"]) == label for row in selected)
                for label in (0, 1)
            },
        }
    val_samples = {row["sample_id"] for row in val}
    val_cases = {row["case_id"] for row in val}
    train_samples = {row["sample_id"] for row in by_split["train"]}
    test_samples = {row["sample_id"] for row in by_split["test"]}
    train_cases = {row["case_id"] for row in by_split["train"]}
    test_cases = {row["case_id"] for row in by_split["test"]}
    return {
        "schema_version": "midogpp_uniform_b_validation_reservation_v1",
        "status": "RESERVED_UNSCORED_BUT_BELOW_CONFIRMATION_CASE_MINIMUM",
        "eligible_row_count": len(val),
        "eligible_case_count": len(val_cases),
        "eligible_centers": list(eligible_centers),
        "per_center_aggregate_coverage": center_rows,
        "sample_id_hash": _string_hash(sorted(val_samples)),
        "case_id_hash": _string_hash(sorted(val_cases)),
        "sample_overlap_train": len(val_samples & train_samples),
        "sample_overlap_test": len(val_samples & test_samples),
        "case_overlap_train": len(val_cases & train_cases),
        "case_overlap_test": len(val_cases & test_cases),
        "project_eval_cases_min_per_center": 10,
        "all_centers_meet_case_minimum": all(
            int(row["cases"]) >= 10 for row in center_rows.values()
        ),
        "features_generated": False,
        "predictions_generated": False,
        "sample_level_predictions_inspected": False,
        "labels_used_for_aggregate_coverage_audit_only": True,
        "formal_confirmation_ready": False,
    }


def _frozen_protocol(
    config: NonlinearProbeConfig,
    input_hashes: Mapping[str, str],
    class_weights: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "midogpp_uniform_b_nonlinear_frozen_protocol_v1",
        "experiment_name": config.name,
        "heldout_centers": list(config.heldout_centers),
        "representation_id": REPRESENTATION_ID,
        "expected_feature_dim": config.expected_feature_dim,
        "candidate_grid": [candidate.to_payload() for candidate in config.candidates],
        "primary_landmark_seed": config.primary_landmark_seed,
        "stability_landmark_seeds": list(config.stability_landmark_seeds),
        "gamma_sample_seed": config.gamma_sample_seed,
        "gamma_sample_cap": config.gamma_sample_cap,
        "gamma_formula": "sigma=m*d_median; gamma=1/(2*sigma^2)",
        "median_estimator": "upper_triangle_nonzero_pairwise_distances",
        "per_outer_inherited_class_weight": {
            center: _class_weight_name(value)
            for center, value in class_weights.items()
        },
        "selection": (
            "outer H; source-inner LODO V among other eight; fit scaler, gamma "
            "sample, landmarks, and classifier on remaining seven only"
        ),
        "primary_estimand": "equal_center_mean_bacc_nonlinear_minus_linear_b",
        "gate": config.gate.__dict__,
        "bootstrap_supportive_only": True,
        "claim_scope": "diagnostic_only",
        "validation_scored": False,
        "input_hashes": dict(sorted(input_hashes.items())),
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def _input_hashes(
    config: NonlinearProbeConfig, cache_hash: str, manifest_hash: str
) -> dict[str, str]:
    return {
        "dataset_manifest": manifest_hash,
        "canonical_b_feature_cache": cache_hash,
        "canonical_b_reference_content_index": _sha256_file(
            config.canonical_reference_root / "manifests/content_index.json"
        ),
        "canonical_b_results": _sha256_file(
            config.canonical_reference_root / "tables/classifier_tuned_source_results.csv"
        ),
        "canonical_b_predictions": _sha256_file(
            config.canonical_reference_root / "tables/classifier_tuned_predictions.csv"
        ),
    }


def _write_content_index(root: Path) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        if relative == "manifests/content_index.json":
            continue
        files.append({"path": relative, "sha256": _sha256_file(path)})
    payload = {
        "schema_version": "midogpp_uniform_b_nonlinear_content_index_v1",
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
        raise ProtocolError(f"Uniform-B nonlinear JSON must be an object: {path}.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_hash(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _class_weight_name(value: object) -> str:
    return "none" if value is None else str(value)
