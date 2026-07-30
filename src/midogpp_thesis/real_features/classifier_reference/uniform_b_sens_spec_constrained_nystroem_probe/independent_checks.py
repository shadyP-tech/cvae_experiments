"""Cross-artifact reconstruction checks kept separate from the runner."""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import expit

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from ..uniform_b_nonlinear_probe.statistics import (
    binary_metrics,
    paired_case_bootstrap,
)
from .config import ConstrainedNystroemConfig, load_constrained_nystroem_config
from .estimator import (
    load_canonical_specs,
    load_selected_nystroem_candidates,
    select_constrained_candidates,
)


def run_independent_checks(
    root: Path, config: ConstrainedNystroemConfig
) -> dict[str, object]:
    resolved = load_constrained_nystroem_config(root / "config.resolved.yaml")
    if resolved != config:
        raise ProtocolError("Resolved constrained config is not the validated config.")
    provenance = _json(root / "provenance/input_artifacts.json")
    if provenance.get("input_hashes") != _actual_input_hashes(config):
        raise ProtocolError("Constrained input or lineage hashes drifted.")
    frozen = _json(root / "manifests/frozen_protocol_snapshot.json")
    observed_config = dict(frozen.get("config", {}))
    expected_config = dict(_jsonable(asdict(config)))
    if config.source_inner_replay_root is None:
        observed_config.pop("source_inner_replay_root", None)
        expected_config.pop("source_inner_replay_root", None)
    if (
        frozen.get("experiment") != config.name
        or observed_config != expected_config
        or frozen.get("threshold") != 0.5
        or frozen.get("threshold_selected") is not False
    ):
        raise ProtocolError("Constrained frozen config binding drifted.")
    grid = _json(root / "manifests/blend_capacity_grid_lock.json")
    inherited = _json(root / "manifests/inherited_candidate_lock_index.json")
    if (
        grid.get("objectives") != list(config.objectives)
        or grid.get("alphas") != list(config.alphas)
        or grid.get("fallback_alpha") != config.fallback_alpha
        or grid.get("fallback_role") != config.fallback_role
        or inherited.get("robust_lineage_only") is not True
        or inherited.get("robust_outer_tables_read_for_selection") is not False
    ):
        raise ProtocolError("Constrained inherited/grid lock drifted.")

    baseline = _csv(root / "tables/source_inner_linear_baseline_cells.csv")
    cells = _csv(root / "tables/source_inner_blend_cells.csv")
    scores = _csv(root / "tables/source_inner_base_scores.csv")
    summaries = _csv(root / "tables/source_inner_candidate_summary.csv")
    decisions = _csv(root / "tables/blend_capacity_decisions.csv")
    expected_summaries, expected_selected = select_constrained_candidates(
        cells, config
    )
    _compare_summaries(summaries, expected_summaries)
    _compare_selection(root, decisions, expected_selected)
    _check_endpoint_replay(baseline, cells, config)
    _check_base_score_coverage(scores, config)

    primary = _csv(root / "tables/outer_predictions.csv")
    stability = _csv(root / "tables/stability_predictions.csv")
    results = _csv(root / "tables/outer_results.csv")
    comparisons = _csv(root / "tables/center_comparison.csv")
    exchange = _csv(root / "tables/error_exchange.csv")
    canonical = _csv(config.canonical_root / "tables/classifier_tuned_predictions.csv")
    baseline_outer = _check_outer_predictions(
        primary, stability, canonical, decisions, config
    )
    _check_outer_metrics(
        primary,
        stability,
        baseline_outer,
        results,
        comparisons,
        config,
    )
    _check_exchange(exchange, primary, baseline_outer)
    _check_reports(
        root,
        comparisons,
        expected_selected,
        expected_summaries,
        primary,
        baseline_outer,
        config,
    )
    return {
        "input_hashes_recomputed": True,
        "resolved_config_bound": True,
        "candidate_selection_reconstructed": True,
        "endpoint_replays_checked": True,
        "outer_tables_reconstructed": True,
        "decision_reconstructed": True,
    }


def _compare_summaries(
    observed: list[dict[str, str]],
    expected: list[dict[str, object]],
) -> None:
    key = lambda row: (
        str(row["outer_center"]),
        str(row["objective"]),
        float(row["alpha"]),
    )
    observed_index = {key(row): row for row in observed}
    expected_index = {key(row): row for row in expected}
    if set(observed_index) != set(expected_index):
        raise ProtocolError("Constrained candidate-summary keys drifted.")
    numeric = (
        "mean_inner_bacc",
        "mean_delta_bacc",
        "mean_delta_recall",
        "mean_delta_specificity",
        "worst_delta_bacc",
        "worst_direction_delta",
    )
    for item, expected_row in expected_index.items():
        row = observed_index[item]
        if (
            row["candidate_id"] != expected_row["candidate_id"]
            or (row["hard_feasible"].lower() == "true")
            != bool(expected_row["hard_feasible"])
            or (row["selected"].lower() == "true")
            != bool(expected_row["selected"])
            or any(
                not np.isclose(float(row[field]), float(expected_row[field]), atol=1e-12)
                for field in numeric
            )
        ):
            raise ProtocolError("Constrained candidate summary is not reproducible.")


def _compare_selection(
    root: Path,
    decisions: list[dict[str, str]],
    expected: Mapping[str, Mapping[str, object]],
) -> None:
    observed = {row["outer_center"]: row for row in decisions}
    if set(observed) != set(expected):
        raise ProtocolError("Constrained decision-center coverage drifted.")
    for center, lock in expected.items():
        row = observed[center]
        if (
            row["candidate_id"] != str(lock["candidate_id"])
            or row["objective"] != str(lock["objective"])
            or not np.isclose(float(row["alpha"]), float(lock["alpha"]))
            or (row["fallback"].lower() == "true") != bool(lock["fallback"])
            or int(row["feasible_nonlinear_candidates"])
            != int(lock["feasible_nonlinear_candidates"])
        ):
            raise ProtocolError("Constrained decision row drifted.")
    locked = _json(root / "manifests/source_only_candidate_locks.json")
    if (
        locked.get("selected") != expected
        or locked.get("selection_hash") != stable_hash(expected)
        or locked.get("outer_labels_read_before_lock") is not False
    ):
        raise ProtocolError("Constrained JSON selection lock drifted.")


def _check_endpoint_replay(
    baseline: list[dict[str, str]],
    cells: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> None:
    specs = load_canonical_specs(config.canonical_root)
    for row in baseline:
        expected = specs[row["outer_center"]]["source_inner_bacc_vector"][
            row["inner_center"]
        ]
        if not np.isclose(float(row["bacc"]), float(expected), atol=1e-12):
            raise ProtocolError("Independent linear-B source-inner replay failed.")
    kernels = load_selected_nystroem_candidates(config.nonlinear_root)
    prior = _csv(config.nonlinear_root / "tables/source_inner_selector_cells.csv")
    prior_index = {
        (row["outer_center"], row["inner_center"], row["candidate_id"]): row
        for row in prior
    }
    for row in cells:
        if row["objective"] != "canonical_class_weight" or not np.isclose(
            float(row["alpha"]), 1.0
        ):
            continue
        expected = prior_index[
            (
                row["outer_center"],
                row["inner_center"],
                kernels[row["outer_center"]].candidate_id,
            )
        ]
        if any(
            not np.isclose(float(row[field]), float(expected[field]), atol=1e-12)
            for field in ("bacc", "positive_recall", "specificity")
        ):
            raise ProtocolError("Independent Nyström endpoint replay failed.")


def _check_base_score_coverage(
    scores: list[dict[str, str]], config: ConstrainedNystroemConfig
) -> None:
    keys = [
        (row["outer_center"], row["inner_center"], row["sample_id"])
        for row in scores
    ]
    if len(set(keys)) != len(scores):
        raise ProtocolError("Constrained base-score rows are not unique.")
    counts: dict[tuple[str, str], int] = {}
    for outer, inner, _ in keys:
        counts[(outer, inner)] = counts.get((outer, inner), 0) + 1
    if set(counts) != {
        (outer, inner)
        for outer in config.heldout_centers
        for inner in config.heldout_centers
        if outer != inner
    }:
        raise ProtocolError("Constrained base-score fold coverage drifted.")


def _check_outer_predictions(
    primary: list[dict[str, str]],
    stability: list[dict[str, str]],
    canonical: list[dict[str, str]],
    decisions: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> list[dict[str, object]]:
    canonical_index = {row["sample_id"]: row for row in canonical}
    if len(canonical_index) != config.expected_rows:
        raise ProtocolError("Canonical outer baseline IDs drifted.")
    if {row["sample_id"] for row in primary} != set(canonical_index):
        raise ProtocolError("Constrained primary sample coverage drifted.")
    locks = {row["outer_center"]: row for row in decisions}
    for row in primary + stability:
        reference = canonical_index[row["sample_id"]]
        lock = locks[row["outer_center"]]
        mixed = float(row["linear_logit"]) + float(lock["alpha"]) * (
            float(row["nonlinear_logit"]) - float(row["linear_logit"])
        )
        if (
            row["outer_center"] != reference["heldout_center"]
            or row["center"] != row["outer_center"]
            or int(row["y_true"]) != int(reference["y_true"])
            or row["objective"] != lock["objective"]
            or not np.isclose(float(row["alpha"]), float(lock["alpha"]))
            or not np.isclose(float(row["mixed_logit"]), mixed, atol=1e-12)
            or not np.isclose(float(row["prob_pos"]), expit(mixed), atol=1e-12)
            or int(row["y_pred"]) != int(mixed >= 0.0)
            or row["fit_used_outer_center"].lower() != "false"
        ):
            raise ProtocolError("Independent constrained outer prediction check failed.")
    return [
        {
            "sample_id": row["sample_id"],
            "case_id": row["case_id"],
            "center": row["heldout_center"],
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }
        for row in canonical
    ]


def _check_outer_metrics(
    primary: list[dict[str, str]],
    stability: list[dict[str, str]],
    baseline: list[dict[str, object]],
    results: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> None:
    all_rows = primary + stability
    result_index = {
        (row["outer_center"], int(row["seed"])): row for row in results
    }
    comparison_index = {
        (row["outer_center"], int(row["seed"])): row for row in comparisons
    }
    expected_keys = {
        (center, seed)
        for center in config.heldout_centers
        for seed in (config.primary_seed, *config.stability_seeds)
    }
    if set(result_index) != expected_keys or set(comparison_index) != expected_keys:
        raise ProtocolError("Constrained outer metric key coverage drifted.")
    baseline_by_center = {
        center: [row for row in baseline if row["center"] == center]
        for center in config.heldout_centers
    }
    for center, seed in expected_keys:
        rows = [
            row
            for row in all_rows
            if row["outer_center"] == center
            and int(row["landmark_seed"]) == seed
        ]
        metrics = binary_metrics(
            np.asarray([int(row["y_true"]) for row in rows]),
            np.asarray([int(row["y_pred"]) for row in rows]),
        )
        base_rows = baseline_by_center[center]
        base_metrics = binary_metrics(
            np.asarray([int(row["y_true"]) for row in base_rows]),
            np.asarray([int(row["y_pred"]) for row in base_rows]),
        )
        for artifact in (result_index[(center, seed)], comparison_index[(center, seed)]):
            for field, value in metrics.items():
                if field in artifact and not np.isclose(
                    float(artifact[field]), float(value), atol=1e-12
                ):
                    raise ProtocolError("Constrained outer metric drifted.")
        comparison = comparison_index[(center, seed)]
        for field, metric_field in (
            ("delta_bacc", "bacc"),
            ("delta_recall", "positive_recall"),
            ("delta_specificity", "specificity"),
        ):
            expected_delta = metrics[metric_field] - base_metrics[metric_field]
            if not np.isclose(float(comparison[field]), expected_delta, atol=1e-12):
                raise ProtocolError("Constrained comparison delta drifted.")


def _check_exchange(
    exchange: list[dict[str, str]],
    primary: list[dict[str, str]],
    baseline: list[dict[str, object]],
) -> None:
    candidate = {row["sample_id"]: row for row in primary}
    base = {str(row["sample_id"]): row for row in baseline}
    observed = {row["sample_id"]: row for row in exchange}
    if set(observed) != set(candidate) or set(base) != set(candidate):
        raise ProtocolError("Constrained error-exchange coverage drifted.")
    for sample_id, row in candidate.items():
        candidate_correct = int(row["y_pred"]) == int(row["y_true"])
        baseline_correct = int(base[sample_id]["y_pred"]) == int(row["y_true"])
        expected = (
            "rescue"
            if candidate_correct and not baseline_correct
            else "regression"
            if baseline_correct and not candidate_correct
            else "both_correct"
            if baseline_correct
            else "both_wrong"
        )
        if observed[sample_id]["outcome"] != expected:
            raise ProtocolError("Constrained error exchange drifted.")


def _check_reports(
    root: Path,
    comparisons: list[dict[str, str]],
    selected: Mapping[str, Mapping[str, object]],
    summaries: list[dict[str, object]],
    primary: list[dict[str, str]],
    baseline: list[dict[str, object]],
    config: ConstrainedNystroemConfig,
) -> None:
    from .runner import _feasibility_report, _progression_decision

    expected_feasibility = _feasibility_report(summaries, selected, config)
    if _json(root / "reports/constraint_feasibility.json") != expected_feasibility:
        raise ProtocolError("Constrained feasibility report drifted.")
    expected_progression = _progression_decision(comparisons, selected, config)
    observed_progression = _json(root / "reports/progression_decision.json")
    observed_bootstrap = observed_progression.pop("bootstrap_supportive_only")
    if observed_progression != expected_progression:
        raise ProtocolError("Constrained progression decision drifted.")
    expected_bootstrap = paired_case_bootstrap(
        primary,
        baseline,
        centers=config.heldout_centers,
        replicates=2000,
        seed=42,
    )
    if observed_bootstrap != expected_bootstrap:
        raise ProtocolError("Constrained paired bootstrap drifted.")


def _actual_input_hashes(config: ConstrainedNystroemConfig) -> dict[str, str]:
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


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json(path: Path) -> dict[str, object]:
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
