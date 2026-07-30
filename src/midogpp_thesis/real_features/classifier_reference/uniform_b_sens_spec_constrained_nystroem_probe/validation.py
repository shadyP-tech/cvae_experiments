"""Independent fail-closed validation for the constrained Nyström bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from ..uniform_b_nonlinear_probe.statistics import binary_metrics
from .config import ConstrainedNystroemConfig
from .estimator import select_constrained_candidates


REQUIRED = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/blend_capacity_grid_lock.json",
    "manifests/inherited_candidate_lock_index.json",
    "manifests/source_only_candidate_locks.json",
    "manifests/content_index.json",
    "tables/source_inner_linear_baseline_cells.csv",
    "tables/source_inner_blend_cells.csv",
    "tables/source_inner_base_scores.csv",
    "tables/source_inner_candidate_summary.csv",
    "tables/blend_capacity_decisions.csv",
    "tables/outer_results.csv",
    "tables/outer_predictions.csv",
    "tables/stability_predictions.csv",
    "tables/center_comparison.csv",
    "tables/error_exchange.csv",
    "reports/constraint_feasibility.json",
    "reports/paired_bootstrap.json",
    "reports/progression_decision.json",
    "reports/diagnostic_summary.json",
    "reports/diagnostic_report.md",
    "reports/runtime_summary.json",
    "reports/leakage_provenance_report.json",
)


def validate_constrained_nystroem_bundle(
    root: str | Path,
    *,
    config: ConstrainedNystroemConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED)
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(item for item in required if not (path / item).is_file())
    if missing:
        raise ProtocolError(f"Constrained-Nyström bundle is incomplete: {missing}.")
    frozen = _json(path / "manifests/frozen_protocol_snapshot.json")
    protocol = _json(path / "manifests/protocol_manifest.json")
    leakage = _json(path / "reports/leakage_provenance_report.json")
    grid = _json(path / "manifests/blend_capacity_grid_lock.json")
    if (
        stable_hash({key: value for key, value in frozen.items() if key != "protocol_hash"})
        != frozen.get("protocol_hash")
        or protocol.get("protocol_hash") != frozen.get("protocol_hash")
        or protocol.get("claim_scope") != "diagnostic_only"
        or protocol.get("selection_used_current_outer_labels") is not False
        or protocol.get("independent_outer_confirmation") is not False
        or protocol.get("validation_scored") is not False
        or protocol.get("test_scored") is not False
        or leakage.get("threshold") != 0.5
        or leakage.get("threshold_selected") is not False
        or leakage.get("center_specific_thresholds") is not False
        or leakage.get("robust_outer_tables_read_for_selection") is not False
        or grid.get("fallback_alpha") != 0.0
        or grid.get("fallback_role") != "exact_linear_b"
    ):
        raise ProtocolError("Constrained-Nyström protocol firewall failed.")

    baseline = _csv(path / "tables/source_inner_linear_baseline_cells.csv")
    cells = _csv(path / "tables/source_inner_blend_cells.csv")
    scores = _csv(path / "tables/source_inner_base_scores.csv")
    summaries = _csv(path / "tables/source_inner_candidate_summary.csv")
    decisions = _csv(path / "tables/blend_capacity_decisions.csv")
    primary = _csv(path / "tables/outer_predictions.csv")
    stability = _csv(path / "tables/stability_predictions.csv")
    comparisons = _csv(path / "tables/center_comparison.csv")
    exchange = _csv(path / "tables/error_exchange.csv")
    if (
        len(baseline) != 9 * 8
        or len(cells)
        != 9 * 8 * len(config.objectives) * len(config.alphas)
        or len(scores) != 8 * config.expected_rows
        or len(summaries)
        != 9 * len(config.objectives) * len(config.alphas)
        or len(decisions) != 9
        or len(primary) != config.expected_rows
        or len(stability) != 2 * config.expected_rows
        or len(comparisons) != 9 * 3
        or len(exchange) != config.expected_rows
    ):
        raise ProtocolError("Constrained-Nyström artifact cardinality drifted.")
    _validate_source_inner_rows(baseline, cells, config)
    _validate_source_inner_replay(baseline, scores, config)
    _validate_blend_reconstruction(baseline, cells, scores)
    _validate_selection(cells, decisions, config)
    _validate_outer(primary, stability, comparisons, decisions, config)
    from .independent_checks import run_independent_checks

    independent = run_independent_checks(path, config)
    _validate_content(path)
    checks = {
        "status": "PASS",
        "linear_cells": len(baseline),
        "blend_cells": len(cells),
        "base_score_rows": len(scores),
        "candidate_summaries": len(summaries),
        "selection_locks": len(decisions),
        "primary_predictions": len(primary),
        "stability_predictions": len(stability),
        "validation_scored": False,
        "test_scored": False,
        "decision": _json(path / "reports/progression_decision.json")["decision"],
        "independent_reconstruction": independent,
    }
    if not allow_pending:
        report = _json(path / "reports/validation_report.json")
        if (
            protocol.get("status") != "PASS"
            or leakage.get("status") != "PASS"
            or report.get("status") != "PASS"
            or report.get("checks") != checks
        ):
            raise ProtocolError("Constrained-Nyström final validation failed.")
    return checks


def _validate_source_inner_replay(
    baseline: list[dict[str, str]],
    scores: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> None:
    if config.source_inner_replay_root is None:
        return
    expected_baseline = _csv(
        config.source_inner_replay_root
        / "tables/source_inner_linear_baseline_cells.csv"
    )
    expected_scores = _csv(
        config.source_inner_replay_root / "tables/source_inner_base_scores.csv"
    )
    if baseline != expected_baseline or scores != expected_scores:
        raise ProtocolError(
            "Bounded-shrinkage source-inner base-score replay drifted."
        )


def _validate_source_inner_rows(
    baseline: list[dict[str, str]],
    cells: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> None:
    expected_centers = set(config.heldout_centers)
    baseline_keys = set()
    for row in baseline:
        outer, inner = row["outer_center"], row["inner_center"]
        baseline_keys.add((outer, inner))
        if (
            set(json.loads(row["train_centers"]))
            != expected_centers.difference({outer, inner})
            or row["selection_used_outer_labels"].lower() != "false"
            or row["fit_used_outer_or_inner_center"].lower() != "false"
        ):
            raise ProtocolError("Constrained linear-cell leakage detected.")
    expected_keys = {
        (outer, inner)
        for outer in config.heldout_centers
        for inner in config.heldout_centers
        if inner != outer
    }
    if baseline_keys != expected_keys:
        raise ProtocolError("Constrained linear-cell coverage is not exact.")
    candidate_keys = set()
    for row in cells:
        key = (
            row["outer_center"],
            row["inner_center"],
            row["objective"],
            float(row["alpha"]),
        )
        candidate_keys.add(key)
        if (
            float(row["threshold"]) != 0.5
            or row["selection_used_outer_labels"].lower() != "false"
            or row["fit_used_outer_or_inner_center"].lower() != "false"
        ):
            raise ProtocolError("Constrained blend-cell firewall failed.")
    expected_candidate_keys = {
        (outer, inner, objective, alpha)
        for outer, inner in expected_keys
        for objective in config.objectives
        for alpha in config.alphas
    }
    if candidate_keys != expected_candidate_keys:
        raise ProtocolError("Constrained blend-cell coverage is not exact.")


def _validate_blend_reconstruction(
    baseline: list[dict[str, str]],
    cells: list[dict[str, str]],
    scores: list[dict[str, str]],
) -> None:
    score_index: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in scores:
        score_index.setdefault((row["outer_center"], row["inner_center"]), []).append(row)
    baseline_index = {
        (row["outer_center"], row["inner_center"]): row for row in baseline
    }
    for cell in cells:
        key = (cell["outer_center"], cell["inner_center"])
        rows = score_index[key]
        objective = cell["objective"]
        alpha = float(cell["alpha"])
        truth = np.asarray([int(row["y_true"]) for row in rows])
        linear = np.asarray([float(row["linear_logit"]) for row in rows])
        nonlinear = np.asarray([float(row[f"{objective}_logit"]) for row in rows])
        mixed = linear + alpha * (nonlinear - linear)
        metrics = binary_metrics(truth, (mixed >= 0.0).astype(np.int8))
        base = baseline_index[key]
        for field in ("bacc", "positive_recall", "specificity"):
            if not np.isclose(float(cell[field]), metrics[field], atol=1e-12):
                raise ProtocolError("Constrained blend reconstruction failed.")
        if (
            not np.isclose(
                float(cell["delta_bacc"]),
                metrics["bacc"] - float(base["bacc"]),
                atol=1e-12,
            )
            or not np.isclose(
                float(cell["delta_recall"]),
                metrics["positive_recall"] - float(base["positive_recall"]),
                atol=1e-12,
            )
            or not np.isclose(
                float(cell["delta_specificity"]),
                metrics["specificity"] - float(base["specificity"]),
                atol=1e-12,
            )
        ):
            raise ProtocolError("Constrained blend delta reconstruction failed.")


def _validate_selection(
    cells: list[dict[str, str]],
    decisions: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> None:
    _, expected = select_constrained_candidates(cells, config)
    observed = {row["outer_center"]: row for row in decisions}
    if set(observed) != set(config.heldout_centers):
        raise ProtocolError("Constrained selection-lock coverage drifted.")
    for center in config.heldout_centers:
        row = observed[center]
        lock = expected[center]
        if (
            row["objective"] != str(lock["objective"])
            or not np.isclose(float(row["alpha"]), float(lock["alpha"]))
            or row["candidate_id"] != str(lock["candidate_id"])
            or (row["fallback"].lower() == "true") != bool(lock["fallback"])
        ):
            raise ProtocolError("Constrained selection lock is not reproducible.")


def _validate_outer(
    primary: list[dict[str, str]],
    stability: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    decisions: list[dict[str, str]],
    config: ConstrainedNystroemConfig,
) -> None:
    locks = {row["outer_center"]: row for row in decisions}
    primary_ids = [row["sample_id"] for row in primary]
    if len(set(primary_ids)) != config.expected_rows:
        raise ProtocolError("Constrained primary prediction IDs are not unique.")
    for seed in config.stability_seeds:
        rows = [row for row in stability if int(row["landmark_seed"]) == seed]
        if len(rows) != config.expected_rows or set(
            row["sample_id"] for row in rows
        ) != set(primary_ids):
            raise ProtocolError("Constrained stability prediction coverage drifted.")
    all_rows = primary + stability
    for row in all_rows:
        lock = locks[row["outer_center"]]
        linear = float(row["linear_logit"])
        nonlinear = float(row["nonlinear_logit"])
        alpha = float(lock["alpha"])
        expected_mixed = linear + alpha * (nonlinear - linear)
        if (
            row["center"] != row["outer_center"]
            or row["objective"] != lock["objective"]
            or not np.isclose(float(row["alpha"]), alpha)
            or not np.isclose(float(row["mixed_logit"]), expected_mixed, atol=1e-12)
            or int(row["y_pred"]) != int(expected_mixed >= 0.0)
            or float(row["threshold"]) != 0.5
            or row["selection_used_outer_labels"].lower() != "false"
        ):
            raise ProtocolError("Constrained outer lock application failed.")
    for comparison in comparisons:
        rows = [
            row
            for row in all_rows
            if row["outer_center"] == comparison["outer_center"]
            and int(row["landmark_seed"]) == int(comparison["seed"])
        ]
        metrics = binary_metrics(
            np.asarray([int(row["y_true"]) for row in rows]),
            np.asarray([int(row["y_pred"]) for row in rows]),
        )
        if not np.isclose(float(comparison["bacc"]), metrics["bacc"]):
            raise ProtocolError("Constrained outer metric reconstruction failed.")


def _validate_content(root: Path) -> None:
    payload = _json(root / "manifests/content_index.json")
    if (
        stable_hash({key: value for key, value in payload.items() if key != "content_hash"})
        != payload.get("content_hash")
    ):
        raise ProtocolError("Constrained content hash drifted.")
    observed = set()
    for row in payload["files"]:
        member = root / row["path"]
        if not member.is_file() or _sha256(member) != row["sha256"]:
            raise ProtocolError(f"Constrained artifact member drifted: {row['path']}.")
        observed.add(row["path"])
    expected = {
        str(member.relative_to(root))
        for member in root.rglob("*")
        if member.is_file() and member.name != "content_index.json"
    }
    if observed != expected:
        raise ProtocolError("Constrained content-index coverage drifted.")


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
