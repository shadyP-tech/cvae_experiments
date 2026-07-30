"""Independent fail-closed validation of nonlinear-probe artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from .config import (
    EXPECTED_NYSTROEM_TRANSFORMS,
    EXPECTED_PAIR_FRAMES,
    EXPECTED_SELECTOR_CELLS,
    NonlinearProbeConfig,
)
from .statistics import (
    binary_metrics,
    paired_case_bootstrap,
    progression_decision,
)


REQUIRED = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/frozen_protocol_snapshot.json",
    "manifests/protocol_manifest.json",
    "manifests/nystroem_grid_lock.json",
    "manifests/baseline_identity_lock.json",
    "manifests/validation_split_reservation_ledger.json",
    "manifests/content_index.json",
    "tables/source_inner_selector_cells.csv",
    "tables/source_inner_candidate_summary.csv",
    "tables/kernel_fit_audit.csv",
    "tables/outer_results.csv",
    "tables/outer_predictions.csv",
    "tables/seed_stability_predictions.csv",
    "tables/paired_center_comparison.csv",
    "tables/seed_stability.csv",
    "tables/error_exchange.csv",
    "tables/centroid_conflict_exchange.csv",
    "reports/conditional_bootstrap.json",
    "reports/progression_decision.json",
    "reports/diagnostic_summary.json",
    "reports/diagnostic_report.md",
    "reports/leakage_provenance_report.json",
    "reports/runtime_summary.json",
)


def validate_nonlinear_probe_bundle(
    root: str | Path,
    *,
    config: NonlinearProbeConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    path = Path(root)
    required = set(REQUIRED)
    if not allow_pending:
        required.add("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Uniform-B nonlinear bundle is incomplete: {missing}.")
    frozen = _read_json(path / "manifests/frozen_protocol_snapshot.json")
    protocol = _read_json(path / "manifests/protocol_manifest.json")
    baseline_lock = _read_json(path / "manifests/baseline_identity_lock.json")
    reservation = _read_json(
        path / "manifests/validation_split_reservation_ledger.json"
    )
    leakage = _read_json(path / "reports/leakage_provenance_report.json")
    runtime = _read_json(path / "reports/runtime_summary.json")
    if (
        frozen.get("claim_scope") != "diagnostic_only"
        or protocol.get("claim_scope") != "diagnostic_only"
        or protocol.get("non_adoptive") is not True
        or protocol.get("may_replace_canonical_reference") is not False
        or protocol.get("may_feed_recipe_selection") is not False
        or protocol.get("may_feed_deployable_selection") is not False
        or protocol.get("validation_scored") is not False
        or protocol.get("test_scored") is not False
        or baseline_lock.get("baseline_refit") is not False
        or reservation.get("features_generated") is not False
        or reservation.get("predictions_generated") is not False
        or reservation.get("formal_confirmation_ready") is not False
        or leakage.get("test_split_used") is not False
        or runtime.get("gpu_used") is not False
    ):
        raise ProtocolError("Uniform-B nonlinear claim or leakage firewall failed.")
    unhashed = {key: value for key, value in frozen.items() if key != "protocol_hash"}
    if stable_hash(unhashed) != frozen.get("protocol_hash"):
        raise ProtocolError("Uniform-B nonlinear frozen protocol hash drifted.")
    if protocol.get("protocol_hash") != frozen.get("protocol_hash"):
        raise ProtocolError("Uniform-B nonlinear protocol identity drifted.")
    _validate_content_index(path)

    cells = _read_csv(path / "tables/source_inner_selector_cells.csv")
    summaries = _read_csv(path / "tables/source_inner_candidate_summary.csv")
    kernel = _read_csv(path / "tables/kernel_fit_audit.csv")
    results = _read_csv(path / "tables/outer_results.csv")
    predictions = _read_csv(path / "tables/outer_predictions.csv")
    stability_predictions = _read_csv(
        path / "tables/seed_stability_predictions.csv"
    )
    comparisons = _read_csv(path / "tables/paired_center_comparison.csv")
    stability = _read_csv(path / "tables/seed_stability.csv")
    error_exchange = _read_csv(path / "tables/error_exchange.csv")
    centroid_exchange = _read_csv(path / "tables/centroid_conflict_exchange.csv")
    if (
        len(cells) != EXPECTED_SELECTOR_CELLS
        or len(summaries) != 9 * 36
        or len(kernel) != EXPECTED_NYSTROEM_TRANSFORMS + 27
        or len(results) != 18
        or len(predictions) != 2 * 9648
        or len(stability_predictions) != 2 * 9648
        or len(comparisons) != 9
        or len(stability) != 18
        or len(error_exchange) != 12
        or len(centroid_exchange) != 10
    ):
        raise ProtocolError("Uniform-B nonlinear artifact cardinality drifted.")
    pair_audits = [row for row in kernel if row["fit_role"] == "source_inner_pair"]
    outer_audits = [row for row in kernel if row["fit_role"] == "outer_final"]
    if (
        len({row["fit_key"] for row in pair_audits}) != EXPECTED_PAIR_FRAMES
        or len(pair_audits) != EXPECTED_NYSTROEM_TRANSFORMS
        or len(outer_audits) != 27
    ):
        raise ProtocolError("Uniform-B nonlinear pair-reuse audit drifted.")
    _validate_kernel_audits(pair_audits, outer_audits, config)
    _validate_selector_cells(
        cells, config, baseline_lock["per_outer_inherited_class_weight"]
    )
    _validate_selected_summaries(summaries, cells, config)
    baseline_rows = [
        row for row in predictions if row["model_role"] == "canonical_b_linear_baseline"
    ]
    primary_rows = [
        row for row in predictions if row["model_role"] == "canonical_b_nystroem_primary"
    ]
    _validate_baseline_identity(baseline_rows, config)
    _validate_outer_metrics(primary_rows, baseline_rows, comparisons)
    recomputed_bootstrap = paired_case_bootstrap(
        primary_rows,
        baseline_rows,
        centers=config.heldout_centers,
        replicates=config.gate.bootstrap_replicates,
        seed=config.gate.bootstrap_seed,
    )
    stored_bootstrap = _read_json(path / "reports/conditional_bootstrap.json")
    _assert_nested_close(stored_bootstrap, recomputed_bootstrap)
    recomputed_decision = progression_decision(
        comparisons, stability, recomputed_bootstrap, config.gate
    )
    stored_decision = _read_json(path / "reports/progression_decision.json")
    _assert_nested_close(stored_decision, recomputed_decision)
    if set(row["sample_id"] for row in primary_rows) & _manifest_split_ids(
        config.manifest_path, {"val", "test"}
    ):
        raise ProtocolError("Validation/test sample IDs leaked into nonlinear predictions.")
    checks = {
        "status": "PASS",
        "selector_cells": len(cells),
        "pair_preprocessing_frames": EXPECTED_PAIR_FRAMES,
        "primary_nystroem_transforms": EXPECTED_NYSTROEM_TRANSFORMS,
        "outer_predictions": len(predictions),
        "stability_predictions": len(stability_predictions),
        "baseline_identity": "EXACT",
        "validation_scored": False,
        "test_scored": False,
        "decision": stored_decision["decision"],
    }
    if not allow_pending:
        if protocol.get("status") != "PASS" or leakage.get("status") != "PASS":
            raise ProtocolError("Uniform-B nonlinear final status is not PASS.")
        report = _read_json(path / "reports/validation_report.json")
        if report.get("status") != "PASS" or report.get("checks") != checks:
            raise ProtocolError("Uniform-B nonlinear validation report drifted.")
    return checks


def _validate_selector_cells(
    cells: list[dict[str, str]],
    config: NonlinearProbeConfig,
    class_weights: Mapping[str, object],
) -> None:
    expected_candidates = {candidate.candidate_id for candidate in config.candidates}
    coverage = set()
    for row in cells:
        outer = row["outer_center"]
        inner = row["inner_center"]
        train_centers = set(json.loads(row["train_centers"]))
        if (
            outer == inner
            or train_centers
            != set(config.heldout_centers).difference({outer, inner})
            or row["candidate_id"] not in expected_candidates
            or row["selection_used_outer_labels"].lower() != "false"
            or row["fit_used_outer_center"].lower() != "false"
            or row["inherited_outer_class_weight"] != str(class_weights[outer])
        ):
            raise ProtocolError("Uniform-B nonlinear selector leakage/identity failed.")
        coverage.add((outer, inner, row["candidate_id"]))
    expected = {
        (outer, inner, candidate.candidate_id)
        for outer in config.heldout_centers
        for inner in config.heldout_centers
        if inner != outer
        for candidate in config.candidates
    }
    if coverage != expected:
        raise ProtocolError("Uniform-B nonlinear selector coverage drifted.")


def _validate_kernel_audits(
    pair_audits: list[dict[str, str]],
    outer_audits: list[dict[str, str]],
    config: NonlinearProbeConfig,
) -> None:
    for fit_key in sorted({row["fit_key"] for row in pair_audits}):
        rows = [row for row in pair_audits if row["fit_key"] == fit_key]
        pair = set(fit_key.split(","))
        expected_grid = {
            (width, components)
            for width in config.width_multipliers
            for components in config.components
        }
        if (
            len(rows) != 9
            or set(json.loads(rows[0]["train_centers"]))
            != set(config.heldout_centers).difference(pair)
            or len({row["fit_row_hash"] for row in rows}) != 1
            or len({row["scaler_state_hash"] for row in rows}) != 1
            or len({row["gamma_sample_row_hash"] for row in rows}) != 1
            or len({row["median_distance"] for row in rows}) != 1
            or {int(row["landmark_seed"]) for row in rows}
            != {config.primary_landmark_seed}
            or {
                (float(row["width_multiplier"]), int(row["n_components"]))
                for row in rows
            }
            != expected_grid
        ):
            raise ProtocolError("Uniform-B nonlinear pair kernel reuse drifted.")
        for row in rows:
            sigma = float(row["width_multiplier"]) * float(row["median_distance"])
            if not np.isclose(float(row["sigma"]), sigma) or not np.isclose(
                float(row["effective_gamma"]), 1.0 / (2.0 * sigma * sigma)
            ):
                raise ProtocolError("Uniform-B nonlinear kernel formula drifted.")
    for outer in config.heldout_centers:
        rows = [row for row in outer_audits if row["fit_key"] == outer]
        if (
            len(rows) != 3
            or {int(row["landmark_seed"]) for row in rows}
            != {config.primary_landmark_seed, *config.stability_landmark_seeds}
            or len({row["gamma_sample_row_hash"] for row in rows}) != 1
            or len({row["median_distance"] for row in rows}) != 1
        ):
            raise ProtocolError("Uniform-B nonlinear outer seed isolation drifted.")


def _validate_selected_summaries(
    summaries: list[dict[str, str]],
    cells: list[dict[str, str]],
    config: NonlinearProbeConfig,
) -> None:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for outer in config.heldout_centers:
        for candidate in config.candidates:
            by_key[(outer, candidate.candidate_id)] = [
                row
                for row in cells
                if row["outer_center"] == outer
                and row["candidate_id"] == candidate.candidate_id
            ]
    for outer in config.heldout_centers:
        outer_rows = [row for row in summaries if row["outer_center"] == outer]
        selected = [row for row in outer_rows if row["selected"].lower() == "true"]
        if len(selected) != 1:
            raise ProtocolError("Uniform-B nonlinear selected candidate count drifted.")
        for row in outer_rows:
            source = by_key[(outer, row["candidate_id"])]
            mean_bacc = float(np.mean([float(item["bacc"]) for item in source]))
            worst_bacc = min(float(item["bacc"]) for item in source)
            if not np.isclose(float(row["mean_inner_bacc"]), mean_bacc) or not np.isclose(
                float(row["worst_inner_bacc"]), worst_bacc
            ):
                raise ProtocolError("Uniform-B nonlinear selector summary drifted.")
        ordered = sorted(
            outer_rows,
            key=lambda row: (
                -float(row["mean_inner_bacc"]),
                -float(row["worst_inner_bacc"]),
                int(row["n_components"]),
                {1.0: 0, 2.0: 1, 0.5: 2}[float(row["width_multiplier"])],
                float(row["logistic_c"]),
                row["candidate_id"],
            ),
        )
        if ordered[0]["candidate_id"] != selected[0]["candidate_id"]:
            raise ProtocolError("Uniform-B nonlinear tie-break selection drifted.")


def _validate_baseline_identity(
    baseline_rows: list[dict[str, str]], config: NonlinearProbeConfig
) -> None:
    canonical = _read_csv(
        config.canonical_reference_root / "tables/classifier_tuned_predictions.csv"
    )
    expected = {
        row["sample_id"]: (
            row["heldout_center"],
            row["case_id"],
            row["center"],
            int(row["y_true"]),
            int(row["y_pred"]),
            float(row["prob_pos"]),
            row["eval_row_hash"],
        )
        for row in canonical
    }
    observed = {
        row["sample_id"]: (
            row["outer_center"],
            row["case_id"],
            row["center"],
            int(row["y_true"]),
            int(row["y_pred"]),
            float(row["prob_pos"]),
            row["eval_row_hash"],
        )
        for row in baseline_rows
    }
    if expected != observed:
        raise ProtocolError("Canonical-B baseline row/hash equality failed.")


def _validate_outer_metrics(
    primary: list[dict[str, str]],
    baseline: list[dict[str, str]],
    comparisons: list[dict[str, str]],
) -> None:
    for comparison in comparisons:
        center = comparison["outer_center"]
        nonlinear_rows = [row for row in primary if row["center"] == center]
        baseline_rows = [row for row in baseline if row["center"] == center]
        truth = np.asarray([int(row["y_true"]) for row in nonlinear_rows])
        nonlinear = binary_metrics(
            truth, np.asarray([int(row["y_pred"]) for row in nonlinear_rows])
        )
        linear = binary_metrics(
            np.asarray([int(row["y_true"]) for row in baseline_rows]),
            np.asarray([int(row["y_pred"]) for row in baseline_rows]),
        )
        observed = {
            "baseline_bacc": linear["bacc"],
            "nonlinear_bacc": nonlinear["bacc"],
            "delta_bacc": nonlinear["bacc"] - linear["bacc"],
            "delta_positive_recall": nonlinear["positive_recall"]
            - linear["positive_recall"],
            "delta_specificity": nonlinear["specificity"] - linear["specificity"],
        }
        if any(
            not np.isclose(float(comparison[key]), value)
            for key, value in observed.items()
        ):
            raise ProtocolError("Uniform-B nonlinear outer metrics drifted.")


def _validate_content_index(root: Path) -> None:
    payload = _read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    if stable_hash(unhashed) != payload.get("content_hash"):
        raise ProtocolError("Uniform-B nonlinear content hash drifted.")
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ProtocolError("Uniform-B nonlinear content index is malformed.")
    expected = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "content_index.json"
    }
    observed = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Uniform-B nonlinear content-index row is malformed.")
        relative = str(row.get("path", ""))
        member = root / relative
        if not member.is_file() or _sha256_file(member) != row.get("sha256"):
            raise ProtocolError(f"Uniform-B nonlinear member drifted: {relative}.")
        observed.add(relative)
    if observed != expected:
        raise ProtocolError("Uniform-B nonlinear content-index coverage drifted.")


def _manifest_split_ids(path: Path, splits: set[str]) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["sample_id"]
            for row in csv.DictReader(handle)
            if str(row["split"]).strip().lower() in splits
        }


def _assert_nested_close(left: object, right: object, path: str = "root") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ProtocolError(f"Nonlinear validation keys differ at {path}.")
        for key in left:
            _assert_nested_close(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ProtocolError(f"Nonlinear validation list differs at {path}.")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_nested_close(a, b, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12):
            raise ProtocolError(f"Nonlinear validation numeric drift at {path}.")
        return
    if left != right:
        raise ProtocolError(f"Nonlinear validation value drift at {path}.")


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
