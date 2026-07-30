"""Metrics, deterministic selection, bootstrap, and progression gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ..protocol import ProtocolError
from .config import Candidate, GateConfig


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int8)
    pred = np.asarray(y_pred, dtype=np.int8)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    if tp + fn == 0 or tn + fp == 0:
        raise ProtocolError("Balanced accuracy requires both classes.")
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "bacc": (recall + specificity) / 2.0,
        "positive_recall": recall,
        "specificity": specificity,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def candidate_sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    width = float(row["width_multiplier"])
    width_rank = {1.0: 0, 2.0: 1, 0.5: 2}[width]
    return (
        -float(row["mean_inner_bacc"]),
        -float(row["worst_inner_bacc"]),
        int(row["n_components"]),
        width_rank,
        float(row["logistic_c"]),
        str(row["candidate_id"]),
    )


def summarize_and_select(
    selector_cells: Sequence[Mapping[str, object]],
    candidates: Sequence[Candidate],
    centers: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    by_key: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in selector_cells:
        by_key[(str(row["outer_center"]), str(row["candidate_id"]))].append(row)
    summaries: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for outer in centers:
        outer_rows = []
        for candidate in candidates:
            rows = by_key[(outer, candidate.candidate_id)]
            if len(rows) != len(centers) - 1:
                raise ProtocolError("Nonlinear selector-cell coverage is incomplete.")
            observed_inner = {str(row["inner_center"]) for row in rows}
            if observed_inner != set(centers).difference({outer}):
                raise ProtocolError("Nonlinear selector inner-center coverage drifted.")
            summary = {
                "schema_version": "midogpp_uniform_b_nonlinear_candidate_summary_v1",
                "outer_center": outer,
                "candidate_id": candidate.candidate_id,
                **candidate.to_payload(),
                "inner_fold_count": len(rows),
                "mean_inner_bacc": float(
                    np.mean([float(row["bacc"]) for row in rows])
                ),
                "worst_inner_bacc": float(
                    np.min([float(row["bacc"]) for row in rows])
                ),
                "mean_inner_positive_recall": float(
                    np.mean([float(row["positive_recall"]) for row in rows])
                ),
                "mean_inner_specificity": float(
                    np.mean([float(row["specificity"]) for row in rows])
                ),
                "selected": False,
            }
            summaries.append(summary)
            outer_rows.append(summary)
        winner = min(outer_rows, key=candidate_sort_key)
        winner["selected"] = True
        selected[outer] = {
            **winner,
            "candidate": candidate_by_id[str(winner["candidate_id"])],
        }
    return summaries, selected


def paired_case_bootstrap(
    primary_predictions: Sequence[Mapping[str, object]],
    baseline_predictions: Sequence[Mapping[str, object]],
    *,
    centers: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    nonlinear = {str(row["sample_id"]): row for row in primary_predictions}
    baseline = {str(row["sample_id"]): row for row in baseline_predictions}
    if set(nonlinear) != set(baseline):
        raise ProtocolError("Bootstrap prediction rows are not paired.")
    by_center_case: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample_id, row in nonlinear.items():
        by_center_case[str(row["center"])][str(row["case_id"])].append(sample_id)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        center_deltas = []
        for center in centers:
            cases = sorted(by_center_case[center])
            sampled = rng.choice(cases, size=len(cases), replace=True)
            sample_ids = [
                sample_id
                for case_id in sampled
                for sample_id in by_center_case[center][str(case_id)]
            ]
            truth = np.asarray(
                [int(nonlinear[sample_id]["y_true"]) for sample_id in sample_ids]
            )
            nonlinear_pred = np.asarray(
                [int(nonlinear[sample_id]["y_pred"]) for sample_id in sample_ids]
            )
            baseline_pred = np.asarray(
                [int(baseline[sample_id]["y_pred"]) for sample_id in sample_ids]
            )
            center_deltas.append(
                binary_metrics(truth, nonlinear_pred)["bacc"]
                - binary_metrics(truth, baseline_pred)["bacc"]
            )
        draws[replicate] = float(np.mean(center_deltas))
    return {
        "schema_version": "midogpp_uniform_b_nonlinear_paired_case_bootstrap_v1",
        "estimand": "equal_center_mean_bacc_nonlinear_minus_linear_b",
        "sampling_unit": "case_within_observed_center",
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(np.mean(draws)),
        "percentile_2_5": float(np.quantile(draws, 0.025)),
        "percentile_97_5": float(np.quantile(draws, 0.975)),
        "supportive_only": True,
        "covers_new_center_uncertainty": False,
    }


def progression_decision(
    comparisons: Sequence[Mapping[str, object]],
    stability_rows: Sequence[Mapping[str, object]],
    bootstrap: Mapping[str, object],
    gate: GateConfig,
) -> dict[str, object]:
    primary_delta = float(np.mean([float(row["delta_bacc"]) for row in comparisons]))
    recall_delta = float(
        np.mean([float(row["delta_positive_recall"]) for row in comparisons])
    )
    specificity_delta = float(
        np.mean([float(row["delta_specificity"]) for row in comparisons])
    )
    worst_delta = min(float(row["delta_bacc"]) for row in comparisons)
    strict_wins = sum(float(row["delta_bacc"]) > 0.0 for row in comparisons)
    seed_groups: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in stability_rows:
        seed_groups[int(row["landmark_seed"])].append(row)
    seed_checks = []
    for seed, rows in sorted(seed_groups.items()):
        mean_delta = float(np.mean([float(row["delta_bacc"]) for row in rows]))
        seed_worst = min(float(row["delta_bacc"]) for row in rows)
        passed = (
            mean_delta > gate.supplemental_mean_delta_min_exclusive
            and abs(mean_delta - primary_delta)
            <= gate.supplemental_primary_deviation_max
            and seed_worst >= gate.supplemental_worst_center_delta_min
        )
        seed_checks.append(
            {
                "landmark_seed": seed,
                "equal_center_mean_bacc_delta": mean_delta,
                "absolute_deviation_from_primary": abs(mean_delta - primary_delta),
                "worst_center_delta": seed_worst,
                "passed": passed,
            }
        )
    checks = {
        "mean_bacc_delta": primary_delta >= gate.mean_bacc_delta_min,
        "strict_center_wins": strict_wins >= gate.strict_center_wins_min,
        "worst_center_delta": worst_delta >= gate.worst_center_delta_min,
        "positive_recall": recall_delta
        > gate.mean_positive_recall_delta_min_exclusive,
        "specificity": specificity_delta >= gate.mean_specificity_delta_min,
        "landmark_seed_stability": bool(seed_checks)
        and all(bool(row["passed"]) for row in seed_checks),
    }
    passed = all(checks.values())
    return {
        "schema_version": "midogpp_uniform_b_nonlinear_progression_v1",
        "decision": (
            "NONLINEAR_B_DIAGNOSTIC_GATE_PASS"
            if passed
            else "NONLINEAR_B_DIAGNOSTIC_GATE_FAIL_BUILD_B_SPATIAL_NEXT"
        ),
        "passed": passed,
        "checks": checks,
        "observed": {
            "equal_center_mean_bacc_delta": primary_delta,
            "strict_center_wins": strict_wins,
            "worst_center_delta": worst_delta,
            "equal_center_mean_positive_recall_delta": recall_delta,
            "equal_center_mean_specificity_delta": specificity_delta,
            "bootstrap_lower_supportive": float(bootstrap["percentile_2_5"]),
        },
        "thresholds": gate.__dict__,
        "supplemental_landmark_seed_checks": seed_checks,
        "bootstrap_is_supportive_not_conjunctive": True,
        "diagnostic_only": True,
        "does_not_authorize_validation_scoring": True,
    }
