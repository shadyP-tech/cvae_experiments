"""Coverage, aggregation, and predeclared decisions for outer prior recovery."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .prior_recovery_common import mean
from .prior_recovery_config import OuterPriorRecoveryConfig
from .source_inner_selection import RecipeLock


def aggregate_outer(
    config: OuterPriorRecoveryConfig,
    rows: Sequence[Mapping[str, object]],
    locks: Mapping[str, RecipeLock],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object]]:
    coverage = outer_coverage(config, rows)
    valid_rows = [row for row in rows if row["status"] == "ok"]
    prior = [row for row in valid_rows if row["representation_role"] == "prior"]
    cell_means: dict[tuple[str, int, str], dict[str, float]] = {}
    for outer in config.heldout_centers:
        for seed in config.training_seeds:
            for arm in ("A", "B", "C", "D"):
                selected = [
                    row
                    for row in prior
                    if row["outer_target_center"] == outer
                    and int(row["training_seed"]) == seed
                    and row["arm"] == arm
                ]
                if len(selected) != len(config.generation_seeds):
                    continue
                cell_means[(outer, seed, arm)] = {
                    "ratio": mean(float(row["preservation_ratio"]) for row in selected),
                    "bacc": mean(float(row["bacc"]) for row in selected),
                }
    paired = []
    for (outer, seed, arm), values in cell_means.items():
        if arm == "A" or (outer, seed, "A") not in cell_means:
            continue
        paired.append(
            {
                "outer_target_center": outer,
                "training_seed": seed,
                "arm": arm,
                "ratio_delta_vs_a": values["ratio"] - cell_means[(outer, seed, "A")]["ratio"],
                "bacc_delta_vs_a": values["bacc"] - cell_means[(outer, seed, "A")]["bacc"],
            }
        )
    aggregation = []
    for arm in ("A", "B", "C", "D"):
        center_means = []
        for outer in config.heldout_centers:
            values = [
                cell_means[(outer, seed, arm)]["ratio"]
                for seed in config.training_seeds
                if (outer, seed, arm) in cell_means
            ]
            if len(values) == len(config.training_seeds):
                center_means.append((outer, mean(values)))
                aggregation.append(
                    {
                        "aggregation_level": "center",
                        "outer_target_center": outer,
                        "arm": arm,
                        "mean_preservation_ratio": mean(values),
                        "n_training_seeds": len(values),
                    }
                )
        if len(center_means) == len(config.heldout_centers):
            aggregation.append(
                {
                    "aggregation_level": "overall",
                    "outer_target_center": "ALL",
                    "arm": arm,
                    "mean_preservation_ratio": mean(value for _, value in center_means),
                    "n_training_seeds": len(config.training_seeds),
                }
            )
    policy_cells = []
    for outer, lock in locks.items():
        for seed in config.training_seeds:
            key = (outer, seed, lock.primary_arm)
            baseline_key = (outer, seed, "A")
            if key in cell_means and baseline_key in cell_means:
                policy_cells.append((outer, seed, cell_means[key], cell_means[baseline_key]))
    valid_locks = all(lock.status == "VALID" for lock in locks.values())
    policy_ratio = mean(item[2]["ratio"] for item in policy_cells)
    seed_deltas = {
        seed: mean(
            item[2]["ratio"] - item[3]["ratio"]
            for item in policy_cells
            if item[1] == seed
        )
        for seed in config.training_seeds
    }
    center_deltas = {
        outer: mean(
            item[2]["ratio"] - item[3]["ratio"]
            for item in policy_cells
            if item[0] == outer
        )
        for outer in config.heldout_centers
    }
    center_wins = sum(value > 0.0 for value in center_deltas.values())
    policy_center_bacc = {
        outer: mean(item[2]["bacc"] for item in policy_cells if item[0] == outer)
        for outer in config.heldout_centers
    }
    baseline_center_bacc = {
        outer: mean(item[3]["bacc"] for item in policy_cells if item[0] == outer)
        for outer in config.heldout_centers
    }
    worst_center_guard, paired_center_bacc_deltas = paired_worst_center_guard(
        policy_center_bacc,
        baseline_center_bacc,
        tolerance=config.safety_max_bacc_regression,
    )
    role_deltas: dict[str, float] = {}
    for role in ("decode", "posterior"):
        deltas = []
        for outer, lock in locks.items():
            for seed in config.training_seeds:
                policy_values = _role_values(valid_rows, role, outer, seed, lock.primary_arm)
                baseline_values = _role_values(valid_rows, role, outer, seed, "A")
                if policy_values and baseline_values:
                    deltas.append(mean(policy_values) - mean(baseline_values))
        role_deltas[role] = mean(deltas)
    representation_safety = all(
        math.isfinite(value) and value >= -config.safety_max_bacc_regression
        for value in role_deltas.values()
    )
    execution_complete = (
        coverage["status"] == "PASS"
        and valid_locks
        and len(policy_cells) == len(config.heldout_centers) * len(config.training_seeds)
    )
    positive = (
        execution_complete
        and policy_ratio >= config.positive_claim_min_ratio
        and all(math.isfinite(value) and value > 0.0 for value in seed_deltas.values())
        and center_wins >= min(config.positive_claim_min_center_wins, len(config.heldout_centers))
        and worst_center_guard
        and representation_safety
    )
    status = (
        "POSITIVE_PRESERVATION"
        if positive
        else "NEGATIVE_PRESERVATION"
        if execution_complete
        else "INCOMPLETE_OR_INVALID_DIAGNOSTIC"
    )
    decision = {
        "status": status,
        "claim_scope": "cvae_preservation_only" if execution_complete else "diagnostic_only",
        "mean_policy_preservation_ratio": policy_ratio,
        "training_seed_deltas_vs_a": seed_deltas,
        "center_ratio_deltas_vs_a": center_deltas,
        "strict_center_wins_vs_a": center_wins,
        "paired_center_bacc_deltas_vs_a": paired_center_bacc_deltas,
        "worst_center_guard_pass": worst_center_guard,
        "decode_posterior_mean_bacc_deltas_vs_a": role_deltas,
        "decode_posterior_safety_pass": representation_safety,
        "all_recipe_locks_valid": valid_locks,
        "factorial_coverage_pass": coverage["status"] == "PASS",
        "execution_complete": execution_complete,
        "aggregation_order": ["generation_seed", "training_seed", "outer_center_equal_weight"],
        "routing_performed": False,
        "composition_performed": False,
    }
    return aggregation, paired, decision, coverage


def outer_coverage(
    config: OuterPriorRecoveryConfig,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected_by_role = {
        "prior": len(config.heldout_centers)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * 4,
        "posterior": len(config.heldout_centers)
        * len(config.training_seeds)
        * len(config.generation_seeds)
        * 4,
        "decode": len(config.heldout_centers) * len(config.training_seeds) * 4,
    }
    observed_by_role = {
        role: sum(row["representation_role"] == role for row in rows)
        for role in expected_by_role
    }
    valid_by_role = {
        role: sum(row["representation_role"] == role and row["status"] == "ok" for row in rows)
        for role in expected_by_role
    }
    expected_total = sum(expected_by_role.values())
    valid_total = sum(valid_by_role.values())
    status = (
        "PASS"
        if observed_by_role == expected_by_role
        and valid_by_role == expected_by_role
        and len(rows) == expected_total
        and valid_total == expected_total
        else "FAIL"
    )
    return {
        "schema_version": "midogpp_prior_recovery_coverage_v1",
        "status": status,
        "expected_rows_by_role": expected_by_role,
        "observed_rows_by_role": observed_by_role,
        "valid_rows_by_role": valid_by_role,
        "expected_all_representation_rows": expected_total,
        "observed_all_representation_rows": len(rows),
        "valid_all_representation_rows": valid_total,
        "heldout_centers": list(config.heldout_centers),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "arms": ["A", "B", "C", "D"],
        "invalid_status_counts": _status_counts(rows),
    }


def paired_worst_center_guard(
    policy: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    tolerance: float,
) -> tuple[bool, dict[str, float]]:
    if set(policy) != set(baseline) or not policy:
        return False, {}
    deltas = {
        center: round(float(policy[center]) - float(baseline[center]), 12)
        for center in baseline
    }
    return all(value >= -float(tolerance) for value in deltas.values()), deltas


def _role_values(
    rows: Sequence[Mapping[str, object]],
    role: str,
    outer: str,
    seed: int,
    arm: str,
) -> list[float]:
    return [
        float(row["bacc"])
        for row in rows
        if row["representation_role"] == role
        and row["outer_target_center"] == outer
        and int(row["training_seed"]) == seed
        and row["arm"] == arm
    ]


def _status_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts
