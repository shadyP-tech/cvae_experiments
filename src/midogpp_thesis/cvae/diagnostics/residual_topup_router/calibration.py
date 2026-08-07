"""Fixed outer-fold action-value gate for the two top-up actions."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    ENERGY_TOPUP_ACTION_ID,
    GENERATION_SEEDS,
    SELECTION_CONFIDENCE_LEVEL,
    SELECTION_RULE,
    SELECTION_THRESHOLD,
    TRAINING_SEEDS,
    UNIFORM_TOPUP_ACTION_ID,
    development_queries,
)


QUERY_GAIN_COLUMNS = (
    "schema_version",
    "outer_target",
    "query_center",
    "seed_cell_count",
    "mean_paired_bacc_gain",
    "minimum_seed_cell_gain",
    "maximum_seed_cell_gain",
    "target_H_excluded",
    "query_cluster_unit",
    "diagnostic_only",
)

SELECTION_COLUMNS = (
    "schema_version",
    "outer_target",
    "selected_action_id",
    "fallback_applied",
    "fallback_reason",
    "query_cluster_count",
    "mean_query_center_gain",
    "sample_standard_deviation",
    "standard_error",
    "one_sided_t_critical",
    "one_sided_lower_confidence_bound",
    "confidence_level",
    "selection_threshold",
    "selection_rule",
    "minimum_training_seed_mean_gain",
    "minimum_generation_seed_mean_gain",
    "seed_risk_role",
    "target_H_labels_used_for_selection",
    "fixed_no_parameter_gate",
    "diagnostic_only",
)


def calibrate_outer_actions(
    gain_rows: Sequence[Mapping[str, object]],
    *,
    config_contract_hash: str,
    global_prediction_seal_hash: str,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    """Apply one fixed center-cluster LCB gate with exact uniform fallback."""

    rows = tuple(gain_rows)
    query_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for outer in CENTERS:
        outer_rows = [row for row in rows if str(row.get("outer_target")) == outer]
        expected_count = len(development_queries(outer)) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
        if len(outer_rows) != expected_count:
            raise ProtocolError("Residual top-up calibration fold coverage drifted.")
        query_means: list[float] = []
        for query in development_queries(outer):
            values = np.asarray(
                [
                    float(row["paired_bacc_gain"])
                    for row in outer_rows
                    if str(row["query_center"]) == query
                ],
                dtype=np.float64,
            )
            if len(values) != len(TRAINING_SEEDS) * len(GENERATION_SEEDS) or not np.isfinite(values).all():
                raise ProtocolError("Residual top-up query gain coverage drifted.")
            mean = float(np.mean(values))
            query_means.append(mean)
            query_rows.append(
                {
                    "schema_version": "midogpp_residual_topup_query_gain_v1",
                    "outer_target": outer,
                    "query_center": query,
                    "seed_cell_count": len(values),
                    "mean_paired_bacc_gain": mean,
                    "minimum_seed_cell_gain": float(np.min(values)),
                    "maximum_seed_cell_gain": float(np.max(values)),
                    "target_H_excluded": True,
                    "query_cluster_unit": True,
                    "diagnostic_only": True,
                }
            )
        values = np.asarray(query_means, dtype=np.float64)
        n = len(values)
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / float(np.sqrt(n))
        critical = float(student_t.ppf(SELECTION_CONFIDENCE_LEVEL, df=n - 1))
        mean = float(np.mean(values))
        lower = mean - critical * standard_error
        selected = (
            ENERGY_TOPUP_ACTION_ID
            if lower > SELECTION_THRESHOLD
            else UNIFORM_TOPUP_ACTION_ID
        )
        training_seed_means = [
            float(
                np.mean(
                    [
                        float(row["paired_bacc_gain"])
                        for row in outer_rows
                        if int(row["training_seed"]) == seed
                    ]
                )
            )
            for seed in TRAINING_SEEDS
        ]
        generation_seed_means = [
            float(
                np.mean(
                    [
                        float(row["paired_bacc_gain"])
                        for row in outer_rows
                        if int(row["generation_seed"]) == seed
                    ]
                )
            )
            for seed in GENERATION_SEEDS
        ]
        selections.append(
            {
                "schema_version": "midogpp_residual_topup_diagnostic_selection_v1",
                "outer_target": outer,
                "selected_action_id": selected,
                "fallback_applied": selected == UNIFORM_TOPUP_ACTION_ID,
                "fallback_reason": "" if selected == ENERGY_TOPUP_ACTION_ID else "one_sided_query_cluster_lcb_not_strictly_positive",
                "query_cluster_count": n,
                "mean_query_center_gain": mean,
                "sample_standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "one_sided_t_critical": critical,
                "one_sided_lower_confidence_bound": lower,
                "confidence_level": SELECTION_CONFIDENCE_LEVEL,
                "selection_threshold": SELECTION_THRESHOLD,
                "selection_rule": SELECTION_RULE,
                "minimum_training_seed_mean_gain": min(training_seed_means),
                "minimum_generation_seed_mean_gain": min(generation_seed_means),
                "seed_risk_role": "report_only_no_seed_selection",
                "target_H_labels_used_for_selection": False,
                "fixed_no_parameter_gate": True,
                "diagnostic_only": True,
            }
        )
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_calibration_lock_v1",
        "status": "LOCKED_FROM_Q_NOT_H_CONSUMED_LABELS_AFTER_GLOBAL_ALL_ACTION_SEAL",
        "config_contract_hash": config_contract_hash,
        "global_prediction_seal_hash": global_prediction_seal_hash,
        "selection_rule": SELECTION_RULE,
        "confidence_level": SELECTION_CONFIDENCE_LEVEL,
        "threshold": SELECTION_THRESHOLD,
        "action_menu": [UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID],
        "query_gain_rows_hash": stable_hash(query_rows),
        "selection_rows_hash": stable_hash(selections),
        "outer_target_count": len(CENTERS),
        "query_clusters_per_outer_target": len(CENTERS) - 1,
        "hyperparameters_fitted": False,
        "target_H_labels_used": False,
        "seed_or_expert_selected": False,
        "diagnostic_only": True,
    }
    lock = {**unhashed, "calibration_lock_hash": stable_hash(unhashed)}
    return tuple(query_rows), tuple(selections), lock


__all__ = (
    "QUERY_GAIN_COLUMNS",
    "SELECTION_COLUMNS",
    "calibrate_outer_actions",
)
