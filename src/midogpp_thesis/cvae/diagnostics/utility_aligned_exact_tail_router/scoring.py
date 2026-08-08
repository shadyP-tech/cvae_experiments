"""Terminal seed-cell, probability-ensemble, and Hxe scoring."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ...metrics import balanced_accuracy, macro_f1, spearman
from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GENERATION_SEEDS,
    R2_ACTION_ID,
    TRAINING_SEEDS,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
)
from .input_contracts import FixedPartitionSurface
from .r2_policy import Stage90R2PlanSet
from .target_prediction_contracts import TargetPredictionStore


SEED_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "action_id",
    "training_seed",
    "generation_seed",
    "row_count",
    "case_count",
    "balanced_accuracy",
    "macro_f1",
    "inference_unit",
    "diagnostic_only",
)
ENSEMBLE_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "action_id",
    "seed_pair_count",
    "row_count",
    "case_count",
    "balanced_accuracy",
    "macro_f1",
    "primary_endpoint",
    "inference_unit",
    "diagnostic_only",
)
ORACLE_DIAGNOSTIC_COLUMNS = (
    "schema_version",
    "target_center",
    "R2_selected_source",
    "oracle_source_ids_json",
    "R2_top1_exact_agreement",
    "R2_top1_tie_agreement",
    "predicted_gain_utility_spearman",
    "R2_Hxe_bacc",
    "oracle_Hxe_bacc",
    "base_bacc",
    "normalized_oracle_gap",
    "row_role",
    "may_update_policy",
    "diagnostic_only",
)


def score_target_seed_cells(
    store: TargetPredictionStore,
    labels_by_sample_id: Mapping[str, int],
    partitions: FixedPartitionSurface,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    by_key = store.by_key
    for target in CENTERS:
        identities = partitions.evaluation_rows_by_center[target]
        truth = [labels_by_sample_id[row.sample_id] for row in identities]
        case_count = len({row.case_id for row in identities})
        for action_id in expected_target_action_ids(target):
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    cell = by_key[(target, action_id, training_seed, generation_seed)]
                    rows.append(
                        {
                            "schema_version": "midogpp_utility_aligned_stage90_seed_metric_v1",
                            "target_center": target,
                            "action_id": action_id,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "row_count": len(truth),
                            "case_count": case_count,
                            "balanced_accuracy": balanced_accuracy(truth, cell.predictions),
                            "macro_f1": macro_f1(truth, cell.predictions),
                            "inference_unit": "technical_seed_cell_descriptive_only",
                            "diagnostic_only": True,
                        }
                    )
    return tuple(rows)


def score_target_probability_ensembles(
    store: TargetPredictionStore,
    labels_by_sample_id: Mapping[str, int],
    partitions: FixedPartitionSurface,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    by_key = store.by_key
    for target in CENTERS:
        identities = partitions.evaluation_rows_by_center[target]
        truth = [labels_by_sample_id[row.sample_id] for row in identities]
        case_count = len({row.case_id for row in identities})
        for action_id in expected_target_action_ids(target):
            probabilities = np.stack(
                [
                    by_key[(target, action_id, training_seed, generation_seed)].probabilities
                    for training_seed in TRAINING_SEEDS
                    for generation_seed in GENERATION_SEEDS
                ]
            )
            mean_probability = np.mean(probabilities, axis=0, dtype=np.float64)
            prediction = (mean_probability >= 0.5).astype(np.uint8)
            rows.append(
                {
                    "schema_version": "midogpp_utility_aligned_stage90_ensemble_metric_v1",
                    "target_center": target,
                    "action_id": action_id,
                    "seed_pair_count": 9,
                    "row_count": len(truth),
                    "case_count": case_count,
                    "balanced_accuracy": balanced_accuracy(truth, prediction),
                    "macro_f1": macro_f1(truth, prediction),
                    "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
                    "inference_unit": "target_center",
                    "diagnostic_only": True,
                }
            )
    return tuple(rows)


def build_hxe_oracle_diagnostics(
    ensemble_rows: tuple[Mapping[str, object], ...],
    plans: Stage90R2PlanSet,
) -> tuple[dict[str, object], ...]:
    import json

    metric = {
        (str(row["target_center"]), str(row["action_id"])): float(
            row["balanced_accuracy"]
        )
        for row in ensemble_rows
    }
    output: list[dict[str, object]] = []
    for target in CENTERS:
        sources = candidate_sources(target)
        plan = plans.by_target[target]
        selected = plan.proposed_source_by_router[R2_ACTION_ID]
        utilities = {
            source: metric[(target, h_x_e_action_id(source))] for source in sources
        }
        predicted = plan.mean_prediction_by_router_source[R2_ACTION_ID]
        maximum = max(utilities.values())
        minimum = min(utilities.values())
        oracle_sources = tuple(
            source for source in sources if math.isclose(utilities[source], maximum, abs_tol=1e-15)
        )
        denominator = maximum - minimum
        gap = (
            0.0
            if denominator == 0.0
            else (maximum - utilities[selected]) / denominator
        )
        rho = spearman(
            [predicted[source] for source in sources],
            [utilities[source] for source in sources],
        )
        output.append(
            {
                "schema_version": "midogpp_utility_aligned_stage90_hxe_oracle_v1",
                "target_center": target,
                "R2_selected_source": selected,
                "oracle_source_ids_json": json.dumps(list(oracle_sources), separators=(",", ":")),
                "R2_top1_exact_agreement": selected == min(oracle_sources),
                "R2_top1_tie_agreement": selected in oracle_sources,
                "predicted_gain_utility_spearman": rho,
                "R2_Hxe_bacc": utilities[selected],
                "oracle_Hxe_bacc": maximum,
                "base_bacc": metric[(target, BASE_ACTION_ID)],
                "normalized_oracle_gap": gap,
                "row_role": "terminal_oracle_diagnostic",
                "may_update_policy": False,
                "diagnostic_only": True,
            }
        )
    return tuple(output)


__all__ = (
    "ENSEMBLE_METRIC_COLUMNS",
    "ORACLE_DIAGNOSTIC_COLUMNS",
    "SEED_METRIC_COLUMNS",
    "build_hxe_oracle_diagnostics",
    "score_target_probability_ensembles",
    "score_target_seed_cells",
)
