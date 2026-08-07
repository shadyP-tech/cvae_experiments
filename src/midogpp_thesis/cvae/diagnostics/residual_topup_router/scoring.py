"""Post-seal metrics for fixed residual top-up actions."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np

from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from .contracts import (
    BASE_ONLY_ACTION_ID,
    CENTERS,
    ENERGY_TOPUP_ACTION_ID,
    GENERATION_SEEDS,
    TARGET_ACTION_IDS,
    TRAINING_SEEDS,
    UNIFORM_TOPUP_ACTION_ID,
)


METRIC_COLUMNS = (
    "schema_version",
    "phase",
    "outer_target",
    "query_center",
    "action_id",
    "arm_role",
    "budget_role",
    "training_seed",
    "generation_seed",
    "evaluation_row_count",
    "evaluation_class_0_count",
    "evaluation_class_1_count",
    "bacc",
    "macro_f1",
    "primary_metric",
    "metric_role",
    "labels_used_only_after_global_prediction_seal",
    "target_H_labels_used_for_own_selection",
)

DEVELOPMENT_GAIN_COLUMNS = (
    "schema_version",
    "outer_target",
    "query_center",
    "training_seed",
    "generation_seed",
    "routed_bacc",
    "uniform_topup_bacc",
    "paired_bacc_gain",
    "routed_macro_f1",
    "uniform_topup_macro_f1",
    "paired_macro_f1_gain_descriptive",
    "primary_comparison_matched_budget",
    "target_H_labels_used",
    "diagnostic_only",
)

TARGET_DELTA_COLUMNS = (
    "schema_version",
    "target_center",
    "training_seed",
    "generation_seed",
    "selected_action_id",
    "energy_topup_bacc",
    "uniform_topup_bacc",
    "base_only_bacc",
    "raw_energy_vs_uniform_bacc_delta",
    "selected_vs_uniform_bacc_delta",
    "uniform_topup_vs_base_budget_delta",
    "raw_energy_vs_uniform_macro_f1_delta_descriptive",
    "primary_comparison_matched_budget",
    "base_only_is_budget_reference",
    "diagnostic_only",
)

ENSEMBLE_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "action_id",
    "seed_probability_aggregation",
    "seed_cell_count",
    "evaluation_row_count",
    "bacc",
    "macro_f1",
    "threshold",
    "selected_for_policy",
    "descriptive_only",
)


def score_prediction_store(
    store: object,
    *,
    labels_by_sample_id: Mapping[str, int],
) -> tuple[dict[str, object], ...]:
    """Score immutable cells; labels cannot be passed to any fitting API."""

    index_rows = tuple(getattr(store, "index_rows", ()))
    slice_for = getattr(store, "slice_for", None)
    if not index_rows or not callable(slice_for):
        raise ProtocolError("Residual top-up prediction store is unavailable.")
    output: list[dict[str, object]] = []
    for row in index_rows:
        sample_ids = _json_strings(row.get("evaluation_row_ids_json"))
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ProtocolError("Residual top-up evaluation identities are invalid.")
        try:
            labels = np.asarray(
                [int(labels_by_sample_id[value]) for value in sample_ids],
                dtype=np.uint8,
            )
        except KeyError as exc:
            raise ProtocolError("Residual top-up labels do not cover a cell.") from exc
        if set(labels.tolist()) != {0, 1}:
            raise ProtocolError("Residual top-up scoring cell lacks both classes.")
        predictions, _ = slice_for(row)
        predictions = np.asarray(predictions, dtype=np.uint8)
        if predictions.shape != labels.shape:
            raise ProtocolError("Residual top-up labels and predictions drifted.")
        output.append(
            {
                "schema_version": "midogpp_residual_topup_metric_cell_v1",
                "phase": str(row["phase"]),
                "outer_target": str(row["outer_target"]),
                "query_center": str(row["query_center"]),
                "action_id": str(row["action_id"]),
                "arm_role": str(row["arm_role"]),
                "budget_role": str(row["budget_role"]),
                "training_seed": int(row["training_seed"]),
                "generation_seed": int(row["generation_seed"]),
                "evaluation_row_count": len(labels),
                "evaluation_class_0_count": int(np.sum(labels == 0)),
                "evaluation_class_1_count": int(np.sum(labels == 1)),
                "bacc": float(balanced_accuracy(labels.tolist(), predictions.tolist())),
                "macro_f1": float(macro_f1(labels.tolist(), predictions.tolist())),
                "primary_metric": "balanced_accuracy",
                "metric_role": (
                    "q_not_H_diagnostic_calibration_and_scoring"
                    if str(row["phase"]) == "development"
                    else "terminal_descriptive_scoring_only"
                ),
                "labels_used_only_after_global_prediction_seal": True,
                "target_H_labels_used_for_own_selection": False,
            }
        )
    return tuple(output)


def development_paired_gains(
    metric_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = [row for row in metric_rows if str(row.get("phase")) == "development"]
    by_key = {
        (
            str(row["outer_target"]),
            str(row["query_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            str(row["action_id"]),
        ): row
        for row in rows
    }
    if len(by_key) != len(rows):
        raise ProtocolError("Residual top-up development metrics duplicate.")
    output: list[dict[str, object]] = []
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    prefix = (outer, query, training_seed, generation_seed)
                    try:
                        routed = by_key[(*prefix, ENERGY_TOPUP_ACTION_ID)]
                        control = by_key[(*prefix, UNIFORM_TOPUP_ACTION_ID)]
                    except KeyError as exc:
                        raise ProtocolError(
                            "Residual top-up development pairing is incomplete."
                        ) from exc
                    output.append(
                        {
                            "schema_version": "midogpp_residual_topup_development_gain_v1",
                            "outer_target": outer,
                            "query_center": query,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "routed_bacc": float(routed["bacc"]),
                            "uniform_topup_bacc": float(control["bacc"]),
                            "paired_bacc_gain": float(routed["bacc"]) - float(control["bacc"]),
                            "routed_macro_f1": float(routed["macro_f1"]),
                            "uniform_topup_macro_f1": float(control["macro_f1"]),
                            "paired_macro_f1_gain_descriptive": float(routed["macro_f1"]) - float(control["macro_f1"]),
                            "primary_comparison_matched_budget": True,
                            "target_H_labels_used": False,
                            "diagnostic_only": True,
                        }
                    )
    return tuple(output)


def target_paired_deltas(
    metric_rows: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    selected_by_target = {
        str(row["outer_target"]): str(row["selected_action_id"])
        for row in selections
    }
    target_rows = [row for row in metric_rows if str(row.get("phase")) == "target"]
    by_key = {
        (
            str(row["outer_target"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            str(row["action_id"]),
        ): row
        for row in target_rows
    }
    if len(by_key) != len(target_rows) or set(selected_by_target) != set(CENTERS):
        raise ProtocolError("Residual top-up target metric geometry drifted.")
    output: list[dict[str, object]] = []
    for target in CENTERS:
        selected_id = selected_by_target[target]
        if selected_id not in {UNIFORM_TOPUP_ACTION_ID, ENERGY_TOPUP_ACTION_ID}:
            raise ProtocolError("Residual top-up selected action is invalid.")
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                prefix = (target, training_seed, generation_seed)
                try:
                    base = by_key[(*prefix, BASE_ONLY_ACTION_ID)]
                    control = by_key[(*prefix, UNIFORM_TOPUP_ACTION_ID)]
                    routed = by_key[(*prefix, ENERGY_TOPUP_ACTION_ID)]
                except KeyError as exc:
                    raise ProtocolError("Residual top-up target pairing is incomplete.") from exc
                selected = routed if selected_id == ENERGY_TOPUP_ACTION_ID else control
                output.append(
                    {
                        "schema_version": "midogpp_residual_topup_target_delta_v1",
                        "target_center": target,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "selected_action_id": selected_id,
                        "energy_topup_bacc": float(routed["bacc"]),
                        "uniform_topup_bacc": float(control["bacc"]),
                        "base_only_bacc": float(base["bacc"]),
                        "raw_energy_vs_uniform_bacc_delta": float(routed["bacc"]) - float(control["bacc"]),
                        "selected_vs_uniform_bacc_delta": float(selected["bacc"]) - float(control["bacc"]),
                        "uniform_topup_vs_base_budget_delta": float(control["bacc"]) - float(base["bacc"]),
                        "raw_energy_vs_uniform_macro_f1_delta_descriptive": float(routed["macro_f1"]) - float(control["macro_f1"]),
                        "primary_comparison_matched_budget": True,
                        "base_only_is_budget_reference": True,
                        "diagnostic_only": True,
                    }
                )
    return tuple(output)


def target_probability_ensemble_metrics(
    store: object,
    *,
    labels_by_sample_id: Mapping[str, int],
    selected_action_by_target: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    """Predeclared all-seed probability average, descriptive and nonselective."""

    index_rows = tuple(getattr(store, "index_rows", ()))
    slice_for = getattr(store, "slice_for", None)
    output: list[dict[str, object]] = []
    for target in CENTERS:
        for action_id in TARGET_ACTION_IDS:
            rows = [
                row
                for row in index_rows
                if row.get("phase") == "target"
                and str(row.get("outer_target")) == target
                and str(row.get("action_id")) == action_id
            ]
            rows.sort(key=lambda row: (int(row["training_seed"]), int(row["generation_seed"])))
            if len(rows) != len(TRAINING_SEEDS) * len(GENERATION_SEEDS):
                raise ProtocolError("Residual top-up ensemble seed coverage drifted.")
            sample_ids = _json_strings(rows[0]["evaluation_row_ids_json"])
            probabilities = []
            for row in rows:
                if _json_strings(row["evaluation_row_ids_json"]) != sample_ids:
                    raise ProtocolError("Residual top-up ensemble row order drifted.")
                _, probability = slice_for(row)
                probabilities.append(np.asarray(probability, dtype=np.float64))
            mean_probability = np.mean(np.stack(probabilities), axis=0)
            prediction = (mean_probability >= 0.5).astype(np.uint8)
            labels = np.asarray([labels_by_sample_id[item] for item in sample_ids], dtype=np.uint8)
            output.append(
                {
                    "schema_version": "midogpp_residual_topup_probability_ensemble_metric_v1",
                    "target_center": target,
                    "action_id": action_id,
                    "seed_probability_aggregation": "arithmetic_mean_all_nine_cells_no_seed_selection",
                    "seed_cell_count": len(rows),
                    "evaluation_row_count": len(labels),
                    "bacc": float(balanced_accuracy(labels.tolist(), prediction.tolist())),
                    "macro_f1": float(macro_f1(labels.tolist(), prediction.tolist())),
                    "threshold": 0.5,
                    "selected_for_policy": selected_action_by_target[target] == action_id,
                    "descriptive_only": True,
                }
            )
    return tuple(output)


def _json_strings(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Residual top-up JSON row list is malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProtocolError("Residual top-up JSON row identities are malformed.")
    return tuple(parsed)


__all__ = (
    "DEVELOPMENT_GAIN_COLUMNS",
    "ENSEMBLE_METRIC_COLUMNS",
    "METRIC_COLUMNS",
    "TARGET_DELTA_COLUMNS",
    "development_paired_gains",
    "score_prediction_store",
    "target_paired_deltas",
    "target_probability_ensemble_metrics",
)
