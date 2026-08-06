"""Consumed-label scoring and fixed-action diagnostic selection."""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import numpy as np

from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from .contracts import (
    ACTION_IDS,
    CONTROL_ACTION_ID,
    GENERATION_SEEDS,
    RHO_VALUES,
    TRAINING_SEEDS,
    development_queries,
)
from .prediction_io import FlatPredictionStore


METRIC_COLUMNS = (
    "schema_version",
    "phase",
    "outer_target",
    "query_center",
    "action_id",
    "arm_role",
    "training_seed",
    "generation_seed",
    "evaluation_row_count",
    "evaluation_class_0_count",
    "evaluation_class_1_count",
    "bacc",
    "macro_f1",
    "primary_metric",
    "macro_f1_role",
    "labels_used_for_scoring_only_after_prediction_seal",
)

ACTION_SUMMARY_COLUMNS = (
    "schema_version",
    "outer_target",
    "action_id",
    "rho",
    "development_cell_count",
    "mean_bacc",
    "mean_macro_f1_descriptive",
    "mean_paired_bacc_delta_vs_control",
    "mean_regret",
    "upper_quartile_cvar_regret",
    "mean_squared_l2_distance_from_uniform",
    "selection_objective",
    "objective_direction",
    "secondary_metric_may_select",
)

SELECTION_COLUMNS = (
    "schema_version",
    "outer_target",
    "selected_action_id",
    "selected_rho",
    "selected_mean_paired_bacc_delta_vs_control",
    "fallback_applied",
    "fallback_reason",
    "selection_uses_only_q_not_H_labels",
    "target_H_labels_used_for_selection",
    "diagnostic_only",
)

PAIRED_DELTA_COLUMNS = (
    "schema_version",
    "outer_target",
    "training_seed",
    "generation_seed",
    "selected_action_id",
    "selected_bacc",
    "control_bacc",
    "paired_bacc_delta",
    "selected_macro_f1",
    "control_macro_f1",
    "paired_macro_f1_delta_descriptive",
    "diagnostic_only",
)


def score_prediction_cells(
    store: FlatPredictionStore,
    *,
    labels_by_sample_id: Mapping[str, int],
    phase: str,
    outer_target: str,
    selected_target_action_id: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Score one already-sealed outer fold; fitting labels are not accepted."""

    if phase not in {"development", "target"}:
        raise ProtocolError("Dense residual metric phase is invalid.")
    rows = [
        row
        for row in store.index_rows
        if str(row.get("phase")) == phase
        and str(row.get("outer_target")) == str(outer_target)
        and (
            phase != "target"
            or str(row.get("arm_role")) == "control"
            or (
                str(row.get("arm_role")) == "selected"
                and str(row.get("action_id")) == selected_target_action_id
            )
        )
    ]
    if phase == "target" and selected_target_action_id not in ACTION_IDS:
        raise ProtocolError(
            "Dense residual target scoring requires one sealed selected action."
        )
    if not rows:
        raise ProtocolError("Dense residual scoring fold has no sealed cells.")
    output: list[dict[str, object]] = []
    for row in rows:
        sample_ids = _json_strings(row.get("evaluation_row_ids_json"))
        if len(sample_ids) != len(set(sample_ids)) or not sample_ids:
            raise ProtocolError("Dense residual scoring row identities are invalid.")
        try:
            labels = np.asarray(
                [int(labels_by_sample_id[sample_id]) for sample_id in sample_ids],
                dtype=np.uint8,
            )
        except KeyError as exc:
            raise ProtocolError(
                "Dense residual scoring labels do not cover a prediction slice."
            ) from exc
        if set(int(value) for value in labels.tolist()) != {0, 1}:
            raise ProtocolError(
                "Dense residual evaluation slice lacks both binary classes."
            )
        predictions, _ = store.slice_for(row)
        if len(predictions) != len(labels):
            raise ProtocolError("Dense residual labels and predictions do not align.")
        output.append(
            {
                "schema_version": "midogpp_dense_residual_metric_cell_v1",
                "phase": phase,
                "outer_target": str(outer_target),
                "query_center": str(row.get("query_center")),
                "action_id": str(row.get("action_id")),
                "arm_role": str(row.get("arm_role")),
                "training_seed": int(row.get("training_seed")),
                "generation_seed": int(row.get("generation_seed")),
                "evaluation_row_count": len(labels),
                "evaluation_class_0_count": int(np.sum(labels == 0)),
                "evaluation_class_1_count": int(np.sum(labels == 1)),
                "bacc": float(balanced_accuracy(labels.tolist(), predictions.tolist())),
                "macro_f1": float(macro_f1(labels.tolist(), predictions.tolist())),
                "primary_metric": "balanced_accuracy",
                "macro_f1_role": "secondary_descriptive_only",
                "labels_used_for_scoring_only_after_prediction_seal": True,
            }
        )
    return tuple(output)


def summarize_development_actions(
    metric_rows: Sequence[Mapping[str, object]],
    prediction_index_rows: Sequence[Mapping[str, object]],
    *,
    outer_target: str,
) -> tuple[dict[str, object], ...]:
    """Apply the predeclared robust-regret objective to fixed actions."""

    metrics = [
        row
        for row in metric_rows
        if str(row.get("phase")) == "development"
        and str(row.get("outer_target")) == str(outer_target)
    ]
    index_rows = [
        row
        for row in prediction_index_rows
        if str(row.get("phase")) == "development"
        and str(row.get("outer_target")) == str(outer_target)
    ]
    expected_base_cells = (
        len(development_queries(outer_target))
        * len(TRAINING_SEEDS)
        * len(GENERATION_SEEDS)
    )
    expected_cells = len(ACTION_IDS) * expected_base_cells
    if len(metrics) != expected_cells or len(index_rows) != expected_cells:
        raise ProtocolError("Dense residual development action coverage drifted.")
    metric_by_key: dict[tuple[str, str, int, int], Mapping[str, object]] = {}
    index_by_key: dict[tuple[str, str, int, int], Mapping[str, object]] = {}
    for row in metrics:
        key = _development_key(row)
        if key in metric_by_key:
            raise ProtocolError("Dense residual development metric cell duplicated.")
        metric_by_key[key] = row
    for row in index_rows:
        key = _development_key(row)
        if key in index_by_key:
            raise ProtocolError("Dense residual development index cell duplicated.")
        index_by_key[key] = row
    if set(metric_by_key) != set(index_by_key):
        raise ProtocolError("Dense residual metric/index cells do not align.")

    base_cells = {
        (query, train_seed, generation_seed)
        for _, query, train_seed, generation_seed in metric_by_key
    }
    if len(base_cells) != expected_base_cells:
        raise ProtocolError("Dense residual development seed/query geometry drifted.")
    best_by_cell = {
        cell: max(
            float(metric_by_key[(action, *cell)]["bacc"])
            for action in ACTION_IDS
        )
        for cell in base_cells
    }
    control_by_cell = {
        cell: float(metric_by_key[(CONTROL_ACTION_ID, *cell)]["bacc"])
        for cell in base_cells
    }

    summaries: list[dict[str, object]] = []
    for action_id, rho in zip(ACTION_IDS, RHO_VALUES, strict=True):
        action_metrics = [metric_by_key[(action_id, *cell)] for cell in sorted(base_cells)]
        baccs = np.asarray([float(row["bacc"]) for row in action_metrics], dtype=float)
        f1s = np.asarray([float(row["macro_f1"]) for row in action_metrics], dtype=float)
        regrets = np.asarray(
            [
                best_by_cell[cell]
                - float(metric_by_key[(action_id, *cell)]["bacc"])
                for cell in sorted(base_cells)
            ],
            dtype=float,
        )
        tail_count = int(math.ceil(0.25 * len(regrets)))
        cvar = float(np.mean(np.sort(regrets)[-tail_count:]))
        deltas = np.asarray(
            [
                float(metric_by_key[(action_id, *cell)]["bacc"])
                - control_by_cell[cell]
                for cell in sorted(base_cells)
            ],
            dtype=float,
        )
        deviations = []
        for cell in sorted(base_cells):
            weights = _json_float_mapping(index_by_key[(action_id, *cell)]["weights_json"])
            uniform = 1.0 / float(len(weights))
            deviations.append(sum((value - uniform) ** 2 for value in weights.values()))
        mean_regret = float(np.mean(regrets))
        mean_deviation = float(np.mean(deviations))
        objective = mean_regret + (0.5 * cvar) + (0.01 * mean_deviation)
        summaries.append(
            {
                "schema_version": "midogpp_dense_residual_action_summary_v1",
                "outer_target": str(outer_target),
                "action_id": action_id,
                "rho": float(rho),
                "development_cell_count": len(action_metrics),
                "mean_bacc": float(np.mean(baccs)),
                "mean_macro_f1_descriptive": float(np.mean(f1s)),
                "mean_paired_bacc_delta_vs_control": float(np.mean(deltas)),
                "mean_regret": mean_regret,
                "upper_quartile_cvar_regret": cvar,
                "mean_squared_l2_distance_from_uniform": mean_deviation,
                "selection_objective": objective,
                "objective_direction": "minimize",
                "secondary_metric_may_select": False,
            }
        )
    return tuple(summaries)


def choose_diagnostic_action(
    action_summaries: Sequence[Mapping[str, object]],
    *,
    outer_target: str,
) -> dict[str, object]:
    """Choose by objective and enforce the positive-delta safety fallback."""

    rows = [row for row in action_summaries if str(row.get("outer_target")) == str(outer_target)]
    if {str(row.get("action_id")) for row in rows} != set(ACTION_IDS):
        raise ProtocolError("Dense residual selection lacks the complete action library.")
    winner = min(
        rows,
        key=lambda row: (
            float(row["selection_objective"]),
            float(row["rho"]),
            str(row["action_id"]),
        ),
    )
    selected = winner
    fallback_applied = False
    fallback_reason = ""
    if (
        str(winner["action_id"]) != CONTROL_ACTION_ID
        and float(winner["mean_paired_bacc_delta_vs_control"]) <= 0.0
    ):
        selected = next(row for row in rows if str(row["action_id"]) == CONTROL_ACTION_ID)
        fallback_applied = True
        fallback_reason = "nonuniform_action_failed_strictly_positive_mean_paired_bacc_gate"
    return {
        "schema_version": "midogpp_dense_residual_diagnostic_selection_v1",
        "outer_target": str(outer_target),
        "selected_action_id": str(selected["action_id"]),
        "selected_rho": float(selected["rho"]),
        "selected_mean_paired_bacc_delta_vs_control": float(
            selected["mean_paired_bacc_delta_vs_control"]
        ),
        "fallback_applied": fallback_applied,
        "fallback_reason": fallback_reason,
        "selection_uses_only_q_not_H_labels": True,
        "target_H_labels_used_for_selection": False,
        "diagnostic_only": True,
    }


def paired_target_deltas(
    target_metrics: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Pair selected and exact-control target cells without seed selection."""

    selection_by_target = {
        str(row["outer_target"]): str(row["selected_action_id"])
        for row in selections
    }
    rows: list[dict[str, object]] = []
    for target in sorted(selection_by_target):
        target_rows = [row for row in target_metrics if str(row["outer_target"]) == target]
        by_key = {
            (
                str(row["arm_role"]),
                int(row["training_seed"]),
                int(row["generation_seed"]),
            ): row
            for row in target_rows
        }
        if len(by_key) != 2 * len(TRAINING_SEEDS) * len(GENERATION_SEEDS):
            raise ProtocolError("Dense residual target metric pairing is incomplete.")
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                selected = by_key[("selected", training_seed, generation_seed)]
                control = by_key[("control", training_seed, generation_seed)]
                rows.append(
                    {
                        "schema_version": "midogpp_dense_residual_paired_delta_v1",
                        "outer_target": target,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "selected_action_id": selection_by_target[target],
                        "selected_bacc": float(selected["bacc"]),
                        "control_bacc": float(control["bacc"]),
                        "paired_bacc_delta": float(selected["bacc"])
                        - float(control["bacc"]),
                        "selected_macro_f1": float(selected["macro_f1"]),
                        "control_macro_f1": float(control["macro_f1"]),
                        "paired_macro_f1_delta_descriptive": float(
                            selected["macro_f1"]
                        )
                        - float(control["macro_f1"]),
                        "diagnostic_only": True,
                    }
                )
    return tuple(rows)


def _development_key(row: Mapping[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row.get("action_id")),
        str(row.get("query_center")),
        int(row.get("training_seed")),
        int(row.get("generation_seed")),
    )


def _json_strings(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Dense residual JSON string list is malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProtocolError("Dense residual JSON row identities are malformed.")
    return tuple(parsed)


def _json_float_mapping(value: object) -> dict[str, float]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Dense residual JSON weight mapping is malformed.") from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ProtocolError("Dense residual JSON weight mapping is empty.")
    output = {str(key): float(raw) for key, raw in parsed.items()}
    if not np.isfinite(tuple(output.values())).all():
        raise ProtocolError("Dense residual JSON weights are non-finite.")
    return output


__all__ = (
    "ACTION_SUMMARY_COLUMNS",
    "METRIC_COLUMNS",
    "PAIRED_DELTA_COLUMNS",
    "SELECTION_COLUMNS",
    "choose_diagnostic_action",
    "paired_target_deltas",
    "score_prediction_cells",
    "summarize_development_actions",
)
