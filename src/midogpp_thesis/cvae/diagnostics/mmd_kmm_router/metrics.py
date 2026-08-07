"""Descriptive consumed-validation scoring after the global seal."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .prediction import PredictionStore


TARGET_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "arm_role",
    "training_seed",
    "generation_seed",
    "balanced_accuracy",
    "macro_f1",
    "evaluation_row_count",
    "prediction_sha256",
    "labels_used_for_scoring_only",
    "diagnostic_only",
)
PAIRED_DELTA_COLUMNS = (
    "schema_version",
    "target_center",
    "training_seed",
    "generation_seed",
    "mmd_kmm_bacc",
    "equal_union_bacc",
    "paired_bacc_delta",
    "mmd_kmm_macro_f1",
    "equal_union_macro_f1",
    "paired_macro_f1_delta",
    "control_fit_aliased",
    "diagnostic_only",
)


def score_predictions(
    store: PredictionStore,
    *,
    labels_by_sample_id: Mapping[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("MMD/KMM scoring requires scikit-learn.") from exc
    metrics: list[dict[str, object]] = []
    by_key: dict[tuple[str, int, int, str], dict[str, object]] = {}
    for row in store.index_rows:
        sample_ids = _json_list(row["evaluation_row_ids_json"])
        try:
            y_true = np.asarray([labels_by_sample_id[value] for value in sample_ids], dtype=np.uint8)
        except KeyError as exc:
            raise ProtocolError("MMD/KMM scoring labels do not cover sealed predictions.") from exc
        y_pred, _ = store.slice_for(row)
        if len(y_true) != len(y_pred) or set(y_true.tolist()) != {0, 1}:
            raise ProtocolError("MMD/KMM scoring label geometry drifted.")
        metric = {
            "schema_version": "midogpp_mmd_kmm_target_metric_v1",
            "target_center": str(row["target_center"]),
            "arm_role": str(row["arm_role"]),
            "training_seed": int(row["training_seed"]),
            "generation_seed": int(row["generation_seed"]),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "evaluation_row_count": len(y_true),
            "prediction_sha256": str(row["prediction_sha256"]),
            "labels_used_for_scoring_only": True,
            "diagnostic_only": True,
        }
        key = (
            metric["target_center"],
            metric["training_seed"],
            metric["generation_seed"],
            metric["arm_role"],
        )
        if key in by_key:
            raise ProtocolError("MMD/KMM target metric key duplicated.")
        by_key[key] = metric
        metrics.append(metric)
    deltas: list[dict[str, object]] = []
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                control = by_key[(target, training_seed, generation_seed, "equal_union_control")]
                routed = by_key[(target, training_seed, generation_seed, "mmd_kmm")]
                prediction_row = next(
                    row
                    for row in store.index_rows
                    if str(row["target_center"]) == target
                    and int(row["training_seed"]) == training_seed
                    and int(row["generation_seed"]) == generation_seed
                    and str(row["arm_role"]) == "mmd_kmm"
                )
                deltas.append(
                    {
                        "schema_version": "midogpp_mmd_kmm_paired_delta_v1",
                        "target_center": target,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "mmd_kmm_bacc": routed["balanced_accuracy"],
                        "equal_union_bacc": control["balanced_accuracy"],
                        "paired_bacc_delta": float(routed["balanced_accuracy"]) - float(control["balanced_accuracy"]),
                        "mmd_kmm_macro_f1": routed["macro_f1"],
                        "equal_union_macro_f1": control["macro_f1"],
                        "paired_macro_f1_delta": float(routed["macro_f1"]) - float(control["macro_f1"]),
                        "control_fit_aliased": _truthy(prediction_row["control_fit_aliased"]),
                        "diagnostic_only": True,
                    }
                )
    target_deltas: dict[str, list[float]] = defaultdict(list)
    for row in deltas:
        target_deltas[str(row["target_center"])].append(float(row["paired_bacc_delta"]))
    target_means = np.asarray(
        [np.mean(target_deltas[target], dtype=np.float64) for target in CENTERS],
        dtype=np.float64,
    )
    overall_delta = float(np.mean(target_means))
    cluster_se = float(np.std(target_means, ddof=1) / np.sqrt(len(target_means)))
    report = {
        "schema_version": "midogpp_mmd_kmm_scoring_summary_v1",
        "target_count": len(CENTERS),
        "seed_cells_per_target": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "metric_row_count": len(metrics),
        "paired_delta_row_count": len(deltas),
        "mean_equal_union_bacc": float(np.mean([float(row["equal_union_bacc"]) for row in deltas])),
        "mean_mmd_kmm_bacc": float(np.mean([float(row["mmd_kmm_bacc"]) for row in deltas])),
        "mean_paired_bacc_delta_center_equal": overall_delta,
        "target_cluster_standard_error": cluster_se,
        "normal_approximation_95_percent_interval": [
            overall_delta - 1.96 * cluster_se,
            overall_delta + 1.96 * cluster_se,
        ],
        "independent_cluster_unit": "target_center",
        "independent_cluster_count": len(CENTERS),
        "seeds_treated_as_nested_repeats": True,
        "routing_quality_claimed": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
    }
    return metrics, deltas, report


def _json_list(value: object) -> list[str]:
    import json

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("MMD/KMM prediction row IDs are malformed.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ProtocolError("MMD/KMM prediction row IDs are invalid.")
    return parsed


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


__all__ = (
    "PAIRED_DELTA_COLUMNS",
    "TARGET_METRIC_COLUMNS",
    "score_predictions",
)
