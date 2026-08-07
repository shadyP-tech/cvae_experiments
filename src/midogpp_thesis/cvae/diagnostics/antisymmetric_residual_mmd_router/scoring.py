"""Target-level scoring after the global cross-fit prediction seal."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ARM_ROLES,
    CENTERS,
    CONTROL_ARM,
    GENERATION_SEEDS,
    ROUTED_ARM,
    TRAINING_SEEDS,
)
from .partitions import CrossfitSurface
from .prediction import CrossfitPredictionStore


TARGET_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "arm_role",
    "training_seed",
    "generation_seed",
    "balanced_accuracy",
    "macro_f1",
    "evaluation_case_count",
    "evaluation_row_count",
    "aggregated_prediction_sha256",
    "component_prediction_sha256_json",
    "case_metrics_averaged",
    "labels_used_for_scoring_only",
    "cross_fitted_transductive_diagnostic",
    "diagnostic_only",
)

PAIRED_DELTA_COLUMNS = (
    "schema_version",
    "target_center",
    "training_seed",
    "generation_seed",
    "antisymmetric_residual_mmd_bacc",
    "equal_union_bacc",
    "paired_bacc_delta",
    "antisymmetric_residual_mmd_macro_f1",
    "equal_union_macro_f1",
    "paired_macro_f1_delta",
    "evaluation_case_count",
    "evaluation_row_count",
    "all_routed_cells_control_aliased",
    "case_metrics_averaged",
    "diagnostic_only",
)

# Explicit aliases make bundle/runner code readable without changing the
# established Stage-90 table vocabulary.
CROSSFIT_TARGET_METRIC_COLUMNS = TARGET_METRIC_COLUMNS
CROSSFIT_PAIRED_DELTA_COLUMNS = PAIRED_DELTA_COLUMNS


def score_case_crossfit_predictions(
    store: CrossfitPredictionStore,
    crossfit: CrossfitSurface,
    *,
    labels_by_sample_id: Mapping[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Concatenate case slices first, then score one target-level vector.

    Balanced accuracy is undefined as a useful comparison for many individual
    cases.  More importantly, averaging case BACCs would change the estimand.
    This function therefore reconstructs each complete target/seed/arm vector
    in deterministic fold order and computes exactly one metric row.
    """

    try:
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from scipy.stats import t as student_t
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Antisymmetric scoring requires scikit-learn.") from exc

    expected_label_ids = {
        row.sample_id for fold in crossfit.folds for row in fold.heldout_rows
    }
    if set(labels_by_sample_id) != expected_label_ids:
        raise ProtocolError(
            "Antisymmetric scoring label capability includes missing or extra rows."
        )

    indexed: dict[tuple[str, int, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in store.index_rows:
        key = (
            str(row.get("target_center")),
            _integer(row.get("training_seed"), "training seed"),
            _integer(row.get("generation_seed"), "generation seed"),
            str(row.get("arm_role")),
        )
        indexed[key].append(row)

    metrics: list[dict[str, object]] = []
    metric_by_key: dict[tuple[str, int, int, str], dict[str, object]] = {}
    routed_alias_by_seed: dict[tuple[str, int, int], bool] = {}
    for target in CENTERS:
        target_folds = crossfit.folds_by_target[target]
        expected_fold_ids = [fold.fold_id for fold in target_folds]
        expected_row_ids = [
            row.sample_id for fold in target_folds for row in fold.heldout_rows
        ]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for arm in ARM_ROLES:
                    key = (target, training_seed, generation_seed, arm)
                    cells = sorted(
                        indexed.get(key, ()),
                        key=lambda row: _integer(row.get("fold_ordinal"), "fold ordinal"),
                    )
                    if (
                        [str(row.get("fold_id")) for row in cells]
                        != expected_fold_ids
                        or len(cells) != len(target_folds)
                    ):
                        raise ProtocolError(
                            "Antisymmetric target scoring lacks complete fold coverage."
                        )
                    sample_ids: list[str] = []
                    predictions: list[np.ndarray] = []
                    component_hashes: list[str] = []
                    for row, fold in zip(cells, target_folds, strict=True):
                        ids = _json_list(row.get("evaluation_row_ids_json"))
                        if ids != [item.sample_id for item in fold.heldout_rows]:
                            raise ProtocolError(
                                "Antisymmetric scoring row IDs escaped their fold."
                            )
                        y_pred, _ = store.slice_for(row)
                        if len(y_pred) != len(ids):
                            raise ProtocolError(
                                "Antisymmetric scoring prediction length drifted."
                            )
                        sample_ids.extend(ids)
                        predictions.append(y_pred)
                        component_hashes.append(str(row["prediction_sha256"]))
                    if sample_ids != expected_row_ids or len(sample_ids) != len(set(sample_ids)):
                        raise ProtocolError(
                            "Antisymmetric scoring did not reconstruct the target exactly once."
                        )
                    try:
                        y_true = np.asarray(
                            [labels_by_sample_id[sample_id] for sample_id in sample_ids],
                            dtype=np.uint8,
                        )
                    except KeyError as exc:
                        raise ProtocolError(
                            "Antisymmetric scoring labels do not cover sealed rows."
                        ) from exc
                    y_pred = np.concatenate(predictions).astype(np.uint8, copy=False)
                    if (
                        y_true.shape != y_pred.shape
                        or set(y_true.tolist()) != {0, 1}
                        or not np.isin(y_pred, (0, 1)).all()
                    ):
                        raise ProtocolError(
                            "Antisymmetric target-level scoring geometry drifted."
                        )
                    metric = {
                        "schema_version": (
                            "midogpp_antisymmetric_residual_mmd_target_metric_v1"
                        ),
                        "target_center": target,
                        "arm_role": arm,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "balanced_accuracy": float(
                            balanced_accuracy_score(y_true, y_pred)
                        ),
                        "macro_f1": float(
                            f1_score(
                                y_true,
                                y_pred,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "evaluation_case_count": len(target_folds),
                        "evaluation_row_count": len(y_true),
                        "aggregated_prediction_sha256": _sha256_array(y_pred),
                        "component_prediction_sha256_json": _compact(
                            component_hashes
                        ),
                        "case_metrics_averaged": False,
                        "labels_used_for_scoring_only": True,
                        "cross_fitted_transductive_diagnostic": True,
                        "diagnostic_only": True,
                    }
                    if key in metric_by_key:
                        raise ProtocolError("Antisymmetric metric key duplicated.")
                    metric_by_key[key] = metric
                    metrics.append(metric)
                    if arm == ROUTED_ARM:
                        routed_alias_by_seed[(target, training_seed, generation_seed)] = all(
                            _truthy(row.get("control_fit_aliased")) for row in cells
                        )

    deltas: list[dict[str, object]] = []
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                control = metric_by_key[
                    (target, training_seed, generation_seed, CONTROL_ARM)
                ]
                routed = metric_by_key[
                    (target, training_seed, generation_seed, ROUTED_ARM)
                ]
                deltas.append(
                    {
                        "schema_version": (
                            "midogpp_antisymmetric_residual_mmd_paired_delta_v1"
                        ),
                        "target_center": target,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "antisymmetric_residual_mmd_bacc": routed[
                            "balanced_accuracy"
                        ],
                        "equal_union_bacc": control["balanced_accuracy"],
                        "paired_bacc_delta": float(routed["balanced_accuracy"])
                        - float(control["balanced_accuracy"]),
                        "antisymmetric_residual_mmd_macro_f1": routed["macro_f1"],
                        "equal_union_macro_f1": control["macro_f1"],
                        "paired_macro_f1_delta": float(routed["macro_f1"])
                        - float(control["macro_f1"]),
                        "evaluation_case_count": routed["evaluation_case_count"],
                        "evaluation_row_count": routed["evaluation_row_count"],
                        "all_routed_cells_control_aliased": routed_alias_by_seed[
                            (target, training_seed, generation_seed)
                        ],
                        "case_metrics_averaged": False,
                        "diagnostic_only": True,
                    }
                )

    expected_metric_count = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS) * len(ARM_ROLES)
    expected_delta_count = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
    if len(metrics) != expected_metric_count or len(deltas) != expected_delta_count:
        raise ProtocolError("Antisymmetric target-level metric counts drifted.")

    deltas_by_target: dict[str, list[float]] = defaultdict(list)
    control_by_target: dict[str, list[float]] = defaultdict(list)
    routed_by_target: dict[str, list[float]] = defaultdict(list)
    for row in deltas:
        target = str(row["target_center"])
        deltas_by_target[target].append(float(row["paired_bacc_delta"]))
        control_by_target[target].append(float(row["equal_union_bacc"]))
        routed_by_target[target].append(
            float(row["antisymmetric_residual_mmd_bacc"])
        )
    target_deltas = np.asarray(
        [np.mean(deltas_by_target[target], dtype=np.float64) for target in CENTERS],
        dtype=np.float64,
    )
    target_control = np.asarray(
        [np.mean(control_by_target[target], dtype=np.float64) for target in CENTERS],
        dtype=np.float64,
    )
    target_routed = np.asarray(
        [np.mean(routed_by_target[target], dtype=np.float64) for target in CENTERS],
        dtype=np.float64,
    )
    overall_delta = float(np.mean(target_deltas))
    cluster_se = float(np.std(target_deltas, ddof=1) / np.sqrt(len(target_deltas)))
    degrees_of_freedom = len(target_deltas) - 1
    two_sided_critical = float(student_t.ppf(0.975, degrees_of_freedom))
    one_sided_critical = float(student_t.ppf(0.95, degrees_of_freedom))
    report = {
        "schema_version": (
            "midogpp_antisymmetric_residual_mmd_scoring_summary_v1"
        ),
        "target_count": len(CENTERS),
        "evaluation_case_count": len(crossfit.folds),
        "seed_cells_per_target": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "metric_row_count": len(metrics),
        "paired_delta_row_count": len(deltas),
        "mean_equal_union_bacc_center_equal": float(np.mean(target_control)),
        "mean_antisymmetric_residual_mmd_bacc_center_equal": float(
            np.mean(target_routed)
        ),
        "mean_paired_bacc_delta_center_equal": overall_delta,
        "strict_target_mean_bacc_wins": int(np.sum(target_deltas > 0.0)),
        "strict_target_mean_bacc_losses": int(np.sum(target_deltas < 0.0)),
        "target_cluster_standard_error": cluster_se,
        "student_t_95_percent_interval": [
            overall_delta - two_sided_critical * cluster_se,
            overall_delta + two_sided_critical * cluster_se,
        ],
        "one_sided_95_percent_lower_confidence_bound": (
            overall_delta - one_sided_critical * cluster_se
        ),
        "cluster_degrees_of_freedom": degrees_of_freedom,
        "cluster_inference_descriptive_only": True,
        "independent_cluster_unit": "target_center",
        "independent_cluster_count": len(CENTERS),
        "seeds_treated_as_nested_repeats": True,
        "case_predictions_concatenated_before_target_metric": True,
        "case_level_metrics_averaged": False,
        "cross_fitted_transductive_diagnostic": True,
        "routing_quality_claimed": False,
        "heldout_target_utility_claimed": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
    }
    return metrics, deltas, report


def _json_list(value: object) -> list[str]:
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Antisymmetric scoring row IDs are malformed.") from exc
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ProtocolError("Antisymmetric scoring row IDs are invalid.")
    return payload


def _integer(value: object, role: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Antisymmetric {role} is invalid.") from exc


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


score_predictions = score_case_crossfit_predictions


__all__ = (
    "CROSSFIT_PAIRED_DELTA_COLUMNS",
    "CROSSFIT_TARGET_METRIC_COLUMNS",
    "PAIRED_DELTA_COLUMNS",
    "TARGET_METRIC_COLUMNS",
    "score_case_crossfit_predictions",
    "score_predictions",
)
