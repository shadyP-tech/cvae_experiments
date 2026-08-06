"""Label-consuming metric scoring for persisted source-inner predictions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from .cache_inputs import ScoringLabels
from .contracts import (
    CENTERS,
    EXPECTED_UTILITY_ROW_COUNT,
    evaluation_order_hash,
    expected_utility_keys,
)
from .prediction import PredictionPass, array_sha256


UTILITY_COLUMNS = (
    "schema_version",
    "utility_row_id",
    "pseudo_target_center",
    "candidate_source_center",
    "training_seed",
    "generation_seed",
    "fit_id",
    "fit_ordinal",
    "prediction_array_row",
    "source_stream_id",
    "expert_lock_hash",
    "checkpoint_hash",
    "checkpoint_file_sha256",
    "frame_hash",
    "frame_file_sha256",
    "sampler_state_hash",
    "sampler_file_sha256",
    "generated_block_sha256",
    "generated_row_count",
    "generated_rows_per_class",
    "classifier_family",
    "classifier_config_hash",
    "scaler_state_hash",
    "classifier_classes",
    "classifier_n_iter",
    "classifier_converged",
    "eval_row_count",
    "eval_class_0_count",
    "eval_class_1_count",
    "eval_case_count",
    "eval_row_hash",
    "eval_prediction_sha256",
    "bacc",
    "macro_f1",
    "primary_metric",
    "macro_f1_role",
    "eval_labels_used_for_scoring_only",
    "pseudo_target_expert_excluded",
    "outer_target_instantiated",
    "candidate_ranking_performed",
    "policy_selection_performed",
    "seed_selection_performed",
)

CASE_CONFUSION_COLUMNS = (
    "schema_version",
    "case_confusion_row_id",
    "utility_row_id",
    "pseudo_target_center",
    "candidate_source_center",
    "training_seed",
    "generation_seed",
    "case_id",
    "case_row_hash",
    "tn",
    "fp",
    "fn",
    "tp",
    "n",
    "true_class_0_count",
    "true_class_1_count",
    "eval_labels_used_for_scoring_only",
)


def score_prediction_pass(
    prediction_pass: PredictionPass,
    scoring_labels: ScoringLabels,
    *,
    expected_utility_rows: int = EXPECTED_UTILITY_ROW_COUNT,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Slice complete predictions by ``q`` and consume labels for metrics only."""

    if scoring_labels.evaluation_order_hash != prediction_pass.evaluation_order_hash:
        raise ProtocolError("Scoring labels do not match the persisted prediction order.")
    labels = np.asarray(scoring_labels.labels, dtype=np.uint8)
    if labels.shape != (len(prediction_pass.evaluation_rows),):
        raise ProtocolError("Scoring labels do not align with prediction rows.")
    rows = prediction_pass.evaluation_rows
    centers = np.asarray([str(getattr(row, "center")) for row in rows], dtype=str)
    sample_ids = np.asarray(
        [str(getattr(row, "sample_id")) for row in rows], dtype=str
    )
    case_ids = np.asarray([str(getattr(row, "case_id")) for row in rows], dtype=str)
    utility_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for fit in prediction_pass.fit_rows:
        candidate = str(fit["source_center"])
        prediction = prediction_pass.y_pred[int(fit["prediction_array_row"])]
        for pseudo_target in CENTERS:
            if pseudo_target == candidate:
                continue
            mask = centers == pseudo_target
            q_labels = labels[mask]
            q_predictions = prediction[mask]
            q_rows = tuple(
                row for row in rows if str(getattr(row, "center")) == pseudo_target
            )
            if (
                q_labels.size == 0
                or sorted(set(int(value) for value in q_labels.tolist())) != [0, 1]
                or len(q_rows) != int(q_labels.size)
            ):
                raise ProtocolError("Pseudo-target scoring slice lacks binary coverage.")
            q_sample_ids = sample_ids[mask]
            q_case_ids = case_ids[mask]
            eval_row_hash = evaluation_order_hash(q_rows)  # type: ignore[arg-type]
            utility_row_id = stable_hash(
                {
                    "pseudo_target_center": pseudo_target,
                    "candidate_source_center": candidate,
                    "training_seed": int(fit["training_seed"]),
                    "generation_seed": int(fit["generation_seed"]),
                    "fit_id": fit["fit_id"],
                    "eval_row_hash": eval_row_hash,
                }
            )
            utility = {
                "schema_version": (
                    "midogpp_uniform_b_v2_source_inner_candidate_utility_row_v1"
                ),
                "utility_row_id": utility_row_id,
                "pseudo_target_center": pseudo_target,
                "candidate_source_center": candidate,
                "training_seed": int(fit["training_seed"]),
                "generation_seed": int(fit["generation_seed"]),
                "fit_id": fit["fit_id"],
                "fit_ordinal": int(fit["fit_ordinal"]),
                "prediction_array_row": int(fit["prediction_array_row"]),
                "source_stream_id": fit["source_stream_id"],
                "expert_lock_hash": fit["expert_lock_hash"],
                "checkpoint_hash": fit["checkpoint_hash"],
                "checkpoint_file_sha256": fit["checkpoint_file_sha256"],
                "frame_hash": fit["frame_hash"],
                "frame_file_sha256": fit["frame_file_sha256"],
                "sampler_state_hash": fit["sampler_state_hash"],
                "sampler_file_sha256": fit["sampler_file_sha256"],
                "generated_block_sha256": fit["generated_block_sha256"],
                "generated_row_count": int(fit["generated_row_count"]),
                "generated_rows_per_class": int(fit["generated_rows_per_class"]),
                "classifier_family": fit["classifier_family"],
                "classifier_config_hash": fit["classifier_config_hash"],
                "scaler_state_hash": fit["scaler_state_hash"],
                "classifier_classes": fit["classifier_classes"],
                "classifier_n_iter": fit["classifier_n_iter"],
                "classifier_converged": bool(fit["classifier_converged"]),
                "eval_row_count": int(q_labels.size),
                "eval_class_0_count": int(np.sum(q_labels == 0)),
                "eval_class_1_count": int(np.sum(q_labels == 1)),
                "eval_case_count": len(set(q_case_ids.tolist())),
                "eval_row_hash": eval_row_hash,
                "eval_prediction_sha256": array_sha256(
                    q_predictions.astype(np.uint8)
                ),
                "bacc": float(
                    balanced_accuracy(q_labels.tolist(), q_predictions.tolist())
                ),
                "macro_f1": float(macro_f1(q_labels.tolist(), q_predictions.tolist())),
                "primary_metric": "balanced_accuracy",
                "macro_f1_role": "secondary_descriptive_only",
                "eval_labels_used_for_scoring_only": True,
                "pseudo_target_expert_excluded": True,
                "outer_target_instantiated": False,
                "candidate_ranking_performed": False,
                "policy_selection_performed": False,
                "seed_selection_performed": False,
            }
            utility_rows.append(utility)
            for case_id in sorted(set(q_case_ids.tolist())):
                case_mask = q_case_ids == case_id
                case_truth = q_labels[case_mask]
                case_prediction = q_predictions[case_mask]
                tn = int(np.sum((case_truth == 0) & (case_prediction == 0)))
                fp = int(np.sum((case_truth == 0) & (case_prediction == 1)))
                fn = int(np.sum((case_truth == 1) & (case_prediction == 0)))
                tp = int(np.sum((case_truth == 1) & (case_prediction == 1)))
                case_sample_ids = q_sample_ids[case_mask].tolist()
                case_row_hash = stable_hash(
                    {
                        "pseudo_target_center": pseudo_target,
                        "case_id": case_id,
                        "ordered_sample_ids": case_sample_ids,
                    }
                )
                case_rows.append(
                    {
                        "schema_version": "midogpp_uniform_b_v2_case_confusion_v1",
                        "case_confusion_row_id": stable_hash(
                            {"utility_row_id": utility_row_id, "case_id": case_id}
                        ),
                        "utility_row_id": utility_row_id,
                        "pseudo_target_center": pseudo_target,
                        "candidate_source_center": candidate,
                        "training_seed": int(fit["training_seed"]),
                        "generation_seed": int(fit["generation_seed"]),
                        "case_id": case_id,
                        "case_row_hash": case_row_hash,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                        "n": int(case_truth.size),
                        "true_class_0_count": int(np.sum(case_truth == 0)),
                        "true_class_1_count": int(np.sum(case_truth == 1)),
                        "eval_labels_used_for_scoring_only": True,
                    }
                )
    keys = {
        (
            str(row["pseudo_target_center"]),
            str(row["candidate_source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        )
        for row in utility_rows
    }
    if (
        len(utility_rows) != int(expected_utility_rows)
        or len(keys) != len(utility_rows)
        or (
            int(expected_utility_rows) == EXPECTED_UTILITY_ROW_COUNT
            and keys != set(expected_utility_keys())
        )
    ):
        raise ProtocolError("Source-inner candidate utility coverage drifted.")
    if any(
        str(row["pseudo_target_center"]) == str(row["candidate_source_center"])
        for row in utility_rows
    ):
        raise ProtocolError("Source-inner utility contains q == e.")
    _validate_case_confusion_aggregation(utility_rows, case_rows)
    return tuple(utility_rows), tuple(case_rows)


def reconstruct_metrics_from_case_confusions(
    rows: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    tn = sum(int(row["tn"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    if tn + fp <= 0 or tp + fn <= 0:
        raise ProtocolError("Case-confusion aggregation lacks one true class.")
    bacc = 0.5 * (float(tn) / float(tn + fp) + float(tp) / float(tp + fn))
    f1_zero_denominator = (2 * tn) + fn + fp
    f1_one_denominator = (2 * tp) + fp + fn
    f1_zero = (
        0.0
        if f1_zero_denominator == 0
        else float(2 * tn) / float(f1_zero_denominator)
    )
    f1_one = (
        0.0
        if f1_one_denominator == 0
        else float(2 * tp) / float(f1_one_denominator)
    )
    return float(bacc), float(0.5 * (f1_zero + f1_one))


def _validate_case_confusion_aggregation(
    utility_rows: Sequence[Mapping[str, object]],
    case_rows: Sequence[Mapping[str, object]],
) -> None:
    by_utility: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in case_rows:
        by_utility[str(row["utility_row_id"])].append(row)
        if (
            int(row["tn"]) + int(row["fp"]) + int(row["fn"]) + int(row["tp"])
            != int(row["n"])
            or int(row["tn"]) + int(row["fp"])
            != int(row["true_class_0_count"])
            or int(row["fn"]) + int(row["tp"])
            != int(row["true_class_1_count"])
        ):
            raise ProtocolError("Case-confusion counts are internally inconsistent.")
    for utility in utility_rows:
        rows = by_utility.get(str(utility["utility_row_id"]), [])
        if len(rows) != int(utility["eval_case_count"]):
            raise ProtocolError("Case-confusion coverage differs from utility case count.")
        if sum(int(row["n"]) for row in rows) != int(utility["eval_row_count"]):
            raise ProtocolError("Case-confusion rows do not cover the utility rows.")
        bacc, f1 = reconstruct_metrics_from_case_confusions(rows)
        if not np.isclose(
            bacc, float(utility["bacc"]), rtol=0.0, atol=1e-15
        ) or not np.isclose(
            f1, float(utility["macro_f1"]), rtol=0.0, atol=1e-15
        ):
            raise ProtocolError(
                "Case-confusion metrics do not reconstruct utility metrics."
            )


__all__ = (
    "CASE_CONFUSION_COLUMNS",
    "UTILITY_COLUMNS",
    "reconstruct_metrics_from_case_confusions",
    "score_prediction_pass",
)
