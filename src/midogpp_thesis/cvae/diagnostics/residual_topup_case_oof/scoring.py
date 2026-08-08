"""Terminal scoring of immutable case-OOF predictions."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np

from ...metrics import balanced_accuracy, macro_f1
from ...protocol import ProtocolError
from .contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    expected_action_ids,
)


CENTER_SEED_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "action_id",
    "action_role",
    "training_seed",
    "generation_seed",
    "fold_count",
    "evaluation_row_count",
    "evaluation_class_0_count",
    "evaluation_class_1_count",
    "bacc",
    "macro_f1",
    "endpoint_role",
    "labels_used_only_after_global_prediction_seal",
    "selector_or_fallback_performed",
    "diagnostic_only",
)

CENTER_ENSEMBLE_METRIC_COLUMNS = (
    "schema_version",
    "target_center",
    "action_id",
    "action_role",
    "seed_probability_aggregation",
    "seed_cell_count",
    "fold_count",
    "evaluation_row_count",
    "evaluation_class_0_count",
    "evaluation_class_1_count",
    "bacc",
    "macro_f1",
    "threshold",
    "endpoint_role",
    "technical_seed_repeats_are_not_independent_units",
    "selector_or_fallback_performed",
    "diagnostic_only",
)


def score_center_seed_cells(
    store: object,
    *,
    labels_by_sample_id: Mapping[str, int],
    crossfit: object,
) -> tuple[dict[str, object], ...]:
    """Aggregate whole-case OOF slices into one target/action/seed score."""

    index_rows = tuple(getattr(store, "index_rows", ()))
    slice_for = getattr(store, "slice_for", None)
    if not index_rows or not callable(slice_for):
        raise ProtocolError("Case-OOF prediction store is unavailable.")
    output: list[dict[str, object]] = []
    for target in CENTERS:
        expected_folds = tuple(crossfit.folds_by_target[target])
        for action_id in expected_action_ids(target):
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    rows = _group_rows(
                        index_rows,
                        target=target,
                        action_id=action_id,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                    )
                    if tuple(str(row["fold_id"]) for row in rows) != tuple(
                        fold.fold_id for fold in expected_folds
                    ):
                        raise ProtocolError(
                            "Case-OOF center/seed fold coverage drifted."
                        )
                    sample_ids: list[str] = []
                    predictions: list[np.ndarray] = []
                    for row in rows:
                        ids = _json_strings(row["evaluation_row_ids_json"])
                        pred, _ = slice_for(row)
                        if len(ids) != len(pred):
                            raise ProtocolError(
                                "Case-OOF prediction identity coverage drifted."
                            )
                        sample_ids.extend(ids)
                        predictions.append(np.asarray(pred, dtype=np.uint8))
                    labels = _labels(sample_ids, labels_by_sample_id)
                    prediction = np.concatenate(predictions)
                    if prediction.shape != labels.shape:
                        raise ProtocolError(
                            "Case-OOF labels and predictions drifted."
                        )
                    output.append(
                        {
                            "schema_version": "midogpp_residual_topup_case_oof_center_seed_metric_v1",
                            "target_center": target,
                            "action_id": action_id,
                            "action_role": str(rows[0]["action_role"]),
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "fold_count": len(rows),
                            "evaluation_row_count": len(labels),
                            "evaluation_class_0_count": int(np.sum(labels == 0)),
                            "evaluation_class_1_count": int(np.sum(labels == 1)),
                            "bacc": float(
                                balanced_accuracy(
                                    labels.tolist(), prediction.tolist()
                                )
                            ),
                            "macro_f1": float(
                                macro_f1(labels.tolist(), prediction.tolist())
                            ),
                            "endpoint_role": "descriptive_paired_seed_cell_not_independent_inference_unit",
                            "labels_used_only_after_global_prediction_seal": True,
                            "selector_or_fallback_performed": False,
                            "diagnostic_only": True,
                        }
                    )
    return tuple(output)


def score_center_probability_ensembles(
    store: object,
    *,
    labels_by_sample_id: Mapping[str, int],
    crossfit: object,
) -> tuple[dict[str, object], ...]:
    """Compute the predeclared all-nine-seed probability endpoint per center."""

    index_rows = tuple(getattr(store, "index_rows", ()))
    slice_for = getattr(store, "slice_for", None)
    if not index_rows or not callable(slice_for):
        raise ProtocolError("Case-OOF prediction store is unavailable.")
    output: list[dict[str, object]] = []
    for target in CENTERS:
        expected_folds = tuple(crossfit.folds_by_target[target])
        for action_id in expected_action_ids(target):
            probability_by_seed: list[np.ndarray] = []
            canonical_ids: tuple[str, ...] | None = None
            action_role: str | None = None
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    rows = _group_rows(
                        index_rows,
                        target=target,
                        action_id=action_id,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                    )
                    if tuple(str(row["fold_id"]) for row in rows) != tuple(
                        fold.fold_id for fold in expected_folds
                    ):
                        raise ProtocolError(
                            "Case-OOF ensemble fold coverage drifted."
                        )
                    ids: list[str] = []
                    probabilities: list[np.ndarray] = []
                    for row in rows:
                        row_ids = _json_strings(row["evaluation_row_ids_json"])
                        _, prob = slice_for(row)
                        if len(row_ids) != len(prob):
                            raise ProtocolError(
                                "Case-OOF ensemble identity coverage drifted."
                            )
                        ids.extend(row_ids)
                        probabilities.append(np.asarray(prob, dtype=np.float64))
                    identity = tuple(ids)
                    if canonical_ids is None:
                        canonical_ids = identity
                        action_role = str(rows[0]["action_role"])
                    elif identity != canonical_ids:
                        raise ProtocolError(
                            "Case-OOF ensemble row order drifted across seeds."
                        )
                    probability_by_seed.append(np.concatenate(probabilities))
            if canonical_ids is None or action_role is None:
                raise ProtocolError("Case-OOF ensemble is empty.")
            if len(probability_by_seed) != len(TRAINING_SEEDS) * len(
                GENERATION_SEEDS
            ):
                raise ProtocolError("Case-OOF ensemble seed coverage drifted.")
            labels = _labels(canonical_ids, labels_by_sample_id)
            mean_probability = np.mean(
                np.stack(probability_by_seed), axis=0, dtype=np.float64
            )
            prediction = (mean_probability >= 0.5).astype(np.uint8)
            output.append(
                {
                    "schema_version": "midogpp_residual_topup_case_oof_center_ensemble_metric_v1",
                    "target_center": target,
                    "action_id": action_id,
                    "action_role": action_role,
                    "seed_probability_aggregation": "arithmetic_mean_all_nine_seed_cells_no_seed_selection",
                    "seed_cell_count": len(probability_by_seed),
                    "fold_count": len(expected_folds),
                    "evaluation_row_count": len(labels),
                    "evaluation_class_0_count": int(np.sum(labels == 0)),
                    "evaluation_class_1_count": int(np.sum(labels == 1)),
                    "bacc": float(
                        balanced_accuracy(labels.tolist(), prediction.tolist())
                    ),
                    "macro_f1": float(
                        macro_f1(labels.tolist(), prediction.tolist())
                    ),
                    "threshold": 0.5,
                    "endpoint_role": "predeclared_operational_primary_endpoint",
                    "technical_seed_repeats_are_not_independent_units": True,
                    "selector_or_fallback_performed": False,
                    "diagnostic_only": True,
                }
            )
    return tuple(output)


def _group_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    target: str,
    action_id: str,
    training_seed: int,
    generation_seed: int,
) -> tuple[Mapping[str, object], ...]:
    selected = tuple(
        row
        for row in rows
        if str(row["target_center"]) == target
        and str(row["action_id"]) == action_id
        and int(row["training_seed"]) == training_seed
        and int(row["generation_seed"]) == generation_seed
    )
    return tuple(sorted(selected, key=lambda row: int(row["fold_ordinal"])))


def _labels(
    sample_ids: Sequence[str], labels_by_sample_id: Mapping[str, int]
) -> np.ndarray:
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ProtocolError("Case-OOF center scoring identities are invalid.")
    try:
        labels = np.asarray(
            [int(labels_by_sample_id[value]) for value in sample_ids],
            dtype=np.uint8,
        )
    except KeyError as exc:
        raise ProtocolError("Case-OOF labels do not cover a center.") from exc
    if set(labels.tolist()) != {0, 1}:
        raise ProtocolError("Case-OOF center score lacks both classes.")
    return labels


def _json_strings(value: object) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Case-OOF prediction row IDs are malformed.") from exc
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) for item in parsed
    ):
        raise ProtocolError("Case-OOF prediction row IDs are malformed.")
    return tuple(parsed)


__all__ = (
    "CENTER_ENSEMBLE_METRIC_COLUMNS",
    "CENTER_SEED_METRIC_COLUMNS",
    "score_center_probability_ensembles",
    "score_center_seed_cells",
)
