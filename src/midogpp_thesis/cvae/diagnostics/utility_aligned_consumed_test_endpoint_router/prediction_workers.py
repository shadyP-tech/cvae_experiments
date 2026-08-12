"""Spawn-safe, CPU-only classifier task worker."""

from __future__ import annotations

import os
from typing import Mapping

import numpy as np

from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from .checkpoint_store import PredictionCheckpoint, write_task_checkpoint
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .prediction_contracts import PredictionTask


SOURCE_ROWS_PER_CLASS = 270
SOURCE_FEATURE_DIM = 3_840


def execute_prediction_task(task: PredictionTask) -> PredictionCheckpoint:
    """Fit every physical action in one exact seed cell, without labels."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, ""}:
        raise ProtocolError("Endpoint-router classifier worker must be CUDA-free.")
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("Endpoint-router fitting requires threadpoolctl.") from exc
    sources = np.load(task.source_array_path, mmap_mode="r", allow_pickle=False)
    target = np.load(task.target_array_path, mmap_mode="r", allow_pickle=False)
    if (
        sources.shape != (81, 2 * SOURCE_ROWS_PER_CLASS, SOURCE_FEATURE_DIM)
        or sources.dtype != np.float32
        or target.ndim != 2
        or target.shape[1] != SOURCE_FEATURE_DIM
        or target.dtype != np.float32
    ):
        raise ProtocolError("Endpoint-router worker memmap geometry drifted.")
    eval_ordinals = np.asarray(
        (*task.support_row_ordinals, *task.evaluation_row_ordinals), dtype=np.int64
    )
    evaluation = np.ascontiguousarray(target[eval_ordinals], dtype=np.float32)
    spec = _classifier_from_payload(task.classifier_payload)
    probability_rows: list[np.ndarray] = []
    action_records: list[dict[str, object]] = []
    with threadpool_limits(limits=3):
        for action in task.actions:
            train_x, train_y = _compose_training(
                sources,
                action.rows_per_class_by_source,
                training_seed=task.training_seed,
                generation_seed=task.generation_seed,
            )
            composition_hash = canonical_sha256(
                {
                    "schema_version": "midogpp_endpoint_router_composition_v1",
                    "source_stream_lock_hash": task.source_stream_lock_hash,
                    "action_hash": action.action_hash,
                    "training_seed": task.training_seed,
                    "generation_seed": task.generation_seed,
                    "train_row_count": len(train_y),
                    "class_row_counts": [int(np.sum(train_y == value)) for value in (0, 1)],
                    "source_prefix_only": True,
                }
            )
            fitted = fit_logistic_classifier(
                train_x,
                train_y,
                evaluation,
                spec=spec,
            )
            matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or matrix.shape != (len(evaluation), 2)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
                or fitted.classifier_config_hash != spec.config_hash
            ):
                raise ProtocolError("Endpoint-router classifier fit drifted.")
            positive = np.ascontiguousarray(matrix[:, 1], dtype=np.float32)
            probability_hash = array_sha256(positive)
            fit_unhashed = {
                "schema_version": "midogpp_endpoint_router_classifier_fit_v1",
                "task_hash": task.task_hash,
                "action_id": action.action_id,
                "action_hash": action.action_hash,
                "composition_hash": composition_hash,
                "classifier_config_hash": fitted.classifier_config_hash,
                "scaler_state_hash": fitted.scaler_state_hash,
                "probability_sha256": probability_hash,
                "support_row_count": len(task.support_row_ids),
                "evaluation_row_count": len(task.evaluation_row_ids),
                "labels_available": False,
                "scaler_fit": "synthetic_train_only",
            }
            probability_rows.append(positive)
            action_records.append(
                {
                    "action_id": action.action_id,
                    "action_hash": action.action_hash,
                    "probability_sha256": probability_hash,
                    "prediction_sha256": array_sha256(
                        (positive >= np.float32(0.5)).astype(np.uint8)
                    ),
                    "composition_hash": composition_hash,
                    "scaler_state_hash": str(fitted.scaler_state_hash),
                    "fit_provenance_hash": canonical_sha256(fit_unhashed),
                    "converged": True,
                }
            )
    return write_task_checkpoint(
        task,
        probabilities=np.ascontiguousarray(np.stack(probability_rows), dtype=np.float32),
        action_records=action_records,
    )


def _compose_training(
    source_array: np.ndarray,
    rows_per_class_by_source: Mapping[str, int],
    *,
    training_seed: int,
    generation_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if training_seed not in TRAINING_SEEDS or generation_seed not in GENERATION_SEEDS:
        raise ProtocolError("Endpoint-router source seed is invalid.")
    negative: list[np.ndarray] = []
    positive: list[np.ndarray] = []
    for source, count in rows_per_class_by_source.items():
        if source not in CENTERS or int(count) <= 0 or int(count) > SOURCE_ROWS_PER_CLASS:
            raise ProtocolError("Endpoint-router source prefix geometry drifted.")
        ordinal = (
            CENTERS.index(source) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
            + TRAINING_SEEDS.index(training_seed) * len(GENERATION_SEEDS)
            + GENERATION_SEEDS.index(generation_seed)
        )
        block = source_array[ordinal]
        negative.append(block[: int(count)])
        positive.append(block[SOURCE_ROWS_PER_CLASS : SOURCE_ROWS_PER_CLASS + int(count)])
    x = np.ascontiguousarray(np.concatenate((*negative, *positive)), dtype=np.float32)
    per_class = sum(map(int, rows_per_class_by_source.values()))
    y = np.concatenate(
        (np.zeros(per_class, dtype=np.int64), np.ones(per_class, dtype=np.int64))
    )
    if x.shape != (2 * per_class, SOURCE_FEATURE_DIM):
        raise ProtocolError("Endpoint-router composed training matrix drifted.")
    return x, y


def _classifier_from_payload(payload: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=str(payload["family"]),
            C=float(payload["C"]),
            penalty=str(payload["penalty"]),
            solver=str(payload["solver"]),
            max_iter=int(payload["max_iter"]),
            class_weight=(
                None if payload["class_weight"] is None else str(payload["class_weight"])
            ),
            random_state=int(payload["random_state"]),
            l1_ratio=(None if payload["l1_ratio"] is None else float(payload["l1_ratio"])),
            threshold_policy=str(payload["threshold_policy"]),
            scaler_fit=str(payload["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Endpoint-router classifier payload is malformed.") from exc


__all__ = ("execute_prediction_task",)
