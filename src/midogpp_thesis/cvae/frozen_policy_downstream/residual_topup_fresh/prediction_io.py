"""Atomic prediction-task checkpoint persistence and resume validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .contracts import (
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EvaluationPlan,
    PredictionCell,
    expected_action_ids,
)
from .prediction_contracts import (
    PREDICTION_CELL_SCHEMA,
    PREDICTION_TASK_SCHEMA,
    PredictionTaskRecord,
    PredictionTaskSpec,
)


def write_task_checkpoint(
    task: PredictionTaskSpec,
    *,
    rows: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    payload = task.payload
    probability_path = Path(str(payload["probability_path"]))
    prediction_path = Path(str(payload["prediction_path"]))
    _atomic_save_npy(
        probability_path,
        np.ascontiguousarray(probabilities, dtype=np.float32),
    )
    _atomic_save_npy(
        prediction_path,
        np.ascontiguousarray(predictions, dtype=np.uint8),
    )
    unhashed = {
        "schema_version": PREDICTION_TASK_SCHEMA,
        "status": "COMPLETE",
        "task_id": payload["task_id"],
        "task_hash": payload["task_hash"],
        "target_center": payload["target_center"],
        "training_seed": payload["training_seed"],
        "generation_seed": payload["generation_seed"],
        "plan_hash": payload["plan_hash"],
        "probability_member": str(
            probability_path.relative_to(Path(str(payload["root"])))
        ),
        "prediction_member": str(
            prediction_path.relative_to(Path(str(payload["root"])))
        ),
        "probability_file_sha256": sha256_file(probability_path),
        "prediction_file_sha256": sha256_file(prediction_path),
        "row_ids": list(payload["evaluation_row_ids"]),
        "cells": [dict(row) for row in rows],
        "all_13_actions_fitted": True,
        "labels_available_to_fit_or_predict": False,
    }
    checkpoint = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    atomic_json(Path(str(payload["metadata_path"])), checkpoint)


def try_load_task(
    task: PredictionTaskSpec,
    *,
    plan: EvaluationPlan,
) -> tuple[
    PredictionTaskRecord,
    tuple[Mapping[str, object], ...],
    tuple[PredictionCell, ...],
] | None:
    payload = task.payload
    metadata_path = Path(str(payload["metadata_path"]))
    probability_path = Path(str(payload["probability_path"]))
    prediction_path = Path(str(payload["prediction_path"]))
    if (
        not metadata_path.is_file()
        or not probability_path.is_file()
        or not prediction_path.is_file()
    ):
        return None
    try:
        metadata = read_json(metadata_path)
        unhashed = {
            key: value
            for key, value in metadata.items()
            if key != "checkpoint_hash"
        }
        cells = metadata.get("cells")
        if (
            metadata.get("checkpoint_hash") != stable_hash(unhashed)
            or metadata.get("status") != "COMPLETE"
            or metadata.get("task_hash") != payload["task_hash"]
            or metadata.get("task_id") != payload["task_id"]
            or metadata.get("plan_hash") != plan.plan_hash
            or metadata.get("all_13_actions_fitted") is not True
            or metadata.get("labels_available_to_fit_or_predict") is not False
            or metadata.get("probability_file_sha256")
            != sha256_file(probability_path)
            or metadata.get("prediction_file_sha256")
            != sha256_file(prediction_path)
            or not isinstance(cells, list)
            or len(cells) != EXPECTED_ACTION_COUNT_PER_TARGET
        ):
            return None
        probabilities = np.load(
            probability_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        predictions = np.load(
            prediction_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        row_ids = tuple(str(value) for value in metadata.get("row_ids", ()))
        if (
            probabilities.dtype != np.float32
            or probabilities.shape
            != (EXPECTED_ACTION_COUNT_PER_TARGET, len(row_ids))
            or predictions.dtype != np.uint8
            or predictions.shape != probabilities.shape
            or not np.isfinite(probabilities).all()
        ):
            return None
        target = str(payload["target_center"])
        training_seed = int(payload["training_seed"])
        generation_seed = int(payload["generation_seed"])
        expected_actions = expected_action_ids(target)
        output_rows: list[Mapping[str, object]] = []
        output_predictions: list[PredictionCell] = []
        for index, (action_id, raw) in enumerate(
            zip(expected_actions, cells, strict=True)
        ):
            if not isinstance(raw, Mapping):
                return None
            row = {str(key): value for key, value in raw.items()}
            planned = plan.action_for(target, action_id)
            probability = np.ascontiguousarray(
                probabilities[index],
                dtype=np.float32,
            )
            prediction = np.ascontiguousarray(
                predictions[index],
                dtype=np.uint8,
            )
            if (
                row.get("schema_version") != PREDICTION_CELL_SCHEMA
                or row.get("target_center") != target
                or row.get("training_seed") != training_seed
                or row.get("generation_seed") != generation_seed
                or row.get("action_id") != action_id
                or row.get("action_hash") != planned.action_hash
                or row.get("probability_sha256") != array_sha256(probability)
                or row.get("prediction_sha256") != array_sha256(prediction)
                or row.get("labels_available_to_fit_or_predict") is not False
                or row.get("classifier_converged") is not True
            ):
                return None
            output_rows.append(MappingProxyType(row))
            output_predictions.append(
                PredictionCell(
                    target_center=target,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    action_id=action_id,
                    action_hash=planned.action_hash,
                    evaluation_row_ids=row_ids,
                    probabilities=probability,
                )
            )
        root = Path(str(payload["root"]))
        record = PredictionTaskRecord(
            target_center=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
            task_id=str(payload["task_id"]),
            task_hash=str(payload["task_hash"]),
            metadata_member=str(metadata_path.relative_to(root)),
            probability_member=str(probability_path.relative_to(root)),
            prediction_member=str(prediction_path.relative_to(root)),
            metadata_sha256=sha256_file(metadata_path),
            probability_file_sha256=sha256_file(probability_path),
            prediction_file_sha256=sha256_file(prediction_path),
        )
        return record, tuple(output_rows), tuple(output_predictions)
    except (OSError, ValueError, KeyError, ProtocolError):
        return None


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read fresh Stage-70 JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Fresh Stage-70 JSON must be a mapping.")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


__all__ = (
    "array_sha256",
    "atomic_json",
    "read_json",
    "sha256_file",
    "try_load_task",
    "write_task_checkpoint",
)
