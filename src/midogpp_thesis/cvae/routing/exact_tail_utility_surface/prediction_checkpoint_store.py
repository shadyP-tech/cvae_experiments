"""Atomic, hash-valid exact-tail prediction checkpoint storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .prediction_contracts import (
    CHECKPOINT_SCHEMA,
    CoarsePredictionRecord,
    PredictionWorkerInput,
)
from .runtime import CoarsePredictionTask, build_coarse_task_checkpoint
from .scoring import array_sha256


def write_checkpoint(
    item: PredictionWorkerInput,
    *,
    classifier_config_hash: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    support_probabilities: np.ndarray | None = None,
    action_prediction_sha256: Mapping[str, str],
    action_probability_sha256: Mapping[str, str],
    action_support_probability_sha256: Mapping[str, str] | None = None,
    action_composition_sha256: Mapping[str, str],
    action_scaler_state_hash: Mapping[str, str],
    evaluation_row_count: int,
    support_row_count: int = 0,
) -> CoarsePredictionRecord:
    if support_probabilities is None:
        support_probabilities = np.empty(
            (len(item.task.action_ids), 0), dtype=np.float32
        )
    support_probabilities = np.ascontiguousarray(
        support_probabilities, dtype=np.float32
    )
    if action_support_probability_sha256 is None:
        action_support_probability_sha256 = {
            action_id: array_sha256(support_probabilities[index])
            for index, action_id in enumerate(item.task.action_ids)
        }
    checkpoint = build_coarse_task_checkpoint(
        task=item.task,
        action_prediction_sha256=action_prediction_sha256,
        action_probability_sha256=action_probability_sha256,
        action_support_probability_sha256=action_support_probability_sha256,
    )
    path = checkpoint_path(item)
    atomic_save_npz(
        path,
        predictions=predictions,
        probabilities=probabilities,
        support_probabilities=support_probabilities,
    )
    file_sha = sha256_file(path)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "task_key": list(item.task.key),
        "task_hash": item.task.task_hash,
        "partition_hash": item.partition_hash,
        "evaluation_row_identity_hash": item.evaluation_row_identity_hash,
        "evaluation_array_sha256": sha256_file(Path(item.evaluation_array_path)),
        "support_row_identity_hash": item.support_row_identity_hash or None,
        "support_array_present": bool(item.support_array_path),
        "support_array_sha256": (
            sha256_file(Path(item.support_array_path))
            if item.support_array_path
            else None
        ),
        "source_cache_hash": item.source_cache_hash,
        "classifier_config_hash": classifier_config_hash,
        "action_ids": list(item.task.action_ids),
        "action_prediction_sha256": dict(action_prediction_sha256),
        "action_probability_sha256": dict(action_probability_sha256),
        "action_support_probability_sha256": dict(
            action_support_probability_sha256
        ),
        "action_composition_sha256": dict(action_composition_sha256),
        "action_scaler_state_hash": dict(action_scaler_state_hash),
        "array_member": path.name,
        "array_file_sha256": file_sha,
        "evaluation_row_count": evaluation_row_count,
        "support_row_count": support_row_count,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "labels_available_to_fit_or_predict": False,
        "support_labels_available_to_fit_or_predict": False,
        "all_eight_actions_materialized": True,
    }
    atomic_json(checkpoint_metadata_path(item), metadata)
    return record_from_metadata(item, metadata, path)


def load_checkpoint(item: PredictionWorkerInput) -> CoarsePredictionRecord | None:
    metadata_path = checkpoint_metadata_path(item)
    array_path = checkpoint_path(item)
    if not metadata_path.exists() and not array_path.exists():
        return None
    if not metadata_path.is_file() or not array_path.is_file():
        # An interrupted pre-publication orphan is safe to replace only when no
        # COMPLETE metadata exists. A COMPLETE metadata/member mismatch is not.
        if metadata_path.is_file():
            raise ProtocolError("Exact-tail COMPLETE checkpoint member is absent.")
        return None
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Exact-tail checkpoint metadata is invalid.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("Exact-tail checkpoint metadata is not an object.")
    expected_keys = {
        "schema_version",
        "status",
        "task_key",
        "task_hash",
        "partition_hash",
        "evaluation_row_identity_hash",
        "evaluation_array_sha256",
        "support_row_identity_hash",
        "support_array_present",
        "support_array_sha256",
        "source_cache_hash",
        "classifier_config_hash",
        "action_ids",
        "action_prediction_sha256",
        "action_probability_sha256",
        "action_support_probability_sha256",
        "action_composition_sha256",
        "action_scaler_state_hash",
        "array_member",
        "array_file_sha256",
        "evaluation_row_count",
        "support_row_count",
        "checkpoint_hash",
        "labels_available_to_fit_or_predict",
        "support_labels_available_to_fit_or_predict",
        "all_eight_actions_materialized",
    }
    if set(raw) != expected_keys:
        raise ProtocolError("Exact-tail checkpoint schema drifted.")
    # Deferred import avoids a store/worker import cycle while retaining the
    # original strict classifier-payload reconstruction check.
    from .prediction_cpu_worker import classifier_from_payload

    if (
        raw.get("schema_version") != CHECKPOINT_SCHEMA
        or raw.get("status") != "COMPLETE"
        or tuple(raw.get("task_key", ())) != item.task.key
        or raw.get("task_hash") != item.task.task_hash
        or raw.get("partition_hash") != item.partition_hash
        or raw.get("evaluation_row_identity_hash")
        != item.evaluation_row_identity_hash
        or raw.get("evaluation_array_sha256")
        != sha256_file(Path(item.evaluation_array_path))
        or raw.get("support_row_identity_hash")
        != (item.support_row_identity_hash or None)
        or raw.get("support_array_present") is not bool(item.support_array_path)
        or raw.get("support_array_sha256")
        != (
            sha256_file(Path(item.support_array_path))
            if item.support_array_path
            else None
        )
        or raw.get("source_cache_hash") != item.source_cache_hash
        or raw.get("classifier_config_hash")
        != classifier_from_payload(item.classifier_payload).config_hash
        or raw.get("action_ids") != list(item.task.action_ids)
        or raw.get("array_member") != array_path.name
        or raw.get("array_file_sha256") != sha256_file(array_path)
        or raw.get("labels_available_to_fit_or_predict") is not False
        or raw.get("support_labels_available_to_fit_or_predict") is not False
        or raw.get("all_eight_actions_materialized") is not True
    ):
        raise ProtocolError("Exact-tail COMPLETE checkpoint binding drifted.")
    return record_from_metadata(item, raw, array_path)


def record_from_metadata(
    item: PredictionWorkerInput, raw: Mapping[str, object], array_path: Path
) -> CoarsePredictionRecord:
    with np.load(array_path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "predictions",
            "probabilities",
            "support_probabilities",
        }:
            raise ProtocolError("Exact-tail checkpoint array schema drifted.")
        predictions = np.asarray(payload["predictions"])
        probabilities = np.asarray(payload["probabilities"])
        support_probabilities = np.asarray(payload["support_probabilities"])
    row_count = int(raw["evaluation_row_count"])
    support_row_count = int(raw["support_row_count"])
    if (
        predictions.shape != (8, row_count)
        or predictions.dtype != np.uint8
        or probabilities.shape != (8, row_count)
        or probabilities.dtype != np.float32
        or support_probabilities.shape != (8, support_row_count)
        or support_probabilities.dtype != np.float32
        or not np.isin(predictions, (0, 1)).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or not np.isfinite(support_probabilities).all()
        or np.any(
            (support_probabilities < 0.0) | (support_probabilities > 1.0)
        )
        or (bool(item.support_array_path) != (support_row_count > 0))
    ):
        raise ProtocolError("Exact-tail checkpoint arrays drifted.")
    pred_hashes = action_hash_map(raw["action_prediction_sha256"], item.task)
    prob_hashes = action_hash_map(raw["action_probability_sha256"], item.task)
    support_prob_hashes = action_hash_map(
        raw["action_support_probability_sha256"], item.task
    )
    for index, action_id in enumerate(item.task.action_ids):
        if (
            array_sha256(predictions[index]) != pred_hashes[action_id]
            or array_sha256(probabilities[index]) != prob_hashes[action_id]
            or array_sha256(support_probabilities[index])
            != support_prob_hashes[action_id]
        ):
            raise ProtocolError("Exact-tail checkpoint action bytes drifted.")
    checkpoint = build_coarse_task_checkpoint(
        task=item.task,
        action_prediction_sha256=pred_hashes,
        action_probability_sha256=prob_hashes,
        action_support_probability_sha256=support_prob_hashes,
    )
    if raw.get("checkpoint_hash") != checkpoint.checkpoint_hash:
        raise ProtocolError("Exact-tail checkpoint semantic hash drifted.")
    compositions = action_hash_map(raw["action_composition_sha256"], item.task)
    scalers = action_hash_map(
        raw["action_scaler_state_hash"], item.task, allowed_lengths={16, 64}
    )
    return CoarsePredictionRecord(
        task=item.task,
        checkpoint_relative_path=str(array_path),
        checkpoint_file_sha256=str(raw["array_file_sha256"]),
        evaluation_row_count=row_count,
        action_composition_sha256=compositions,
        action_scaler_state_hash=scalers,
        checkpoint_hash=checkpoint.checkpoint_hash,
    )


def action_hash_map(
    raw: object,
    task: CoarsePredictionTask,
    *,
    allowed_lengths: set[int] = {64},
) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Exact-tail checkpoint action hash map is malformed.")
    values = {str(key): str(value) for key, value in raw.items()}
    if tuple(values) != task.action_ids:
        raise ProtocolError("Exact-tail checkpoint action hash order drifted.")
    if any(
        len(value) not in allowed_lengths
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise ProtocolError("Exact-tail checkpoint action hash is malformed.")
    return values


def checkpoint_stem(task: CoarsePredictionTask) -> str:
    return (
        f"H{task.outer_target}_q{task.pseudo_query}_"
        f"train{task.training_seed}_gen{task.generation_seed}"
    )


def checkpoint_path(item: PredictionWorkerInput) -> Path:
    return Path(item.checkpoint_root) / f"{checkpoint_stem(item.task)}.npz"


def checkpoint_metadata_path(item: PredictionWorkerInput) -> Path:
    return Path(item.checkpoint_root) / f"{checkpoint_stem(item.task)}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


__all__ = (
    "action_hash_map",
    "atomic_json",
    "atomic_save_npz",
    "checkpoint_metadata_path",
    "checkpoint_path",
    "checkpoint_stem",
    "load_checkpoint",
    "record_from_metadata",
    "sha256_file",
    "write_checkpoint",
)
