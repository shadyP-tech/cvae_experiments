"""Hash-valid checkpoints and a complete label-free prediction store."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import build_inner_exact_tail_action_library
from .development_prediction_contracts import (
    EXPECTED_PREDICTION_CELL_COUNT,
    CoarseDevelopmentTask,
    PredictionCheckpointRecord,
    PredictionWorkerInput,
    action_library_for,
    expected_prediction_keys,
)
from .source_cache_store import atomic_write_json, sha256_file


PREDICTION_CHECKPOINT_SCHEMA = (
    "midogpp_stage90_utility_aligned_prediction_checkpoint_v1"
)
DEVELOPMENT_PREDICTION_ARRAY_MEMBER = (
    "arrays/utility_aligned_development_predictions.npz"
)
DEVELOPMENT_PREDICTION_INDEX_MEMBER = (
    "manifests/utility_aligned_development_prediction_index.json"
)


PredictionKey = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class DevelopmentPredictionStore:
    predictions: np.ndarray
    probabilities: np.ndarray
    offsets: np.ndarray
    index_rows: tuple[Mapping[str, object], ...]
    prediction_index_hash: str

    def __post_init__(self) -> None:
        predictions = np.asarray(self.predictions)
        probabilities = np.asarray(self.probabilities)
        offsets = np.asarray(self.offsets)
        rows = tuple(MappingProxyType(dict(row)) for row in self.index_rows)
        if (
            predictions.ndim != 1
            or probabilities.shape != predictions.shape
            or predictions.dtype != np.uint8
            or probabilities.dtype != np.float32
            or offsets.dtype != np.int64
            or offsets.shape != (len(rows) + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(predictions)
            or not np.isin(predictions, (0, 1)).all()
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or len(rows) != EXPECTED_PREDICTION_CELL_COUNT
        ):
            raise ProtocolError("Stage-90 development prediction store drifted.")
        expected_keys = expected_prediction_keys()
        for ordinal, (row, key) in enumerate(zip(rows, expected_keys, strict=True)):
            start, stop = int(offsets[ordinal]), int(offsets[ordinal + 1])
            observed_key = (
                str(row.get("outer_target")),
                str(row.get("query_center")),
                str(row.get("action_id")),
                int(row.get("training_seed", -1)),
                int(row.get("generation_seed", -1)),
            )
            if (
                int(row.get("cell_ordinal", -1)) != ordinal
                or observed_key != key
                or int(row.get("array_start", -1)) != start
                or int(row.get("array_stop", -1)) != stop
                or stop <= start
                or array_sha256(predictions[start:stop])
                != row.get("prediction_sha256")
                or array_sha256(probabilities[start:stop])
                != row.get("probability_sha256")
                or row.get("labels_stored") is not False
            ):
                raise ProtocolError("Stage-90 prediction cell binding drifted.")
        unhashed = {
            "schema_version": "midogpp_stage90_utility_aligned_prediction_index_v1",
            "cell_count": len(rows),
            "cells": [dict(row) for row in rows],
            "labels_stored": False,
            "all_predictions_materialized_before_development_labels": True,
        }
        if self.prediction_index_hash != stable_hash(unhashed):
            raise ProtocolError("Stage-90 prediction index semantic hash drifted.")
        predictions.setflags(write=False)
        probabilities.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "index_rows", rows)

    @property
    def cell_by_key(self) -> Mapping[PredictionKey, Mapping[str, object]]:
        return MappingProxyType(
            {
                (
                    str(row["outer_target"]),
                    str(row["query_center"]),
                    str(row["action_id"]),
                    int(row["training_seed"]),
                    int(row["generation_seed"]),
                ): row
                for row in self.index_rows
            }
        )

    def prediction_for(self, key: PredictionKey) -> np.ndarray:
        try:
            row = self.cell_by_key[key]
        except KeyError as exc:
            raise ProtocolError("Stage-90 prediction key is absent.") from exc
        return self.predictions[int(row["array_start"]) : int(row["array_stop"])]


def write_prediction_checkpoint(
    item: PredictionWorkerInput,
    *,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classifier_config_hash: str,
    action_prediction_sha256: Mapping[str, str],
    action_probability_sha256: Mapping[str, str],
    action_composition_sha256: Mapping[str, str],
    action_scaler_state_hash: Mapping[str, str],
) -> PredictionCheckpointRecord:
    expected_shape = (len(item.task.action_ids), len(item.evaluation_row_ids))
    if (
        predictions.shape != expected_shape
        or predictions.dtype != np.uint8
        or probabilities.shape != expected_shape
        or probabilities.dtype != np.float32
    ):
        raise ProtocolError("Stage-90 prediction checkpoint array geometry drifted.")
    semantic = _checkpoint_semantic_hash(
        item.task,
        action_prediction_sha256,
        action_probability_sha256,
    )
    npz_path = Path(item.checkpoint_npz_path)
    json_path = Path(item.checkpoint_json_path)
    atomic_save_npz(npz_path, predictions=predictions, probabilities=probabilities)
    unhashed = {
        "schema_version": PREDICTION_CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "task_key": list(item.task.key),
        "task_hash": item.task.task_hash,
        "config_contract_hash": item.config_contract_hash,
        "generation_lock_hash": item.generation_lock_hash,
        "source_cache_lock_hash": item.source_cache_lock_hash,
        "partition_lock_hash": item.partition_lock_hash,
        "support_partition_hash": item.support_partition_hash,
        "evaluation_array_sha256": item.evaluation_array_sha256,
        "evaluation_row_ids": list(item.evaluation_row_ids),
        "evaluation_row_identity_hash": item.evaluation_row_identity_hash,
        "classifier_config_hash": classifier_config_hash,
        "action_ids": list(item.task.action_ids),
        "action_prediction_sha256": dict(action_prediction_sha256),
        "action_probability_sha256": dict(action_probability_sha256),
        "action_composition_sha256": dict(action_composition_sha256),
        "action_scaler_state_hash": dict(action_scaler_state_hash),
        "checkpoint_npz_path": str(npz_path),
        "checkpoint_npz_sha256": sha256_file(npz_path),
        "evaluation_row_count": len(item.evaluation_row_ids),
        "semantic_checkpoint_hash": semantic,
        "labels_available_to_fit_or_predict": False,
        "strict_H_q_e_exclusion": True,
        "all_eight_actions_materialized_atomically": True,
    }
    payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    atomic_write_json(json_path, payload)
    return _record_from_payload(item, payload)


def load_prediction_checkpoint(
    item: PredictionWorkerInput,
) -> PredictionCheckpointRecord | None:
    json_path = Path(item.checkpoint_json_path)
    npz_path = Path(item.checkpoint_npz_path)
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file():
        return None
    if not npz_path.is_file():
        raise ProtocolError("Stage-90 COMPLETE prediction checkpoint member is absent.")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Stage-90 prediction checkpoint metadata is invalid.") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Stage-90 prediction checkpoint must be an object.")
    expected_fields = {
        "schema_version", "status", "task_key", "task_hash", "config_contract_hash",
        "generation_lock_hash", "source_cache_lock_hash", "partition_lock_hash",
        "support_partition_hash", "evaluation_array_sha256", "evaluation_row_ids",
        "evaluation_row_identity_hash", "classifier_config_hash", "action_ids",
        "action_prediction_sha256", "action_probability_sha256",
        "action_composition_sha256", "action_scaler_state_hash",
        "checkpoint_npz_path", "checkpoint_npz_sha256", "evaluation_row_count",
        "semantic_checkpoint_hash", "labels_available_to_fit_or_predict",
        "strict_H_q_e_exclusion", "all_eight_actions_materialized_atomically",
        "checkpoint_hash",
    }
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    if (
        set(payload) != expected_fields
        or payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("schema_version") != PREDICTION_CHECKPOINT_SCHEMA
        or payload.get("status") != "COMPLETE"
        or tuple(payload.get("task_key", ())) != item.task.key
        or payload.get("task_hash") != item.task.task_hash
        or payload.get("config_contract_hash") != item.config_contract_hash
        or payload.get("generation_lock_hash") != item.generation_lock_hash
        or payload.get("source_cache_lock_hash") != item.source_cache_lock_hash
        or payload.get("partition_lock_hash") != item.partition_lock_hash
        or payload.get("support_partition_hash") != item.support_partition_hash
        or payload.get("evaluation_array_sha256") != item.evaluation_array_sha256
        or payload.get("evaluation_row_ids") != list(item.evaluation_row_ids)
        or payload.get("evaluation_row_identity_hash") != item.evaluation_row_identity_hash
        or payload.get("classifier_config_hash")
        != _classifier_config_hash(item.classifier_payload)
        or payload.get("action_ids") != list(item.task.action_ids)
        or payload.get("checkpoint_npz_path") != str(npz_path)
        or payload.get("checkpoint_npz_sha256") != sha256_file(npz_path)
        or int(payload.get("evaluation_row_count", -1)) != len(item.evaluation_row_ids)
        or payload.get("labels_available_to_fit_or_predict") is not False
        or payload.get("strict_H_q_e_exclusion") is not True
        or payload.get("all_eight_actions_materialized_atomically") is not True
    ):
        raise ProtocolError("Stage-90 prediction checkpoint binding drifted.")
    return _record_from_payload(item, payload)


def _record_from_payload(
    item: PredictionWorkerInput, payload: Mapping[str, object]
) -> PredictionCheckpointRecord:
    npz_path = Path(item.checkpoint_npz_path)
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"predictions", "probabilities"}:
                raise ProtocolError("Stage-90 checkpoint NPZ schema drifted.")
            predictions = np.asarray(arrays["predictions"])
            probabilities = np.asarray(arrays["probabilities"])
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError("Stage-90 checkpoint arrays are unreadable.") from exc
    expected_shape = (len(item.task.action_ids), len(item.evaluation_row_ids))
    if (
        predictions.shape != expected_shape
        or predictions.dtype != np.uint8
        or probabilities.shape != expected_shape
        or probabilities.dtype != np.float32
        or not np.isin(predictions, (0, 1)).all()
        or not np.isfinite(probabilities).all()
    ):
        raise ProtocolError("Stage-90 checkpoint arrays drifted.")
    pred_hashes = _action_hash_map(payload.get("action_prediction_sha256"), item.task)
    prob_hashes = _action_hash_map(payload.get("action_probability_sha256"), item.task)
    compositions = _action_hash_map(payload.get("action_composition_sha256"), item.task)
    scalers = _action_hash_map(
        payload.get("action_scaler_state_hash"), item.task, allowed_lengths={16, 64}
    )
    for ordinal, action_id in enumerate(item.task.action_ids):
        if (
            array_sha256(predictions[ordinal]) != pred_hashes[action_id]
            or array_sha256(probabilities[ordinal]) != prob_hashes[action_id]
        ):
            raise ProtocolError("Stage-90 checkpoint action bytes drifted.")
    semantic = _checkpoint_semantic_hash(item.task, pred_hashes, prob_hashes)
    if payload.get("semantic_checkpoint_hash") != semantic:
        raise ProtocolError("Stage-90 checkpoint semantic hash drifted.")
    return PredictionCheckpointRecord(
        task=item.task,
        checkpoint_json_path=item.checkpoint_json_path,
        checkpoint_npz_path=item.checkpoint_npz_path,
        checkpoint_file_sha256=str(payload["checkpoint_npz_sha256"]),
        checkpoint_hash=str(payload["checkpoint_hash"]),
        evaluation_row_count=len(item.evaluation_row_ids),
        action_prediction_sha256=pred_hashes,
        action_probability_sha256=prob_hashes,
        action_composition_sha256=compositions,
        action_scaler_state_hash=scalers,
    )


def consolidate_prediction_records(
    records: Sequence[PredictionCheckpointRecord], *, root: Path
) -> DevelopmentPredictionStore:
    by_key = {record.task.key: record for record in records}
    expected_tasks = tuple(
        record.task.key for record in sorted(records, key=lambda item: item.task.task_ordinal)
    )
    if len(by_key) != len(records) or len(records) * 8 != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("Stage-90 prediction checkpoint coverage drifted.")
    canonical_library_hash = (
        build_inner_exact_tail_action_library().action_library_hash
    )
    flat_predictions: list[np.ndarray] = []
    flat_probabilities: list[np.ndarray] = []
    offsets = [0]
    index_rows: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.task.task_ordinal):
        with np.load(record.checkpoint_npz_path, allow_pickle=False) as arrays:
            task_predictions = np.asarray(arrays["predictions"])
            task_probabilities = np.asarray(arrays["probabilities"])
        actions = action_library_for(
            outer_target=record.task.outer_target,
            query_center=record.task.query_center,
        )
        metadata = json.loads(Path(record.checkpoint_json_path).read_text(encoding="utf-8"))
        for action_ordinal, action in enumerate(actions):
            prediction = np.ascontiguousarray(task_predictions[action_ordinal], dtype=np.uint8)
            probability = np.ascontiguousarray(task_probabilities[action_ordinal], dtype=np.float32)
            start, stop = offsets[-1], offsets[-1] + len(prediction)
            offsets.append(stop)
            flat_predictions.append(prediction)
            flat_probabilities.append(probability)
            index_rows.append(
                {
                    "schema_version": "midogpp_stage90_utility_aligned_prediction_cell_v1",
                    "cell_ordinal": len(index_rows),
                    "outer_target": record.task.outer_target,
                    "query_center": record.task.query_center,
                    "action_id": action.action_id,
                    "selected_source": action.selected_source,
                    "training_seed": record.task.training_seed,
                    "generation_seed": record.task.generation_seed,
                    "task_hash": record.task.task_hash,
                    "action_hash": action.action_hash,
                    "canonical_inner_action_library_hash": canonical_library_hash,
                    "source_cache_lock_hash": metadata["source_cache_lock_hash"],
                    "partition_lock_hash": metadata["partition_lock_hash"],
                    "support_partition_hash": metadata["support_partition_hash"],
                    "evaluation_row_ids": metadata["evaluation_row_ids"],
                    "evaluation_row_identity_hash": metadata["evaluation_row_identity_hash"],
                    "composition_sha256": record.action_composition_sha256[action.action_id],
                    "classifier_config_hash": metadata["classifier_config_hash"],
                    "scaler_state_hash": record.action_scaler_state_hash[action.action_id],
                    "prediction_sha256": record.action_prediction_sha256[action.action_id],
                    "probability_sha256": record.action_probability_sha256[action.action_id],
                    "array_start": start,
                    "array_stop": stop,
                    "labels_stored": False,
                    "strict_H_q_e_exclusion": True,
                    "diagnostic_only": True,
                }
            )
    observed_keys = tuple(
        (
            str(row["outer_target"]), str(row["query_center"]), str(row["action_id"]),
            int(row["training_seed"]), int(row["generation_seed"]),
        )
        for row in index_rows
    )
    if observed_keys != expected_prediction_keys():
        raise ProtocolError("Stage-90 consolidated prediction order drifted.")
    predictions = np.concatenate(flat_predictions).astype(np.uint8, copy=False)
    probabilities = np.concatenate(flat_probabilities).astype(np.float32, copy=False)
    offset_array = np.asarray(offsets, dtype=np.int64)
    arrays_path = root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER
    atomic_save_npz(
        arrays_path,
        predictions=predictions,
        probabilities=probabilities,
        offsets=offset_array,
    )
    index_unhashed = {
        "schema_version": "midogpp_stage90_utility_aligned_prediction_index_v1",
        "cell_count": len(index_rows),
        "cells": index_rows,
        "labels_stored": False,
        "all_predictions_materialized_before_development_labels": True,
    }
    index = {**index_unhashed, "prediction_index_hash": stable_hash(index_unhashed)}
    atomic_write_json(root / DEVELOPMENT_PREDICTION_INDEX_MEMBER, index)
    return DevelopmentPredictionStore(
        predictions=predictions,
        probabilities=probabilities,
        offsets=offset_array,
        index_rows=tuple(index_rows),
        prediction_index_hash=str(index["prediction_index_hash"]),
    )


def load_development_prediction_store(root: Path) -> DevelopmentPredictionStore:
    try:
        index = json.loads(
            (root / DEVELOPMENT_PREDICTION_INDEX_MEMBER).read_text(encoding="utf-8")
        )
        with np.load(root / DEVELOPMENT_PREDICTION_ARRAY_MEMBER, allow_pickle=False) as arrays:
            if set(arrays.files) != {"predictions", "probabilities", "offsets"}:
                raise ProtocolError("Stage-90 prediction store NPZ schema drifted.")
            predictions = np.asarray(arrays["predictions"])
            probabilities = np.asarray(arrays["probabilities"])
            offsets = np.asarray(arrays["offsets"])
    except ProtocolError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("Stage-90 prediction store is unreadable.") from exc
    if not isinstance(index, Mapping) or set(index) != {
        "schema_version", "cell_count", "cells", "labels_stored",
        "all_predictions_materialized_before_development_labels", "prediction_index_hash",
    }:
        raise ProtocolError("Stage-90 prediction index schema drifted.")
    return DevelopmentPredictionStore(
        predictions=predictions,
        probabilities=probabilities,
        offsets=offsets,
        index_rows=tuple(index["cells"]),
        prediction_index_hash=str(index["prediction_index_hash"]),
    )


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_semantic_hash(
    task: CoarseDevelopmentTask,
    predictions: Mapping[str, str],
    probabilities: Mapping[str, str],
) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_stage90_utility_aligned_checkpoint_semantics_v1",
            "task_hash": task.task_hash,
            "action_prediction_sha256": dict(predictions),
            "action_probability_sha256": dict(probabilities),
            "atomic_base_plus_seven_tails": True,
        }
    )


def _action_hash_map(
    raw: object,
    task: CoarseDevelopmentTask,
    *,
    allowed_lengths: set[int] = {64},
) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Stage-90 checkpoint action hash map is malformed.")
    values = {str(key): str(value) for key, value in raw.items()}
    if tuple(values) != task.action_ids or any(
        len(value) not in allowed_lengths
        or any(character not in "0123456789abcdef" for character in value)
        for value in values.values()
    ):
        raise ProtocolError("Stage-90 checkpoint action hash map drifted.")
    return values


def _classifier_config_hash(raw: Mapping[str, object]) -> str:
    from .development_prediction_worker import classifier_from_payload

    return classifier_from_payload(raw).config_hash


__all__ = (
    "DEVELOPMENT_PREDICTION_ARRAY_MEMBER",
    "DEVELOPMENT_PREDICTION_INDEX_MEMBER",
    "PREDICTION_CHECKPOINT_SCHEMA",
    "DevelopmentPredictionStore",
    "PredictionKey",
    "array_sha256",
    "atomic_save_npz",
    "consolidate_prediction_records",
    "load_development_prediction_store",
    "load_prediction_checkpoint",
    "write_prediction_checkpoint",
)
