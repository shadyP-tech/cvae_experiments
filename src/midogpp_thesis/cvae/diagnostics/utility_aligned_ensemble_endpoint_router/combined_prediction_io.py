"""Hash-valid checkpoints and canonical combined probability persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_json, atomic_npz, read_json, sha256_file
from .prediction_contracts import (
    CombinedPredictionCell,
    CombinedPredictionStore,
    array_sha256,
    build_store,
)


_CHECKPOINT_ARRAY_HASH_FIELDS = (
    ("support_predictions", "support_prediction_sha256"),
    ("support_probabilities", "support_probability_sha256"),
    ("evaluation_predictions", "evaluation_prediction_sha256"),
    ("evaluation_probabilities", "evaluation_probability_sha256"),
)


def write_task_checkpoint(
    task: Mapping[str, object],
    *,
    support_predictions: np.ndarray,
    support_probabilities: np.ndarray,
    evaluation_predictions: np.ndarray,
    evaluation_probabilities: np.ndarray,
    action_rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    action_count = len(action_rows)
    support_count = int(task["support_row_count"])
    evaluation_count = int(task["evaluation_row_count"])
    if (
        support_predictions.shape != (action_count, support_count)
        or support_probabilities.shape != support_predictions.shape
        or evaluation_predictions.shape != (action_count, evaluation_count)
        or evaluation_probabilities.shape != evaluation_predictions.shape
        or support_predictions.dtype != np.uint8
        or evaluation_predictions.dtype != np.uint8
        or support_probabilities.dtype != np.float32
        or evaluation_probabilities.dtype != np.float32
    ):
        raise ProtocolError("Combined checkpoint array geometry drifted.")
    for ordinal, row in enumerate(action_rows):
        if (
            row.get("support_prediction_sha256") != array_sha256(support_predictions[ordinal])
            or row.get("support_probability_sha256") != array_sha256(support_probabilities[ordinal])
            or row.get("evaluation_prediction_sha256") != array_sha256(evaluation_predictions[ordinal])
            or row.get("evaluation_probability_sha256") != array_sha256(evaluation_probabilities[ordinal])
        ):
            raise ProtocolError("Combined checkpoint action hash drifted.")
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    atomic_npz(
        npz_path,
        support_predictions=support_predictions,
        support_probabilities=support_probabilities,
        evaluation_predictions=evaluation_predictions,
        evaluation_probabilities=evaluation_probabilities,
    )
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_task_checkpoint_v1",
        "status": "COMPLETE",
        "task_id": str(task["task_id"]),
        "task_hash": str(task["task_hash"]),
        "task_role": str(task["task_role"]),
        "config_contract_hash": str(task["config_contract_hash"]),
        "source_cache_lock_hash": str(task["source_cache_lock_hash"]),
        "partition_lock_hash": str(task["partition_lock_hash"]),
        "support_row_identity_hash": str(task["support_row_identity_hash"]),
        "evaluation_row_identity_hash": str(task["evaluation_row_identity_hash"]),
        "support_row_count": support_count,
        "evaluation_row_count": evaluation_count,
        "actions": [dict(row) for row in action_rows],
        "checkpoint_npz_sha256": sha256_file(npz_path),
        "labels_available_during_fit_or_predict": False,
        "support_and_evaluation_predicted_by_same_fit": True,
    }
    payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    atomic_json(json_path, payload)
    return payload


def load_task_checkpoint(task: Mapping[str, object]) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("Combined COMPLETE checkpoint member is absent.")
    payload = read_json(json_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    expected_bindings = (
        ("task_id", task["task_id"]),
        ("task_hash", task["task_hash"]),
        ("task_role", task["task_role"]),
        ("config_contract_hash", task["config_contract_hash"]),
        ("source_cache_lock_hash", task["source_cache_lock_hash"]),
        ("partition_lock_hash", task["partition_lock_hash"]),
        ("support_row_identity_hash", task["support_row_identity_hash"]),
        ("evaluation_row_identity_hash", task["evaluation_row_identity_hash"]),
    )
    if (
        payload.get("schema_version") != "midogpp_stage90_ensemble_endpoint_task_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("checkpoint_npz_sha256") != sha256_file(npz_path)
        or payload.get("labels_available_during_fit_or_predict") is not False
        or payload.get("support_and_evaluation_predicted_by_same_fit") is not True
        or any(payload.get(key) != value for key, value in expected_bindings)
    ):
        raise ProtocolError("Combined checkpoint binding drifted.")
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            expected_members = {
                member_name for member_name, _ in _CHECKPOINT_ARRAY_HASH_FIELDS
            }
            if set(arrays.files) != expected_members:
                raise ProtocolError("Combined checkpoint NPZ member set drifted.")
            loaded = {
                member_name: np.asarray(arrays[member_name])
                for member_name, _ in _CHECKPOINT_ARRAY_HASH_FIELDS
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError("Combined checkpoint arrays are unreadable.") from exc
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ProtocolError("Combined checkpoint action rows are malformed.")
    action_count = len(actions)
    support_count = int(task["support_row_count"])
    evaluation_count = int(task["evaluation_row_count"])
    expected_layouts = {
        "support_predictions": ((action_count, support_count), np.dtype(np.uint8)),
        "support_probabilities": ((action_count, support_count), np.dtype(np.float32)),
        "evaluation_predictions": ((action_count, evaluation_count), np.dtype(np.uint8)),
        "evaluation_probabilities": ((action_count, evaluation_count), np.dtype(np.float32)),
    }
    if any(
        loaded[member_name].shape != expected_shape
        or loaded[member_name].dtype != expected_dtype
        for member_name, (expected_shape, expected_dtype) in expected_layouts.items()
    ):
        raise ProtocolError("Combined checkpoint array geometry drifted.")
    for ordinal, row in enumerate(actions):
        if not isinstance(row, Mapping):
            raise ProtocolError("Combined checkpoint action row is malformed.")
        for member_name, hash_field in _CHECKPOINT_ARRAY_HASH_FIELDS:
            if row.get(hash_field) != array_sha256(loaded[member_name][ordinal]):
                raise ProtocolError("Combined checkpoint vector bytes drifted.")
    return {**payload, **loaded}


def write_combined_store(
    array_path: Path, index_path: Path, store: CombinedPredictionStore
) -> None:
    offsets = [0]
    vectors: dict[str, list[np.ndarray]] = {
        "support_predictions": [],
        "support_probabilities": [],
        "evaluation_predictions": [],
        "evaluation_probabilities": [],
    }
    rows: list[dict[str, object]] = []
    support_offsets = [0]
    evaluation_offsets = [0]
    for ordinal, cell in enumerate(store.cells):
        for role in ("support", "evaluation"):
            for kind in ("predictions", "probabilities"):
                vectors[f"{role}_{kind}"].append(getattr(cell, f"{role}_{kind}"))
        support_offsets.append(support_offsets[-1] + len(cell.support_predictions))
        evaluation_offsets.append(evaluation_offsets[-1] + len(cell.evaluation_predictions))
        rows.append(
            {
                "cell_ordinal": ordinal,
                **cell.hash_payload(),
                "support_start": support_offsets[-2],
                "support_stop": support_offsets[-1],
                "evaluation_start": evaluation_offsets[-2],
                "evaluation_stop": evaluation_offsets[-1],
            }
        )
    del offsets
    atomic_npz(
        array_path,
        support_predictions=np.concatenate(vectors["support_predictions"]).astype(np.uint8, copy=False),
        support_probabilities=np.concatenate(vectors["support_probabilities"]).astype(np.float32, copy=False),
        evaluation_predictions=np.concatenate(vectors["evaluation_predictions"]).astype(np.uint8, copy=False),
        evaluation_probabilities=np.concatenate(vectors["evaluation_probabilities"]).astype(np.float32, copy=False),
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        evaluation_offsets=np.asarray(evaluation_offsets, dtype=np.int64),
    )
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_prediction_index_v1",
        "store": store.to_payload(),
        "cells": rows,
        "array_member": array_path.name,
        "array_sha256": sha256_file(array_path),
        "labels_stored": False,
    }
    atomic_json(index_path, {**unhashed, "prediction_index_hash": stable_hash(unhashed)})


def read_combined_store(array_path: Path, index_path: Path) -> CombinedPredictionStore:
    index = read_json(index_path)
    unhashed = {key: value for key, value in index.items() if key != "prediction_index_hash"}
    if (
        index.get("schema_version") != "midogpp_stage90_ensemble_endpoint_prediction_index_v1"
        or index.get("prediction_index_hash") != stable_hash(unhashed)
        or index.get("array_member") != array_path.name
        or index.get("array_sha256") != sha256_file(array_path)
        or index.get("labels_stored") is not False
    ):
        raise ProtocolError("Combined prediction index binding drifted.")
    rows = index.get("cells")
    meta = index.get("store")
    if not isinstance(rows, list) or not isinstance(meta, Mapping):
        raise ProtocolError("Combined prediction index payload is malformed.")
    try:
        with np.load(array_path, allow_pickle=False) as arrays:
            expected_members = {
                "support_predictions", "support_probabilities",
                "evaluation_predictions", "evaluation_probabilities",
                "support_offsets", "evaluation_offsets",
            }
            if set(arrays.files) != expected_members:
                raise ProtocolError("Combined prediction NPZ member set drifted.")
            loaded = {name: np.asarray(arrays[name]) for name in (
                "support_predictions", "support_probabilities",
                "evaluation_predictions", "evaluation_probabilities",
                "support_offsets", "evaluation_offsets",
            )}
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError("Combined prediction store arrays are unreadable.") from exc
    for role in ("support", "evaluation"):
        predictions = loaded[f"{role}_predictions"]
        probabilities = loaded[f"{role}_probabilities"]
        offsets = loaded[f"{role}_offsets"]
        if (
            predictions.ndim != 1
            or predictions.dtype != np.uint8
            or probabilities.shape != predictions.shape
            or probabilities.dtype != np.float32
            or offsets.shape != (len(rows) + 1,)
            or offsets.dtype != np.int64
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(predictions)
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise ProtocolError("Combined prediction flat-array layout drifted.")
    cells: list[CombinedPredictionCell] = []
    expected_row_fields = {
        "cell_ordinal", "key", "action_hash", "support_row_identity_hash",
        "evaluation_row_identity_hash", "support_prediction_sha256",
        "support_probability_sha256", "evaluation_prediction_sha256",
        "evaluation_probability_sha256", "composition_hash", "scaler_state_hash",
        "fit_provenance_hash", "aliased_from_action_id", "support_start",
        "support_stop", "evaluation_start", "evaluation_stop",
    }
    for ordinal, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
            or int(row.get("cell_ordinal", -1)) != ordinal
        ):
            raise ProtocolError("Combined prediction index row order drifted.")
        key = tuple(row.get("key", ()))
        if len(key) != 4:
            raise ProtocolError("Combined prediction key drifted.")
        s0, s1 = int(row["support_start"]), int(row["support_stop"])
        e0, e1 = int(row["evaluation_start"]), int(row["evaluation_stop"])
        if (
            (s0, s1) != tuple(int(value) for value in loaded["support_offsets"][ordinal:ordinal + 2])
            or (e0, e1) != tuple(int(value) for value in loaded["evaluation_offsets"][ordinal:ordinal + 2])
        ):
            raise ProtocolError("Combined prediction cell offsets drifted.")
        cell = CombinedPredictionCell(
            scope_id=str(key[0]), action_id=str(key[1]),
            training_seed=int(key[2]), generation_seed=int(key[3]),
            action_hash=str(row["action_hash"]),
            support_row_identity_hash=str(row["support_row_identity_hash"]),
            evaluation_row_identity_hash=str(row["evaluation_row_identity_hash"]),
            support_predictions=np.ascontiguousarray(loaded["support_predictions"][s0:s1], dtype=np.uint8),
            support_probabilities=np.ascontiguousarray(loaded["support_probabilities"][s0:s1], dtype=np.float32),
            evaluation_predictions=np.ascontiguousarray(loaded["evaluation_predictions"][e0:e1], dtype=np.uint8),
            evaluation_probabilities=np.ascontiguousarray(loaded["evaluation_probabilities"][e0:e1], dtype=np.float32),
            composition_hash=str(row["composition_hash"]),
            scaler_state_hash=str(row["scaler_state_hash"]),
            fit_provenance_hash=str(row["fit_provenance_hash"]),
            aliased_from_action_id=(None if row.get("aliased_from_action_id") is None else str(row["aliased_from_action_id"])),
        )
        if any(cell.hash_payload().get(key) != row.get(key) for key in cell.hash_payload()):
            raise ProtocolError("Combined prediction persisted cell drifted.")
        cells.append(cell)
    rebuilt = build_store(
        role=str(meta["role"]),
        cells=cells,
        source_cache_lock_hash=str(meta["source_cache_lock_hash"]),
        partition_lock_hash=str(meta["partition_lock_hash"]),
        action_library_hash=str(meta["action_library_hash"]),
        expected_cell_count=int(meta["expected_cell_count"]),
        unique_classifier_fit_count=int(meta["unique_classifier_fit_count"]),
    )
    if set(meta) != set(rebuilt.to_payload()) or dict(meta) != rebuilt.to_payload():
        raise ProtocolError("Combined prediction store metadata drifted.")
    return rebuilt


__all__ = (
    "load_task_checkpoint",
    "read_combined_store",
    "write_combined_store",
    "write_task_checkpoint",
)
