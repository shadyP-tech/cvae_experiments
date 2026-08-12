"""Hash-valid atomic task checkpoints for spawned classifier workers."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from .artifact_io import atomic_bytes, atomic_json, read_json, sha256_file
from .prediction_contracts import PredictionTask


@dataclass(frozen=True)
class PredictionCheckpoint:
    task_hash: str
    task_key: tuple[str, str, int, int]
    probabilities: np.ndarray
    action_records: tuple[Mapping[str, object], ...]
    checkpoint_hash: str
    npz_path: Path
    json_path: Path

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.probabilities, dtype=np.float32)
        records = tuple(MappingProxyType(dict(row)) for row in self.action_records)
        if (
            values.ndim != 2
            or values.shape[0] != len(records)
            or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))
        ):
            raise ProtocolError("Endpoint-router checkpoint arrays drifted.")
        values.setflags(write=False)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "action_records", records)


def write_task_checkpoint(
    task: PredictionTask,
    *,
    probabilities: np.ndarray,
    action_records: Sequence[Mapping[str, object]],
) -> PredictionCheckpoint:
    values = np.ascontiguousarray(probabilities, dtype=np.float32)
    records = tuple(dict(record) for record in action_records)
    expected_rows = len(task.support_row_ids) + len(task.evaluation_row_ids)
    if (
        values.shape != (len(task.actions), expected_rows)
        or len(records) != len(task.actions)
        or tuple(record.get("action_id") for record in records)
        != tuple(action.action_id for action in task.actions)
        or any(
            record.get("action_hash") != action.action_hash
            for record, action in zip(records, task.actions, strict=True)
        )
    ):
        raise ProtocolError("Endpoint-router checkpoint result geometry drifted.")
    stream = io.BytesIO()
    np.savez(stream, probabilities=values)
    npz_path = Path(task.checkpoint_npz_path)
    json_path = Path(task.checkpoint_json_path)
    atomic_bytes(npz_path, stream.getvalue())
    unhashed = {
        "schema_version": "midogpp_endpoint_router_prediction_checkpoint_v1",
        "status": "COMPLETE",
        "phase": task.phase,
        "task_hash": task.task_hash,
        "task_key": list(task.key),
        "action_records": records,
        "probability_matrix_sha256": array_sha256(values),
        "npz_sha256": sha256_file(npz_path),
        "npz_path": str(npz_path),
        "array_shape": list(values.shape),
        "array_dtype": "float32",
        "support_row_count": len(task.support_row_ids),
        "evaluation_row_count": len(task.evaluation_row_ids),
        "source_stream_lock_hash": task.source_stream_lock_hash,
        "partition_lock_hash": task.partition_lock_hash,
        "cache_binding_hash": task.cache_binding_hash,
        "target_array_sha256": task.target_array_sha256,
        "labels_available": False,
        "classifier_fit_count": len(task.actions),
    }
    payload = {**unhashed, "checkpoint_hash": canonical_sha256(unhashed)}
    atomic_json(json_path, payload)
    return _checkpoint_from(task, payload, values)


def load_task_checkpoint(task: PredictionTask) -> PredictionCheckpoint | None:
    json_path = Path(task.checkpoint_json_path)
    npz_path = Path(task.checkpoint_npz_path)
    _validate_checkpoint_paths(task, npz_path=npz_path, json_path=json_path)
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        # Each member is wholly derived from the immutable task and written
        # atomically.  A process death between the NPZ and JSON replacements
        # can therefore leave exactly one safe-to-discard orphan.  Delete only
        # after validating the complete task-derived sibling path pair; any
        # other incomplete or non-regular shape remains fail closed.
        existing = tuple(path for path in (npz_path, json_path) if path.exists())
        if len(existing) != 1 or existing[0].is_symlink() or not existing[0].is_file():
            raise ProtocolError("Endpoint-router prediction checkpoint is incomplete.")
        existing[0].unlink()
        return None
    payload = read_json(json_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    if (
        payload.get("checkpoint_hash") != canonical_sha256(unhashed)
        or payload.get("schema_version")
        != "midogpp_endpoint_router_prediction_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("phase") != task.phase
        or payload.get("task_hash") != task.task_hash
        or tuple(payload.get("task_key", ())) != task.key
        or payload.get("npz_path") != str(npz_path)
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or payload.get("source_stream_lock_hash") != task.source_stream_lock_hash
        or payload.get("partition_lock_hash") != task.partition_lock_hash
        or payload.get("cache_binding_hash") != task.cache_binding_hash
        or payload.get("target_array_sha256") != task.target_array_sha256
        or payload.get("labels_available") is not False
        or int(payload.get("classifier_fit_count", -1)) != len(task.actions)
    ):
        raise ProtocolError("Endpoint-router prediction checkpoint binding drifted.")
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"probabilities"}:
                raise ProtocolError("Endpoint-router checkpoint NPZ schema drifted.")
            values = np.ascontiguousarray(arrays["probabilities"], dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Endpoint-router checkpoint NPZ is unreadable.") from exc
    records = payload.get("action_records")
    expected_shape = (
        len(task.actions), len(task.support_row_ids) + len(task.evaluation_row_ids)
    )
    if (
        values.shape != expected_shape
        or payload.get("array_shape") != list(expected_shape)
        or payload.get("array_dtype") != "float32"
        or payload.get("probability_matrix_sha256") != array_sha256(values)
        or not isinstance(records, list)
        or len(records) != len(task.actions)
        or any(not isinstance(row, Mapping) for row in records)
        or any(
            row.get("action_id") != action.action_id
            or row.get("action_hash") != action.action_hash
            for row, action in zip(records, task.actions, strict=True)
        )
    ):
        raise ProtocolError("Endpoint-router checkpoint contents drifted.")
    return _checkpoint_from(task, payload, values)


def _validate_checkpoint_paths(
    task: PredictionTask, *, npz_path: Path, json_path: Path
) -> None:
    expected_stem = (
        f"{task.phase}_H{task.outer_target}_q{task.query_center}_"
        f"train{task.training_seed}_gen{task.generation_seed}"
    )
    if (
        not npz_path.is_absolute()
        or not json_path.is_absolute()
        or npz_path.parent != json_path.parent
        or npz_path.parent.name
        not in {"development_predictions", "target_predictions"}
        or npz_path.name != f"{expected_stem}.npz"
        or json_path.name != f"{expected_stem}.json"
        or ".." in npz_path.parts
        or ".." in json_path.parts
    ):
        raise ProtocolError("Endpoint-router checkpoint task paths drifted.")


def _checkpoint_from(
    task: PredictionTask,
    payload: Mapping[str, object],
    values: np.ndarray,
) -> PredictionCheckpoint:
    records = payload.get("action_records")
    if not isinstance(records, list):
        raise ProtocolError("Endpoint-router checkpoint action records are absent.")
    return PredictionCheckpoint(
        task_hash=task.task_hash,
        task_key=task.key,
        probabilities=values,
        action_records=tuple(dict(row) for row in records if isinstance(row, Mapping)),
        checkpoint_hash=str(payload["checkpoint_hash"]),
        npz_path=Path(task.checkpoint_npz_path),
        json_path=Path(task.checkpoint_json_path),
    )


__all__ = (
    "PredictionCheckpoint",
    "load_task_checkpoint",
    "write_task_checkpoint",
)
