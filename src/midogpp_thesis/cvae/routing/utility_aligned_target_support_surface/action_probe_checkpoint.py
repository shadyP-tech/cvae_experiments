"""Atomic, hash-valid action-probe checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .action_probe_contracts import ActionProbeCheckpoint, ActionProbeTask


CHECKPOINT_SCHEMA = "midogpp_target_support_action_probe_checkpoint_v1"


def write_action_probe_checkpoint(
    task: ActionProbeTask,
    probabilities: np.ndarray,
) -> ActionProbeCheckpoint:
    values = np.ascontiguousarray(probabilities, dtype=np.float32)
    action_ids = ("B", *tuple(f"Hxe::{source}" for source in task.candidate_sources))
    if (
        values.shape != (len(action_ids), len(task.support_case_ids))
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ProtocolError("Target-support action-probe probability geometry drifted.")
    root = Path(task.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    probability_path = root / f"{task.checkpoint_stem}.npy"
    metadata_path = root / f"{task.checkpoint_stem}.json"
    _atomic_npy(probability_path, values)
    probability_sha = sha256_file(probability_path)
    unhashed = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "task_hash": task.task_hash,
        "task_identity": task.identity_payload(),
        "probability_member": probability_path.name,
        "probability_file_sha256": probability_sha,
        "probability_shape": list(values.shape),
        "probability_dtype": "float32",
        "action_ids": list(action_ids),
        "support_row_count": len(task.support_case_ids),
        "labels_used": False,
        "target_evaluation_rows_opened": False,
    }
    payload = {**unhashed, "checkpoint_hash": canonical_sha256(unhashed)}
    _atomic_json(metadata_path, payload)
    return ActionProbeCheckpoint(
        task_hash=task.task_hash,
        checkpoint_hash=str(payload["checkpoint_hash"]),
        probability_member=probability_path.name,
        probability_file_sha256=probability_sha,
        action_ids=action_ids,
        support_row_count=len(task.support_case_ids),
    )


def load_action_probe_checkpoint(
    task: ActionProbeTask,
) -> ActionProbeCheckpoint | None:
    root = Path(task.checkpoint_root)
    metadata_path = root / f"{task.checkpoint_stem}.json"
    probability_path = root / f"{task.checkpoint_stem}.npy"
    if not metadata_path.is_file() or not probability_path.is_file():
        return None
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Target-support action-probe checkpoint is unreadable.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("Target-support action-probe checkpoint must be an object.")
    unhashed = {key: value for key, value in raw.items() if key != "checkpoint_hash"}
    action_ids = ("B", *tuple(f"Hxe::{source}" for source in task.candidate_sources))
    required = {
        "schema_version", "status", "task_hash", "task_identity",
        "probability_member", "probability_file_sha256", "probability_shape",
        "probability_dtype", "action_ids", "support_row_count", "labels_used",
        "target_evaluation_rows_opened", "checkpoint_hash",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != CHECKPOINT_SCHEMA
        or raw.get("status") != "COMPLETE"
        or raw.get("task_hash") != task.task_hash
        or raw.get("task_identity") != task.identity_payload()
        or raw.get("probability_member") != probability_path.name
        or raw.get("probability_file_sha256") != sha256_file(probability_path)
        or raw.get("probability_shape") != [len(action_ids), len(task.support_case_ids)]
        or raw.get("probability_dtype") != "float32"
        or raw.get("action_ids") != list(action_ids)
        or raw.get("support_row_count") != len(task.support_case_ids)
        or raw.get("labels_used") is not False
        or raw.get("target_evaluation_rows_opened") is not False
        or raw.get("checkpoint_hash") != canonical_sha256(unhashed)
    ):
        raise ProtocolError("Target-support action-probe checkpoint failed validation.")
    try:
        probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Target-support action-probe probability array is unreadable.") from exc
    if (
        probabilities.shape != (len(action_ids), len(task.support_case_ids))
        or probabilities.dtype != np.float32
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ProtocolError("Target-support action-probe probability array drifted.")
    return ActionProbeCheckpoint(
        task_hash=task.task_hash,
        checkpoint_hash=str(raw["checkpoint_hash"]),
        probability_member=probability_path.name,
        probability_file_sha256=str(raw["probability_file_sha256"]),
        action_ids=action_ids,
        support_row_count=len(task.support_case_ids),
    )


def load_checkpoint_probabilities(
    task: ActionProbeTask,
    checkpoint: ActionProbeCheckpoint,
) -> np.ndarray:
    if checkpoint.task_hash != task.task_hash:
        raise ProtocolError("Target-support action-probe checkpoint/task drifted.")
    path = Path(task.checkpoint_root) / checkpoint.probability_member
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return np.asarray(values, dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            __import__("os").fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ProtocolError("Cannot publish target-support action-probe array.") from exc


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + f".{__import__('os').getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            __import__("os").fsync(handle.fileno())
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise ProtocolError("Cannot publish target-support action-probe metadata.") from exc


__all__ = (
    "CHECKPOINT_SCHEMA",
    "load_action_probe_checkpoint",
    "load_checkpoint_probabilities",
    "sha256_file",
    "write_action_probe_checkpoint",
)
