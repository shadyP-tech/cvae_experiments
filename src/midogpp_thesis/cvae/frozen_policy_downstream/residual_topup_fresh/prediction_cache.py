"""Canonical prediction-cache orchestration and index publication."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    ResidualTopupFreshConfig,
)
from .contracts import CENTERS, EXPECTED_PLAN_CELL_COUNT, EvaluationPlan, PredictionCell
from .policy_loading import FrozenPolicySurface
from .prediction_contracts import (
    EXPECTED_PREDICTION_TASK_COUNT,
    PREDICTION_CACHE_SCHEMA,
    PREDICTION_INDEX_COLUMNS,
    PredictionCache,
    PredictionTaskExecutor,
    PredictionTaskRecord,
    PredictionTaskSpec,
)
from .prediction_io import (
    atomic_json,
    read_json,
    sha256_file,
    try_load_task,
)
from .prediction_tasks import build_prediction_tasks, spawn_prediction_tasks
from .source_cache import FreshSourceCache
from .target_cache import FreshTargetSurface


def materialize_prediction_cache(
    config: ResidualTopupFreshConfig,
    *,
    plan: EvaluationPlan,
    policy: FrozenPolicySurface,
    source_cache: FreshSourceCache,
    target_surface: FreshTargetSurface,
    generation_lock_hash: str,
    root: str | Path,
    executor: PredictionTaskExecutor | None = None,
) -> PredictionCache:
    """Checkpoint 81 target/seed tasks, each fitting all 13 actions."""

    cache_root = Path(root)
    cache_root.mkdir(parents=True, exist_ok=True)
    if (
        plan.actions_by_target != policy.actions_by_target
        or source_cache.generation_lock_hash != generation_lock_hash
    ):
        raise ProtocolError("Fresh prediction execution input identities drifted.")
    tasks = build_prediction_tasks(
        config,
        plan=plan,
        policy=policy,
        source_cache=source_cache,
        target_surface=target_surface,
        generation_lock_hash=generation_lock_hash,
        root=cache_root,
    )
    pending = [task for task in tasks if try_load_task(task, plan=plan) is None]
    if pending:
        if executor is None:
            spawn_prediction_tasks(pending)
        else:
            executor(tuple(pending))

    records, rows, predictions = _load_complete_tasks(
        tasks,
        plan=plan,
        incomplete_message="Fresh prediction executor left an incomplete task.",
    )
    unhashed = {
        "schema_version": PREDICTION_CACHE_SCHEMA,
        "status": "COMPLETE",
        "plan_hash": plan.plan_hash,
        "policy_lock_hash": policy.policy_lock_hash,
        "action_library_hash": policy.action_library_hash,
        "source_cache_hash": source_cache.cache_hash,
        "bank_lock_hash": source_cache.bank_lock_hash,
        "generation_lock_hash": generation_lock_hash,
        "target_cache_content_hash": target_surface.cache_content_hash,
        "target_cache_protocol_hash": target_surface.cache_protocol_hash,
        "reservation_hash": target_surface.reservation.reservation_hash,
        "target_frame_sha256_by_center": {
            target: target_surface.frames_by_center[target].file_sha256
            for target in CENTERS
        },
        "prediction_task_count": len(records),
        "prediction_cell_count": len(rows),
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "all_actions_fitted": True,
        "labels_available_to_fit_or_predict": False,
        "records": [record.to_payload() for record in records],
    }
    lock = {**unhashed, "prediction_cache_hash": stable_hash(unhashed)}
    atomic_json(cache_root / "prediction_cache.json", lock)
    return PredictionCache(
        root=cache_root,
        plan_hash=plan.plan_hash,
        source_cache_hash=source_cache.cache_hash,
        generation_lock_hash=generation_lock_hash,
        records=records,
        index_rows=rows,
        predictions=predictions,
        cache_hash=str(lock["prediction_cache_hash"]),
    )


def load_prediction_cache(
    root: str | Path,
    *,
    plan: EvaluationPlan,
    config: ResidualTopupFreshConfig,
    policy: FrozenPolicySurface,
    source_cache: FreshSourceCache,
    target_surface: FreshTargetSurface,
    generation_lock_hash: str,
) -> PredictionCache:
    """Reload and reconstructively validate every cache/task input binding."""

    cache_root = Path(root)
    lock = read_json(cache_root / "prediction_cache.json")
    observed = lock.get("prediction_cache_hash")
    unhashed = {
        key: value
        for key, value in lock.items()
        if key != "prediction_cache_hash"
    }
    if (
        observed != stable_hash(unhashed)
        or lock.get("schema_version") != PREDICTION_CACHE_SCHEMA
        or lock.get("status") != "COMPLETE"
        or lock.get("plan_hash") != plan.plan_hash
        or lock.get("prediction_task_count")
        != EXPECTED_PREDICTION_TASK_COUNT
        or lock.get("prediction_cell_count") != EXPECTED_PLAN_CELL_COUNT
        or lock.get("labels_available_to_fit_or_predict") is not False
    ):
        raise ProtocolError("Fresh prediction-cache lock drifted.")
    tasks = build_prediction_tasks(
        config,
        plan=plan,
        policy=policy,
        source_cache=source_cache,
        target_surface=target_surface,
        generation_lock_hash=generation_lock_hash,
        root=cache_root,
    )
    raw_records = lock.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(tasks):
        raise ProtocolError("Fresh prediction-cache record coverage drifted.")
    for task, raw in zip(tasks, raw_records, strict=True):
        if not isinstance(raw, Mapping):
            raise ProtocolError("Fresh prediction-cache record is malformed.")
        payload = task.payload
        expected_members = {
            "metadata_member": str(
                Path(str(payload["metadata_path"])).relative_to(
                    cache_root.resolve()
                )
            ),
            "probability_member": str(
                Path(str(payload["probability_path"])).relative_to(
                    cache_root.resolve()
                )
            ),
            "prediction_member": str(
                Path(str(payload["prediction_path"])).relative_to(
                    cache_root.resolve()
                )
            ),
        }
        if (
            raw.get("task_id") != payload["task_id"]
            or raw.get("task_hash") != payload["task_hash"]
            or any(
                raw.get(key) != value
                for key, value in expected_members.items()
            )
            or raw.get("metadata_sha256")
            != sha256_file(Path(str(payload["metadata_path"])))
            or raw.get("probability_file_sha256")
            != sha256_file(Path(str(payload["probability_path"])))
            or raw.get("prediction_file_sha256")
            != sha256_file(Path(str(payload["prediction_path"])))
        ):
            raise ProtocolError("Fresh prediction-cache task binding drifted.")

    records, rows, predictions = _load_complete_tasks(
        tasks,
        plan=plan,
        incomplete_message="Fresh prediction-cache member drifted.",
    )
    return PredictionCache(
        root=cache_root,
        plan_hash=plan.plan_hash,
        source_cache_hash=str(lock.get("source_cache_hash", "")),
        generation_lock_hash=str(lock.get("generation_lock_hash", "")),
        records=records,
        index_rows=rows,
        predictions=predictions,
        cache_hash=str(observed),
    )


def write_prediction_index(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row[column] for column in PREDICTION_INDEX_COLUMNS}
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _load_complete_tasks(
    tasks: Sequence[PredictionTaskSpec],
    *,
    plan: EvaluationPlan,
    incomplete_message: str,
) -> tuple[
    tuple[PredictionTaskRecord, ...],
    tuple[Mapping[str, object], ...],
    tuple[PredictionCell, ...],
]:
    records: list[PredictionTaskRecord] = []
    rows: list[Mapping[str, object]] = []
    predictions: list[PredictionCell] = []
    for task in tasks:
        loaded = try_load_task(task, plan=plan)
        if loaded is None:
            raise ProtocolError(incomplete_message)
        record, task_rows, task_predictions = loaded
        records.append(record)
        rows.extend(task_rows)
        predictions.extend(task_predictions)
    if tuple(cell.key for cell in predictions) != tuple(
        cell.key for cell in plan.cells
    ):
        raise ProtocolError("Fresh prediction-cache canonical order drifted.")
    return tuple(records), tuple(rows), tuple(predictions)


__all__ = (
    "load_prediction_cache",
    "materialize_prediction_cache",
    "write_prediction_index",
)
