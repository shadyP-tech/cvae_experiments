"""Hash-valid resume, publication and reconstruction of prediction tasks."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    UtilityAlignedResidualFreshConfig,
)
from .contracts import (
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_LOGICAL_PREDICTION_COUNT,
    EvaluationPlan,
    PredictionCell,
    expected_action_ids,
)
from .policy_loading import FrozenUtilityAlignedPolicySurface
from .prediction_contracts import (
    EXPECTED_PREDICTION_TASK_COUNT,
    PREDICTION_CACHE_SCHEMA,
    PREDICTION_INDEX_COLUMNS,
    PredictionCache,
    PredictionTaskExecutor,
    PredictionTaskRecord,
    PredictionTaskSpec,
)
from .prediction_io import array_sha256, atomic_json, read_json, sha256_file
from .prediction_planning import build_prediction_tasks
from .prediction_worker import spawn_prediction_tasks
from .source_cache import FreshSourceCache
from .target_surface import FreshTargetSurface


def materialize_prediction_cache(
    config: UtilityAlignedResidualFreshConfig,
    *,
    plan: EvaluationPlan,
    policy: FrozenUtilityAlignedPolicySurface,
    source_cache: FreshSourceCache,
    target_surface: FreshTargetSurface,
    generation_lock_hash: str,
    root: str | Path,
    scratch_root: str | Path | None = None,
    executor: PredictionTaskExecutor | None = None,
) -> PredictionCache:
    """Resume 81 atomic tasks, then publish a hash-bound cache lock."""

    cache_root = Path(root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    scratch = None if scratch_root is None else Path(scratch_root).resolve()
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    tasks = build_prediction_tasks(
        config,
        plan=plan,
        policy=policy,
        source_cache=source_cache,
        target_surface=target_surface,
        generation_lock_hash=generation_lock_hash,
        root=cache_root,
        scratch_root=scratch,
    )
    pending = [task for task in tasks if try_load_task(task, plan) is None]
    if pending:
        (spawn_prediction_tasks if executor is None else executor)(tuple(pending))
    records: list[PredictionTaskRecord] = []
    predictions: list[PredictionCell] = []
    for task in tasks:
        loaded = try_load_task(task, plan)
        if loaded is None:
            raise ProtocolError("Utility-aligned prediction executor left an incomplete task.")
        record, cells = loaded
        records.append(record)
        predictions.extend(cells)
    if tuple(cell.key for cell in predictions) != tuple(cell.key for cell in plan.logical_cells):
        raise ProtocolError("Utility-aligned prediction-cache canonical order drifted.")
    total_unique = sum(record.unique_composition_fit_count for record in records)
    unhashed = {
        "schema_version": PREDICTION_CACHE_SCHEMA,
        "status": "COMPLETE",
        "plan_hash": plan.plan_hash,
        "policy_lock_hash": policy.policy_lock_hash,
        "action_library_hash": policy.action_library_hash,
        "source_cache_hash": source_cache.cache_hash,
        "generation_lock_hash": generation_lock_hash,
        "target_cache_content_hash": target_surface.cache_content_hash,
        "reservation_hash": target_surface.reservation.reservation_hash,
        "prediction_task_count": len(records),
        "logical_prediction_count": len(predictions),
        "unique_composition_fit_count": total_unique,
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "probability_dtype": "float32",
        "hash_validated_resume": True,
        "labels_available_to_fit_or_predict": False,
        "records": [_record_payload(record) for record in records],
    }
    lock = {**unhashed, "prediction_cache_hash": stable_hash(unhashed)}
    atomic_json(cache_root / "prediction_cache.json", lock)
    return PredictionCache(
        root=cache_root,
        plan_hash=plan.plan_hash,
        source_cache_hash=source_cache.cache_hash,
        generation_lock_hash=generation_lock_hash,
        records=tuple(records),
        predictions=tuple(predictions),
        cache_hash=str(lock["prediction_cache_hash"]),
        unique_composition_fit_count=total_unique,
    )


def load_prediction_cache(
    config: UtilityAlignedResidualFreshConfig,
    *,
    plan: EvaluationPlan,
    policy: FrozenUtilityAlignedPolicySurface,
    source_cache: FreshSourceCache,
    target_surface: FreshTargetSurface,
    generation_lock_hash: str,
    root: str | Path,
) -> PredictionCache:
    """Reconstruct every logical prediction from hash-bound checkpoint bytes."""

    cache_root = Path(root).resolve()
    lock = read_json(cache_root / "prediction_cache.json")
    observed_hash = lock.get("prediction_cache_hash")
    unhashed = {key: value for key, value in lock.items() if key != "prediction_cache_hash"}
    if (
        observed_hash != stable_hash(unhashed)
        or lock.get("schema_version") != PREDICTION_CACHE_SCHEMA
        or lock.get("status") != "COMPLETE"
        or lock.get("plan_hash") != plan.plan_hash
        or lock.get("policy_lock_hash") != policy.policy_lock_hash
        or lock.get("action_library_hash") != policy.action_library_hash
        or lock.get("source_cache_hash") != source_cache.cache_hash
        or lock.get("generation_lock_hash") != generation_lock_hash
        or lock.get("target_cache_content_hash") != target_surface.cache_content_hash
        or lock.get("reservation_hash") != target_surface.reservation.reservation_hash
        or lock.get("prediction_task_count") != EXPECTED_PREDICTION_TASK_COUNT
        or lock.get("logical_prediction_count") != EXPECTED_LOGICAL_PREDICTION_COUNT
        or lock.get("labels_available_to_fit_or_predict") is not False
    ):
        raise ProtocolError("Utility-aligned prediction-cache lock drifted.")
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
        raise ProtocolError("Utility-aligned prediction-cache record coverage drifted.")
    records: list[PredictionTaskRecord] = []
    predictions: list[PredictionCell] = []
    for task, raw in zip(tasks, raw_records, strict=True):
        loaded = try_load_task(task, plan)
        if loaded is None or not isinstance(raw, Mapping):
            raise ProtocolError("Utility-aligned published prediction task is incomplete.")
        record, cells = loaded
        if dict(raw) != _record_payload(record):
            raise ProtocolError("Utility-aligned prediction-cache record drifted.")
        records.append(record)
        predictions.extend(cells)
    if tuple(cell.key for cell in predictions) != tuple(cell.key for cell in plan.logical_cells):
        raise ProtocolError("Utility-aligned published prediction order drifted.")
    unique_count = sum(record.unique_composition_fit_count for record in records)
    if unique_count != lock.get("unique_composition_fit_count"):
        raise ProtocolError("Utility-aligned composition-fit count drifted.")
    return PredictionCache(
        root=cache_root,
        plan_hash=plan.plan_hash,
        source_cache_hash=source_cache.cache_hash,
        generation_lock_hash=generation_lock_hash,
        records=tuple(records),
        predictions=tuple(predictions),
        cache_hash=str(observed_hash),
        unique_composition_fit_count=unique_count,
    )


def write_prediction_index(path: str | Path, cache: PredictionCache) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record_by_task = {record.task_id: record for record in cache.records}
    rows: list[dict[str, object]] = []
    action_row_by_task: dict[str, int] = {}
    for cell in cache.predictions:
        task_id = (
            f"target_{cell.target_center}__train_{cell.training_seed}"
            f"__gen_{cell.generation_seed}"
        )
        probability_row = action_row_by_task.get(task_id, 0)
        action_row_by_task[task_id] = probability_row + 1
        record = record_by_task[task_id]
        rows.append(
            {
                "schema_version": "midogpp_utility_aligned_prediction_index_row_v1",
                "target_center": cell.target_center,
                "training_seed": cell.training_seed,
                "generation_seed": cell.generation_seed,
                "action_id": cell.action_id,
                "action_hash": cell.action_hash,
                "composition_hash": cell.composition_hash,
                "evaluation_row_ids_hash": stable_hash(list(cell.evaluation_row_ids)),
                "probability_member": f"checkpoints/predictions/{record.probability_member}",
                "probability_row": probability_row,
                "probability_sha256": array_sha256(cell.probabilities),
            }
        )
    if len(rows) != EXPECTED_LOGICAL_PREDICTION_COUNT:
        raise ProtocolError("Utility-aligned prediction index coverage drifted.")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def try_load_task(
    task: PredictionTaskSpec,
    plan: EvaluationPlan,
) -> tuple[PredictionTaskRecord, tuple[PredictionCell, ...]] | None:
    payload = task.payload
    metadata_path = Path(str(payload["metadata_path"]))
    probability_path = Path(str(payload["probability_path"]))
    if not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError, ProtocolError) as exc:
        raise ProtocolError("Utility-aligned resume metadata exists but is unreadable.") from exc
    if metadata.get("status") != "COMPLETE":
        return None
    if not probability_path.is_file():
        raise ProtocolError("Utility-aligned COMPLETE checkpoint lacks its probability array.")
    try:
        unhashed = {key: value for key, value in metadata.items() if key != "checkpoint_hash"}
        if (
            metadata.get("checkpoint_hash") != stable_hash(unhashed)
            or metadata.get("task_hash") != payload["task_hash"]
            or metadata.get("plan_hash") != plan.plan_hash
            or metadata.get("probability_file_sha256") != sha256_file(probability_path)
            or metadata.get("logical_prediction_count") != EXPECTED_ACTION_COUNT_PER_TARGET
            or metadata.get("labels_available_to_fit_or_predict") is not False
        ):
            raise ProtocolError("Utility-aligned COMPLETE checkpoint binding/hash drifted.")
        probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
        rows = tuple(str(row) for row in metadata.get("row_ids", ()))
        logical = metadata.get("logical_rows")
        if (
            probabilities.dtype != np.float32
            or probabilities.shape != (EXPECTED_ACTION_COUNT_PER_TARGET, len(rows))
            or not np.isfinite(probabilities).all()
            or not isinstance(logical, list)
        ):
            raise ProtocolError("Utility-aligned COMPLETE probability array drifted.")
        target = str(payload["target_center"])
        cells: list[PredictionCell] = []
        for index, (action_id, raw) in enumerate(
            zip(expected_action_ids(target), logical, strict=True)
        ):
            if not isinstance(raw, Mapping):
                raise ProtocolError("Utility-aligned COMPLETE logical row is malformed.")
            planned = plan.action_for(target, action_id)
            probability = np.ascontiguousarray(probabilities[index], dtype=np.float32)
            if (
                raw.get("action_id") != action_id
                or raw.get("action_hash") != planned.action_hash
                or raw.get("composition_hash") != planned.composition_hash
                or raw.get("probability_sha256") != array_sha256(probability)
                or raw.get("labels_available_to_fit_or_predict") is not False
            ):
                raise ProtocolError("Utility-aligned COMPLETE logical prediction drifted.")
            cells.append(
                PredictionCell(
                    target_center=target,
                    training_seed=int(payload["training_seed"]),
                    generation_seed=int(payload["generation_seed"]),
                    action_id=action_id,
                    action_hash=planned.action_hash,
                    composition_hash=planned.composition_hash,
                    evaluation_row_ids=rows,
                    probabilities=probability,
                )
            )
        root = Path(str(payload["canonical_root"]))
        return (
            PredictionTaskRecord(
                task_id=str(payload["task_id"]),
                task_hash=str(payload["task_hash"]),
                metadata_member=metadata_path.relative_to(root).as_posix(),
                probability_member=probability_path.relative_to(root).as_posix(),
                metadata_sha256=sha256_file(metadata_path),
                probability_sha256=sha256_file(probability_path),
                unique_composition_fit_count=int(metadata["unique_composition_fit_count"]),
            ),
            tuple(cells),
        )
    except ProtocolError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError("Utility-aligned COMPLETE checkpoint validation failed.") from exc


def _record_payload(record: PredictionTaskRecord) -> dict[str, object]:
    return {
        "task_id": record.task_id,
        "task_hash": record.task_hash,
        "metadata_member": record.metadata_member,
        "probability_member": record.probability_member,
        "metadata_sha256": record.metadata_sha256,
        "probability_sha256": record.probability_sha256,
        "unique_composition_fit_count": record.unique_composition_fit_count,
    }


__all__ = (
    "load_prediction_cache",
    "materialize_prediction_cache",
    "try_load_task",
    "write_prediction_index",
)
