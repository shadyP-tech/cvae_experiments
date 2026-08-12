"""Atomic float32 prediction stores with complete reconstructive indexes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from .artifact_io import atomic_bytes, atomic_json, read_json, sha256_file
from .checkpoint_store import PredictionCheckpoint
from .prediction_contracts import (
    DEVELOPMENT_ARRAY_MEMBER,
    DEVELOPMENT_CELL_COUNT,
    DEVELOPMENT_INDEX_MEMBER,
    DEVELOPMENT_ROLE,
    TARGET_ARRAY_MEMBER,
    TARGET_CELL_COUNT,
    TARGET_INDEX_MEMBER,
    TARGET_ROLE,
    PredictionCell,
    PredictionStore,
    PredictionTask,
    canonical_cell_keys,
    canonical_scopes,
    cell_scope,
    prediction_store_hash,
)
from .prediction_planning import PredictionPlan


def materialize_prediction_store(
    plan: PredictionPlan,
    checkpoints: Sequence[PredictionCheckpoint],
    *,
    root: Path,
) -> PredictionStore:
    """Collapse task checkpoints into the canonical store, or validate resume."""

    array_member, index_member = _members(plan.phase)
    array_path = root / array_member
    index_path = root / index_member
    if array_path.is_file() and index_path.is_file():
        return load_prediction_store(
            root,
            phase=plan.phase,
            expected_plan=plan,
        )
    by_hash = {checkpoint.task_hash: checkpoint for checkpoint in checkpoints}
    if (
        len(by_hash) != len(plan.tasks)
        or set(by_hash) != {task.task_hash for task in plan.tasks}
    ):
        raise ProtocolError("Endpoint-router checkpoint coverage is incomplete.")
    cells: list[PredictionCell] = []
    for task in plan.tasks:
        checkpoint = by_hash[task.task_hash]
        split = len(task.support_row_ids)
        for ordinal, (action, record) in enumerate(
            zip(task.actions, checkpoint.action_records, strict=True)
        ):
            values = checkpoint.probabilities[ordinal]
            cells.append(
                PredictionCell(
                    phase=plan.phase,
                    outer_target=task.outer_target,
                    query_center=task.query_center,
                    action_id=action.action_id,
                    action_hash=action.action_hash,
                    training_seed=task.training_seed,
                    generation_seed=task.generation_seed,
                    support_row_identity_hash=task.support_row_identity_hash,
                    evaluation_row_identity_hash=task.evaluation_row_identity_hash,
                    support_probabilities=values[:split],
                    evaluation_probabilities=values[split:],
                    composition_hash=str(record["composition_hash"]),
                    scaler_state_hash=str(record["scaler_state_hash"]),
                    fit_provenance_hash=str(record["fit_provenance_hash"]),
                )
            )
    mappings = _scope_mappings(plan.tasks)
    store_hash = prediction_store_hash(
        plan.phase,
        cells,
        support_row_ids_by_scope=mappings[0],
        evaluation_row_ids_by_scope=mappings[1],
        support_case_ids_by_scope=mappings[2],
        evaluation_case_ids_by_scope=mappings[3],
        source_stream_lock_hash=plan.tasks[0].source_stream_lock_hash,
        partition_lock_hash=plan.tasks[0].partition_lock_hash,
        cache_binding_hash=plan.tasks[0].cache_binding_hash,
        action_library_hash=plan.action_library_hash,
    )
    store = PredictionStore(
        phase=plan.phase,
        cells=tuple(cells),
        support_row_ids_by_scope=mappings[0],
        evaluation_row_ids_by_scope=mappings[1],
        support_case_ids_by_scope=mappings[2],
        evaluation_case_ids_by_scope=mappings[3],
        source_stream_lock_hash=plan.tasks[0].source_stream_lock_hash,
        partition_lock_hash=plan.tasks[0].partition_lock_hash,
        cache_binding_hash=plan.tasks[0].cache_binding_hash,
        action_library_hash=plan.action_library_hash,
        store_hash=store_hash,
    )
    _persist_store(store, root=root, plan=plan)
    return load_prediction_store(root, phase=plan.phase, expected_plan=plan)


def load_prediction_store(
    root: Path,
    *,
    phase: str,
    expected_plan: PredictionPlan | None = None,
) -> PredictionStore:
    array_member, index_member = _members(phase)
    array_path = root / array_member
    index_path = root / index_member
    index = read_json(index_path)
    expected_index_fields = {
        "schema_version", "phase", "cell_count", "task_count",
        "unique_fit_count", "prediction_plan_hash", "prediction_store_hash",
        "source_stream_lock_hash", "partition_lock_hash", "cache_binding_hash",
        "action_library_hash", "array_member", "array_sha256",
        "support_row_ids_by_scope", "evaluation_row_ids_by_scope",
        "support_case_ids_by_scope", "evaluation_case_ids_by_scope", "cells",
        "labels_stored", "storage_dtype", "scientific_reductions_dtype",
        "prediction_index_hash",
    }
    index_unhashed = {
        key: value for key, value in index.items() if key != "prediction_index_hash"
    }
    expected_count = (
        DEVELOPMENT_CELL_COUNT if phase == DEVELOPMENT_ROLE else TARGET_CELL_COUNT
    )
    expected_tasks = 648 if phase == DEVELOPMENT_ROLE else 81
    if (
        set(index) != expected_index_fields
        or index.get("prediction_index_hash") != canonical_sha256(index_unhashed)
        or
        index.get("schema_version")
        != "midogpp_endpoint_router_prediction_index_v1"
        or index.get("phase") != phase
        or index.get("array_member") != array_member
        or index.get("array_sha256") != sha256_file(array_path)
        or index.get("labels_stored") is not False
        or index.get("storage_dtype") != "float32"
        or index.get("scientific_reductions_dtype") != "float64"
        or int(index.get("cell_count", -1)) != expected_count
        or int(index.get("task_count", -1)) != expected_tasks
        or int(index.get("unique_fit_count", -1)) != expected_count
        or (
            expected_plan is not None
            and (
                index.get("prediction_plan_hash") != expected_plan.plan_hash
                or index.get("action_library_hash")
                != expected_plan.action_library_hash
            )
        )
    ):
        raise ProtocolError("Endpoint-router prediction index binding drifted.")
    try:
        with np.load(array_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {
                "support_probabilities",
                "support_offsets",
                "evaluation_probabilities",
                "evaluation_offsets",
            }:
                raise ProtocolError("Endpoint-router prediction NPZ schema drifted.")
            support = np.asarray(arrays["support_probabilities"])
            support_offsets = np.asarray(arrays["support_offsets"])
            evaluation = np.asarray(arrays["evaluation_probabilities"])
            evaluation_offsets = np.asarray(arrays["evaluation_offsets"])
    except (OSError, ValueError) as exc:
        raise ProtocolError("Endpoint-router prediction NPZ is unreadable.") from exc
    raw_cells = index.get("cells")
    if (
        support.dtype != np.float32
        or evaluation.dtype != np.float32
        or support_offsets.dtype != np.int64
        or evaluation_offsets.dtype != np.int64
        or not isinstance(raw_cells, list)
        or len(raw_cells) != expected_count
        or len(support_offsets) != expected_count + 1
        or len(evaluation_offsets) != expected_count + 1
        or support_offsets[0] != 0
        or evaluation_offsets[0] != 0
        or support_offsets[-1] != len(support)
        or evaluation_offsets[-1] != len(evaluation)
        or np.any(support_offsets < 0)
        or np.any(evaluation_offsets < 0)
        or np.any(np.diff(support_offsets) <= 0)
        or np.any(np.diff(evaluation_offsets) <= 0)
    ):
        raise ProtocolError("Endpoint-router prediction store array geometry drifted.")
    cells: list[PredictionCell] = []
    expected_cell_fields = {
        "schema_version", "phase", "outer_target", "query_center", "action_id",
        "action_hash", "training_seed", "generation_seed",
        "support_row_identity_hash", "evaluation_row_identity_hash",
        "support_row_count", "evaluation_row_count",
        "support_probability_sha256", "evaluation_probability_sha256",
        "composition_hash", "scaler_state_hash", "fit_provenance_hash",
        "labels_stored", "cell_hash", "cell_ordinal", "support_array_start",
        "support_array_stop", "evaluation_array_start", "evaluation_array_stop",
    }
    planned_cells = {}
    if expected_plan is not None:
        planned_cells = {
            (
                task.outer_target, task.query_center, action.action_id,
                task.training_seed, task.generation_seed,
            ): (task, action)
            for task in expected_plan.tasks
            for action in task.actions
        }
    for ordinal, (raw, expected_key) in enumerate(
        zip(raw_cells, canonical_cell_keys(phase), strict=True)
    ):
        if not isinstance(raw, Mapping) or set(raw) != expected_cell_fields:
            raise ProtocolError("Endpoint-router prediction cell index is malformed.")
        start_s, stop_s = int(support_offsets[ordinal]), int(support_offsets[ordinal + 1])
        start_e, stop_e = int(evaluation_offsets[ordinal]), int(evaluation_offsets[ordinal + 1])
        observed_key = (
            str(raw.get("outer_target")),
            str(raw.get("query_center")),
            str(raw.get("action_id")),
            int(raw.get("training_seed", -1)),
            int(raw.get("generation_seed", -1)),
        )
        cell = PredictionCell(
            phase=phase,
            outer_target=observed_key[0],
            query_center=observed_key[1],
            action_id=observed_key[2],
            action_hash=str(raw.get("action_hash")),
            training_seed=observed_key[3],
            generation_seed=observed_key[4],
            support_row_identity_hash=str(raw.get("support_row_identity_hash")),
            evaluation_row_identity_hash=str(raw.get("evaluation_row_identity_hash")),
            support_probabilities=support[start_s:stop_s],
            evaluation_probabilities=evaluation[start_e:stop_e],
            composition_hash=str(raw.get("composition_hash")),
            scaler_state_hash=str(raw.get("scaler_state_hash")),
            fit_provenance_hash=str(raw.get("fit_provenance_hash")),
        )
        planned = planned_cells.get(expected_key)
        if (
            observed_key != expected_key
            or raw.get("cell_ordinal") != ordinal
            or raw.get("support_array_start") != start_s
            or raw.get("support_array_stop") != stop_s
            or raw.get("evaluation_array_start") != start_e
            or raw.get("evaluation_array_stop") != stop_e
            or raw.get("cell_hash") != cell.cell_hash
            or raw.get("support_probability_sha256")
            != cell.support_probability_sha256
            or raw.get("evaluation_probability_sha256")
            != cell.evaluation_probability_sha256
            or (expected_plan is not None and planned is None)
            or (
                planned is not None
                and (
                    raw.get("action_hash") != planned[1].action_hash
                    or raw.get("support_row_identity_hash")
                    != planned[0].support_row_identity_hash
                    or raw.get("evaluation_row_identity_hash")
                    != planned[0].evaluation_row_identity_hash
                    or raw.get("support_row_count") != len(planned[0].support_row_ids)
                    or raw.get("evaluation_row_count")
                    != len(planned[0].evaluation_row_ids)
                )
            )
        ):
            raise ProtocolError("Endpoint-router prediction cell reconstruction drifted.")
        cells.append(cell)
    mappings = tuple(
        _string_tuple_mapping(index.get(key), role=key, phase=phase)
        for key in (
            "support_row_ids_by_scope",
            "evaluation_row_ids_by_scope",
            "support_case_ids_by_scope",
            "evaluation_case_ids_by_scope",
        )
    )
    store = PredictionStore(
        phase=phase,
        cells=tuple(cells),
        support_row_ids_by_scope=mappings[0],
        evaluation_row_ids_by_scope=mappings[1],
        support_case_ids_by_scope=mappings[2],
        evaluation_case_ids_by_scope=mappings[3],
        source_stream_lock_hash=str(index.get("source_stream_lock_hash")),
        partition_lock_hash=str(index.get("partition_lock_hash")),
        cache_binding_hash=str(index.get("cache_binding_hash")),
        action_library_hash=str(index.get("action_library_hash")),
        store_hash=str(index.get("prediction_store_hash")),
    )
    if (
        index.get("source_stream_lock_hash") != store.source_stream_lock_hash
        or index.get("partition_lock_hash") != store.partition_lock_hash
        or index.get("cache_binding_hash") != store.cache_binding_hash
        or index.get("action_library_hash") != store.action_library_hash
    ):
        raise ProtocolError("Endpoint-router prediction-store lineage drifted.")
    if expected_plan is not None and any(
        value != expected
        for value, expected in (
            (store.source_stream_lock_hash, expected_plan.tasks[0].source_stream_lock_hash),
            (store.partition_lock_hash, expected_plan.tasks[0].partition_lock_hash),
            (store.cache_binding_hash, expected_plan.tasks[0].cache_binding_hash),
            (store.action_library_hash, expected_plan.action_library_hash),
        )
    ):
        raise ProtocolError("Endpoint-router prediction store escaped its plan.")
    return store


def _persist_store(store: PredictionStore, *, root: Path, plan: PredictionPlan) -> None:
    array_member, index_member = _members(store.phase)
    support_offsets = [0]
    evaluation_offsets = [0]
    support_parts: list[np.ndarray] = []
    evaluation_parts: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for ordinal, cell in enumerate(store.cells):
        support_parts.append(cell.support_probabilities)
        evaluation_parts.append(cell.evaluation_probabilities)
        support_offsets.append(support_offsets[-1] + len(cell.support_probabilities))
        evaluation_offsets.append(
            evaluation_offsets[-1] + len(cell.evaluation_probabilities)
        )
        rows.append(
            {
                **cell.index_payload(),
                "cell_ordinal": ordinal,
                "support_array_start": support_offsets[-2],
                "support_array_stop": support_offsets[-1],
                "evaluation_array_start": evaluation_offsets[-2],
                "evaluation_array_stop": evaluation_offsets[-1],
            }
        )
    stream = io.BytesIO()
    np.savez(
        stream,
        support_probabilities=np.ascontiguousarray(np.concatenate(support_parts), dtype=np.float32),
        support_offsets=np.asarray(support_offsets, dtype=np.int64),
        evaluation_probabilities=np.ascontiguousarray(np.concatenate(evaluation_parts), dtype=np.float32),
        evaluation_offsets=np.asarray(evaluation_offsets, dtype=np.int64),
    )
    array_path = root / array_member
    atomic_bytes(array_path, stream.getvalue())
    index = {
        "schema_version": "midogpp_endpoint_router_prediction_index_v1",
        "phase": store.phase,
        "cell_count": len(store.cells),
        "task_count": len(plan.tasks),
        "unique_fit_count": len(store.cells),
        "prediction_plan_hash": plan.plan_hash,
        "prediction_store_hash": store.store_hash,
        "source_stream_lock_hash": store.source_stream_lock_hash,
        "partition_lock_hash": store.partition_lock_hash,
        "cache_binding_hash": store.cache_binding_hash,
        "action_library_hash": store.action_library_hash,
        "array_member": array_member,
        "array_sha256": sha256_file(array_path),
        "support_row_ids_by_scope": {key: list(value) for key, value in store.support_row_ids_by_scope.items()},
        "evaluation_row_ids_by_scope": {key: list(value) for key, value in store.evaluation_row_ids_by_scope.items()},
        "support_case_ids_by_scope": {key: list(value) for key, value in store.support_case_ids_by_scope.items()},
        "evaluation_case_ids_by_scope": {key: list(value) for key, value in store.evaluation_case_ids_by_scope.items()},
        "cells": rows,
        "labels_stored": False,
        "storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
    }
    index["prediction_index_hash"] = canonical_sha256(index)
    atomic_json(root / index_member, index)


def _scope_mappings(
    tasks: Sequence[PredictionTask],
) -> tuple[dict[str, tuple[str, ...]], ...]:
    support_rows: dict[str, tuple[str, ...]] = {}
    evaluation_rows: dict[str, tuple[str, ...]] = {}
    support_cases: dict[str, tuple[str, ...]] = {}
    evaluation_cases: dict[str, tuple[str, ...]] = {}
    for task in tasks:
        scope = cell_scope(task.phase, task.outer_target, task.query_center)
        values = (
            task.support_row_ids,
            task.evaluation_row_ids,
            task.support_case_ids,
            task.evaluation_case_ids,
        )
        destinations = (support_rows, evaluation_rows, support_cases, evaluation_cases)
        for destination, value in zip(destinations, values, strict=True):
            if scope in destination and destination[scope] != value:
                raise ProtocolError("Endpoint-router task scope rows drifted across seeds.")
            destination[scope] = value
    return support_rows, evaluation_rows, support_cases, evaluation_cases


def _string_tuple_mapping(
    value: object, *, role: str, phase: str
) -> dict[str, tuple[str, ...]]:
    if (
        not isinstance(value, Mapping)
        or tuple(map(str, value)) != canonical_scopes(phase)
        or any(not isinstance(values, list) for values in value.values())
    ):
        raise ProtocolError(f"Endpoint-router prediction index {role} is malformed.")
    return {
        str(key): tuple(str(item) for item in values)
        for key, values in value.items()
    }


def _members(phase: str) -> tuple[str, str]:
    if phase == DEVELOPMENT_ROLE:
        return DEVELOPMENT_ARRAY_MEMBER, DEVELOPMENT_INDEX_MEMBER
    if phase == TARGET_ROLE:
        return TARGET_ARRAY_MEMBER, TARGET_INDEX_MEMBER
    raise ProtocolError("Endpoint-router prediction-store phase is invalid.")


__all__ = ("load_prediction_store", "materialize_prediction_store")
