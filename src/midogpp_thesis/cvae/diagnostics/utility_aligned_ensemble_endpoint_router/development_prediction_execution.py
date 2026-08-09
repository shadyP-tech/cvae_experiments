"""Combined support/evaluation development prediction execution.

Each spawned task fits the canonical base plus seven exact tails once and uses
that same fitted classifier for both label-free support and evaluation rows.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import fit_logistic_classifier
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .artifact_io import atomic_json, atomic_npy
from .actions import (
    build_inner_ensemble_endpoint_action_library,
    inner_action_library_for,
)
from .combined_prediction_io import (
    load_task_checkpoint,
    read_combined_store,
    write_combined_store,
    write_task_checkpoint,
)
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS, inner_candidate_sources
from .input_contracts import row_identity_hash
from .prediction_contracts import CombinedPredictionCell, CombinedPredictionStore, array_sha256, build_store
from .source_cache import SOURCE_ROWS_PER_CLASS, SourceCache
from ..utility_aligned_exact_tail_router.development_prediction_worker import (
    classifier_from_payload,
)


DEVELOPMENT_ARRAY_MEMBER = "arrays/ensemble_endpoint_development_predictions.npz"
DEVELOPMENT_INDEX_MEMBER = "manifests/ensemble_endpoint_development_prediction_index.json"
DEVELOPMENT_CHECKPOINT_DIRECTORY = "checkpoints/ensemble_endpoint_development"
EXPECTED_DEVELOPMENT_TASK_COUNT = 9 * 8 * 3 * 3
EXPECTED_DEVELOPMENT_CELL_COUNT = EXPECTED_DEVELOPMENT_TASK_COUNT * 8


def materialize_development_predictions(
    config: object,
    generation_lock: object,
    source_cache: SourceCache,
    frame: object,
    partitions: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> CombinedPredictionStore:
    del generation_lock  # Its semantic hash is already bound by the source-cache lock.
    array_path = root / DEVELOPMENT_ARRAY_MEMBER
    index_path = root / DEVELOPMENT_INDEX_MEMBER
    library_hash = build_inner_ensemble_endpoint_action_library().action_library_hash
    if array_path.is_file() and index_path.is_file():
        store = read_combined_store(array_path, index_path)
        _validate_development_store(
            store,
            source_cache_lock_hash=source_cache_lock_hash,
            partition_lock_hash=str(getattr(partitions, "lock_hash")),
            action_library_hash=library_hash,
        )
        _remove_working_directory(root)
        return store
    scratch = _write_combined_scratch(
        root,
        frame=frame,
        partitions=partitions,
        role="development",
        expected_support_case_count=_support_case_count(config),
    )
    tasks = _build_tasks(
        config,
        source_cache,
        partitions,
        scratch=scratch,
        source_cache_lock_hash=source_cache_lock_hash,
        root=root,
    )
    completed = _execute_or_resume(tasks, workers=int(getattr(config, "runtime")["classifier_workers"]))
    cells: list[CombinedPredictionCell] = []
    for task in tasks:
        payload = completed[str(task["task_id"])]
        for ordinal, action in enumerate(payload["actions"]):
            cells.append(_cell_from_result(task, payload, action, ordinal))
    store = build_store(
        role="development",
        cells=cells,
        source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=str(getattr(partitions, "lock_hash")),
        action_library_hash=library_hash,
        expected_cell_count=EXPECTED_DEVELOPMENT_CELL_COUNT,
        unique_classifier_fit_count=EXPECTED_DEVELOPMENT_CELL_COUNT,
    )
    _validate_development_store(
        store,
        source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=str(getattr(partitions, "lock_hash")),
        action_library_hash=library_hash,
    )
    write_combined_store(array_path, index_path, store)
    verified = read_combined_store(array_path, index_path)
    _validate_development_store(
        verified,
        source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=str(getattr(partitions, "lock_hash")),
        action_library_hash=library_hash,
    )
    _remove_working_directory(root)
    return verified


def _write_combined_scratch(
    root: Path,
    *,
    frame: object,
    partitions: object,
    role: str,
    expected_support_case_count: int = 2,
) -> Mapping[str, object]:
    checkpoint_root = root / DEVELOPMENT_CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows: list[object] = []
    centers: dict[str, object] = {}
    cursor = 0
    for center in CENTERS:
        support = tuple(partitions.support_rows_by_center[center])
        evaluation = tuple(partitions.evaluation_rows_by_center[center])
        if (
            len({row.case_id for row in support}) != expected_support_case_count
            or {row.case_id for row in support}
            & {row.case_id for row in evaluation}
        ):
            raise ProtocolError("Development combined scratch partition drifted.")
        selected = (*support, *evaluation)
        rows.extend(selected)
        support_start, support_stop = cursor, cursor + len(support)
        evaluation_start = support_stop
        evaluation_stop = evaluation_start + len(evaluation)
        cursor = evaluation_stop
        centers[center] = {
            "support_start": support_start,
            "support_stop": support_stop,
            "evaluation_start": evaluation_start,
            "evaluation_stop": evaluation_stop,
            "support_row_count": len(support),
            "evaluation_row_count": len(evaluation),
            "support_row_identity_hash": row_identity_hash(support),
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
        }
    embeddings = np.ascontiguousarray(frame.embeddings_for(rows), dtype=np.float32)
    array_path = checkpoint_root / "combined_validation_embeddings.npy"
    atomic_npy(array_path, embeddings)
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_combined_scratch_v1",
        "role": role,
        "array_path": str(array_path.resolve()),
        "array_sha256": array_sha256(embeddings),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "partition_lock_hash": str(partitions.lock_hash),
        "centers": centers,
        "support_labels_stored": False,
        "evaluation_labels_stored": False,
    }
    payload = {**unhashed, "scratch_hash": stable_hash(unhashed)}
    atomic_json(checkpoint_root / "combined_validation_index.json", payload)
    return payload


def _build_tasks(
    config: object,
    source_cache: SourceCache,
    partitions: object,
    *,
    scratch: Mapping[str, object],
    source_cache_lock_hash: str,
    root: Path,
) -> tuple[dict[str, object], ...]:
    centers = scratch.get("centers")
    if not isinstance(centers, Mapping):
        raise ProtocolError("Development combined scratch lacks centers.")
    checkpoint_root = root / DEVELOPMENT_CHECKPOINT_DIRECTORY / "tasks"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    source_rows = [record.to_row() for record in source_cache.source_records]
    tasks: list[dict[str, object]] = []
    for outer in CENTERS:
        for query in CENTERS:
            if outer == query:
                continue
            offset = centers[query]
            if not isinstance(offset, Mapping):
                raise ProtocolError("Development combined center offset drifted.")
            actions = inner_action_library_for(outer, query)
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    task_id = f"H{outer}_q{query}_train{training_seed}_gen{generation_seed}"
                    task: dict[str, object] = {
                        "schema_version": "midogpp_stage90_ensemble_endpoint_development_task_v1",
                        "task_id": task_id,
                        "task_role": "development",
                        "config_contract_hash": str(getattr(config, "contract_hash")),
                        "source_cache_lock_hash": source_cache_lock_hash,
                        "partition_lock_hash": str(partitions.lock_hash),
                        "outer_target": outer,
                        "query_center": query,
                        "scope_id": f"{outer}::{query}",
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "candidate_sources": list(inner_candidate_sources(outer, query)),
                        "source_array_path": str(source_cache.source_array_path.resolve()),
                        "source_index_rows": source_rows,
                        "combined_array_path": str(scratch["array_path"]),
                        "combined_array_sha256": str(scratch["array_sha256"]),
                        **dict(offset),
                        "action_ids": [action.action_id for action in actions],
                        "action_hashes": [action.action_hash for action in actions],
                        "classifier": getattr(config, "classifier").to_payload(),
                        "threads_per_fit": int(getattr(config, "runtime")["classifier_threads_per_worker"]),
                        "checkpoint_json_path": str(checkpoint_root / f"{task_id}.json"),
                        "checkpoint_npz_path": str(checkpoint_root / f"{task_id}.npz"),
                        "labels_available": False,
                    }
                    hash_payload = {
                        key: value for key, value in task.items()
                        if key not in {"checkpoint_json_path", "checkpoint_npz_path"}
                    }
                    task["task_hash"] = stable_hash(hash_payload)
                    tasks.append(task)
    if len(tasks) != EXPECTED_DEVELOPMENT_TASK_COUNT:
        raise ProtocolError("Development exact H/q task count drifted.")
    return tuple(tasks)


def development_prediction_task(task: Mapping[str, object]) -> Mapping[str, object]:
    if task.get("labels_available") is not False or task.get("task_role") != "development":
        raise ProtocolError("Development task escaped the label-free boundary.")
    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r", allow_pickle=False)
    if source_array.shape != (81, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or source_array.dtype != np.float32:
        raise ProtocolError("Development source cache geometry drifted.")
    source_index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): row
        for row in task["source_index_rows"]
    }
    training_seed, generation_seed = int(task["training_seed"]), int(task["generation_seed"])
    sources: dict[str, np.ndarray] = {}
    stream_ids: dict[str, str] = {}
    for source in task["candidate_sources"]:
        record = source_index[(str(source), training_seed, generation_seed)]
        sources[str(source)] = source_array[int(record["block_ordinal"])]
        stream_ids[str(source)] = str(record["stream_id"])
    combined_all = np.load(Path(str(task["combined_array_path"])), mmap_mode="r", allow_pickle=False)
    s0, s1 = int(task["support_start"]), int(task["support_stop"])
    e0, e1 = int(task["evaluation_start"]), int(task["evaluation_stop"])
    combined = np.ascontiguousarray(np.concatenate((combined_all[s0:s1], combined_all[e0:e1])), dtype=np.float32)
    support_count = s1 - s0
    actions = inner_action_library_for(task["outer_target"], task["query_center"])
    if [action.action_id for action in actions] != list(task["action_ids"]):
        raise ProtocolError("Development action order drifted.")
    spec = classifier_from_payload(task["classifier"])
    support_predictions: list[np.ndarray] = []
    support_probabilities: list[np.ndarray] = []
    evaluation_predictions: list[np.ndarray] = []
    evaluation_probabilities: list[np.ndarray] = []
    action_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Development fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for action in actions:
            train_x, train_y = _compose_development_action(sources, action)
            result = fit_logistic_classifier(train_x, train_y, combined, spec=spec)
            predictions = np.asarray(result.predictions, dtype=np.uint8)
            probabilities = np.asarray(result.probabilities, dtype=np.float64)
            if (
                tuple(int(value) for value in result.classes) != (0, 1)
                or predictions.shape != (len(combined),)
                or probabilities.shape != (len(combined), 2)
                or not result.converged
                or result.classifier_config_hash != spec.config_hash
            ):
                raise ProtocolError("Development classifier fit drifted.")
            positive = probabilities[:, 1].astype(np.float32, copy=False)
            support_pred = np.ascontiguousarray(predictions[:support_count], dtype=np.uint8)
            support_prob = np.ascontiguousarray(positive[:support_count], dtype=np.float32)
            evaluation_pred = np.ascontiguousarray(predictions[support_count:], dtype=np.uint8)
            evaluation_prob = np.ascontiguousarray(positive[support_count:], dtype=np.float32)
            composition_hash = stable_hash({
                "schema_version": "midogpp_stage90_ensemble_endpoint_development_composition_v1",
                "action_hash": action.action_hash,
                "source_stream_ids": stream_ids,
            })
            fit_hash = stable_hash({
                "task_hash": task["task_hash"], "action_hash": action.action_hash,
                "composition_hash": composition_hash,
                "scaler_state_hash": str(result.scaler_state_hash),
                "support_probability_sha256": array_sha256(support_prob),
                "evaluation_probability_sha256": array_sha256(evaluation_prob),
            })
            support_predictions.append(support_pred)
            support_probabilities.append(support_prob)
            evaluation_predictions.append(evaluation_pred)
            evaluation_probabilities.append(evaluation_prob)
            action_rows.append({
                "action_id": action.action_id,
                "action_hash": action.action_hash,
                "composition_hash": composition_hash,
                "scaler_state_hash": str(result.scaler_state_hash),
                "fit_provenance_hash": fit_hash,
                "aliased_from_action_id": None,
                "support_prediction_sha256": array_sha256(support_pred),
                "support_probability_sha256": array_sha256(support_prob),
                "evaluation_prediction_sha256": array_sha256(evaluation_pred),
                "evaluation_probability_sha256": array_sha256(evaluation_prob),
            })
    return write_task_checkpoint(
        task,
        support_predictions=np.stack(support_predictions).astype(np.uint8, copy=False),
        support_probabilities=np.stack(support_probabilities).astype(np.float32, copy=False),
        evaluation_predictions=np.stack(evaluation_predictions).astype(np.uint8, copy=False),
        evaluation_probabilities=np.stack(evaluation_probabilities).astype(np.float32, copy=False),
        action_rows=action_rows,
    )


def _execute_or_resume(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Development execution requires four classifier workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = load_task_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(development_prediction_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                loaded = load_task_checkpoint(task)
                if loaded is None:
                    raise ProtocolError("Development worker returned without checkpoint.")
                completed[str(task["task_id"])] = loaded
                print(f"[utility-ensemble-endpoint] development tasks {len(completed)}/{len(tasks)}", flush=True)
    if len(completed) != len(tasks):
        raise ProtocolError("Development checkpoint coverage is incomplete.")
    return completed


def _compose_development_action(
    sources: Mapping[str, np.ndarray], action: object
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(sources) != tuple(action.source_order):
        raise ProtocolError("Development composition source order drifted.")
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in (0, 1):
        counts = action.final_counts_by_class[label]
        for source in action.source_order:
            count = int(counts[source])
            values = np.asarray(sources[source])
            if (
                values.shape != (2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
                or count not in {144, 270}
            ):
                raise ProtocolError("Development source-prefix geometry drifted.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(np.asarray(values[start:start + count], dtype=np.float32))
            labels.append(np.full(count, label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    if embeddings.shape[0] != len(truth) or embeddings.shape[1] != COMMON_OUTPUT_DIM:
        raise ProtocolError("Development composed action geometry drifted.")
    return embeddings, truth


def _cell_from_result(
    task: Mapping[str, object], payload: Mapping[str, object], action: Mapping[str, object], ordinal: int
) -> CombinedPredictionCell:
    return CombinedPredictionCell(
        scope_id=str(task["scope_id"]), action_id=str(action["action_id"]),
        action_hash=str(action["action_hash"]), training_seed=int(task["training_seed"]),
        generation_seed=int(task["generation_seed"]),
        support_row_identity_hash=str(task["support_row_identity_hash"]),
        evaluation_row_identity_hash=str(task["evaluation_row_identity_hash"]),
        support_predictions=np.ascontiguousarray(payload["support_predictions"][ordinal], dtype=np.uint8),
        support_probabilities=np.ascontiguousarray(payload["support_probabilities"][ordinal], dtype=np.float32),
        evaluation_predictions=np.ascontiguousarray(payload["evaluation_predictions"][ordinal], dtype=np.uint8),
        evaluation_probabilities=np.ascontiguousarray(payload["evaluation_probabilities"][ordinal], dtype=np.float32),
        composition_hash=str(action["composition_hash"]), scaler_state_hash=str(action["scaler_state_hash"]),
        fit_provenance_hash=str(action["fit_provenance_hash"]), aliased_from_action_id=None,
    )


def _validate_development_store(
    store: CombinedPredictionStore, *, source_cache_lock_hash: str,
    partition_lock_hash: str, action_library_hash: str,
) -> None:
    if (
        store.role != "development"
        or store.expected_cell_count != EXPECTED_DEVELOPMENT_CELL_COUNT
        or store.unique_classifier_fit_count != EXPECTED_DEVELOPMENT_CELL_COUNT
        or store.source_cache_lock_hash != source_cache_lock_hash
        or store.partition_lock_hash != partition_lock_hash
        or store.action_library_hash != action_library_hash
    ):
        raise ProtocolError("Development combined store binding drifted.")
    expected_scopes = tuple(f"{outer}::{query}" for outer in CENTERS for query in CENTERS if outer != query)
    if {cell.scope_id for cell in store.cells} != set(expected_scopes):
        raise ProtocolError("Development H/q coverage drifted.")
    for scope in expected_scopes:
        outer, query = scope.split("::")
        action_ids = tuple(action.action_id for action in inner_action_library_for(outer, query))
        observed = {cell.action_id for cell in store.cells if cell.scope_id == scope}
        if observed != set(action_ids):
            raise ProtocolError("Development exact-tail action coverage drifted.")
    expected_keys = tuple(
        (f"{outer}::{query}", action.action_id, training_seed, generation_seed)
        for outer in CENTERS
        for query in CENTERS if outer != query
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for action in inner_action_library_for(outer, query)
    )
    if tuple(cell.key for cell in store.cells) != expected_keys:
        raise ProtocolError("Development combined cell ordering drifted.")
    for cell in store.cells:
        outer, query = cell.scope_id.split("::")
        expected_hash = {
            action.action_id: action.action_hash
            for action in inner_action_library_for(outer, query)
        }[cell.action_id]
        if cell.action_hash != expected_hash:
            raise ProtocolError("Development combined cell action hash drifted.")


def validate_development_prediction_store(
    store: CombinedPredictionStore, *, source_cache_lock_hash: str,
    partition_lock_hash: str,
) -> None:
    """Public canonical-order/action reconstruction gate for validators."""

    _validate_development_store(
        store, source_cache_lock_hash=source_cache_lock_hash,
        partition_lock_hash=partition_lock_hash,
        action_library_hash=build_inner_ensemble_endpoint_action_library().action_library_hash,
    )


def _remove_working_directory(root: Path) -> None:
    path = root / DEVELOPMENT_CHECKPOINT_DIRECTORY
    if path.exists():
        shutil.rmtree(path)


def _support_case_count(config: object) -> int:
    direct = getattr(config, "fixed_support_case_count_per_center", None)
    if direct is not None:
        return int(direct)
    protocol = getattr(config, "protocol", {})
    if isinstance(protocol, Mapping):
        return int(protocol.get("fixed_support_case_count_per_center", 2))
    return 2


__all__ = (
    "DEVELOPMENT_ARRAY_MEMBER",
    "DEVELOPMENT_INDEX_MEMBER",
    "EXPECTED_DEVELOPMENT_CELL_COUNT",
    "EXPECTED_DEVELOPMENT_TASK_COUNT",
    "development_prediction_task",
    "materialize_development_predictions",
    "validate_development_prediction_store",
)
