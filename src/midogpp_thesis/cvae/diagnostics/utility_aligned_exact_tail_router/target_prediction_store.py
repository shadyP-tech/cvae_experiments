"""Resumable target execution, consolidation, and canonical persistence."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import FrozenExactTailActionLibrary
from .artifact_io import atomic_csv, atomic_json, atomic_npz, read_json, sha256_file
from .input_contracts import FixedPartitionSurface
from .partitions import CaseFoldSurface
from .source_cache_contracts import SourceCache
from .target_prediction_contracts import (
    TARGET_PREDICTION_ARRAY_MEMBER,
    TARGET_PREDICTION_CACHE_MEMBER,
    TARGET_PREDICTION_INDEX_COLUMNS,
    TARGET_PREDICTION_INDEX_MEMBER,
    TargetPredictionCell,
    TargetPredictionStore,
    array_sha256,
    canonical_target_cell_keys,
)
from .target_prediction_planning import (
    build_target_prediction_tasks,
    write_target_evaluation_scratch,
)
from .target_prediction_worker import target_prediction_task


def materialize_target_predictions(
    config: object,
    source_cache: SourceCache,
    library: FrozenExactTailActionLibrary,
    frame: object,
    partitions: FixedPartitionSurface,
    case_folds: CaseFoldSurface,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> TargetPredictionStore:
    final_members = (
        root / TARGET_PREDICTION_ARRAY_MEMBER,
        root / TARGET_PREDICTION_INDEX_MEMBER,
        root / TARGET_PREDICTION_CACHE_MEMBER,
    )
    if all(path.is_file() for path in final_members):
        return read_target_prediction_store(
            root,
            library=library,
            source_cache_lock_hash=source_cache_lock_hash,
            case_fold_lock_hash=case_folds.lock_hash,
        )
    scratch = write_target_evaluation_scratch(
        root, frame=frame, partitions=partitions
    )
    tasks = build_target_prediction_tasks(
        config,
        source_cache,
        library,
        partitions,
        source_cache_lock_hash=source_cache_lock_hash,
        case_fold_lock_hash=case_folds.lock_hash,
        scratch=scratch,
        root=root,
    )
    completed = _execute_or_resume(tasks, workers=int(getattr(config, "runtime")["classifier_workers"]))
    store = _assemble_store(
        tasks,
        completed,
        library=library,
        source_cache_lock_hash=source_cache_lock_hash,
        case_fold_lock_hash=case_folds.lock_hash,
    )
    write_target_prediction_store(root, store)
    verified = read_target_prediction_store(
        root,
        library=library,
        source_cache_lock_hash=source_cache_lock_hash,
        case_fold_lock_hash=case_folds.lock_hash,
    )
    shutil.rmtree(root / "checkpoints/target_predictions", ignore_errors=True)
    return verified


def _execute_or_resume(
    tasks: Sequence[Mapping[str, object]],
    *,
    workers: int,
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Utility-aligned target execution requires four workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = load_target_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
            futures = {executor.submit(target_prediction_task, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                future.result()
                loaded = load_target_checkpoint(task)
                if loaded is None:
                    raise ProtocolError("Target worker returned without a checkpoint.")
                completed[str(task["task_id"])] = loaded
                print(
                    f"[utility-exact-tail] target tasks {len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Utility-aligned target checkpoint coverage is incomplete.")
    return completed


def load_target_checkpoint(
    task: Mapping[str, object],
) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if not json_path.is_file() and not npz_path.is_file():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        return None
    payload = read_json(json_path)
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    if (
        payload.get("schema_version")
        != "midogpp_utility_aligned_stage90_target_checkpoint_v1"
        or payload.get("task_id") != task["task_id"]
        or payload.get("task_hash") != task["task_hash"]
        or payload.get("array_file_sha256") != sha256_file(npz_path)
        or payload.get("checkpoint_hash") != stable_hash(unhashed)
        or payload.get("labels_used") is not False
    ):
        raise ProtocolError("Utility-aligned target checkpoint binding drifted.")
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != len(task["actions"]):
        raise ProtocolError("Utility-aligned target checkpoint action coverage drifted.")
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            predictions = np.asarray(arrays["predictions"])
            probabilities = np.asarray(arrays["probabilities"])
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError("Utility-aligned target checkpoint arrays are unreadable.") from exc
    expected_shape = (len(actions), int(payload["evaluation_row_count"]))
    if predictions.shape != expected_shape or probabilities.shape != expected_shape:
        raise ProtocolError("Utility-aligned target checkpoint array shape drifted.")
    for index, action in enumerate(actions):
        if (
            action.get("action_id") != task["actions"][index]["action_id"]
            or action.get("action_hash") != task["actions"][index]["action_hash"]
            or action.get("prediction_sha256") != array_sha256(predictions[index])
            or action.get("probability_sha256") != array_sha256(probabilities[index])
        ):
            raise ProtocolError("Utility-aligned target checkpoint cell drifted.")
    return {**payload, "predictions": predictions, "probabilities": probabilities}


def _assemble_store(
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
    *,
    library: FrozenExactTailActionLibrary,
    source_cache_lock_hash: str,
    case_fold_lock_hash: str,
) -> TargetPredictionStore:
    cells: list[TargetPredictionCell] = []
    unique_fits = 0
    for task in tasks:
        result = completed[str(task["task_id"])]
        predictions = np.asarray(result["predictions"])
        probabilities = np.asarray(result["probabilities"])
        for index, action in enumerate(result["actions"]):
            cells.append(
                TargetPredictionCell(
                    target_center=str(task["target_center"]),
                    action_id=str(action["action_id"]),
                    action_hash=str(action["action_hash"]),
                    training_seed=int(task["training_seed"]),
                    generation_seed=int(task["generation_seed"]),
                    evaluation_row_identity_hash=str(result["evaluation_row_identity_hash"]),
                    predictions=np.ascontiguousarray(predictions[index], dtype=np.uint8),
                    probabilities=np.ascontiguousarray(probabilities[index], dtype=np.float32),
                    composition_sha256=str(action["composition_sha256"]),
                    scaler_state_hash=str(action["scaler_state_hash"]),
                    aliased_fit=bool(action["aliased_fit"]),
                )
            )
        unique_fits += int(result["unique_classifier_fit_count"])
    if tuple(cell.key for cell in cells) != canonical_target_cell_keys(library):
        raise ProtocolError("Utility-aligned target cell ordering drifted.")
    provisional = {
        "schema_version": "midogpp_utility_aligned_stage90_target_prediction_store_v1",
        "action_library_hash": library.action_library_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "case_fold_lock_hash": case_fold_lock_hash,
        "cell_count": len(cells),
        "cell_keys": [list(cell.key) for cell in cells],
        "cell_action_hashes": [cell.action_hash for cell in cells],
        "cell_prediction_hashes": [array_sha256(cell.predictions) for cell in cells],
        "cell_probability_hashes": [array_sha256(cell.probabilities) for cell in cells],
        "composition_hashes": [cell.composition_sha256 for cell in cells],
        "unique_classifier_fit_count": unique_fits,
        "labels_stored": False,
    }
    return TargetPredictionStore(
        cells=tuple(cells),
        action_library_hash=library.action_library_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        case_fold_lock_hash=case_fold_lock_hash,
        unique_classifier_fit_count=unique_fits,
        store_hash=stable_hash(provisional),
    )


def write_target_prediction_store(root: Path, store: TargetPredictionStore) -> None:
    offsets = [0]
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for ordinal, cell in enumerate(store.cells):
        start = offsets[-1]
        stop = start + len(cell.predictions)
        offsets.append(stop)
        predictions.append(cell.predictions)
        probabilities.append(cell.probabilities)
        rows.append(
            {
                "schema_version": "midogpp_utility_aligned_stage90_target_prediction_cell_v1",
                "cell_ordinal": ordinal,
                "target_center": cell.target_center,
                "action_id": cell.action_id,
                "action_hash": cell.action_hash,
                "training_seed": cell.training_seed,
                "generation_seed": cell.generation_seed,
                "evaluation_row_count": len(cell.predictions),
                "evaluation_row_identity_hash": cell.evaluation_row_identity_hash,
                "prediction_sha256": array_sha256(cell.predictions),
                "probability_sha256": array_sha256(cell.probabilities),
                "composition_sha256": cell.composition_sha256,
                "scaler_state_hash": cell.scaler_state_hash,
                "array_start": start,
                "array_stop": stop,
                "aliased_fit": cell.aliased_fit,
                "labels_available": False,
            }
        )
    array_path = root / TARGET_PREDICTION_ARRAY_MEMBER
    index_path = root / TARGET_PREDICTION_INDEX_MEMBER
    atomic_npz(
        array_path,
        predictions=np.concatenate(predictions).astype(np.uint8, copy=False),
        probabilities=np.concatenate(probabilities).astype(np.float32, copy=False),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    atomic_csv(index_path, rows, TARGET_PREDICTION_INDEX_COLUMNS)
    atomic_json(
        root / TARGET_PREDICTION_CACHE_MEMBER,
        {
            **store.to_payload(),
            "prediction_array_member": TARGET_PREDICTION_ARRAY_MEMBER,
            "prediction_array_sha256": sha256_file(array_path),
            "prediction_index_member": TARGET_PREDICTION_INDEX_MEMBER,
            "prediction_index_sha256": sha256_file(index_path),
        },
    )


def read_target_prediction_store(
    root: Path,
    *,
    library: FrozenExactTailActionLibrary,
    source_cache_lock_hash: str,
    case_fold_lock_hash: str,
) -> TargetPredictionStore:
    meta = read_json(root / TARGET_PREDICTION_CACHE_MEMBER)
    array_path = root / TARGET_PREDICTION_ARRAY_MEMBER
    if (
        meta.get("action_library_hash") != library.action_library_hash
        or meta.get("source_cache_lock_hash") != source_cache_lock_hash
        or meta.get("case_fold_lock_hash") != case_fold_lock_hash
        or meta.get("prediction_array_sha256") != sha256_file(array_path)
        or meta.get("prediction_index_sha256")
        != sha256_file(root / TARGET_PREDICTION_INDEX_MEMBER)
    ):
        raise ProtocolError("Utility-aligned target prediction cache binding drifted.")
    import csv

    with (root / TARGET_PREDICTION_INDEX_MEMBER).open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    try:
        with np.load(array_path, allow_pickle=False) as payload:
            flat_pred = np.asarray(payload["predictions"])
            flat_prob = np.asarray(payload["probabilities"])
            offsets = np.asarray(payload["offsets"])
    except (OSError, ValueError, KeyError) as exc:
        raise ProtocolError("Utility-aligned target prediction arrays are unreadable.") from exc
    if (
        flat_pred.ndim != 1
        or flat_pred.dtype != np.uint8
        or flat_prob.shape != flat_pred.shape
        or flat_prob.dtype != np.float32
        or offsets.ndim != 1
        or offsets.dtype != np.int64
        or len(offsets) != len(rows) + 1
        or len(offsets) == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(flat_pred)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ProtocolError("Utility-aligned target prediction array layout drifted.")
    expected_keys = canonical_target_cell_keys(library)
    cells: list[TargetPredictionCell] = []
    for ordinal, row in enumerate(rows):
        if set(row) != set(TARGET_PREDICTION_INDEX_COLUMNS):
            raise ProtocolError("Utility-aligned target prediction index schema drifted.")
        start, stop = int(row["array_start"]), int(row["array_stop"])
        key = (
            row["target_center"],
            row["action_id"],
            int(row["training_seed"]),
            int(row["generation_seed"]),
        )
        expected_action_hash = library.action(key[0], key[1]).action_hash
        if (
            row["schema_version"]
            != "midogpp_utility_aligned_stage90_target_prediction_cell_v1"
            or int(row["cell_ordinal"]) != ordinal
            or key != expected_keys[ordinal]
            or row["action_hash"] != expected_action_hash
            or start != int(offsets[ordinal])
            or stop != int(offsets[ordinal + 1])
            or stop - start != int(row["evaluation_row_count"])
            or start < 0
            or stop < start
            or stop > len(flat_pred)
            or row["labels_available"].lower() != "false"
            or row["prediction_sha256"] != array_sha256(flat_pred[start:stop])
            or row["probability_sha256"] != array_sha256(flat_prob[start:stop])
        ):
            raise ProtocolError("Utility-aligned target prediction index binding drifted.")
        cells.append(
            TargetPredictionCell(
                target_center=row["target_center"],
                action_id=row["action_id"],
                action_hash=row["action_hash"],
                training_seed=int(row["training_seed"]),
                generation_seed=int(row["generation_seed"]),
                evaluation_row_identity_hash=row["evaluation_row_identity_hash"],
                predictions=np.ascontiguousarray(flat_pred[start:stop], dtype=np.uint8),
                probabilities=np.ascontiguousarray(flat_prob[start:stop], dtype=np.float32),
                composition_sha256=row["composition_sha256"],
                scaler_state_hash=row["scaler_state_hash"],
                aliased_fit=row["aliased_fit"].lower() == "true",
            )
        )
    store = TargetPredictionStore(
        cells=tuple(cells),
        action_library_hash=library.action_library_hash,
        source_cache_lock_hash=source_cache_lock_hash,
        case_fold_lock_hash=case_fold_lock_hash,
        unique_classifier_fit_count=int(meta["unique_classifier_fit_count"]),
        store_hash=str(meta["store_hash"]),
    )
    if tuple(cell.key for cell in store.cells) != expected_keys:
        raise ProtocolError("Utility-aligned persisted target cell order drifted.")
    return store


__all__ = (
    "load_target_checkpoint",
    "materialize_target_predictions",
    "read_target_prediction_store",
    "write_target_prediction_store",
)
