"""Spawn-safe CPU worker and exact checkpoint validation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ...real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ..generation.contracts import COMMON_OUTPUT_DIM
from ..protocol import ProtocolError
from .artifact_io import atomic_json, atomic_npz, read_json, sha256_array, sha256_file
from .fixed_bank_a1_prediction_contracts import (
    ACTION_COUNT_PER_TARGET,
    stable_digest,
)
from .frozen_source_streams import (
    EXPECTED_STREAM_COUNT,
    SOURCE_ROWS_PER_CLASS,
    source_block_sha256,
)


_SOURCE_ARRAY_CACHE: dict[tuple[str, str], np.ndarray] = {}
_TARGET_ARRAY_CACHE: dict[tuple[str, str], np.ndarray] = {}
_SOURCE_BLOCK_HASH_CACHE: dict[tuple[str, str, int], str] = {}
_TARGET_SLICE_HASH_CACHE: dict[tuple[str, str, int, int], str] = {}


def execute_or_resume_prediction_tasks(
    tasks: Sequence[Mapping[str, object]], *, workers: int
) -> Mapping[str, Mapping[str, object]]:
    if workers != 4:
        raise ProtocolError("Fixed-bank A1 predictions require four workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        loaded = load_prediction_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        thread_counts = {int(task["threads_per_fit"]) for task in pending}
        if thread_counts != {3}:
            raise ProtocolError("Fixed-bank A1 worker thread topology drifted.")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_prediction_worker,
            initargs=(3,),
        ) as executor:
            futures = {
                executor.submit(execute_prediction_task, dict(task)): task
                for task in pending
            }
            for future in as_completed(futures):
                future.result()
                task = futures[future]
                loaded = load_prediction_checkpoint(task)
                if loaded is None:
                    raise ProtocolError(
                        "Fixed-bank A1 worker omitted its checkpoint."
                    )
                completed[str(task["task_id"])] = loaded
                print(
                    f"[flip-router:predictions] tasks {len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Fixed-bank A1 checkpoint coverage is incomplete.")
    return MappingProxyType(completed)


def execute_prediction_task(task: Mapping[str, object]) -> None:
    """Execute one plain-dict task and return only through durable checkpoint."""

    unhashed = {
        key: value
        for key, value in task.items()
        if key
        not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    actions = task.get("actions")
    if (
        task.get("task_hash") != stable_hash(unhashed)
        or task.get("labels_available") is not False
        or task.get("target_expert_available") is not False
        or not isinstance(actions, list)
        or len(actions) != ACTION_COUNT_PER_TARGET
    ):
        raise ProtocolError("Fixed-bank A1 worker task boundary drifted.")
    blocks, evaluation = load_task_arrays(task)
    spec = classifier_from_payload(task["classifier"])
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Fixed-bank A1 fitting requires threadpoolctl.") from exc
    values: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for action in actions:
            train_x, train_y, weights, composition_hash = compose_action(
                blocks, action, tuple(task["candidate_sources"])
            )
            fitted = fit_logistic_classifier(
                train_x,
                train_y,
                evaluation,
                spec=spec,
                sample_weight=None if np.all(weights == 1.0) else weights,
            )
            matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or matrix.shape != (len(evaluation), 2)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                or not fitted.converged
            ):
                raise ProtocolError("Fixed-bank A1 classifier fit drifted.")
            positive = np.ascontiguousarray(matrix[:, 1], dtype=np.float32)
            probability_hash = sha256_array(positive)
            prediction_hash = sha256_array(
                (positive >= np.float32(0.5)).astype(np.uint8)
            )
            fit_payload = {
                "schema_version": "fixed_bank_a1_classifier_fit_v1",
                "task_hash": task["task_hash"],
                "action_id": action["action_id"],
                "action_hash": action["action_hash"],
                "composition_hash": composition_hash,
                "classifier_config_hash": fitted.classifier_config_hash,
                "scaler_state_hash": fitted.scaler_state_hash,
                "probability_sha256": probability_hash,
                "prediction_sha256": prediction_hash,
                "sample_weight_scope": "logistic_regression_fit_only",
                "scaler_fit_used_sample_weight": False,
                "labels_available": False,
            }
            values.append(positive)
            metadata.append(
                {
                    "action_id": action["action_id"],
                    "action_hash": action["action_hash"],
                    "probability_sha256": probability_hash,
                    "prediction_sha256": prediction_hash,
                    "fit_provenance_hash": stable_hash(fit_payload),
                }
            )
    matrix = np.ascontiguousarray(np.stack(values), dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    if npz_path.is_symlink() or json_path.is_symlink():
        raise ProtocolError("Fixed-bank A1 checkpoint path is a symlink.")
    atomic_npz(npz_path, probabilities=matrix)
    checkpoint = {
        "schema_version": "fixed_bank_a1_prediction_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "target_center": task["target_center"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "target_row_identity_hash": task["target_row_identity_hash"],
        "array_sha256": sha256_file(npz_path),
        "array_shape": list(matrix.shape),
        "array_dtype": str(matrix.dtype),
        "actions": metadata,
        "labels_available": False,
        "target_expert_available": False,
    }
    atomic_json(
        json_path,
        {**checkpoint, "checkpoint_hash": stable_hash(checkpoint)},
    )


def _initialize_prediction_worker(threads: int) -> None:
    """Bind every persistent CPU child to the frozen 3-thread topology."""

    if int(threads) != 3:
        raise ProtocolError("Fixed-bank A1 worker initializer drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "3"
    _SOURCE_ARRAY_CACHE.clear()
    _TARGET_ARRAY_CACHE.clear()
    _SOURCE_BLOCK_HASH_CACHE.clear()
    _TARGET_SLICE_HASH_CACHE.clear()


def load_prediction_checkpoint(
    task: Mapping[str, object],
) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    if json_path.is_symlink() or npz_path.is_symlink():
        raise ProtocolError("Fixed-bank A1 checkpoint is a symlink.")
    if not json_path.is_file() and not npz_path.is_file():
        return None
    if not json_path.is_file() and npz_path.is_file():
        # Exact crash boundary: the array is written before its hash-bearing
        # JSON.  The worker deterministically overwrites it from the task.
        return None
    if json_path.is_file() and not npz_path.is_file():
        raise ProtocolError("Fixed-bank A1 checkpoint is partial.")
    payload = read_json(json_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        if tuple(archive.files) != ("probabilities",):
            raise ProtocolError("Fixed-bank A1 checkpoint array members drifted.")
        values = np.asarray(archive["probabilities"])
    expected_actions = task.get("actions")
    observed_actions = payload.get("actions")
    expected_rows = int(task["target_stop"]) - int(task["target_start"])
    if (
        payload.get("checkpoint_hash")
        != stable_hash(
            {key: value for key, value in payload.items() if key != "checkpoint_hash"}
        )
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("task_id") != task.get("task_id")
        or payload.get("target_center") != task.get("target_center")
        or payload.get("training_seed") != task.get("training_seed")
        or payload.get("generation_seed") != task.get("generation_seed")
        or payload.get("target_row_identity_hash")
        != task.get("target_row_identity_hash")
        or payload.get("array_sha256") != sha256_file(npz_path)
        or values.shape != (ACTION_COUNT_PER_TARGET, expected_rows)
        or values.dtype != np.float32
        or payload.get("array_shape") != list(values.shape)
        or payload.get("array_dtype") != str(values.dtype)
        or not isinstance(expected_actions, list)
        or not isinstance(observed_actions, list)
        or len(expected_actions) != ACTION_COUNT_PER_TARGET
        or len(observed_actions) != ACTION_COUNT_PER_TARGET
        or payload.get("labels_available") is not False
        or payload.get("target_expert_available") is not False
    ):
        raise ProtocolError("Fixed-bank A1 checkpoint validation failed.")
    for ordinal, row in enumerate(observed_actions):
        expected = expected_actions[ordinal]
        if (
            not isinstance(row, Mapping)
            or row.get("action_id") != expected.get("action_id")
            or row.get("action_hash") != expected.get("action_hash")
            or row.get("probability_sha256") != sha256_array(values[ordinal])
            or row.get("prediction_sha256")
            != sha256_array(
                (values[ordinal] >= np.float32(0.5)).astype(np.uint8)
            )
            or not stable_digest(row.get("fit_provenance_hash"))
        ):
            raise ProtocolError("Fixed-bank A1 checkpoint action record drifted.")
    return payload


def load_task_arrays(
    task: Mapping[str, object]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    records = task["source_index_rows"]
    if stable_hash(records) != task["source_index_rows_hash"]:
        raise ProtocolError("Fixed-bank A1 source index drifted.")
    index = {
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): row
        for row in records
    }
    source_path = Path(str(task["source_array_path"]))
    target_path = Path(str(task["target_array_path"]))
    if source_path.is_symlink() or target_path.is_symlink():
        raise ProtocolError("Fixed-bank A1 task arrays cannot be symlinks.")
    # The parent validated each large file once while planning.  Workers bind
    # the exact resolved path and immutable slice hashes without rehashing the
    # complete 153 MB source memmap for every one of 81 tasks.
    source_key = (str(source_path.resolve()), str(task["source_array_sha256"]))
    source = _SOURCE_ARRAY_CACHE.get(source_key)
    if source is None:
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        _SOURCE_ARRAY_CACHE[source_key] = source
    if (
        source.shape
        != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        or source.dtype != np.float32
    ):
        raise ProtocolError("Fixed-bank A1 source memmap geometry drifted.")
    blocks = {}
    for candidate in task["candidate_sources"]:
        record = index[
            (
                str(candidate),
                int(task["training_seed"]),
                int(task["generation_seed"]),
            )
        ]
        block_ordinal = int(record["block_ordinal"])
        block = source[block_ordinal]
        block_key = (*source_key, block_ordinal)
        observed_block_hash = _SOURCE_BLOCK_HASH_CACHE.get(block_key)
        if observed_block_hash is None:
            observed_block_hash = source_block_sha256(block)
            _SOURCE_BLOCK_HASH_CACHE[block_key] = observed_block_hash
        if observed_block_hash != record["output_sha256"]:
            raise ProtocolError("Fixed-bank A1 source block hash drifted.")
        blocks[str(candidate)] = block
    target_key = (str(target_path.resolve()), str(task["target_array_sha256"]))
    target = _TARGET_ARRAY_CACHE.get(target_key)
    if target is None:
        target = np.load(target_path, mmap_mode="r", allow_pickle=False)
        _TARGET_ARRAY_CACHE[target_key] = target
    start = int(task["target_start"])
    stop = int(task["target_stop"])
    evaluation = np.ascontiguousarray(
        target[start:stop],
        dtype=np.float32,
    )
    slice_key = (*target_key, start, stop)
    observed_slice_hash = _TARGET_SLICE_HASH_CACHE.get(slice_key)
    if observed_slice_hash is None:
        observed_slice_hash = sha256_array(evaluation)
        _TARGET_SLICE_HASH_CACHE[slice_key] = observed_slice_hash
    if (
        not np.isfinite(evaluation).all()
        or observed_slice_hash != task.get("target_slice_sha256")
    ):
        raise ProtocolError("Fixed-bank A1 target slice contains nonfinite values.")
    return blocks, evaluation


def compose_action(
    blocks: Mapping[str, np.ndarray],
    action: Mapping[str, object],
    candidates: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    counts_raw = action["counts_by_class"]
    weights_raw = action["sample_weight_by_source"]
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    canonical_counts = {}
    for label in (0, 1):
        counts = {
            str(key): int(value) for key, value in counts_raw[str(label)].items()
        }
        if tuple(counts) != candidates:
            raise ProtocolError("Fixed-bank A1 composition source order drifted.")
        canonical_counts[str(label)] = counts
        for source, count in counts.items():
            if count <= 0 or count > SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Fixed-bank A1 source prefix exceeds capacity.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(
                np.asarray(blocks[source][start : start + count], dtype=np.float32)
            )
            labels.append(np.full(count, label, dtype=np.uint8))
            weights.append(
                np.full(count, float(weights_raw[source]), dtype=np.float64)
            )
    x = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    y = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    w = np.ascontiguousarray(np.concatenate(weights), dtype=np.float64)
    composition = {
        "counts_by_class": canonical_counts,
        "sample_weight_by_source": dict(weights_raw),
        "action_hash": action["action_hash"],
    }
    return x, y, w, stable_hash(composition)


def classifier_from_payload(raw: Mapping[str, object]) -> ClassifierSpec:
    return ClassifierSpec(
        family=str(raw["family"]),
        C=float(raw["C"]),
        penalty=str(raw["penalty"]),
        solver=str(raw["solver"]),
        max_iter=int(raw["max_iter"]),
        class_weight=(
            None if raw["class_weight"] is None else str(raw["class_weight"])
        ),
        random_state=int(raw["random_state"]),
        l1_ratio=(
            None if raw["l1_ratio"] is None else float(raw["l1_ratio"])
        ),
        threshold_policy=str(raw["threshold_policy"]),
        scaler_fit=str(raw["scaler_fit"]),
    )


__all__ = (
    "classifier_from_payload",
    "compose_action",
    "execute_or_resume_prediction_tasks",
    "execute_prediction_task",
    "load_prediction_checkpoint",
    "load_task_arrays",
)
