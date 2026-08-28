"""Spawn-safe one-thread classifier worker for held source-center pairs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from midogpp_thesis.real_features.classifier_reference.classifiers import fit_logistic_classifier

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, atomic_npz, read_json, sha256_array, sha256_file
from ....runtime.fixed_bank_a1_prediction_worker import classifier_from_payload, compose_action
from ....runtime.frozen_source_streams import (
    EXPECTED_STREAM_COUNT,
    SOURCE_ROWS_PER_CLASS,
    source_block_sha256,
)
from ....generation.contracts import COMMON_OUTPUT_DIM
from ..hashing import canonical_hash, require_sha256
from ..workstation import CPU_WORKER_ENVIRONMENT
from .held_actions import actions_for_held_pair, canonical_held_action_library
from .held_actions import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    held_candidate_sources,
)
from .resume import fsync_directory, fsync_file
from ..identity import CENTERS


_SOURCE_CACHE: dict[tuple[str, str], np.ndarray] = {}
_EVAL_CACHE: dict[tuple[str, str], np.ndarray] = {}
_BLOCK_HASH_CACHE: dict[tuple[str, str, int], str] = {}
_EVAL_HASH_CACHE: dict[tuple[str, str, int, int], str] = {}


def initialize_held_prediction_worker(threads: int = 1) -> None:
    if int(threads) != 1:
        raise ProtocolError("OE-PPUR v3 held worker requires exactly one BLAS thread.")
    for name, value in CPU_WORKER_ENVIRONMENT.items():
        os.environ[name] = value
    _SOURCE_CACHE.clear()
    _EVAL_CACHE.clear()
    _BLOCK_HASH_CACHE.clear()
    _EVAL_HASH_CACHE.clear()


def execute_or_resume_held_prediction_tasks(
    tasks: Sequence[Mapping[str, object]],
    *,
    workers: int = 4,
) -> Mapping[str, Mapping[str, object]]:
    rows = tuple(tasks)
    if workers != 4 or len(rows) != 324:
        raise ProtocolError("OE-PPUR v3 held predictions require 324 tasks on four workers.")
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in rows:
        loaded = load_held_prediction_checkpoint(task)
        if loaded is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = loaded
    if pending:
        with ProcessPoolExecutor(
            max_workers=4,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_held_prediction_worker,
            initargs=(1,),
        ) as executor:
            futures = {executor.submit(execute_held_prediction_task, dict(task)): task for task in pending}
            for future in as_completed(futures):
                future.result()
                task = futures[future]
                loaded = load_held_prediction_checkpoint(task)
                if loaded is None:
                    raise ProtocolError("OE-PPUR v3 held worker omitted its checkpoint.")
                completed[str(task["task_id"])] = loaded
                print(f"[oe-ppur-v3:source-predictions] tasks {len(completed)}/324", flush=True)
    if len(completed) != 324:
        raise ProtocolError("OE-PPUR v3 held prediction coverage is incomplete.")
    return MappingProxyType(completed)


def execute_held_prediction_task(task: Mapping[str, object]) -> None:
    _validate_task(task)
    blocks, evaluations = _load_task_arrays(task)
    spec = classifier_from_payload(task["classifier"])
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("OE-PPUR v3 held fitting requires threadpoolctl.") from exc
    combined = np.ascontiguousarray(np.concatenate(evaluations, axis=0), dtype=np.float32)
    values: list[np.ndarray] = []
    fit_receipts: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        for action in task["actions"]:
            if not isinstance(action, Mapping):
                raise ProtocolError("OE-PPUR v3 held worker action is untyped.")
            train_x, train_y, weights, composition_hash = compose_action(
                blocks, action, tuple(str(value) for value in task["candidate_sources"])
            )
            fitted = fit_logistic_classifier(
                train_x,
                train_y,
                combined,
                spec=spec,
                sample_weight=weights,
            )
            matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or matrix.shape != (len(combined), 2)
                or not fitted.converged
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
            ):
                raise ProtocolError("OE-PPUR v3 held classifier fit drifted.")
            positive = np.ascontiguousarray(matrix[:, 1], dtype=np.float32)
            values.append(positive)
            fit_receipts.append(
                {
                    "action_id": action["action_id"],
                    "action_hash": action["action_hash"],
                    "composition_hash": composition_hash,
                    "classifier_config_hash": fitted.classifier_config_hash,
                    "scaler_state_hash": fitted.scaler_state_hash,
                    "probability_sha256": sha256_array(positive),
                    "scaler_fit_used_sample_weight": False,
                    "sample_weight_scope": "logistic_regression_fit_only",
                    "labels_available": False,
                }
            )
    matrix = np.ascontiguousarray(np.stack(values), dtype=np.float32)
    first_count = len(evaluations[0])
    first = np.ascontiguousarray(matrix[:, :first_count], dtype=np.float32)
    second = np.ascontiguousarray(matrix[:, first_count:], dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    _persist_or_validate_npz(npz_path, first=first, second=second)
    fsync_file(npz_path)
    fsync_directory(npz_path.parent)
    body = {
        "schema_version": "oe_ppur_v3_held_prediction_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "excluded_centers": task["excluded_centers"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "source_frame_hash": task["source_frame_hash"],
        "source_stream_lock_hash": task["source_stream_lock_hash"],
        "held_action_library_sha256": task["held_action_library_sha256"],
        "held_mass_policy_receipt_sha256": task["held_mass_policy_receipt_sha256"],
        "npz_sha256": sha256_file(npz_path),
        "first_shape": list(first.shape),
        "second_shape": list(second.shape),
        "first_sha256": sha256_array(first),
        "second_sha256": sha256_array(second),
        "fit_receipts": fit_receipts,
        "threads_per_fit": 1,
        "labels_available": False,
        "target_rows_present": False,
    }
    payload = {**body, "checkpoint_hash": canonical_hash(body)}
    if json_path.exists():
        if json_path.is_symlink() or not json_path.is_file() or read_json(json_path) != payload:
            raise ProtocolError("OE-PPUR v3 existing held checkpoint differs.")
    else:
        atomic_json(json_path, payload)
    fsync_file(json_path)
    fsync_directory(json_path.parent)


def load_held_prediction_checkpoint(task: Mapping[str, object]) -> Mapping[str, object] | None:
    json_path = Path(str(task["checkpoint_json_path"]))
    npz_path = Path(str(task["checkpoint_npz_path"]))
    _reject_symlink_chain(json_path, require_leaf=False)
    _reject_symlink_chain(npz_path, require_leaf=False)
    if json_path.is_symlink() or npz_path.is_symlink():
        raise ProtocolError("OE-PPUR v3 held checkpoint path is a symlink.")
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.exists() and npz_path.is_file():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise ProtocolError("OE-PPUR v3 held checkpoint is partial.")
    payload = read_json(json_path)
    body = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if tuple(archive.files) != ("first", "second"):
                raise ProtocolError("OE-PPUR v3 held checkpoint array members drifted.")
            first = np.asarray(archive["first"])
            second = np.asarray(archive["second"])
    except (OSError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 held checkpoint array is unreadable.") from exc
    views = task["evaluation_views"]
    expected_first = int(views[0]["stop"]) - int(views[0]["start"])
    expected_second = int(views[1]["stop"]) - int(views[1]["start"])
    if (
        payload.get("checkpoint_hash") != canonical_hash(body)
        or payload.get("task_id") != task.get("task_id")
        or payload.get("task_hash") != task.get("task_hash")
        or payload.get("source_frame_hash") != task.get("source_frame_hash")
        or payload.get("source_stream_lock_hash") != task.get("source_stream_lock_hash")
        or payload.get("held_action_library_sha256") != task.get("held_action_library_sha256")
        or payload.get("held_mass_policy_receipt_sha256") != task.get("held_mass_policy_receipt_sha256")
        or payload.get("npz_sha256") != sha256_file(npz_path)
        or first.shape != (9, expected_first)
        or second.shape != (9, expected_second)
        or first.dtype != np.float32
        or second.dtype != np.float32
        or payload.get("first_sha256") != sha256_array(first)
        or payload.get("second_sha256") != sha256_array(second)
        or payload.get("labels_available") is not False
        or payload.get("target_rows_present") is not False
    ):
        raise ProtocolError("OE-PPUR v3 held checkpoint validation failed.")
    return payload


def load_held_checkpoint_arrays(task: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    if load_held_prediction_checkpoint(task) is None:
        raise ProtocolError("OE-PPUR v3 held checkpoint is absent.")
    with np.load(Path(str(task["checkpoint_npz_path"])), allow_pickle=False) as archive:
        first = np.ascontiguousarray(archive["first"], dtype=np.float32)
        second = np.ascontiguousarray(archive["second"], dtype=np.float32)
    first.setflags(write=False)
    second.setflags(write=False)
    return first, second


def _validate_task(task: Mapping[str, object]) -> None:
    body = {key: value for key, value in task.items() if key not in {"task_hash", "checkpoint_npz_path", "checkpoint_json_path"}}
    pair = tuple(str(value) for value in task.get("excluded_centers", ()))
    expected_actions = [action.to_payload() for action in actions_for_held_pair(*pair)] if len(pair) == 2 else []
    library = canonical_held_action_library()
    legal_pairs = tuple(
        (first, second)
        for index, first in enumerate(CENTERS)
        for second in CENTERS[index + 1 :]
    )
    try:
        training_seed = int(task["training_seed"])
        generation_seed = int(task["generation_seed"])
        pair_ordinal = legal_pairs.index(pair)
        seed_ordinal = TRAINING_SEEDS.index(training_seed) * len(GENERATION_SEEDS) + GENERATION_SEEDS.index(generation_seed)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 held task pair/seed scope drifted.") from exc
    candidates = held_candidate_sources(pair)
    records = task.get("source_records")
    views = task.get("evaluation_views")
    expected_task_id = f"held_{pair[0]}_{pair[1]}_train_{training_seed}_gen_{generation_seed}"
    if not isinstance(records, list) or not isinstance(views, list):
        raise ProtocolError("OE-PPUR v3 held task records/views are untyped.")
    observed_record_keys = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("OE-PPUR v3 held task source record is untyped.")
        try:
            key = (
                str(raw["source_center"]),
                int(raw["training_seed"]),
                int(raw["generation_seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("OE-PPUR v3 held task source record key drifted.") from exc
        observed_record_keys.append(key)
        if key[0] not in CENTERS or key[1] not in TRAINING_SEEDS or key[2] not in GENERATION_SEEDS:
            raise ProtocolError("OE-PPUR v3 held task source record scope drifted.")
        expected_record_ordinal = (
            CENTERS.index(key[0]) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
            + TRAINING_SEEDS.index(key[1]) * len(GENERATION_SEEDS)
            + GENERATION_SEEDS.index(key[2])
        )
        if (
            set(raw)
            != {
                "block_ordinal",
                "source_center",
                "training_seed",
                "generation_seed",
                "stream_id",
                "expert_lock_hash",
                "rows_per_class",
                "row_count",
                "feature_dim",
                "output_sha256",
            }
            or int(raw.get("block_ordinal", -1)) != expected_record_ordinal
            or int(raw.get("rows_per_class", -1)) != SOURCE_ROWS_PER_CLASS
            or int(raw.get("row_count", -1)) != 2 * SOURCE_ROWS_PER_CLASS
            or int(raw.get("feature_dim", -1)) != COMMON_OUTPUT_DIM
            or not str(raw.get("stream_id", ""))
            or len(str(raw.get("expert_lock_hash", ""))) not in (16, 64)
            or require_sha256(raw.get("output_sha256"), "source block output hash")
            != raw.get("output_sha256")
        ):
            raise ProtocolError("OE-PPUR v3 held task source record lineage drifted.")
    expected_record_keys = [
        (source, training_seed, generation_seed) for source in candidates
    ]
    view_centers = tuple(str(raw.get("center")) for raw in views if isinstance(raw, Mapping))
    valid_views = len(views) == 2 and view_centers == pair
    if valid_views:
        for raw in views:
            try:
                start, stop = int(raw["start"]), int(raw["stop"])
                row_ids = tuple(str(value) for value in raw["source_row_ids"])
                cache_indices = tuple(int(value) for value in raw["source_cache_row_indices"])
                require_sha256(raw["row_identity_hash"], "evaluation row identity hash")
                require_sha256(raw["slice_sha256"], "evaluation slice hash")
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("OE-PPUR v3 held task evaluation view drifted.") from exc
            if (
                start < 0
                or stop <= start
                or len(row_ids) != stop - start
                or len(cache_indices) != stop - start
                or len(set(row_ids)) != len(row_ids)
                or len(set(cache_indices)) != len(cache_indices)
                or canonical_hash(row_ids) != raw["row_identity_hash"]
            ):
                valid_views = False
    if (
        task.get("schema_version") != "oe_ppur_v3_held_prediction_task_v1"
        or task.get("task_hash") != canonical_hash(body)
        or task.get("actions") != expected_actions
        or task.get("held_action_library_sha256") != library.library_hash
        or task.get("held_mass_policy_receipt_sha256") != library.mass_policy.receipt_hash
        or pair not in legal_pairs
        or task.get("task_ordinal") != pair_ordinal * 9 + seed_ordinal
        or task.get("task_id") != expected_task_id
        or tuple(str(value) for value in task.get("candidate_sources", ())) != candidates
        or observed_record_keys != expected_record_keys
        or canonical_hash(records) != task.get("source_records_hash")
        or not valid_views
        or task.get("threads_per_fit") != 1
        or task.get("labels_available") is not False
        or task.get("target_rows_present") is not False
        or task.get("scaler_fit_used_sample_weight") is not False
        or task.get("sample_weight_scope") != "logistic_regression_fit_only"
    ):
        raise ProtocolError("OE-PPUR v3 held worker task boundary drifted.")
    for role in (
        "source_array_path",
        "evaluation_array_path",
        "checkpoint_npz_path",
        "checkpoint_json_path",
    ):
        path = Path(str(task.get(role, "")))
        _reject_symlink_chain(
            path,
            require_leaf=role in {"source_array_path", "evaluation_array_path"},
        )


def _load_task_arrays(task: Mapping[str, object]) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    records = task["source_records"]
    if canonical_hash(records) != task["source_records_hash"]:
        raise ProtocolError("OE-PPUR v3 held source records drifted.")
    source_path = Path(str(task["source_array_path"]))
    eval_path = Path(str(task["evaluation_array_path"]))
    if source_path.is_symlink() or eval_path.is_symlink():
        raise ProtocolError("OE-PPUR v3 held task arrays cannot be symlinks.")
    source_key = (str(source_path.resolve()), require_sha256(task["source_array_sha256"], "source array hash"))
    source = _SOURCE_CACHE.get(source_key)
    if source is None:
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        _SOURCE_CACHE[source_key] = source
    if source.shape != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM) or source.dtype != np.float32:
        raise ProtocolError("OE-PPUR v3 held source memmap geometry drifted.")
    blocks = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("OE-PPUR v3 held source record is untyped.")
        center = str(raw["source_center"])
        ordinal = int(raw["block_ordinal"])
        block = source[ordinal]
        key = (*source_key, ordinal)
        observed = _BLOCK_HASH_CACHE.get(key)
        if observed is None:
            observed = source_block_sha256(block)
            _BLOCK_HASH_CACHE[key] = observed
        if observed != raw["output_sha256"]:
            raise ProtocolError("OE-PPUR v3 held source block hash drifted.")
        blocks[center] = block
    eval_key = (str(eval_path.resolve()), require_sha256(task["evaluation_array_sha256"], "evaluation array hash"))
    evaluation = _EVAL_CACHE.get(eval_key)
    if evaluation is None:
        evaluation = np.load(eval_path, mmap_mode="r", allow_pickle=False)
        _EVAL_CACHE[eval_key] = evaluation
    if evaluation.ndim != 2 or evaluation.shape[1] != COMMON_OUTPUT_DIM or evaluation.dtype != np.float32:
        raise ProtocolError("OE-PPUR v3 held evaluation memmap geometry drifted.")
    views = []
    for raw in task["evaluation_views"]:
        start, stop = int(raw["start"]), int(raw["stop"])
        values = np.ascontiguousarray(evaluation[start:stop], dtype=np.float32)
        key = (*eval_key, start, stop)
        observed = _EVAL_HASH_CACHE.get(key)
        if observed is None:
            observed = sha256_array(values)
            _EVAL_HASH_CACHE[key] = observed
        if observed != raw["slice_sha256"] or not np.isfinite(values).all():
            raise ProtocolError("OE-PPUR v3 held evaluation slice drifted.")
        views.append(values)
    return blocks, (views[0], views[1])


def _persist_or_validate_npz(path: Path, *, first: np.ndarray, second: np.ndarray) -> None:
    if path.is_symlink():
        raise ProtocolError("OE-PPUR v3 held checkpoint array is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError("OE-PPUR v3 held checkpoint array is unsafe.")
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != ("first", "second"):
                raise ProtocolError("OE-PPUR v3 held checkpoint array members drifted.")
            if sha256_array(archive["first"]) != sha256_array(first) or sha256_array(archive["second"]) != sha256_array(second):
                raise ProtocolError("OE-PPUR v3 held checkpoint array differs.")
        return
    atomic_npz(path, first=first, second=second)


def _reject_symlink_chain(path: Path, *, require_leaf: bool) -> None:
    if not path.is_absolute() or path == Path(path.anchor):
        raise ProtocolError("OE-PPUR v3 held worker path is not absolute/narrow.")
    current = path
    while True:
        if current.is_symlink():
            raise ProtocolError("OE-PPUR v3 held worker path contains a symlink.")
        if current.exists() and current != path and not current.is_dir():
            raise ProtocolError("OE-PPUR v3 held worker parent is not a directory.")
        if current == current.parent:
            break
        current = current.parent
    if require_leaf and not path.is_file():
        raise ProtocolError("OE-PPUR v3 held worker input array is absent.")
    if not require_leaf and not path.exists() and not path.parent.is_dir():
        raise ProtocolError("OE-PPUR v3 held worker checkpoint parent is absent.")


__all__ = (
    "execute_held_prediction_task",
    "execute_or_resume_held_prediction_tasks",
    "initialize_held_prediction_worker",
    "load_held_checkpoint_arrays",
    "load_held_prediction_checkpoint",
)
