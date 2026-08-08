"""Source-cache scratch construction, task planning, and GPU scheduling."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...generation.contracts import GenerationLock
from ...generation.generation import source_generation_plan
from ...protocol import ProtocolError
from .artifact_io import atomic_save_npy, atomic_write_json
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .source_cache_contracts import (
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
)
from .source_cache_worker import generate_source_task


class SourceTaskConfig(Protocol):
    expert_bank_root: Path
    contract_hash: str


def write_support_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: object,
    crossfit: object,
) -> Mapping[str, object]:
    by_center = getattr(crossfit, "fixed_support_rows_by_center", None)
    if not isinstance(by_center, Mapping):
        raise ProtocolError("Case-OOF fixed-support rows are unavailable.")
    rows = [row for center in CENTERS for row in by_center[center]]
    embeddings = np.ascontiguousarray(
        getattr(frame, "embeddings_for")(rows), dtype=np.float32
    )
    offsets: dict[str, object] = {}
    cursor = 0
    for center in CENTERS:
        center_rows = tuple(by_center[center])
        stop = cursor + len(center_rows)
        offsets[center] = {
            "start": cursor,
            "stop": stop,
            "case_ids": [str(row.case_id) for row in center_rows],
            "sample_ids": [str(row.sample_id) for row in center_rows],
        }
        cursor = stop
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_support_scratch_v1",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "offsets": offsets,
        "array_sha256": _sha256_array(embeddings),
        "crossfit_fold_lock_hash": str(getattr(crossfit, "lock_hash", "")),
        "labels_used": False,
        "evaluation_embeddings_used": False,
    }
    payload = {**unhashed, "support_scratch_hash": stable_hash(unhashed)}
    atomic_save_npy(array_path, embeddings)
    atomic_write_json(index_path, payload)
    return payload


def build_source_tasks(
    config: SourceTaskConfig,
    generation_lock: GenerationLock,
    *,
    checkpoint_root: Path,
    support_array_path: Path,
    support_index_path: Path,
    support_scratch_hash: str,
) -> tuple[tuple[Mapping[str, object], ...], Mapping[tuple[str, int, int], object]]:
    keys = tuple(source_generation_plan(generation_lock))
    key_map = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in keys
    }
    if set(key_map) != set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS)):
        raise ProtocolError("Case-OOF GenerationLock source grid drifted.")
    tasks: list[Mapping[str, object]] = []
    for ordinal, (source, training_seed) in enumerate(
        product(CENTERS, TRAINING_SEEDS)
    ):
        device = GENERATION_DEVICES[ordinal % len(GENERATION_DEVICES)]
        tasks.append(
            {
                "task_ordinal": ordinal,
                "source_center": source,
                "training_seed": training_seed,
                "generation_keys": tuple(
                    key_map[(source, training_seed, seed)]
                    for seed in GENERATION_SEEDS
                ),
                "device": device,
                "expert_bank_root": str(config.expert_bank_root),
                "support_array_path": str(support_array_path),
                "support_index_path": str(support_index_path),
                "checkpoint_path": str(
                    checkpoint_root / f"source_{source}_train_{training_seed}.json"
                ),
                "array_path": str(
                    checkpoint_root / f"source_{source}_train_{training_seed}.npy"
                ),
                "config_contract_hash": config.contract_hash,
                "generation_lock_hash": generation_lock.generation_lock_hash,
                "support_scratch_hash": support_scratch_hash,
            }
        )
    if len(tasks) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("Case-OOF source task count drifted.")
    return tuple(tasks), key_map


def execute_pending_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Use one persistent spawned process per A5000 device."""

    if not tasks:
        return ()
    context = mp.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context)
        for _ in GENERATION_DEVICES
    ]
    future_to_task: dict[Future[dict[str, object]], Mapping[str, object]] = {}
    try:
        for task in tasks:
            device_index = GENERATION_DEVICES.index(str(task["device"]))
            future = executors[device_index].submit(generate_source_task, task)
            future_to_task[future] = task
        return tuple(future.result() for future in as_completed(future_to_task))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def source_task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


def _sha256_array(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "SourceTaskConfig",
    "build_source_tasks",
    "execute_pending_tasks",
    "source_task_key",
    "write_support_scratch",
)
