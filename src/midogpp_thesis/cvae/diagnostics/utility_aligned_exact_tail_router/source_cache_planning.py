"""Label-free scratch construction and persistent-GPU source planning."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import GenerationLock
from ...generation.generation import source_generation_plan
from ...protocol import ProtocolError
from .input_contracts import row_identity_hash
from .source_cache_contracts import (
    EXPECTED_SOURCE_TASK_COUNT,
    GENERATION_DEVICES,
)
from .source_cache_store import atomic_save_npy, atomic_write_json, sha256_array
from .source_cache_worker import generate_source_task


class SourceTaskConfig(Protocol):
    expert_bank_root: Path
    contract_hash: str


def write_support_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: object,
    partitions: object,
) -> Mapping[str, object]:
    by_center = getattr(partitions, "support_rows_by_center", None)
    if not isinstance(by_center, Mapping) or tuple(by_center) != CENTERS:
        raise ProtocolError("Stage-90 fixed support surface is unavailable.")
    rows = tuple(row for center in CENTERS for row in by_center[center])
    embeddings = np.ascontiguousarray(
        getattr(frame, "embeddings_for")(rows), dtype=np.float32
    )
    offsets: dict[str, object] = {}
    cursor = 0
    for center in CENTERS:
        center_rows = tuple(by_center[center])
        case_ids = tuple(sorted({str(row.case_id) for row in center_rows}))
        if len(case_ids) != 2:
            raise ProtocolError(
                "Consumed Stage-90 support must contain exactly two cases per center."
            )
        stop = cursor + len(center_rows)
        offsets[center] = {
            "start": cursor,
            "stop": stop,
            "case_ids_by_row": [str(row.case_id) for row in center_rows],
            "independent_case_ids": list(case_ids),
            "sample_ids": [str(row.sample_id) for row in center_rows],
            "support_partition_hash": row_identity_hash(center_rows),
        }
        cursor = stop
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_stage90_utility_aligned_support_scratch_v1",
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "offsets": offsets,
        "array_sha256": sha256_array(embeddings),
        "validation_cache_binding_hash": str(
            getattr(frame, "cache_binding_hash", "")
        ),
        "support_partition_lock_hash": str(getattr(partitions, "lock_hash", "")),
        "fixed_support_case_count_per_center": 2,
        "labels_consumed": False,
        "evaluation_embeddings_consumed": False,
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
    expected = set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    if set(key_map) != expected:
        raise ProtocolError("Stage-90 GenerationLock source grid drifted.")
    tasks: list[Mapping[str, object]] = []
    for ordinal, (source, training_seed) in enumerate(product(CENTERS, TRAINING_SEEDS)):
        stem = f"source_{source}_train_{training_seed}"
        tasks.append(
            {
                "schema_version": "midogpp_stage90_utility_aligned_source_task_v1",
                "task_ordinal": ordinal,
                "source_center": source,
                "training_seed": training_seed,
                "generation_keys": tuple(
                    key_map[(source, training_seed, seed)] for seed in GENERATION_SEEDS
                ),
                "device": GENERATION_DEVICES[ordinal % len(GENERATION_DEVICES)],
                "expert_bank_root": str(config.expert_bank_root),
                "support_array_path": str(support_array_path),
                "support_index_path": str(support_index_path),
                "checkpoint_path": str(checkpoint_root / f"{stem}.json"),
                "source_array_path": str(checkpoint_root / f"{stem}_streams.npy"),
                "component_array_path": str(
                    checkpoint_root / f"{stem}_components.npy"
                ),
                "config_contract_hash": str(config.contract_hash),
                "generation_lock_hash": generation_lock.generation_lock_hash,
                "support_scratch_hash": str(support_scratch_hash),
                "labels_available": False,
                "amp_enabled": False,
            }
        )
    if len(tasks) != EXPECTED_SOURCE_TASK_COUNT:
        raise ProtocolError("Stage-90 source task count drifted.")
    return tuple(tasks), key_map


def execute_pending_tasks(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Use exactly one spawned persistent process for each frozen GPU."""

    if not tasks:
        return ()
    context = mp.get_context("spawn")
    executors = [
        ProcessPoolExecutor(max_workers=1, mp_context=context)
        for _ in GENERATION_DEVICES
    ]
    futures: dict[Future[dict[str, object]], Mapping[str, object]] = {}
    try:
        for task in tasks:
            device_index = GENERATION_DEVICES.index(str(task["device"]))
            futures[executors[device_index].submit(generate_source_task, task)] = task
        return tuple(future.result() for future in as_completed(futures))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def source_task_key(task: Mapping[str, object]) -> tuple[str, int]:
    return str(task["source_center"]), int(task["training_seed"])


__all__ = (
    "SourceTaskConfig",
    "build_source_tasks",
    "execute_pending_tasks",
    "source_task_key",
    "write_support_scratch",
)
