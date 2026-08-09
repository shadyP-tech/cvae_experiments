"""Label-free target prediction task planning and evaluation scratch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .actions import FrozenExactTailActionLibrary
from .artifact_io import atomic_json, atomic_npy, sha256_file
from .contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS, candidate_sources
from .input_contracts import FixedPartitionSurface, LabelFreeValidationFrame, row_identity_hash
from .source_cache_contracts import SourceCache
from .target_prediction_contracts import TARGET_CHECKPOINT_DIRECTORY


EXPECTED_TARGET_TASK_COUNT = len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
TARGET_EVALUATION_ARRAY_MEMBER = "checkpoints/target_evaluation_embeddings.npy"
TARGET_EVALUATION_INDEX_MEMBER = "checkpoints/target_evaluation_index.json"


def write_target_evaluation_scratch(
    root: Path,
    *,
    frame: LabelFreeValidationFrame,
    partitions: FixedPartitionSurface,
) -> Mapping[str, object]:
    array_path = root / TARGET_EVALUATION_ARRAY_MEMBER
    index_path = root / TARGET_EVALUATION_INDEX_MEMBER
    rows = [row for target in CENTERS for row in partitions.evaluation_rows_by_center[target]]
    embeddings = frame.embeddings_for(rows)
    offsets: dict[str, object] = {}
    cursor = 0
    for target in CENTERS:
        selected = partitions.evaluation_rows_by_center[target]
        stop = cursor + len(selected)
        offsets[target] = {
            "start": cursor,
            "stop": stop,
            "row_count": len(selected),
            "sample_ids": [row.sample_id for row in selected],
            "case_ids": sorted({row.case_id for row in selected}),
            "row_identity_hash": row_identity_hash(selected),
        }
        cursor = stop
    atomic_npy(array_path, embeddings)
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_target_evaluation_scratch_v1",
        "array_path": str(array_path.resolve()),
        "array_sha256": sha256_file(array_path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "support_partition_lock_hash": partitions.lock_hash,
        "targets": offsets,
        "labels_stored": False,
        "route_inputs_stored": False,
        "inference_only": True,
    }
    payload = {**unhashed, "scratch_hash": stable_hash(unhashed)}
    atomic_json(index_path, payload)
    return payload


def build_target_prediction_tasks(
    config: object,
    source_cache: SourceCache,
    library: FrozenExactTailActionLibrary,
    partitions: FixedPartitionSurface,
    *,
    source_cache_lock_hash: str,
    case_fold_lock_hash: str,
    scratch: Mapping[str, object],
    root: Path,
) -> tuple[dict[str, object], ...]:
    targets = scratch.get("targets")
    if not isinstance(targets, Mapping) or tuple(targets) != CENTERS:
        raise ProtocolError("Target evaluation scratch coverage drifted.")
    checkpoint_root = root / TARGET_CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, object]] = []
    source_rows = [record.to_row() for record in source_cache.source_records]
    for target in CENTERS:
        raw_offset = targets[target]
        if not isinstance(raw_offset, Mapping):
            raise ProtocolError("Target scratch offset is malformed.")
        expected_rows = partitions.evaluation_rows_by_center[target]
        if (
            raw_offset.get("row_identity_hash") != row_identity_hash(expected_rows)
            or raw_offset.get("sample_ids") != [row.sample_id for row in expected_rows]
        ):
            raise ProtocolError("Target scratch row identity drifted.")
        actions = library.actions_by_target[target]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_id = f"target_{target}_train{training_seed}_gen{generation_seed}"
                task: dict[str, object] = {
                    "schema_version": "midogpp_utility_aligned_stage90_target_task_v1",
                    "config_contract_hash": str(getattr(config, "contract_hash")),
                    "source_cache_lock_hash": source_cache_lock_hash,
                    "case_fold_lock_hash": case_fold_lock_hash,
                    "action_library_hash": library.action_library_hash,
                    "task_id": task_id,
                    "target_center": target,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "candidate_sources": list(candidate_sources(target)),
                    "actions": [action.to_payload() for action in actions],
                    "source_array_path": str(source_cache.source_array_path.resolve()),
                    "source_index_rows": source_rows,
                    "evaluation_array_path": str(scratch["array_path"]),
                    "evaluation_array_sha256": str(scratch["array_sha256"]),
                    "evaluation_offset": dict(raw_offset),
                    "evaluation_scratch_hash": str(scratch["scratch_hash"]),
                    "classifier": getattr(config, "classifier").to_payload(),
                    "threads_per_fit": 3,
                    "checkpoint_json_path": str(checkpoint_root / f"{task_id}.json"),
                    "checkpoint_npz_path": str(checkpoint_root / f"{task_id}.npz"),
                    "labels_available": False,
                    "target_support_labels_used": False,
                    "target_evaluation_used_for_route": False,
                    "seed_selection_performed": False,
                    "policy_authorized": False,
                }
                hash_payload = {
                    key: value
                    for key, value in task.items()
                    if key not in {"checkpoint_json_path", "checkpoint_npz_path"}
                }
                task["task_hash"] = stable_hash(hash_payload)
                tasks.append(task)
    if len(tasks) != EXPECTED_TARGET_TASK_COUNT:
        raise ProtocolError("Utility-aligned target task count drifted.")
    return tuple(tasks)


__all__ = (
    "EXPECTED_TARGET_TASK_COUNT",
    "TARGET_EVALUATION_ARRAY_MEMBER",
    "TARGET_EVALUATION_INDEX_MEMBER",
    "build_target_prediction_tasks",
    "write_target_evaluation_scratch",
)
