"""Label-free planning for the 81 target/seed prediction tasks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json, json_ready, sha256_file
from .contracts import (
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    candidate_sources,
    expected_action_ids,
)


CLASSIFIER_THREADS_PER_WORKER = 3
EXPECTED_PREDICTION_TASK_COUNT = (
    len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
)


def write_evaluation_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: object,
    crossfit: object,
) -> Mapping[str, object]:
    """Persist evaluation embeddings for inference, never for route building."""

    folds = tuple(getattr(crossfit, "folds", ()))
    if len(folds) != EXPECTED_CASE_OOF_FOLD_COUNT:
        raise ProtocolError("Case-OOF evaluation scratch fold coverage drifted.")
    rows = [row for fold in folds for row in fold.heldout_rows]
    sample_ids = [str(row.sample_id) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ProtocolError("Case-OOF evaluation scratch duplicates samples.")
    embeddings = np.ascontiguousarray(
        getattr(frame, "embeddings_for")(rows), dtype=np.float32
    )
    fold_payloads: dict[str, object] = {}
    cursor = 0
    for fold in folds:
        fold_rows = tuple(fold.heldout_rows)
        stop = cursor + len(fold_rows)
        fold_payloads[fold.fold_id] = {
            "fold_ordinal": int(fold.fold_ordinal),
            "target_center": str(fold.target_center),
            "heldout_case_id": str(fold.heldout_case_id),
            "start": cursor,
            "stop": stop,
            "sample_ids": [str(row.sample_id) for row in fold_rows],
            "row_identity_hash": str(fold.heldout_row_identity_hash),
            "fold_hash": str(fold.fold_hash),
        }
        cursor = stop
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_evaluation_scratch_v1",
        "array_sha256": sha256_file(array_path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "crossfit_fold_lock_hash": str(getattr(crossfit, "lock_hash", "")),
        "folds": fold_payloads,
        "labels_persisted": False,
        "route_inputs_persisted": False,
        "inference_only": True,
    }
    payload = {**unhashed, "evaluation_scratch_hash": stable_hash(unhashed)}
    atomic_write_json(index_path, payload)
    return payload


def build_prediction_tasks(
    config: object,
    generation_lock_hash: str,
    source_cache: object,
    plan: object,
    crossfit: object,
    *,
    source_cache_lock_hash: str,
    scratch: Mapping[str, object],
    scratch_path: Path,
    checkpoint_root: Path,
) -> tuple[dict[str, object], ...]:
    """Build 9 targets x 9 seed cells; each fit is reused across case folds."""

    raw_folds = scratch.get("folds")
    if not isinstance(raw_folds, Mapping):
        raise ProtocolError("Case-OOF evaluation scratch lacks fold offsets.")
    source_rows = json_ready(tuple(getattr(source_cache, "index_rows")))
    tasks: list[dict[str, object]] = []
    folds_by_target = getattr(crossfit, "folds_by_target", {})
    for target in CENTERS:
        candidates = candidate_sources(target)
        actions = tuple(getattr(plan, "actions_for_target")(target))
        if (
            len(actions) != EXPECTED_ACTION_COUNT_PER_TARGET
            or tuple(str(action.action_id) for action in actions)
            != expected_action_ids(target)
            or tuple(str(action.target_center) for action in actions)
            != (target,) * len(actions)
        ):
            raise ProtocolError("Case-OOF task action library drifted.")
        action_payloads = tuple(json_ready(action.to_payload()) for action in actions)
        fold_payloads: list[object] = []
        for fold in folds_by_target[target]:
            scratch_fold = raw_folds.get(fold.fold_id)
            if not isinstance(scratch_fold, Mapping):
                raise ProtocolError("Case-OOF scratch lacks a planned fold.")
            fold_actions = tuple(getattr(plan, "actions_by_fold")[fold.fold_id])
            if fold_actions != actions:
                raise ProtocolError("Case-OOF fold actions are not target-frozen.")
            fold_payloads.append(dict(scratch_fold))
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_id = (
                    f"target_H{target}_train{training_seed}_gen{generation_seed}"
                )
                task: dict[str, object] = {
                    "schema_version": "midogpp_residual_topup_case_oof_prediction_task_v1",
                    "config_contract_hash": str(getattr(config, "contract_hash")),
                    "generation_lock_hash": generation_lock_hash,
                    "source_cache_lock_hash": source_cache_lock_hash,
                    "crossfit_fold_lock_hash": str(getattr(crossfit, "lock_hash")),
                    "router_plan_lock_hash": str(getattr(plan, "lock_hash")),
                    "task_id": task_id,
                    "target_center": target,
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "candidate_sources": list(candidates),
                    "actions": list(action_payloads),
                    "folds": fold_payloads,
                    "source_array_path": str(getattr(source_cache, "array_path")),
                    "source_index_rows": source_rows,
                    "evaluation_array_path": str(scratch_path),
                    "evaluation_scratch_hash": str(
                        scratch["evaluation_scratch_hash"]
                    ),
                    "classifier": getattr(config, "classifier").to_payload(),
                    "threads_per_fit": CLASSIFIER_THREADS_PER_WORKER,
                    "checkpoint_json_path": str(
                        checkpoint_root / f"{task_id}.json"
                    ),
                    "checkpoint_npz_path": str(
                        checkpoint_root / f"{task_id}.npz"
                    ),
                    "labels_available": False,
                    "other_evaluation_embeddings_used_for_route": False,
                    "policy_selection_performed": False,
                    "fallback_performed": False,
                }
                hash_payload = {
                    key: value
                    for key, value in task.items()
                    if key
                    not in {
                        "checkpoint_json_path",
                        "checkpoint_npz_path",
                        "task_hash",
                    }
                }
                task["task_hash"] = stable_hash(hash_payload)
                tasks.append(task)
    if len(tasks) != EXPECTED_PREDICTION_TASK_COUNT:
        raise ProtocolError("Case-OOF prediction task count drifted.")
    return tuple(tasks)


__all__ = (
    "CLASSIFIER_THREADS_PER_WORKER",
    "EXPECTED_PREDICTION_TASK_COUNT",
    "build_prediction_tasks",
    "write_evaluation_scratch",
)
