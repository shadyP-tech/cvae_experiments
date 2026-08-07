"""Label-free task planning and evaluation-scratch construction.

This module is deliberately limited to material that is available before the
global prediction seal: immutable plans, source-cache identities, evaluation
embeddings, and row identities.  It never opens the label-bearing manifest.
"""

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
    CLASSIFIER_THREADS_PER_WORKER,
    DEVELOPMENT_ACTION_IDS,
    EXPECTED_DEVELOPMENT_TASK_COUNT,
    EXPECTED_TARGET_TASK_COUNT,
    GENERATION_SEEDS,
    TARGET_ACTION_IDS,
    TRAINING_SEEDS,
    development_queries,
)


def build_prediction_tasks(
    config: object,
    generation_lock_hash: str,
    source_cache: object,
    plans: object,
    partitions: object,
    *,
    source_cache_lock_hash: str,
    scratch: Mapping[str, object],
    scratch_path: Path,
    checkpoint_root: Path,
) -> tuple[dict[str, object], ...]:
    """Build the complete label-free 648-development plus 81-target task grid."""

    tasks: list[dict[str, object]] = []
    source_rows = json_ready(tuple(getattr(source_cache, "index_rows")))
    offsets = scratch["centers"]
    for outer in CENTERS:
        for query in development_queries(outer):
            candidates = tuple(center for center in CENTERS if center not in {outer, query})
            evaluation = offsets[query]
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    task_id = f"development_H{outer}_q{query}_train{training_seed}_gen{generation_seed}"
                    tasks.append(
                        _task_payload(
                            config=config,
                            generation_lock_hash=generation_lock_hash,
                            source_cache=source_cache,
                            source_rows=source_rows,
                            source_cache_lock_hash=source_cache_lock_hash,
                            plans=tuple(
                                json_ready(plans.plan(
                                    phase="development",
                                    outer_target=outer,
                                    query_center=query,
                                    action_id=action,
                                ))
                                for action in DEVELOPMENT_ACTION_IDS
                            ),
                            phase="development",
                            outer=outer,
                            query=query,
                            candidates=candidates,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            evaluation=evaluation,
                            scratch=scratch,
                            scratch_path=scratch_path,
                            task_id=task_id,
                            checkpoint_root=checkpoint_root,
                        )
                    )
    if len(tasks) != EXPECTED_DEVELOPMENT_TASK_COUNT:
        raise ProtocolError("Residual top-up development task count drifted.")
    for target in CENTERS:
        candidates = tuple(center for center in CENTERS if center != target)
        evaluation = offsets[target]
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                task_id = f"target_H{target}_train{training_seed}_gen{generation_seed}"
                tasks.append(
                    _task_payload(
                        config=config,
                        generation_lock_hash=generation_lock_hash,
                        source_cache=source_cache,
                        source_rows=source_rows,
                        source_cache_lock_hash=source_cache_lock_hash,
                        plans=tuple(
                            json_ready(plans.plan(
                                phase="target",
                                outer_target=target,
                                query_center=target,
                                action_id=action,
                            ))
                            for action in TARGET_ACTION_IDS
                        ),
                        phase="target",
                        outer=target,
                        query=target,
                        candidates=candidates,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        evaluation=evaluation,
                        scratch=scratch,
                        scratch_path=scratch_path,
                        task_id=task_id,
                        checkpoint_root=checkpoint_root,
                    )
                )
    if len(tasks) != EXPECTED_DEVELOPMENT_TASK_COUNT + EXPECTED_TARGET_TASK_COUNT:
        raise ProtocolError("Residual top-up total task count drifted.")
    return bind_task_plan_lock(tuple(tasks), str(getattr(plans, "lock_hash")))


def _task_payload(
    *,
    config: object,
    generation_lock_hash: str,
    source_cache: object,
    source_rows: object,
    source_cache_lock_hash: str,
    plans: tuple[object, ...],
    phase: str,
    outer: str,
    query: str,
    candidates: tuple[str, ...],
    training_seed: int,
    generation_seed: int,
    evaluation: Mapping[str, object],
    scratch: Mapping[str, object],
    scratch_path: Path,
    task_id: str,
    checkpoint_root: Path,
) -> dict[str, object]:
    task = {
        "schema_version": "midogpp_residual_topup_prediction_task_v1",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "generation_lock_hash": generation_lock_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "router_plan_lock_hash": "__SET_BY_CALLER__",
        "task_id": task_id,
        "phase": phase,
        "outer_target": outer,
        "query_center": query,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "candidate_sources": list(candidates),
        "plans": list(plans),
        "source_array_path": str(getattr(source_cache, "array_path")),
        "source_index_rows": source_rows,
        "evaluation_array_path": str(scratch_path),
        "evaluation_scratch_hash": str(scratch["evaluation_scratch_hash"]),
        "evaluation_start": int(evaluation["start"]),
        "evaluation_stop": int(evaluation["stop"]),
        "evaluation_row_ids": list(evaluation["sample_ids"]),
        "evaluation_row_identity_hash": str(evaluation["row_identity_hash"]),
        "classifier": getattr(config, "classifier").to_payload(),
        "threads_per_fit": CLASSIFIER_THREADS_PER_WORKER,
        "plan_hashes": [str(plan["plan_hash"]) for plan in plans],
        "checkpoint_json_path": str(checkpoint_root / f"{task_id}.json"),
        "checkpoint_npz_path": str(checkpoint_root / f"{task_id}.npz"),
    }
    return task


def write_evaluation_scratch(
    array_path: Path,
    index_path: Path,
    *,
    frame: object,
    partitions: object,
) -> Mapping[str, object]:
    """Persist evaluation embeddings and identities without exposing labels."""

    rows_by_center = getattr(partitions, "evaluation_rows_by_center")
    rows = [row for center in CENTERS for row in rows_by_center[center]]
    embeddings = getattr(frame, "embeddings_for")(rows)
    centers: dict[str, object] = {}
    cursor = 0
    for center in CENTERS:
        center_rows = tuple(rows_by_center[center])
        stop = cursor + len(center_rows)
        centers[center] = {
            "start": cursor,
            "stop": stop,
            "sample_ids": [row.sample_id for row in center_rows],
            "row_identity_hash": stable_hash([row.identity_payload() for row in center_rows]),
        }
        cursor = stop
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    unhashed = {
        "schema_version": "midogpp_residual_topup_evaluation_scratch_v1",
        "array_sha256": sha256_file(array_path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "centers": centers,
    }
    payload = {**unhashed, "evaluation_scratch_hash": stable_hash(unhashed)}
    atomic_write_json(index_path, payload)
    return payload


def bind_task_plan_lock(
    tasks: tuple[dict[str, object], ...], plan_lock_hash: str
) -> tuple[dict[str, object], ...]:
    """Finalize the shared plan lock and deterministic task hash."""

    output = []
    for raw in tasks:
        task = dict(raw)
        task["router_plan_lock_hash"] = plan_lock_hash
        hash_payload = {
            key: value
            for key, value in task.items()
            if key not in {"checkpoint_json_path", "checkpoint_npz_path", "task_hash"}
        }
        task["task_hash"] = stable_hash(hash_payload)
        output.append(task)
    return tuple(output)


__all__ = (
    "bind_task_plan_lock",
    "build_prediction_tasks",
    "write_evaluation_scratch",
)
