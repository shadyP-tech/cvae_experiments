"""Evaluation scratch, exact H/q/e planning, and spawned CPU scheduling."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from .actions import build_inner_exact_tail_action_library
from .contracts import CENTERS, inner_candidate_sources
from .development_prediction_contracts import (
    BLAS_THREADS_PER_WORKER,
    EXPECTED_COARSE_TASK_COUNT,
    PREDICTION_WORKERS,
    CoarseDevelopmentTask,
    PredictionCheckpointRecord,
    PredictionWorkerInput,
    SourceSlice,
    action_library_for,
    expected_coarse_task_keys,
)
from .development_prediction_worker import prediction_worker
from .input_contracts import row_identity_hash
from .source_cache_contracts import SOURCE_CACHE_LOCK_MEMBER, SourceCache
from .source_cache_store import (
    atomic_save_npy,
    atomic_write_json,
    read_json,
    sha256_array,
)


class PredictionPlanningConfig(Protocol):
    contract_hash: str
    classifier: ClassifierSpec
    runtime: Mapping[str, object]


def write_evaluation_scratch(
    root: Path, *, frame: object, partitions: object
) -> Mapping[str, object]:
    by_center = getattr(partitions, "evaluation_rows_by_center", None)
    if not isinstance(by_center, Mapping) or tuple(by_center) != CENTERS:
        raise ProtocolError("Stage-90 evaluation partitions are unavailable.")
    centers: dict[str, object] = {}
    for center in CENTERS:
        rows = tuple(by_center[center])
        if not rows or {str(row.partition_role) for row in rows} != {"evaluation"}:
            raise ProtocolError("Stage-90 evaluation scratch row role drifted.")
        values = np.ascontiguousarray(
            getattr(frame, "embeddings_for")(rows), dtype=np.float32
        )
        path = root / f"evaluation_center_{center}.npy"
        atomic_save_npy(path, values)
        centers[center] = {
            "path": str(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "array_sha256": sha256_array(values),
            "sample_ids": [str(row.sample_id) for row in rows],
            "case_ids": [str(row.case_id) for row in rows],
            "row_identity_hash": row_identity_hash(rows),
        }
    unhashed = {
        "schema_version": "midogpp_stage90_utility_aligned_evaluation_scratch_v1",
        "centers": centers,
        "partition_lock_hash": str(getattr(partitions, "lock_hash", "")),
        "labels_consumed": False,
    }
    payload = {**unhashed, "evaluation_scratch_hash": stable_hash(unhashed)}
    atomic_write_json(root / "evaluation_scratch_index.json", payload)
    return payload


def build_prediction_worker_inputs(
    config: PredictionPlanningConfig,
    source_cache: SourceCache,
    partitions: object,
    *,
    scratch: Mapping[str, object],
    checkpoint_root: Path,
    generation_lock_hash: str,
) -> tuple[PredictionWorkerInput, ...]:
    source_lock = read_json(source_cache.root / SOURCE_CACHE_LOCK_MEMBER)
    source_lock_hash = str(source_lock.get("source_cache_lock_hash", ""))
    if not source_lock_hash:
        raise ProtocolError("Stage-90 source-cache lock is unavailable.")
    centers = scratch.get("centers")
    by_eval = getattr(partitions, "evaluation_rows_by_center", None)
    by_support = getattr(partitions, "support_rows_by_center", None)
    if not isinstance(centers, Mapping) or not isinstance(by_eval, Mapping) or not isinstance(by_support, Mapping):
        raise ProtocolError("Stage-90 prediction planning inputs are malformed.")
    tasks: list[PredictionWorkerInput] = []
    canonical_library_hash = (
        build_inner_exact_tail_action_library().action_library_hash
    )
    for ordinal, (outer, query, training_seed, generation_seed) in enumerate(
        expected_coarse_task_keys()
    ):
        candidates = inner_candidate_sources(outer, query)
        actions = action_library_for(outer_target=outer, query_center=query)
        task_unhashed = {
            "schema_version": "midogpp_stage90_utility_aligned_coarse_task_v1",
            "task_ordinal": ordinal,
            "outer_target": outer,
            "query_center": query,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
            "candidate_sources": list(candidates),
            "action_ids": [action.action_id for action in actions],
            "canonical_action_hashes": [action.action_hash for action in actions],
            "canonical_inner_action_library_hash": canonical_library_hash,
            "strict_H_q_e_exclusion": True,
        }
        task = CoarseDevelopmentTask(
            task_ordinal=ordinal,
            outer_target=outer,
            query_center=query,
            training_seed=training_seed,
            generation_seed=generation_seed,
            candidate_sources=candidates,
            action_ids=tuple(action.action_id for action in actions),
            task_hash=stable_hash(task_unhashed),
        )
        source_slices = tuple(
            SourceSlice(
                source_center=source,
                block_ordinal=source_cache.source_by_key[
                    (source, training_seed, generation_seed)
                ].block_ordinal,
                stream_id=source_cache.source_by_key[
                    (source, training_seed, generation_seed)
                ].stream_id,
                expert_lock_hash=source_cache.source_by_key[
                    (source, training_seed, generation_seed)
                ].expert_lock_hash,
                output_sha256=source_cache.source_by_key[
                    (source, training_seed, generation_seed)
                ].output_sha256,
            )
            for source in candidates
        )
        raw_eval = centers.get(query)
        if not isinstance(raw_eval, Mapping):
            raise ProtocolError("Stage-90 query evaluation scratch is absent.")
        stem = (
            f"H{outer}_q{query}_train{training_seed}_gen{generation_seed}"
        )
        tasks.append(
            PredictionWorkerInput(
                task=task,
                source_array_path=str(source_cache.source_array_path),
                source_slices=source_slices,
                source_cache_lock_hash=source_lock_hash,
                evaluation_array_path=str(raw_eval["path"]),
                evaluation_array_sha256=str(raw_eval["array_sha256"]),
                evaluation_row_ids=tuple(str(row.sample_id) for row in by_eval[query]),
                evaluation_row_identity_hash=row_identity_hash(by_eval[query]),
                support_partition_hash=row_identity_hash(by_support[query]),
                partition_lock_hash=str(getattr(partitions, "lock_hash", "")),
                generation_lock_hash=str(generation_lock_hash),
                config_contract_hash=str(config.contract_hash),
                classifier_payload=config.classifier.to_payload(),
                checkpoint_json_path=str(checkpoint_root / f"{stem}.json"),
                checkpoint_npz_path=str(checkpoint_root / f"{stem}.npz"),
                threads_per_fit=BLAS_THREADS_PER_WORKER,
            )
        )
    if len(tasks) != EXPECTED_COARSE_TASK_COUNT:
        raise ProtocolError("Stage-90 coarse prediction task count drifted.")
    return tuple(tasks)


def execute_pending_prediction_tasks(
    tasks: Sequence[PredictionWorkerInput],
) -> tuple[PredictionCheckpointRecord, ...]:
    if not tasks:
        return ()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=PREDICTION_WORKERS, mp_context=context
    ) as executor:
        future_to_task: dict[
            Future[PredictionCheckpointRecord], PredictionWorkerInput
        ] = {executor.submit(prediction_worker, task): task for task in tasks}
        return tuple(
            future.result() for future in as_completed(future_to_task)
        )


def validate_runtime(config: PredictionPlanningConfig) -> None:
    workers = int(config.runtime.get("classifier_workers", PREDICTION_WORKERS))
    threads = int(
        config.runtime.get("classifier_threads_per_worker", BLAS_THREADS_PER_WORKER)
    )
    if workers != PREDICTION_WORKERS or threads != BLAS_THREADS_PER_WORKER:
        raise ProtocolError("Stage-90 CPU schedule must remain four workers by three BLAS threads.")


__all__ = (
    "PredictionPlanningConfig",
    "build_prediction_worker_inputs",
    "execute_pending_prediction_tasks",
    "validate_runtime",
    "write_evaluation_scratch",
)
