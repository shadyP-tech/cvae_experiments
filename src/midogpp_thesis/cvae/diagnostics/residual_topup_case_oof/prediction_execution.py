"""Four-worker materialization of every case-fold/action/seed prediction."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Mapping

from ...protocol import ProtocolError
from .prediction_planning import build_prediction_tasks, write_evaluation_scratch
from .prediction_store import (
    EXPECTED_PREDICTION_CELL_COUNT,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PredictionStore,
    assemble_prediction_store,
    read_prediction_store,
    write_prediction_store,
)
from .prediction_validation import validate_prediction_store_binding
from .prediction_worker import (
    PREDICTION_CHECKPOINT_DIRECTORY,
    load_prediction_checkpoint,
    prediction_task,
)


CLASSIFIER_WORKERS = 4


def materialize_all_action_predictions(
    config: object,
    generation_lock_hash: str,
    source_cache: object,
    plan: object,
    frame: object,
    crossfit: object,
    *,
    source_cache_lock_hash: str,
    root: Path,
) -> PredictionStore:
    """Persist all 3,042 cells before the label capability can open."""

    final_array = root / PREDICTION_ARRAY_MEMBER
    final_index = root / PREDICTION_INDEX_MEMBER
    if final_array.is_file() and final_index.is_file():
        store = read_prediction_store(root)
        validate_prediction_store_binding(
            store,
            config=config,
            generation_lock_hash=generation_lock_hash,
            source_cache=source_cache,
            source_cache_lock_hash=source_cache_lock_hash,
            plan=plan,
            crossfit=crossfit,
        )
        shutil.rmtree(root / PREDICTION_CHECKPOINT_DIRECTORY, ignore_errors=True)
        return store

    checkpoint_root = root / PREDICTION_CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    scratch_path = checkpoint_root / "evaluation_embeddings.npy"
    scratch_index_path = checkpoint_root / "evaluation_index.json"
    scratch = write_evaluation_scratch(
        scratch_path,
        scratch_index_path,
        frame=frame,
        crossfit=crossfit,
    )
    tasks = build_prediction_tasks(
        config,
        generation_lock_hash,
        source_cache,
        plan,
        crossfit,
        source_cache_lock_hash=source_cache_lock_hash,
        scratch=scratch,
        scratch_path=scratch_path,
        checkpoint_root=checkpoint_root,
    )
    completed = _materialize_task_checkpoints(config, tasks)
    store = _assemble_store(tasks, completed)
    write_prediction_store(root, store)
    validate_prediction_store_binding(
        store,
        config=config,
        generation_lock_hash=generation_lock_hash,
        source_cache=source_cache,
        source_cache_lock_hash=source_cache_lock_hash,
        plan=plan,
        crossfit=crossfit,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return store


def _materialize_task_checkpoints(
    config: object,
    tasks: tuple[Mapping[str, object], ...],
) -> dict[str, Mapping[str, object]]:
    completed: dict[str, Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        json_path = Path(str(task["checkpoint_json_path"]))
        npz_path = Path(str(task["checkpoint_npz_path"]))
        if not json_path.is_file() or not npz_path.is_file():
            pending.append(task)
            continue
        completed[str(task["task_id"])] = load_prediction_checkpoint(
            json_path, npz_path, task=task
        )
    if pending:
        if int(getattr(config, "runtime")["classifier_workers"]) != CLASSIFIER_WORKERS:
            raise ProtocolError("Case-OOF requires four classifier workers.")
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=CLASSIFIER_WORKERS, mp_context=context
        ) as executor:
            future_to_task: dict[
                Future[Mapping[str, object]], Mapping[str, object]
            ] = {
                executor.submit(prediction_task, task): task for task in pending
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                future.result()
                completed[str(task["task_id"])] = load_prediction_checkpoint(
                    Path(str(task["checkpoint_json_path"])),
                    Path(str(task["checkpoint_npz_path"])),
                    task=task,
                )
                print(
                    f"[case-oof] classifier tasks {len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Case-OOF prediction checkpoints are incomplete.")
    return completed


def _assemble_store(
    tasks: tuple[Mapping[str, object], ...],
    completed: Mapping[str, Mapping[str, object]],
) -> PredictionStore:
    cells: list[Mapping[str, object]] = []
    unique_fit_count = 0
    for task in tasks:
        result = completed[str(task["task_id"])]
        cells.extend(tuple(result["cells"]))
        unique_fit_count += int(result["unique_classifier_fit_count"])
    if len(cells) != EXPECTED_PREDICTION_CELL_COUNT:
        raise ProtocolError("Case-OOF assembled prediction coverage drifted.")
    return assemble_prediction_store(
        cells, unique_classifier_fit_count=unique_fit_count
    )


__all__ = ("CLASSIFIER_WORKERS", "materialize_all_action_predictions")
