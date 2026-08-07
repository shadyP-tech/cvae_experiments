"""Scheduler and compatibility facade for case-cross-fit predictions."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ...protocol import ProtocolError
from ..mmd_kmm_router.inputs import LabelFreeValidationFrame
from ..mmd_kmm_router.source_products import SourceProducts
from ._prediction_common import (
    atomic_save_npy as _atomic_save_npy,
    atomic_save_npz as _atomic_save_npz,
    compact_json as _compact,
    integer as _integer,
    is_hash as _is_hash,
    parse_json_value as _json_value,
    require_mapping as _mapping,
    sha256_array as _sha256_array,
    truthy as _truthy,
)
from .composition import (
    ClassSpecificComposition as _ClassSpecificComposition,
    arm_plan_payload as _arm_plan_payload,
    compose_class_specific_prefix_blocks as _compose_class_specific_prefix_blocks,
    fit_classifier as _fit_classifier,
    fit_metadata as _fit_metadata,
    normalize_allocations_by_class as _normalize_allocations_by_class,
    source_float_mapping as _source_float_mapping,
    validate_plan as _validate_plan,
)
from .contracts import ARM_ROLES, GENERATION_SEEDS, TRAINING_SEEDS
from .partitions import CrossfitSurface
from .prediction_store import (
    CROSSFIT_PREDICTION_ARRAY_MEMBER,
    CROSSFIT_PREDICTION_INDEX_COLUMNS,
    CROSSFIT_PREDICTION_INDEX_MEMBER,
    CrossfitPredictionStore,
    assemble_crossfit_prediction_store,
    plan_surface as _plan_surface,
    read_crossfit_prediction_store,
    validate_crossfit_prediction_store_binding,
    write_crossfit_prediction_store,
)
from .prediction_worker import (
    PREDICTION_CHECKPOINT_DIRECTORY,
    build_prediction_tasks,
    load_prediction_checkpoint as _load_prediction_checkpoint,
    prediction_task,
    task_key as _task_key,
    write_heldout_scratch as _write_heldout_scratch,
    write_prediction_checkpoint as _write_prediction_checkpoint,
)

if TYPE_CHECKING:  # pragma: no cover
    from .config import AntisymmetricResidualMMDDiagnosticConfig


def materialize_case_crossfit_predictions(
    config: "AntisymmetricResidualMMDDiagnosticConfig",
    generation_lock_hash: str,
    source_products: SourceProducts,
    plans: object,
    frame: LabelFreeValidationFrame,
    crossfit: CrossfitSurface,
    *,
    source_products_lock_hash: str,
    root: Path,
) -> CrossfitPredictionStore:
    """Fit both arms for every fold and retained seed cell, with resume."""

    final_array = root / CROSSFIT_PREDICTION_ARRAY_MEMBER
    final_index = root / CROSSFIT_PREDICTION_INDEX_MEMBER
    if final_array.is_file() and final_index.is_file():
        store = read_crossfit_prediction_store(final_array, final_index)
        validate_crossfit_prediction_store_binding(
            store,
            config=config,
            generation_lock_hash=generation_lock_hash,
            source_products_lock_hash=source_products_lock_hash,
            plans=plans,
            crossfit=crossfit,
        )
        shutil.rmtree(root / PREDICTION_CHECKPOINT_DIRECTORY, ignore_errors=True)
        return store

    plan_map, plan_lock_hash = _plan_surface(plans)
    checkpoint_root = root / PREDICTION_CHECKPOINT_DIRECTORY
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    scratch_path = checkpoint_root / "heldout_embeddings.npy"
    scratch_index_path = checkpoint_root / "heldout_index.json"
    scratch = _write_heldout_scratch(
        scratch_path,
        scratch_index_path,
        frame=frame,
        crossfit=crossfit,
    )
    tasks = build_prediction_tasks(
        config,
        generation_lock_hash,
        source_products,
        plan_map,
        plan_lock_hash,
        crossfit,
        source_products_lock_hash=source_products_lock_hash,
        scratch=scratch,
        scratch_path=scratch_path,
        checkpoint_root=checkpoint_root,
    )

    completed: dict[tuple[str, int, int], Mapping[str, object]] = {}
    pending: list[Mapping[str, object]] = []
    for task in tasks:
        checkpoint = Path(str(task["checkpoint_path"]))
        if not checkpoint.is_file():
            pending.append(task)
            continue
        completed[_task_key(task)] = _load_prediction_checkpoint(
            checkpoint, task=task
        )

    if pending:
        context = mp.get_context("spawn")
        worker_count = int(config.runtime["classifier_workers"])
        if worker_count != 4:
            raise ProtocolError(
                "Antisymmetric cross-fit requires exactly four workers."
            )
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
            future_to_task: dict[
                Future[dict[str, object]], Mapping[str, object]
            ] = {
                executor.submit(_prediction_task, task): task for task in pending
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                checkpoint = Path(str(task["checkpoint_path"]))
                _write_prediction_checkpoint(checkpoint, result)
                completed[_task_key(task)] = _load_prediction_checkpoint(
                    checkpoint, task=task
                )
                print(
                    "[antisymmetric-mmd] classifier tasks "
                    f"{len(completed)}/{len(tasks)}",
                    flush=True,
                )
    if len(completed) != len(tasks):
        raise ProtocolError("Antisymmetric cross-fit checkpoints are incomplete.")

    store = assemble_crossfit_prediction_store(completed, crossfit)
    write_crossfit_prediction_store(final_array, final_index, store)
    validate_crossfit_prediction_store_binding(
        store,
        config=config,
        generation_lock_hash=generation_lock_hash,
        source_products_lock_hash=source_products_lock_hash,
        plans=plans,
        crossfit=crossfit,
    )
    shutil.rmtree(checkpoint_root, ignore_errors=True)
    return store


def _prediction_task(task: Mapping[str, object]) -> dict[str, object]:
    """Compatibility wrapper that remains spawn-picklable and patchable."""

    return prediction_task(
        task,
        fit_classifier_fn=_fit_classifier,
        compose_blocks_fn=_compose_class_specific_prefix_blocks,
    )


def _write_prediction_store(
    path: Path, store: CrossfitPredictionStore
) -> None:
    """Compatibility helper for callers of the former monolithic module."""

    _atomic_save_npz(
        path,
        {
            "y_pred": store.y_pred,
            "prob_pos": store.prob_pos,
            "unique_classifier_fit_count": np.asarray(
                store.unique_classifier_fit_count, dtype=np.int64
            ),
        },
    )


__all__ = (
    "CROSSFIT_PREDICTION_ARRAY_MEMBER",
    "CROSSFIT_PREDICTION_INDEX_COLUMNS",
    "CROSSFIT_PREDICTION_INDEX_MEMBER",
    "CrossfitPredictionStore",
    "materialize_case_crossfit_predictions",
    "read_crossfit_prediction_store",
    "validate_crossfit_prediction_store_binding",
)
