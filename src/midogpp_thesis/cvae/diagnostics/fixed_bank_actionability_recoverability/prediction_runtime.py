"""Thin orchestration facade for the fixed A0/A1 prediction runtime.

The neutral direct-target runtime remains untouched.  This package-owned
facade coordinates label-free target scratch, CPU worker checkpoints, and the
sealed prediction store for this one terminal diagnostic.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import FrozenSourceStreamCache
from .prediction_contracts import (
    CHECKPOINT_DIRECTORY,
    EXPECTED_CELL_COUNT,
    EXPECTED_TASK_COUNT,
    GLOBAL_PREDICTION_SEAL_MEMBER,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    ActionPredictionConfig,
    ActionPredictionStore,
    ActionSpecLike,
    GlobalActionPredictionSeal,
    PredictionCell,
    hash_like as _hash_like,
    prediction_store_hash as _store_hash,
    validate_action_library,
)
from .prediction_store import (
    cell_from_index as _cell_from_index,
    load_global_action_prediction_seal,
    write_final_store as _write_final_store,
    write_global_prediction_seal,
)
from .prediction_tasks import (
    build_tasks as _build_tasks,
    cells_from_checkpoints as _cells_from_checkpoints,
    classifier_from_payload as _classifier_from_payload,
    execute_or_resume as _execute_or_resume,
    load_checkpoint as _load_checkpoint,
    row_id as _row_id,
    target_cache_binding_hash as _target_cache_binding_hash,
)
from .prediction_tasks import compose_action as _compose_action_impl
from .prediction_tasks import load_task_arrays as _load_task_arrays_impl
from .prediction_tasks import prediction_task as _prediction_task_impl
from .prediction_tasks import validate_target_scratch as _validate_scratch_impl
from .prediction_tasks import write_target_scratch as _write_scratch_impl


def materialize_action_predictions(
    config: ActionPredictionConfig,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    *,
    partition_hash: str,
    action_library: Mapping[str, Sequence[ActionSpecLike]],
    root: Path,
) -> GlobalActionPredictionSeal:
    """Fit and seal B, U, eight A0 and eight weighted A1 actions per target."""

    _assert_runtime(config.runtime)
    library_payload, library_hash = _validate_action_library(action_library)
    final_array = root / PREDICTION_ARRAY_MEMBER
    final_index = root / PREDICTION_INDEX_MEMBER
    final_seal = root / GLOBAL_PREDICTION_SEAL_MEMBER
    target_binding = _target_cache_binding_hash(frame)
    if final_array.is_file() and final_index.is_file() and final_seal.is_file():
        result = load_global_action_prediction_seal(
            root,
            expected_config_hash=config.contract_hash,
            expected_source_lock_hash=source_cache.lock_hash,
            expected_partition_hash=partition_hash,
            expected_action_library_hash=library_hash,
            expected_target_cache_binding_hash=target_binding,
        )
        shutil.rmtree(root / CHECKPOINT_DIRECTORY, ignore_errors=True)
        return result

    scratch = _write_target_scratch(
        root,
        frame=frame,
        partition_hash=partition_hash,
        target_cache_binding_hash=target_binding,
    )
    tasks = _build_tasks(
        config,
        source_cache,
        scratch=scratch,
        library_payload=library_payload,
        action_library_hash=library_hash,
        partition_hash=partition_hash,
        root=root,
    )
    completed = _execute_or_resume(tasks, workers=4)
    cells = _cells_from_checkpoints(tasks, completed)
    rows = {
        center: tuple(str(value) for value in scratch["row_ids_by_center"][center])
        for center in CENTERS
    }
    cases = {
        center: tuple(str(value) for value in scratch["case_ids_by_center"][center])
        for center in CENTERS
    }
    store_hash = _store_hash(
        cells,
        rows_by_center=rows,
        case_ids_by_center=cases,
        source_stream_lock_hash=source_cache.lock_hash,
        action_library_hash=library_hash,
        target_cache_binding_hash=target_binding,
    )
    _write_final_store(
        final_array,
        final_index,
        cells=cells,
        rows_by_center=rows,
        case_ids_by_center=cases,
        config_contract_hash=config.contract_hash,
        partition_hash=partition_hash,
        source_stream_lock_hash=source_cache.lock_hash,
        action_library_hash=library_hash,
        target_cache_binding_hash=target_binding,
        store_hash=store_hash,
    )
    write_global_prediction_seal(
        final_seal,
        arrays_path=final_array,
        index_path=final_index,
        config_contract_hash=config.contract_hash,
        partition_hash=partition_hash,
        prediction_store_hash=store_hash,
        source_stream_lock_hash=source_cache.lock_hash,
        action_library_hash=library_hash,
        target_cache_binding_hash=target_binding,
    )
    result = load_global_action_prediction_seal(
        root,
        expected_config_hash=config.contract_hash,
        expected_source_lock_hash=source_cache.lock_hash,
        expected_partition_hash=partition_hash,
        expected_action_library_hash=library_hash,
        expected_target_cache_binding_hash=target_binding,
    )
    shutil.rmtree(root / CHECKPOINT_DIRECTORY, ignore_errors=True)
    return result


def _assert_runtime(runtime: Mapping[str, object]) -> None:
    if (
        int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("scientific_reductions_dtype") != "float64"
        or int(runtime.get("target_task_count", -1)) != EXPECTED_TASK_COUNT
        or int(runtime.get("target_probability_cell_count", -1))
        != EXPECTED_CELL_COUNT
        or int(runtime.get("maximum_total_classifier_fit_count", -1))
        != EXPECTED_CELL_COUNT
    ):
        raise ProtocolError(
            "Actionability prediction execution requires spawn and four 3-thread workers."
        )


def _validate_action_library(
    action_library: Mapping[str, Sequence[ActionSpecLike]],
) -> tuple[dict[str, list[dict[str, object]]], str]:
    return validate_action_library(action_library)


def _write_target_scratch(
    root: Path,
    *,
    frame: object,
    partition_hash: str,
    target_cache_binding_hash: str,
) -> Mapping[str, object]:
    return _write_scratch_impl(
        root,
        frame=frame,
        partition_hash=partition_hash,
        target_cache_binding_hash_value=target_cache_binding_hash,
        output_dim=COMMON_OUTPUT_DIM,
    )


def _validate_target_scratch(
    payload: Mapping[str, object],
    *,
    expected_partition_hash: str,
    expected_target_cache_binding_hash: str,
) -> None:
    _validate_scratch_impl(
        payload,
        expected_partition_hash=expected_partition_hash,
        expected_target_cache_binding_hash=expected_target_cache_binding_hash,
        output_dim=COMMON_OUTPUT_DIM,
    )


def _load_task_arrays(
    task: Mapping[str, object], *, candidates: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    return _load_task_arrays_impl(
        task,
        candidates=candidates,
        output_dim=COMMON_OUTPUT_DIM,
    )


def _prediction_task(task: Mapping[str, object]) -> None:
    _prediction_task_impl(task, output_dim=COMMON_OUTPUT_DIM)


def _compose_action(
    blocks: Mapping[str, np.ndarray],
    action: Mapping[str, object],
    candidates: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    return _compose_action_impl(
        blocks,
        action,
        candidates,
        output_dim=COMMON_OUTPUT_DIM,
    )


__all__ = (
    "ActionPredictionStore",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_TASK_COUNT",
    "GlobalActionPredictionSeal",
    "PredictionCell",
    "load_global_action_prediction_seal",
    "materialize_action_predictions",
)
