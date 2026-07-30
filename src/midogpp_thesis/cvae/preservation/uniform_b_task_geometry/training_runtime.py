"""Deterministic source-panel scheduling with one process per CUDA device."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ..independent_source import IndependentSourceData
from .checkpoint_store import TaskGeometryCheckpointStore
from .config import UniformBTaskGeometryConfig
from .contracts import ARMS
from .execution import RuntimePlan, partition_panel_tasks
from .task_geometry import TaskGeometryState, fit_task_geometry
from .training import train_source_panel


@dataclass(frozen=True)
class PanelTrainingTask:
    source: IndependentSourceData
    projected: np.ndarray
    frame_hash: str
    training_seed: int
    device: str
    config: UniformBTaskGeometryConfig
    artifact_root: Path

    @property
    def key(self) -> tuple[str, int]:
        return (self.source.center, self.training_seed)


@dataclass(frozen=True)
class PanelTrainingResult:
    source_center: str
    training_seed: int
    runtime_device: str
    geometry: TaskGeometryState
    checkpoint_records: tuple[dict[str, object], ...]
    geometry_rows: tuple[dict[str, object], ...]
    rng_rows: tuple[dict[str, object], ...]
    elapsed_seconds: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_center, self.training_seed)


def train_panel_grid(
    *,
    root: Path,
    config: UniformBTaskGeometryConfig,
    sources: Mapping[str, IndependentSourceData],
    projected: Mapping[str, np.ndarray],
    frame_hashes: Mapping[str, str],
    runtime_plan: RuntimePlan,
) -> tuple[PanelTrainingResult, ...]:
    """Train all source/seed tasks and return canonical task order."""

    keys = tuple(
        (center, seed)
        for center in config.heldout_centers
        for seed in config.training_seeds
    )
    partitions = partition_panel_tasks(keys, runtime_plan.training_devices)
    tasks = {
        key: PanelTrainingTask(
            source=sources[key[0]],
            projected=np.asarray(projected[key[0]], dtype=np.float32),
            frame_hash=str(frame_hashes[key[0]]),
            training_seed=int(key[1]),
            device=_device_for_key(key, partitions),
            config=config,
            artifact_root=Path(root),
        )
        for key in keys
    }
    if len(runtime_plan.training_devices) == 1:
        results = tuple(_train_panel_task(tasks[key]) for key in keys)
    else:
        results = _train_multi_device(
            tasks,
            keys=keys,
            partitions=partitions,
        )
    if tuple(result.key for result in results) != keys:
        raise ProtocolError("Panel scheduler changed canonical result order.")
    return results


def _train_multi_device(
    tasks: Mapping[tuple[str, int], PanelTrainingTask],
    *,
    keys: tuple[tuple[str, int], ...],
    partitions: Mapping[str, tuple[tuple[str, int], ...]],
) -> tuple[PanelTrainingResult, ...]:
    context = mp.get_context("spawn")
    executors: dict[str, ProcessPoolExecutor] = {}
    futures: dict[tuple[str, int], Future[PanelTrainingResult]] = {}
    try:
        for device, device_keys in partitions.items():
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
            )
            executors[device] = executor
            for key in device_keys:
                futures[key] = executor.submit(_train_panel_task, tasks[key])
        return tuple(futures[key].result() for key in keys)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)


def _train_panel_task(task: PanelTrainingTask) -> PanelTrainingResult:
    started = perf_counter()
    geometry = fit_task_geometry(
        task.projected,
        task.source.labels,
        task.source.case_ids,
        task.source.sample_ids,
        source_center=task.source.center,
        source_row_hash=task.source.row_hash,
        frame_hash=task.frame_hash,
        config=task.config,
        seed=task.training_seed,
    )
    panel = train_source_panel(
        task.projected,
        task.source.labels,
        task.source.case_ids,
        task.source.sample_ids,
        geometry=geometry,
        config=task.config,
        source_center=task.source.center,
        training_seed=task.training_seed,
        source_identity_hash=task.source.identity_hash,
        frame_hash=task.frame_hash,
        device=task.device,
    )
    store = TaskGeometryCheckpointStore(task.artifact_root, task.config)
    records = []
    rng_rows = []
    for arm in ARMS:
        runtime = panel.arms[arm]
        records.append(
            dict(
                store.save(
                    runtime,
                    source_center=task.source.center,
                    training_seed=task.training_seed,
                    frame_hash=task.frame_hash,
                    geometry_hash=geometry.state_hash,
                )
            )
        )
        rng_rows.append(
            {
                "schema_version": "midogpp_uniform_b_rng_pairing_v1",
                "source_center": task.source.center,
                "training_seed": task.training_seed,
                "arm": arm,
                "schedule_hash": panel.schedule_hash,
                "initialization_hash": panel.shared_initialization_hash,
                "warmup_state_hash": panel.warmup_state_hash,
                "task_branch_state_hash": (
                    panel.task_branch_state_hash
                    if arm in {"BG", "BM", "BT"}
                    else ""
                ),
                "final_stream_hash": runtime.final_stream_hash,
                "outer_or_inner_identity_present": False,
                "runtime_device": task.device,
            }
        )
    return PanelTrainingResult(
        source_center=task.source.center,
        training_seed=task.training_seed,
        runtime_device=task.device,
        geometry=geometry,
        checkpoint_records=tuple(records),
        geometry_rows=_geometry_rows(
            geometry,
            source_center=task.source.center,
            training_seed=task.training_seed,
        ),
        rng_rows=tuple(rng_rows),
        elapsed_seconds=perf_counter() - started,
    )


def _geometry_rows(
    geometry: TaskGeometryState,
    *,
    source_center: str,
    training_seed: int,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "schema_version": "midogpp_task_geometry_diagnostic_v1",
            "source_center": source_center,
            "training_seed": training_seed,
            "fold": fold.fold,
            "geometry_hash": geometry.state_hash,
            "fit_row_hash": fold.fit_row_hash,
            "reference_row_hash": fold.reference_row_hash,
            "reference_per_class": fold.reference_per_class,
            "hessian_min_eigenvalue": float(
                fold.hessian_eigenvalues.min()
            ),
            "hessian_max_eigenvalue": float(
                fold.hessian_eigenvalues.max()
            ),
            "hessian_condition_number": float(
                fold.hessian_eigenvalues.max()
                / fold.hessian_eigenvalues.min()
            ),
            "outer_or_inner_rows_used": False,
        }
        for fold in geometry.folds
    )


def _device_for_key(
    key: tuple[str, int],
    partitions: Mapping[str, tuple[tuple[str, int], ...]],
) -> str:
    matches = [
        device for device, tasks in partitions.items() if key in tasks
    ]
    if len(matches) != 1:
        raise ProtocolError("Panel task has no unique runtime device.")
    return matches[0]


__all__ = (
    "PanelTrainingResult",
    "PanelTrainingTask",
    "train_panel_grid",
)
