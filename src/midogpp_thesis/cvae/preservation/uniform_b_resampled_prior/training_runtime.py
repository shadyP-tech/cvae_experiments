"""Two-GPU fresh BG training with deterministic checkpoint resume."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from ...keyed_training import model_state_hash
from ...protocol import ProtocolError
from ..independent_source import IndependentSourceData
from .checkpoint_store import ResampledPriorCheckpointStore
from .config import UniformBResampledPriorConfig
from .contracts import SourceTrainingKey
from .execution import RuntimePlan, partition_panel_tasks
from .ratio import PosteriorRatioState, fit_posterior_ratio_state
from .training import train_fresh_bg_checkpoint


@dataclass(frozen=True)
class PanelTrainingTask:
    source: IndependentSourceData
    projected: np.ndarray
    training_key: SourceTrainingKey
    device: str
    config: UniformBResampledPriorConfig
    artifact_root: Path

    @property
    def key(self) -> tuple[str, int]:
        return (self.source.center, self.training_key.training_seed)


@dataclass(frozen=True)
class PanelTrainingResult:
    source_center: str
    training_seed: int
    runtime_device: str
    checkpoint_record: dict[str, object]
    ratio_state: PosteriorRatioState
    resumed_checkpoint: bool
    elapsed_seconds: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_center, self.training_seed)


def train_panel_grid(
    *,
    root: Path,
    config: UniformBResampledPriorConfig,
    sources: Mapping[str, IndependentSourceData],
    projected: Mapping[str, np.ndarray],
    training_keys: Mapping[tuple[str, int], SourceTrainingKey],
    runtime_plan: RuntimePlan,
) -> tuple[PanelTrainingResult, ...]:
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
            training_key=training_keys[key],
            device=_device_for_key(key, partitions),
            config=config,
            artifact_root=Path(root),
        )
        for key in keys
    }
    if len(runtime_plan.training_devices) == 1:
        results = tuple(_train_panel_task(tasks[key]) for key in keys)
    else:
        results = _train_multi_device(tasks, keys=keys, partitions=partitions)
    if tuple(item.key for item in results) != keys:
        raise ProtocolError("Fresh BG scheduler changed canonical task order.")
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
            executor = ProcessPoolExecutor(max_workers=1, mp_context=context)
            executors[device] = executor
            for key in device_keys:
                futures[key] = executor.submit(_train_panel_task, tasks[key])
        return tuple(futures[key].result() for key in keys)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)


def _train_panel_task(task: PanelTrainingTask) -> PanelTrainingResult:
    started = perf_counter()
    store = ResampledPriorCheckpointStore(task.artifact_root, task.config)
    state = store.load(task.training_key.hash, device=task.device)
    resumed = state is not None
    if state is None:
        runtime = train_fresh_bg_checkpoint(
            task.projected,
            task.source.labels,
            task.source.case_ids,
            task.source.sample_ids,
            config=task.config,
            training_key=task.training_key,
            source_identity_hash=task.source.identity_hash,
            device=task.device,
        )
        record = store.save(runtime)
        state = runtime.state
    else:
        record = dict(store.records[task.training_key.hash])
    checkpoint_hash = model_state_hash(state.model)
    ratio_state = fit_posterior_ratio_state(
        state.model,
        task.projected,
        task.source.labels,
        task.source.case_ids,
        source_center=task.source.center,
        training_seed=task.training_key.training_seed,
        checkpoint_hash=checkpoint_hash,
        source_row_hash=task.source.row_hash,
        source_case_hash=task.source.case_hash,
        config=task.config,
        device=task.device,
    )
    return PanelTrainingResult(
        source_center=task.source.center,
        training_seed=task.training_key.training_seed,
        runtime_device=task.device,
        checkpoint_record=record,
        ratio_state=ratio_state,
        resumed_checkpoint=resumed,
        elapsed_seconds=perf_counter() - started,
    )


def _device_for_key(
    key: tuple[str, int],
    partitions: Mapping[str, tuple[tuple[str, int], ...]],
) -> str:
    matches = [device for device, values in partitions.items() if key in values]
    if len(matches) != 1:
        raise ProtocolError("Fresh BG task has no unique runtime device.")
    return matches[0]


__all__ = ("PanelTrainingResult", "PanelTrainingTask", "train_panel_grid")
