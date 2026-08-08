"""Spawned four-worker orchestration for exact-tail prediction execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .config import ExactTailUtilitySurfaceConfig, FreshInputAttestation
from .contracts import DevelopmentPartition, EXPECTED_COARSE_TASK_COUNT
from .prediction_checkpoint_store import load_checkpoint
from .prediction_consolidation import consolidate_prediction_records
from .prediction_contracts import (
    CoarsePredictionRecord,
    PredictionExecutionResult,
    PredictionWorkerInput,
)
from .prediction_cpu_worker import prediction_worker
from .prediction_planning import build_prediction_worker_inputs
from .prediction_seal import seal_consolidated_predictions
from .production_inputs import PreparedDevelopmentInputs
from .runtime import CLASSIFIER_WORKERS
from .source_contracts import GeneratedDevelopmentCache


def materialize_exact_tail_predictions(
    config: ExactTailUtilitySurfaceConfig,
    attestation: FreshInputAttestation,
    inputs: PreparedDevelopmentInputs,
    generated: GeneratedDevelopmentCache,
    *,
    root: Path,
    checkpoint_root: Path | None = None,
) -> PredictionExecutionResult:
    """Materialize, consolidate, and globally seal all 5,184 predictions.

    ``root`` remains the canonical arrays/index/seal root. ``checkpoint_root``
    may be placed outside it for resumable workstation scratch; omitting it
    preserves the historical ``root/checkpoints/predictions`` location.
    """

    resume_root = (
        root / "checkpoints/predictions"
        if checkpoint_root is None
        else checkpoint_root
    )
    worker_inputs = build_prediction_worker_inputs(
        config,
        inputs,
        generated,
        checkpoint_root=resume_root,
    )
    records = execute_or_resume(worker_inputs)
    if len(records) != EXPECTED_COARSE_TASK_COUNT:
        raise ProtocolError("Exact-tail coarse prediction execution is incomplete.")
    return consolidate_and_seal(
        config,
        attestation,
        inputs.reservation.partitions,
        records,
        root=root,
    )


def consolidate_and_seal(
    config: ExactTailUtilitySurfaceConfig,
    attestation: FreshInputAttestation,
    partitions: Mapping[str, DevelopmentPartition],
    records: Sequence[CoarsePredictionRecord],
    *,
    root: Path,
) -> PredictionExecutionResult:
    """Compatibility-preserving composition of consolidation and sealing."""

    consolidated = consolidate_prediction_records(
        config,
        partitions,
        records,
        root=root,
    )
    return seal_consolidated_predictions(
        config,
        attestation,
        partitions,
        records,
        consolidated,
        root=root,
    )


def execute_or_resume(
    inputs: Sequence[PredictionWorkerInput],
) -> tuple[CoarsePredictionRecord, ...]:
    completed: dict[tuple[str, str, int, int], CoarsePredictionRecord] = {}
    pending: list[PredictionWorkerInput] = []
    for item in inputs:
        existing = load_checkpoint(item)
        if existing is None:
            pending.append(item)
        else:
            completed[item.task.key] = existing
    if pending:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=CLASSIFIER_WORKERS, mp_context=context
        ) as pool:
            futures = {pool.submit(prediction_worker, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                if item.task.key in completed:
                    raise ProtocolError("Exact-tail prediction task completed twice.")
                completed[item.task.key] = future.result()
    expected = tuple(item.task.key for item in inputs)
    if set(completed) != set(expected):
        raise ProtocolError("Exact-tail checkpoint coverage drifted.")
    return tuple(completed[key] for key in expected)


__all__ = (
    "consolidate_and_seal",
    "execute_or_resume",
    "materialize_exact_tail_predictions",
)
