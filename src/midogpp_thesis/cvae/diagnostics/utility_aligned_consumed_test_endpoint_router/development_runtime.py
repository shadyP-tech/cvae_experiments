"""Four-by-three CPU runtime for globally sealed development predictions."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Sequence

from ...protocol import ProtocolError
from .checkpoint_store import PredictionCheckpoint, load_task_checkpoint
from .prediction_contracts import (
    DEVELOPMENT_ARRAY_MEMBER, DEVELOPMENT_INDEX_MEMBER, DEVELOPMENT_ROLE,
    DEVELOPMENT_SEAL_MEMBER, PredictionStore, PredictionTask,
)
from .prediction_planning import PredictionPlan
from .prediction_store import load_prediction_store, materialize_prediction_store
from .prediction_workers import execute_prediction_task
from .seals import DevelopmentPredictionCapability, seal_development_predictions


DEVELOPMENT_CHECKPOINT_DIRECTORY = "checkpoints/development_predictions"


def materialize_development_predictions(
    plan: PredictionPlan,
    *,
    root: Path,
    workers: int = 4,
) -> DevelopmentPredictionCapability:
    if plan.phase != DEVELOPMENT_ROLE or workers != 4:
        raise ProtocolError("Endpoint-router development runtime topology drifted.")
    resumed = load_development_prediction_capability(plan, root=root)
    if resumed is not None:
        return resumed
    checkpoints = execute_prediction_plan(plan.tasks, workers=workers)
    store = materialize_prediction_store(plan, checkpoints, root=root)
    capability = seal_development_predictions(store, root=root)
    shutil.rmtree(root / DEVELOPMENT_CHECKPOINT_DIRECTORY, ignore_errors=True)
    return capability


def load_development_prediction_capability(
    plan: PredictionPlan, *, root: Path
) -> DevelopmentPredictionCapability | None:
    """Reconstruct a completed phase before considering any classifier work."""

    if plan.phase != DEVELOPMENT_ROLE:
        raise ProtocolError("Development resume received another prediction phase.")
    members = tuple(
        root / member
        for member in (
            DEVELOPMENT_ARRAY_MEMBER,
            DEVELOPMENT_INDEX_MEMBER,
            DEVELOPMENT_SEAL_MEMBER,
        )
    )
    present = tuple(path.is_file() for path in members)
    if not any(present):
        return None
    if not all(present):
        raise ProtocolError("Endpoint-router development final surface is incomplete.")
    store = load_prediction_store(root, phase=DEVELOPMENT_ROLE, expected_plan=plan)
    # Rebuild the expected seal from the reconstructed store.  The seal helper
    # validates an existing member byte-for-byte, including current array and
    # index digests, instead of trusting a self-consistent but rewritten seal.
    return seal_development_predictions(store, root=root)


def execute_prediction_plan(
    tasks: Sequence[PredictionTask], *, workers: int = 4
) -> tuple[PredictionCheckpoint, ...]:
    """Run pending tasks through spawn and revalidate every returned checkpoint."""

    values = tuple(tasks)
    if workers != 4 or not values:
        raise ProtocolError("Endpoint-router classifier pool must contain four workers.")
    completed: dict[str, PredictionCheckpoint] = {}
    pending: list[PredictionTask] = []
    for task in values:
        checkpoint = load_task_checkpoint(task)
        if checkpoint is None:
            pending.append(task)
        else:
            completed[task.task_hash] = checkpoint
    if pending:
        context = mp.get_context("spawn")
        futures: dict[Future[PredictionCheckpoint], PredictionTask] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            for task in pending:
                futures[executor.submit(execute_prediction_task, task)] = task
            for future in as_completed(futures):
                task = futures[future]
                returned = future.result()
                verified = load_task_checkpoint(task)
                if (
                    verified is None
                    or returned.checkpoint_hash != verified.checkpoint_hash
                ):
                    raise ProtocolError(
                        "Endpoint-router worker checkpoint return drifted."
                    )
                completed[task.task_hash] = verified
                print(
                    f"[endpoint-router:{task.phase}] tasks "
                    f"{len(completed)}/{len(values)}",
                    flush=True,
                )
    if set(completed) != {task.task_hash for task in values}:
        raise ProtocolError("Endpoint-router prediction task coverage is incomplete.")
    return tuple(completed[task.task_hash] for task in values)


__all__ = (
    "DEVELOPMENT_CHECKPOINT_DIRECTORY",
    "execute_prediction_plan",
    "load_development_prediction_capability",
    "materialize_development_predictions",
)
