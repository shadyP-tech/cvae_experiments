"""CPU runtime for the 810 physical target-action prediction cells."""

from __future__ import annotations

from pathlib import Path
import shutil

from ...protocol import ProtocolError
from .development_runtime import execute_prediction_plan
from .prediction_contracts import (
    TARGET_ARRAY_MEMBER, TARGET_INDEX_MEMBER, TARGET_ROLE, TARGET_SEAL_MEMBER,
    PredictionStore,
)
from .prediction_planning import PredictionPlan
from .prediction_store import load_prediction_store, materialize_prediction_store


TARGET_CHECKPOINT_DIRECTORY = "checkpoints/target_predictions"


def materialize_target_predictions(
    plan: PredictionPlan,
    *,
    root: Path,
    workers: int = 4,
) -> PredictionStore:
    """Seal physical probabilities in storage; policy seal follows separately."""

    if plan.phase != TARGET_ROLE or workers != 4:
        raise ProtocolError("Endpoint-router target runtime topology drifted.")
    resumed = load_target_prediction_store(plan, root=root)
    if resumed is not None:
        return resumed
    checkpoints = execute_prediction_plan(plan.tasks, workers=workers)
    store = materialize_prediction_store(plan, checkpoints, root=root)
    # The store is now durable and hash-validated; logical G/R/P aliases and
    # all nine target plans must still freeze before a scoring capability exists.
    shutil.rmtree(root / TARGET_CHECKPOINT_DIRECTORY, ignore_errors=True)
    return store


def load_target_prediction_store(
    plan: PredictionPlan, *, root: Path
) -> PredictionStore | None:
    """Load both final target members before considering classifier work."""

    if plan.phase != TARGET_ROLE:
        raise ProtocolError("Target resume received another prediction phase.")
    members = (root / TARGET_ARRAY_MEMBER, root / TARGET_INDEX_MEMBER)
    seal_path = root / TARGET_SEAL_MEMBER
    present = tuple(path.is_file() for path in members)
    if seal_path.exists() and not all(present):
        raise ProtocolError(
            "Endpoint-router target seal exists without its complete store."
        )
    if not any(present):
        return None
    if not all(present):
        raise ProtocolError("Endpoint-router target final surface is incomplete.")
    return load_prediction_store(root, phase=TARGET_ROLE, expected_plan=plan)


__all__ = (
    "TARGET_CHECKPOINT_DIRECTORY",
    "load_target_prediction_store",
    "materialize_target_predictions",
)
