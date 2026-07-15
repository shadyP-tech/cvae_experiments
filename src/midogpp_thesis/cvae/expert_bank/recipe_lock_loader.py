"""Fail-closed Stage-30 loading of fold-aligned stability consensus locks."""

from __future__ import annotations

from pathlib import Path

from ...real_features.classifier_reference.protocol import ProtocolError
from ..preservation.prior_recovery_stability_artifacts import (
    validate_stability_bundle,
)
from ..preservation.prior_recovery_stability_consensus import (
    TrainingSeedConsensusLock,
)


def load_consensus_recipe_for_fold(
    artifact_root: str | Path,
    *,
    outer_target_center: str,
) -> TrainingSeedConsensusLock:
    """Load only lock H for Stage-30 fold H after validating the whole bundle."""

    outer = str(outer_target_center)
    locks = validate_stability_bundle(Path(artifact_root))
    if not locks or any(
        lock.integrity_status != "VALID" or not lock.recipe_export_ready
        for lock in locks.values()
    ):
        raise ProtocolError(
            "Stability bundle is not globally ready for Stage 30 recipe export."
        )
    lock = locks.get(outer)
    if lock is None:
        raise ProtocolError(f"Stability bundle has no consensus lock for fold H={outer}.")
    if (
        lock.outer_target_center != outer
        or lock.integrity_status != "VALID"
        or not lock.recipe_export_ready
    ):
        raise ProtocolError(
            f"Consensus lock for fold H={outer} is not eligible for Stage 30."
        )
    return lock
