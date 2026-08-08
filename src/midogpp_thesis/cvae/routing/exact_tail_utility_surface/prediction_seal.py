"""Global pre-label sealing for consolidated exact-tail predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .config import ExactTailUtilitySurfaceConfig, FreshInputAttestation
from .contracts import DevelopmentPartition
from .prediction_checkpoint_store import atomic_json
from .prediction_contracts import (
    GLOBAL_SEAL_MEMBER,
    CoarsePredictionRecord,
    ConsolidatedPredictionArtifacts,
    PredictionExecutionResult,
)
from .runtime import coarse_prediction_tasks
from .scoring import SealedPredictionSurface
from .seals import build_global_prediction_seal


def seal_consolidated_predictions(
    config: ExactTailUtilitySurfaceConfig,
    attestation: FreshInputAttestation,
    partitions: Mapping[str, DevelopmentPartition],
    records: Sequence[CoarsePredictionRecord],
    consolidated: ConsolidatedPredictionArtifacts,
    *,
    root: Path,
) -> PredictionExecutionResult:
    seal = build_global_prediction_seal(
        config_contract_hash=config.contract_hash,
        reservation_index_hash=attestation.reservation_index_hash,
        development_cache_binding_hash=attestation.development_cache_binding_hash,
        development_manifest_sha256=attestation.development_manifest_sha256,
        target_evaluation_binding_hash=attestation.target_evaluation_binding_hash,
        prediction_index_sha256=consolidated.prediction_index_sha256,
        prediction_arrays_sha256=consolidated.prediction_arrays_sha256,
        partitions=partitions,
        cells=consolidated.cells,
    )
    seal_path = root / GLOBAL_SEAL_MEMBER
    atomic_json(seal_path, seal.to_payload())
    # Constructing this view re-hashes every prediction vector against the seal.
    surface = SealedPredictionSurface(
        predictions_by_key=consolidated.predictions_by_key,
        seal=seal,
    )
    by_task = {record.task.key: record for record in records}
    planned = coarse_prediction_tasks()
    return PredictionExecutionResult(
        predictions=surface,
        seal=seal,
        seal_path=seal_path,
        prediction_index_path=consolidated.prediction_index_path,
        prediction_arrays_path=consolidated.prediction_arrays_path,
        task_records=tuple(by_task[task.key] for task in planned),
    )


__all__ = ("seal_consolidated_predictions",)
