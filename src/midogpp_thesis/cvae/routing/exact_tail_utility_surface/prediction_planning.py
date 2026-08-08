"""Build the canonical exact-tail coarse prediction worker surface."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from .config import ExactTailUtilitySurfaceConfig
from .contracts import row_identity_hash
from .prediction_contracts import PredictionWorkerInput
from .production_inputs import PreparedDevelopmentInputs
from .runtime import coarse_prediction_tasks
from .source_contracts import GeneratedDevelopmentCache


def build_prediction_worker_inputs(
    config: ExactTailUtilitySurfaceConfig,
    inputs: PreparedDevelopmentInputs,
    generated: GeneratedDevelopmentCache,
    *,
    checkpoint_root: Path,
) -> tuple[PredictionWorkerInput, ...]:
    source_by_key = generated.source_by_key
    worker_inputs: list[PredictionWorkerInput] = []
    for task in coarse_prediction_tasks():
        records = tuple(
            source_by_key[(source, task.training_seed, task.generation_seed)]
            for source in task.candidate_sources
        )
        partition = inputs.reservation.partitions[task.pseudo_query]
        worker_inputs.append(
            PredictionWorkerInput(
                task=task,
                cache_root=str(generated.root.resolve()),
                source_records=records,
                evaluation_array_path=str(
                    inputs.evaluation_array_path_by_center[
                        task.pseudo_query
                    ].resolve()
                ),
                evaluation_row_identity_hash=row_identity_hash(
                    partition.evaluation_rows
                ),
                partition_hash=partition.reservation_hash,
                source_cache_hash=generated.cache_hash,
                classifier_payload=MappingProxyType(config.classifier.to_payload()),
                checkpoint_root=str(checkpoint_root.resolve()),
            )
        )
    return tuple(worker_inputs)


__all__ = ("build_prediction_worker_inputs",)
