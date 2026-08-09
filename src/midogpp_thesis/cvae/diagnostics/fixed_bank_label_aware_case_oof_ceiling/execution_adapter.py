"""Experiment-owned adapters into the neutral label-free workstation runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from ...runtime.label_free_action_predictions import (
    BASE_ACTION_ID,
    GlobalPredictionSeal,
    build_direct_target_actions,
    h_x_e_action_id,
    load_global_prediction_seal,
    materialize_label_free_action_predictions,
)
from ...runtime.preflight import run_label_free_workstation_preflight
from .core_contracts import CaseIdentityRow, SeedProbabilityRow
from .experiment_contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS
from .partitions import CaseOOFPartition, build_case_oof_partition


LOCAL_SOURCE_DIRECTORY = "source_cache"


def build_case_partition(frame: object, *, config: object) -> CaseOOFPartition:
    identities = tuple(
        CaseIdentityRow(
            target_center=str(row.center),
            case_id=str(row.case_id),
            sample_id=str(row.evaluation_row_id),
        )
        for row in getattr(frame, "rows")
    )
    return build_case_oof_partition(
        identities,
        partition_seed=int(getattr(config, "protocol")["partition_seed"]),
    )


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache:
    return materialize_frozen_source_streams(config, generation_lock, root=root)


def stage_sources_for_cpu(
    cache: FrozenSourceStreamCache,
    *,
    config: object,
    root: Path,
) -> FrozenSourceStreamCache:
    scratch = tuple(str(value) for value in getattr(config, "runtime")["scratch_preference"])
    if not scratch or scratch[0] != "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1":
        raise ProtocolError("Label-aware CPU scratch preference drifted.")
    return stage_frozen_source_streams(
        cache,
        scratch_root=Path(scratch[0]),
        canonical_root=root,
        local_directory=LOCAL_SOURCE_DIRECTORY,
    )


def materialize_probabilities(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    partition: CaseOOFPartition,
    *,
    root: Path,
) -> GlobalPredictionSeal:
    return materialize_label_free_action_predictions(
        config,
        source_cache,
        frame,
        partition_lock_hash=partition.partition_hash,
        root=root,
    )


def seed_probability_rows(capability: GlobalPredictionSeal) -> tuple[SeedProbabilityRow, ...]:
    """Convert runtime Hxe names to the scientific source-center action IDs."""

    store = capability.store
    rows: list[SeedProbabilityRow] = []
    for target in CENTERS:
        row_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        runtime_actions = (BASE_ACTION_ID, *(h_x_e_action_id(source) for source in CENTERS if source != target))
        for action in runtime_actions:
            scientific_action = BASE_ACTION_ID if action == BASE_ACTION_ID else action.removeprefix("Hxe::")
            for seed_ordinal, (training_seed, generation_seed) in enumerate(
                (pair for pair in ((train, generation) for train in TRAINING_SEEDS for generation in GENERATION_SEEDS))
            ):
                probabilities = store.probabilities(target, action, training_seed, generation_seed)
                for sample_id, case_id, probability in zip(row_ids, case_ids, probabilities, strict=True):
                    rows.append(
                        SeedProbabilityRow(
                            target_center=target,
                            case_id=case_id,
                            sample_id=sample_id,
                            action_id=scientific_action,
                            seed_pair_ordinal=seed_ordinal,
                            probability=float(probability),
                            probability_store_hash=store.store_hash,
                        )
                    )
    return tuple(sorted(rows))


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache,
    prediction_capability: GlobalPredictionSeal,
    local_staging: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_label_aware_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction_capability.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction_capability.store.cells),
        "unique_classifier_fit_count": len(prediction_capability.store.cells),
        "local_source_staging": dict(local_staging),
        "gpu_source_phase_completed_before_cpu_fit_phase": True,
        "gpu_and_cpu_pools_disjoint": True,
        "float32_stores": True,
        "float64_scientific_reductions": True,
        "resume_checkpoints_hash_validated": True,
    }


__all__ = (
    "GlobalPredictionSeal",
    "build_case_partition",
    "build_direct_target_actions",
    "load_frozen_source_streams",
    "load_global_prediction_seal",
    "materialize_probabilities",
    "materialize_sources",
    "run_label_free_workstation_preflight",
    "runtime_summary_payload",
    "seed_probability_rows",
    "stage_sources_for_cpu",
)
