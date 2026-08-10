"""Adapters into the neutral, label-free GPU/CPU workstation runtime."""

from __future__ import annotations

from dataclasses import dataclass
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
from ...runtime.preflight import run_label_free_workstation_preflight as _neutral_preflight
from .case_partitions import CaseOOFPartition, build_case_oof_partition
from .experiment_contracts import CENTERS, GENERATION_SEEDS, TRAINING_SEEDS


SCRATCH_ROOT = "/data/local/fixed_bank_hierarchical_residual_stacker_v1"
LOCAL_SOURCE_DIRECTORY = "source_cache"


@dataclass(frozen=True, order=True)
class RuntimeSeedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_pair_ordinal: int
    probability: float
    probability_store_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "action_id": self.action_id,
            "seed_pair_ordinal": self.seed_pair_ordinal,
            "probability": self.probability,
            "probability_store_hash": self.probability_store_hash,
            "predictions_globally_sealed_before_labels": True,
            "target_expert_used": False,
        }


def build_case_partition(frame: object, *, config: object) -> CaseOOFPartition:
    return build_case_oof_partition(
        tuple(getattr(frame, "rows")),
        partition_seed=int(getattr(config, "protocol")["partition_seed"]),
    )


def run_label_free_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    return _neutral_preflight(root, runtime=runtime, expected_scratch_root=SCRATCH_ROOT)


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache:
    return materialize_frozen_source_streams(config, generation_lock, root=root)


def stage_sources_for_cpu(
    cache: FrozenSourceStreamCache, *, config: object, root: Path
) -> FrozenSourceStreamCache:
    scratch = tuple(str(value) for value in getattr(config, "runtime")["scratch_preference"])
    if scratch != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("Residual-stacker CPU scratch preference drifted.")
    forbidden = (
        "fixed_bank_label_aware_case_oof_ceiling",
        "fixed_bank_pooled_bacc_case_oof_ceiling",
    )
    if any(token in scratch[0] for token in forbidden):
        raise ProtocolError("Residual stacker cannot reuse prior Stage-90 scratch.")
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


def seed_probability_rows(
    capability: GlobalPredictionSeal,
) -> tuple[RuntimeSeedProbabilityRow, ...]:
    """Expose B plus eight legal Hxe actions; target expert is impossible here."""

    store = capability.store
    rows: list[RuntimeSeedProbabilityRow] = []
    seed_pairs = tuple(
        (training, generation)
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
    )
    for target in CENTERS:
        row_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        actions = (BASE_ACTION_ID, *(h_x_e_action_id(e) for e in CENTERS if e != target))
        for runtime_action in actions:
            action = (
                BASE_ACTION_ID
                if runtime_action == BASE_ACTION_ID
                else runtime_action.removeprefix("Hxe::")
            )
            if action == target:
                raise ProtocolError("Target expert entered residual-stacker action surface.")
            for seed_ordinal, (training_seed, generation_seed) in enumerate(seed_pairs):
                values = store.probabilities(
                    target, runtime_action, training_seed, generation_seed
                )
                for sample_id, case_id, probability in zip(
                    row_ids, case_ids, values, strict=True
                ):
                    rows.append(
                        RuntimeSeedProbabilityRow(
                            target_center=target,
                            case_id=case_id,
                            sample_id=sample_id,
                            action_id=action,
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
        "schema_version": "midogpp_hierarchical_residual_stacker_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction_capability.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction_capability.store.cells),
        "unique_classifier_fit_count": len(prediction_capability.store.cells),
        "local_source_staging": dict(local_staging),
        "scratch_root": SCRATCH_ROOT,
        "prior_stage90_artifact_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
        "gpu_source_phase_completed_before_cpu_fit_phase": True,
        "gpu_and_cpu_pools_disjoint": True,
        "parent_cuda_visible_devices": "",
        "persistent_a5000_gpu_worker_count": 2,
        "cpu_model_worker_count": 4,
        "threads_per_cpu_model_worker": 3,
        "blas_threads_per_process": 1,
        "logical_cpu_budget": 12,
        "minimum_ram_gib": 100,
        "minimum_free_gpu_memory_gib": 18,
        "minimum_artifact_disk_gib": 8,
        "float32_memmap_stores": True,
        "float64_scientific_reductions": True,
        "resume_checkpoints_hash_validated": True,
    }


__all__ = (
    "GlobalPredictionSeal",
    "RuntimeSeedProbabilityRow",
    "SCRATCH_ROOT",
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
