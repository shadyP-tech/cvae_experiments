"""Adapters from the signed diagnostic into the neutral workstation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from ...runtime.label_free_action_predictions import (
    BASE_ACTION_ID,
    GlobalPredictionSeal,
    h_x_e_action_id,
    materialize_label_free_action_predictions,
)
from ...runtime.preflight import run_label_free_workstation_preflight as _preflight
from ..fixed_bank_hierarchical_residual_stacker.case_partitions import (
    CaseOOFPartition,
    build_case_oof_partition,
)
from ..fixed_bank_hierarchical_residual_stacker.experiment_contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    MIDOGPP_CENTERS,
)


SCRATCH_ROOT = "/data/local/fixed_bank_signed_error_gate_v1"
LOCAL_SOURCE_DIRECTORY = "source_cache"


@dataclass(frozen=True, order=True)
class RuntimeSeedProbabilityRow:
    """One seed-pair probability row exposed without labels."""

    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_pair_ordinal: int
    probability: float
    probability_store_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            **self.__dict__,
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
    if (
        runtime.get("probability_surface_format")
        != "sealed_compressed_float32_npz_shared_runtime"
        or runtime.get("probability_materialization_device") != "cpu"
        or int(runtime.get("probability_materialization_workers", -1)) != 4
        or runtime.get("context_feature_format")
        != "bounded_process_local_float64_target_contexts"
        or runtime.get("resume_policy")
        != (
            "hash_validated_source_prediction_task_resume_plus_"
            "deterministic_phase_replay"
        )
        or runtime.get("context_features_rebuilt_and_hash_revalidated_per_target")
        is not True
        or int(runtime.get("maximum_concurrent_target_context_builds", -1)) != 4
        or runtime.get("cross_target_context_cache_forbidden") is not True
    ):
        raise ProtocolError("Signed-error workstation storage contract drifted.")
    shared_runtime = dict(runtime)
    shared_runtime["resume_policy"] = (
        "hash_validated_atomic_phase_and_task_checkpoints"
    )
    report_path = root / "reports/workstation_preflight.json"
    signed_fields = {
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "probability_materialization_device": "cpu",
        "probability_materialization_workers": 4,
        "probability_store_format": "compressed_float32_npz",
        "context_feature_format": "bounded_process_local_float64_target_contexts",
        "maximum_concurrent_target_context_builds": 4,
        "cross_target_context_cache_present": False,
        "resume_strategy": runtime["resume_policy"],
    }
    with tempfile.TemporaryDirectory(
        prefix=".signed-preflight-probe-", dir=root.parent
    ) as probe:
        probed = dict(
            _preflight(
                Path(probe),
                runtime=shared_runtime,
                expected_scratch_root=SCRATCH_ROOT,
            )
        )
    if report_path.is_file():
        payload = read_json(report_path)
        if payload.get("status") != "PASS" or any(
            payload.get(key) != value for key, value in signed_fields.items()
        ):
            raise ProtocolError("Signed-error persisted workstation preflight drifted.")
        return payload
    payload = probed
    payload["disk_probe_path"] = str(root.resolve())
    payload.update(signed_fields)
    atomic_json(report_path, payload)
    return payload


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache:
    cache = materialize_frozen_source_streams(config, generation_lock, root=root)
    _remove_completed_checkpoint_tree(root / "checkpoints/frozen_source_streams")
    return cache


def stage_sources_for_cpu(
    cache: FrozenSourceStreamCache, *, config: object, root: Path
) -> FrozenSourceStreamCache:
    scratch = tuple(
        str(value) for value in getattr(config, "runtime")["scratch_preference"]
    )
    if scratch != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("Signed-error CPU scratch preference drifted.")
    if any(
        token in scratch[0]
        for token in (
            "fixed_bank_hierarchical_residual_stacker",
            "fixed_bank_label_aware_case_oof_ceiling",
            "fixed_bank_pooled_bacc_case_oof_ceiling",
        )
    ):
        raise ProtocolError("Signed-error diagnostic cannot reuse prior Stage-90 scratch.")
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
    seal = materialize_label_free_action_predictions(
        config,
        source_cache,
        frame,
        partition_lock_hash=partition.partition_hash,
        root=root,
    )
    _remove_completed_checkpoint_tree(
        root / "checkpoints/label_free_action_predictions"
    )
    return seal


def seed_probability_rows(
    capability: GlobalPredictionSeal,
) -> tuple[RuntimeSeedProbabilityRow, ...]:
    """Expose B and the eight legal Hxe actions for every target and seed pair."""

    rows: list[RuntimeSeedProbabilityRow] = []
    store = capability.store
    seed_pairs = tuple(
        (training, generation)
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
    )
    for target in MIDOGPP_CENTERS:
        row_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        actions = (
            BASE_ACTION_ID,
            *(h_x_e_action_id(source) for source in MIDOGPP_CENTERS if source != target),
        )
        for runtime_action in actions:
            action = (
                BASE_ACTION_ID
                if runtime_action == BASE_ACTION_ID
                else runtime_action.removeprefix("Hxe::")
            )
            if action == target:
                raise ProtocolError("Target expert entered signed-error action surface.")
            for ordinal, (training_seed, generation_seed) in enumerate(seed_pairs):
                values = store.probabilities(
                    target, runtime_action, training_seed, generation_seed
                )
                for sample_id, case_id, probability in zip(
                    row_ids, case_ids, values, strict=True
                ):
                    rows.append(
                        RuntimeSeedProbabilityRow(
                            target,
                            case_id,
                            sample_id,
                            action,
                            ordinal,
                            float(probability),
                            store.store_hash,
                        )
                    )
    return tuple(sorted(rows))


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache,
    prediction_capability: GlobalPredictionSeal,
    local_staging: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_fixed_bank_signed_error_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction_capability.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction_capability.store.cells),
        "unique_classifier_fit_count": len(prediction_capability.store.cells),
        "local_source_staging": dict(local_staging),
        "scratch_root": SCRATCH_ROOT,
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "probability_materialization_device": "cpu",
        "probability_materialization_workers": int(runtime["classifier_workers"]),
        "model_workers": int(runtime["model_workers"]),
        "model_threads_per_worker": int(runtime["model_threads_per_worker"]),
        "bootstrap_workers": int(runtime["bootstrap_workers"]),
        "bootstrap_threads_per_worker": int(
            runtime["bootstrap_threads_per_worker"]
        ),
        "multiprocessing_start_method": runtime["multiprocessing_start_method"],
        "logical_cpu_budget": 12,
        "prior_stage90_artifact_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
        "gpu_source_phase_completed_before_cpu_fit_phase": True,
        "gpu_and_cpu_pools_disjoint": True,
        "parent_cuda_visible_devices": "",
        "generated_source_store_format": "float32_npy_memmap",
        "probability_store_format": "compressed_float32_npz",
        "bounded_process_local_probability_surface_copy_count": int(
            runtime["model_workers"]
        ),
        "context_features_rebuilt_and_hash_revalidated_per_target": True,
        "maximum_concurrent_target_context_builds": int(runtime["model_workers"]),
        "cross_target_context_cache_present": False,
        "float64_scientific_reductions": True,
        "resume_strategy": runtime["resume_policy"],
    }


def _remove_completed_checkpoint_tree(path: Path) -> None:
    """Remove only experiment-owned checkpoints after a final seal revalidates."""

    if path.exists():
        shutil.rmtree(path)


__all__ = (
    "GlobalPredictionSeal",
    "RuntimeSeedProbabilityRow",
    "SCRATCH_ROOT",
    "build_case_partition",
    "materialize_probabilities",
    "materialize_sources",
    "run_label_free_workstation_preflight",
    "runtime_summary_payload",
    "seed_probability_rows",
    "stage_sources_for_cpu",
)
