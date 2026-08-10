"""Adapters from the scientific diagnostic into the neutral workstation runtime."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from ...runtime.preflight import run_label_free_workstation_preflight as _preflight
from .actions import actions_for_target, build_action_library
from .case_partitions import (
    CaseIdentityRow,
    CaseOOFPartition,
    build_case_oof_partition,
)
from .constants import MIDOGPP_CENTERS
from .contracts import SeedProbabilityRow
from .experiment_contracts import (
    GENERATION_SEEDS,
    OOF_FOLD_SEED,
    TRAINING_SEEDS,
)
from .prediction_runtime import (
    ActionPredictionStore,
    GlobalActionPredictionSeal,
    load_global_action_prediction_seal,
    materialize_action_predictions,
)


SCRATCH_ROOT = "/data/local/fixed_bank_actionability_recoverability_v1"
LOCAL_SOURCE_DIRECTORY = "source_cache"


def build_case_partition(frame: object, *, config: object) -> CaseOOFPartition:
    protocol = getattr(config, "protocol")
    seed = int(protocol.get("partition_seed", OOF_FOLD_SEED))
    if seed != OOF_FOLD_SEED:
        raise ProtocolError("Actionability partition seed drifted.")
    identities = tuple(
        CaseIdentityRow(
            target_center=str(row.center),
            case_id=str(row.case_id),
            sample_id=str(row.evaluation_row_id),
        )
        for row in getattr(frame, "rows")
    )
    return build_case_oof_partition(identities, partition_seed=seed)


def run_label_free_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    if (
        runtime.get("probability_surface_format")
        != "sealed_compressed_float32_npz"
        or runtime.get("probability_materialization_device") != "cpu"
        or int(runtime.get("source_generation_worker_count", -1)) != 2
        or int(runtime.get("model_workers", -1)) != 4
        or int(runtime.get("model_threads_per_worker", -1)) != 3
        or int(runtime.get("bootstrap_workers", -1)) != 4
        or int(runtime.get("bootstrap_threads_per_worker", -1)) != 3
        or int(runtime.get("physical_actions_per_target_task", -1)) != 18
        or int(runtime.get("logical_actions_per_target", -1)) != 19
        or int(runtime.get("target_probability_cell_count", -1)) != 1458
        or int(runtime.get("target_unique_classifier_fit_count", -1)) != 1458
        or runtime.get("resume_policy")
        != (
            "hash_validated_source_prediction_task_resume_plus_"
            "deterministic_phase_replay"
        )
        or runtime.get("parent_cuda_context_forbidden_during_cpu_phase")
        is not True
    ):
        raise ProtocolError("Actionability workstation execution contract drifted.")

    # The shared preflight owns hardware and environment checks but retains the
    # historical nine-action topology.  Probe it in a temporary location with
    # only those legacy topology fields adapted, then atomically publish one
    # enriched report for this 18-action experiment.
    shared_runtime = dict(runtime)
    shared_runtime.update(
        {
            "generation_workers_per_device": 1,
            "parent_cuda_context_forbidden": True,
            "target_action_identity_count": 81,
            "target_probability_cell_count": 729,
            "target_unique_classifier_fit_count": 729,
            "maximum_total_classifier_fit_count": 729,
            "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
        }
    )
    actionability_fields = {
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "probability_materialization_device": "cpu",
        "probability_materialization_workers": 4,
        "physical_actions_per_target_task": 18,
        "logical_actions_per_target": 19,
        "target_probability_cell_count": 1458,
        "target_unique_classifier_fit_count": 1458,
        "A0_A1_geometry_selected": False,
        "A1_sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
        "resume_strategy": runtime["resume_policy"],
    }
    report_path = root / "reports/workstation_preflight.json"
    if report_path.is_file():
        payload = read_json(report_path)
        if payload.get("status") != "PASS" or any(
            payload.get(key) != value for key, value in actionability_fields.items()
        ):
            raise ProtocolError("Persisted actionability preflight drifted.")
        return payload
    with tempfile.TemporaryDirectory(
        prefix=".actionability-preflight-probe-", dir=root.parent
    ) as probe:
        payload = dict(
            _preflight(
                Path(probe),
                runtime=shared_runtime,
                expected_scratch_root=SCRATCH_ROOT,
            )
        )
    payload["disk_probe_path"] = str(root.resolve())
    payload.update(actionability_fields)
    atomic_json(report_path, payload)
    return payload


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> FrozenSourceStreamCache:
    cache = materialize_frozen_source_streams(config, generation_lock, root=root)
    # A crash can occur after the final lock is published but before the
    # neutral runtime removes its owned checkpoints.  Successful reload above
    # proves the final bytes, so the checkpoint tree is now redundant.
    shutil.rmtree(root / "checkpoints/frozen_source_streams", ignore_errors=True)
    return cache


def stage_sources_for_cpu(
    cache: FrozenSourceStreamCache, *, config: object, root: Path
) -> FrozenSourceStreamCache:
    preference = tuple(
        str(value) for value in getattr(config, "runtime")["scratch_preference"]
    )
    if preference != (SCRATCH_ROOT, "artifact_parent"):
        raise ProtocolError("Actionability CPU scratch preference drifted.")
    return stage_frozen_source_streams(
        cache,
        scratch_root=Path(preference[0]),
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
) -> GlobalActionPredictionSeal:
    flat = build_action_library()
    library = {
        target: tuple(action for action in flat if action.target_center == target)
        for target in MIDOGPP_CENTERS
    }
    return materialize_action_predictions(
        config,
        source_cache,
        frame,
        partition_hash=partition.partition_hash,
        action_library=library,
        root=root,
    )


def seed_probability_rows(
    capability: GlobalActionPredictionSeal,
) -> tuple[SeedProbabilityRow, ...]:
    store: ActionPredictionStore = capability.store
    seed_pairs = tuple(
        (training, generation)
        for training in TRAINING_SEEDS
        for generation in GENERATION_SEEDS
    )
    rows: list[SeedProbabilityRow] = []
    for target in MIDOGPP_CENTERS:
        row_ids = store.rows_by_center[target]
        case_ids = store.case_ids_by_center[target]
        for action in actions_for_target(target):
            for ordinal, (training, generation) in enumerate(seed_pairs):
                values = store.probabilities(
                    target, action.action_id, training, generation
                )
                for sample_id, case_id, probability in zip(
                    row_ids, case_ids, values, strict=True
                ):
                    rows.append(
                        SeedProbabilityRow(
                            target_center=target,
                            case_id=case_id,
                            sample_id=sample_id,
                            action_id=action.action_id,
                            seed_pair_ordinal=ordinal,
                            probability=float(probability),
                            probability_store_hash=store.store_hash,
                        )
                    )
    return tuple(sorted(rows))


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache,
    prediction_capability: GlobalActionPredictionSeal,
    local_staging: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_actionability_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction_capability.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction_capability.store.cells),
        "unique_classifier_fit_count": len(prediction_capability.store.cells),
        "physical_action_count_per_target": len(actions_for_target("0")),
        "A0_A1_geometry_selected": False,
        "A1_sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
        "local_source_staging": dict(local_staging),
        "scratch_root": SCRATCH_ROOT,
        "previous_stage90_output_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
        "gpu_source_phase_completed_before_cpu_fit_phase": True,
        "gpu_and_cpu_pools_disjoint": True,
        "persistent_a5000_gpu_worker_count": 2,
        "cpu_classifier_worker_count": int(runtime["classifier_workers"]),
        "blas_threads_per_classifier_worker": int(
            runtime["classifier_threads_per_worker"]
        ),
        "multiprocessing_start_method": str(
            runtime["multiprocessing_start_method"]
        ),
        "float32_probability_store": True,
        "float64_scientific_reductions": True,
        "resume_checkpoints_hash_validated": True,
    }


__all__ = (
    "GlobalActionPredictionSeal",
    "SCRATCH_ROOT",
    "build_case_partition",
    "load_frozen_source_streams",
    "load_global_action_prediction_seal",
    "materialize_probabilities",
    "materialize_sources",
    "run_label_free_workstation_preflight",
    "runtime_summary_payload",
    "seed_probability_rows",
    "stage_sources_for_cpu",
)
