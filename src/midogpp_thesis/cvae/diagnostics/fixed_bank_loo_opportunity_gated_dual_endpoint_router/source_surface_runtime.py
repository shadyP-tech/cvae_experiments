"""Neutral source generation, prediction materialization, and exact-nine surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from ...runtime.fixed_bank_a1_action_predictions import (
    GlobalPredictionSeal,
    materialize_fixed_bank_a1_action_predictions,
)
from ...runtime.frozen_source_streams import (
    FrozenSourceStreamCache,
    load_frozen_source_streams,
    materialize_frozen_source_streams,
    stage_frozen_source_streams,
)
from .hashing import canonical_hash
from .scratch_policy import (
    LOCAL_GENERATION_DIRECTORY,
    fresh_scratch_base,
    prediction_scratch,
)


@dataclass(frozen=True)
class ProbabilityIndexRow:
    target_center: str
    action_id: str
    row_count: int
    source_cell_probability_sha256: tuple[str, ...]
    sample_identity_hash: str
    case_identity_hash: str
    exact_nine_probability_sha256: str
    storage_dtype: str = "float32"
    reduction_dtype: str = "float64"

    def to_payload(self) -> dict[str, object]:
        # Keep this explicit: passing ``self`` to ``json_native`` would call
        # this method again before reaching the dataclass fallback.
        return {
            "target_center": self.target_center,
            "action_id": self.action_id,
            "row_count": self.row_count,
            "source_cell_probability_sha256": list(
                self.source_cell_probability_sha256
            ),
            "sample_identity_hash": self.sample_identity_hash,
            "case_identity_hash": self.case_identity_hash,
            "exact_nine_probability_sha256": self.exact_nine_probability_sha256,
            "storage_dtype": self.storage_dtype,
            "reduction_dtype": self.reduction_dtype,
        }


@dataclass(frozen=True)
class MaterializedSourceCaches:
    canonical: FrozenSourceStreamCache
    local: FrozenSourceStreamCache

    @property
    def lock_hash(self) -> str:
        return self.canonical.lock_hash

    @property
    def records(self) -> tuple[object, ...]:
        return self.canonical.records


def materialize_sources(
    config: object, generation_lock: object, *, root: Path
) -> MaterializedSourceCaches:
    base = fresh_scratch_base()
    local_root = base / LOCAL_GENERATION_DIRECTORY
    local_root.mkdir(parents=True, exist_ok=False)
    local = materialize_frozen_source_streams(config, generation_lock, root=local_root)
    stage_frozen_source_streams(
        local,
        scratch_root=root.parent,
        canonical_root=local_root,
        local_directory=root.name,
    )
    canonical = load_frozen_source_streams(
        root,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(
            getattr(generation_lock, "generation_lock_hash")
        ),
    )
    if dict(canonical.lock_payload) != dict(local.lock_payload):
        raise ProtocolError("Dual-endpoint staged source bytes drifted.")
    return MaterializedSourceCaches(canonical=canonical, local=local)


def materialize_probabilities(
    config: object,
    source_cache: FrozenSourceStreamCache,
    frame: object,
    *,
    partition_hash: str,
    action_library: Mapping[str, Sequence[object]],
    root: Path,
) -> GlobalPredictionSeal:
    return materialize_fixed_bank_a1_action_predictions(
        config,
        source_cache,
        frame,
        partition_hash=partition_hash,
        action_library=action_library,
        root=root,
        scratch_root=prediction_scratch(),
    )


def physical_partition_hash(frame: object) -> str:
    rows = tuple(getattr(frame, "rows"))
    return canonical_hash(
        {
            "schema_version": "fixed_bank_dual_endpoint_global_physical_plan_v1",
            "rows": [
                {
                    "target_center": str(getattr(row, "center")),
                    "case_id": str(getattr(row, "case_id")),
                    "sample_id": str(getattr(row, "sample_id")),
                }
                for row in rows
            ],
            "row_count": len(rows),
            "case_count": len(
                {
                    (str(getattr(row, "center")), str(getattr(row, "case_id")))
                    for row in rows
                }
            ),
            "labels_used": False,
            "arbitrary_folds_used": False,
        }
    )


def build_exact_nine_surface(prediction: GlobalPredictionSeal) -> object:
    from .probability_surfaces import ExactNineProbabilityRow, ExactNineProbabilitySurface

    store = prediction.store
    rows: list[object] = []
    for target in CENTERS:
        cells_for_target = tuple(
            cell for cell in store.cells if cell.target_center == target
        )
        action_ids = tuple(dict.fromkeys(cell.action_id for cell in cells_for_target))
        if len(action_ids) != 10:
            raise ProtocolError("Dual-endpoint action coverage drifted.")
        sample_ids = tuple(store.rows_by_center[target])
        case_ids = tuple(store.case_ids_by_center[target])
        for action in action_ids:
            cells = tuple(cell for cell in cells_for_target if cell.action_id == action)
            if len(cells) != 9:
                raise ProtocolError("Dual-endpoint seed-pair coverage drifted.")
            matrix = np.stack([cell.probabilities for cell in cells]).astype(
                np.float64, copy=False
            )
            rows.extend(
                ExactNineProbabilityRow(
                    target,
                    case_id,
                    sample_id,
                    action,
                    tuple(float(value) for value in matrix[:, ordinal]),
                )
                for ordinal, (sample_id, case_id) in enumerate(
                    zip(sample_ids, case_ids, strict=True)
                )
            )
    if len(rows) != 99_280:
        raise ProtocolError("Dual-endpoint exact-nine row topology drifted.")
    return ExactNineProbabilitySurface(tuple(rows), store.store_hash)


def probability_index_rows(
    prediction: GlobalPredictionSeal,
) -> tuple[ProbabilityIndexRow, ...]:
    store = prediction.store
    rows: list[ProbabilityIndexRow] = []
    for target in CENTERS:
        sample_ids = tuple(store.rows_by_center[target])
        case_ids = tuple(store.case_ids_by_center[target])
        action_ids = tuple(
            dict.fromkeys(
                cell.action_id for cell in store.cells if cell.target_center == target
            )
        )
        for action in action_ids:
            cells = tuple(
                cell
                for cell in store.cells
                if cell.target_center == target and cell.action_id == action
            )
            values = np.mean(
                np.stack([cell.probabilities for cell in cells]).astype(np.float64),
                axis=0,
                dtype=np.float64,
            )
            rows.append(
                ProbabilityIndexRow(
                    target,
                    action,
                    len(values),
                    tuple(cell.probability_sha256 for cell in cells),
                    canonical_hash(list(sample_ids)),
                    canonical_hash(list(case_ids)),
                    sha256_array(values),
                )
            )
    if len(rows) != 90:
        raise ProtocolError("Dual-endpoint probability index drifted.")
    return tuple(rows)


def runtime_summary_payload(
    *,
    source_cache: FrozenSourceStreamCache | MaterializedSourceCaches,
    prediction: GlobalPredictionSeal,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    preflight_attestation = {
        "schema_version": preflight.get("schema_version"),
        "status": preflight.get("status"),
        "generation_devices": preflight.get("generation_devices"),
        "persistent_gpu_workers": preflight.get("persistent_gpu_workers"),
        "classifier_workers": preflight.get("classifier_workers"),
        "blas_threads_per_classifier_worker": preflight.get(
            "blas_threads_per_classifier_worker"
        ),
        "target_probability_cell_count": preflight.get(
            "target_probability_cell_count"
        ),
        "scratch_root_id": preflight.get("scratch_root_id"),
        "scratch_fallback_role": preflight.get("scratch_fallback_role"),
    }
    return {
        "schema_version": "fixed_bank_dual_endpoint_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "source_stream_count": len(source_cache.records),
        "classifier_cell_count": len(prediction.store.cells),
        "unique_classifier_fit_count": len(prediction.store.cells),
        "workstation_preflight": preflight_attestation,
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "persistent_generation_worker_count": 2,
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_route_phase": "",
        "classifier_workers": int(runtime["classifier_workers"]),
        "route_workers": int(runtime["route_model_workers"]),
        "classifier_threads_per_worker": int(
            runtime["classifier_threads_per_worker"]
        ),
        "fresh_parent_outer_blas_threads": 1,
        "route_worker_blas_threads": 3,
        "multiprocessing_start_method": runtime["multiprocessing_start_method"],
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        "resume_policy": runtime["resume_policy"],
        "task_checkpoints_are_intra_launch_atomicity_only": True,
        "terminal_or_cross_run_recovery_used": False,
        "dedicated_local_scratch_used_for_throughput": True,
        "scratch_root_id": "fixed_bank_loo_opportunity_gated_dual_endpoint_router_v1",
        "local_and_canonical_source_lock_identical": (
            not isinstance(source_cache, MaterializedSourceCaches)
            or dict(source_cache.local.lock_payload)
            == dict(source_cache.canonical.lock_payload)
        ),
        "prior_run_scratch_used_as_evidence": False,
        "predecessor_stage90_artifact_checkpoint_or_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }


__all__ = (
    "GlobalPredictionSeal",
    "MaterializedSourceCaches",
    "ProbabilityIndexRow",
    "build_exact_nine_surface",
    "load_frozen_source_streams",
    "materialize_probabilities",
    "materialize_sources",
    "physical_partition_hash",
    "probability_index_rows",
    "runtime_summary_payload",
)
