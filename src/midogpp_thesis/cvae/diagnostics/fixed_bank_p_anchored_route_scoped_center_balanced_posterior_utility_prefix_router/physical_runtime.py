"""Neutral frozen-source and exact-nine probability runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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
from .actions import action_library_by_target
from .constants import CENTERS, EXPECTED_TEST_ROW_COUNT, physical_action_ids
from .hashing import canonical_hash
from .probability_surface import build_physical_probability_surface
from .scratch import (
    PREDICTION_DIRECTORY,
    SOURCE_DIRECTORY,
    ScratchLease,
    create_scratch,
)
from .workstation import assert_cuda_free_cpu_phase, enter_cuda_free_cpu_phase


@dataclass(frozen=True)
class MaterializedPhysicalInputs:
    canonical_source_cache: FrozenSourceStreamCache
    local_source_cache: FrozenSourceStreamCache
    prediction: GlobalPredictionSeal
    scratch: ScratchLease


@dataclass(frozen=True)
class ProbabilityIndexRow:
    target_center: str
    action_id: str
    row_count: int
    source_cell_probability_sha256: tuple[str, ...]
    sample_identity_hash: str
    case_identity_hash: str
    exact_nine_probability_sha256: str

    def to_payload(self) -> dict[str, object]:
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
            "storage_dtype": "float32",
            "reduction_dtype": "float64",
        }


def materialize_physical_inputs(
    config: object,
    generation_lock: object,
    frame: object,
    *,
    root: Path,
) -> MaterializedPhysicalInputs:
    """Generate once on two GPUs, then fit the 810 cells in a disjoint CPU phase."""

    lease = create_scratch(root, getattr(config, "runtime"))
    local_root = lease.root / SOURCE_DIRECTORY
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
        raise ProtocolError("CBPUPR staged source bytes drifted.")
    enter_cuda_free_cpu_phase()
    assert_cuda_free_cpu_phase()
    prediction_scratch = lease.root / PREDICTION_DIRECTORY
    prediction_scratch.mkdir(parents=True, exist_ok=False)
    prediction = materialize_fixed_bank_a1_action_predictions(
        config,
        local,
        frame,
        partition_hash=physical_partition_hash(frame),
        action_library=action_library_by_target(),
        root=root,
        scratch_root=prediction_scratch,
    )
    return MaterializedPhysicalInputs(canonical, local, prediction, lease)


def physical_partition_hash(frame: object) -> str:
    rows = tuple(getattr(frame, "rows"))
    if len(rows) != EXPECTED_TEST_ROW_COUNT:
        raise ProtocolError("CBPUPR physical plan row count drifted.")
    return canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_global_physical_plan_v1",
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


def probability_index_rows(
    prediction: GlobalPredictionSeal,
) -> tuple[ProbabilityIndexRow, ...]:
    store = prediction.store
    output: list[ProbabilityIndexRow] = []
    for target in CENTERS:
        samples = tuple(store.rows_by_center[target])
        cases = tuple(store.case_ids_by_center[target])
        for action in physical_action_ids(target):
            cells = tuple(
                cell
                for cell in store.cells
                if cell.target_center == target and cell.action_id == action
            )
            if len(cells) != 9:
                raise ProtocolError("CBPUPR probability index lacks exact nine.")
            values = np.mean(
                np.stack([cell.probabilities for cell in cells]).astype(np.float64),
                axis=0,
                dtype=np.float64,
            )
            output.append(
                ProbabilityIndexRow(
                    target,
                    action,
                    len(values),
                    tuple(cell.probability_sha256 for cell in cells),
                    canonical_hash(list(samples)),
                    canonical_hash(list(cases)),
                    sha256_array(values),
                )
            )
    if len(output) != 90:
        raise ProtocolError("CBPUPR probability index topology drifted.")
    return tuple(output)


def build_surface(physical: MaterializedPhysicalInputs) -> object:
    return build_physical_probability_surface(physical.prediction.store)


def runtime_summary_payload(
    physical: MaterializedPhysicalInputs,
    *,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cbpupr_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": physical.canonical_source_cache.lock_hash,
        "global_prediction_seal_hash": physical.prediction.seal_hash,
        "source_stream_count": len(physical.canonical_source_cache.records),
        "classifier_cell_count": len(physical.prediction.store.cells),
        "unique_classifier_fit_count": len(physical.prediction.store.cells),
        "workstation_preflight_hash": canonical_hash(preflight),
        "persistent_generation_worker_count": 2,
        "classifier_workers": int(runtime["classifier_workers"]),
        "route_workers": int(runtime["route_model_workers"]),
        "route_worker_blas_threads": int(runtime["classifier_threads_per_worker"]),
        "outer_process_blas_threads": int(runtime["classifier_threads_per_worker"]),
        "target_posterior_process_blas_threads": int(
            runtime["target_posterior_threads_per_worker"]
        ),
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_route_phase": "",
        "double_exclusion_state_count": int(runtime["ordered_H_J_pair_count"]),
        "unused_nested_endpoint_fits_eliminated": True,
        "outer_endpoint_model_fit_count": int(runtime["expected_outer_endpoint_model_fit_count"]),
        "donor_response_model_fit_count": int(runtime["donor_response_model_fit_count"]),
        "target_posterior_model_fit_count": int(
            runtime["expected_target_posterior_model_fit_count"]
        ),
        "pseudo_route_count": int(runtime["pseudo_route_count"]),
        "pseudo_posterior_model_fit_count": int(
            runtime["expected_pseudo_posterior_model_fit_count"]
        ),
        "total_posterior_model_fit_count": int(
            runtime["expected_total_posterior_model_fit_count"]
        ),
        "validation_endpoint_optimizer_refit_count": 0,
        "validation_posterior_optimizer_refit_count": 0,
        "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
        "prior_rebinding_additional_endpoint_model_fit_count": 0,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "scratch_root_id": physical.scratch.root.name,
        "scratch_role": physical.scratch.role,
        "local_and_canonical_source_lock_identical": (
            dict(physical.local_source_cache.lock_payload)
            == dict(physical.canonical_source_cache.lock_payload)
        ),
        "previous_stage90_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }


__all__ = (
    "MaterializedPhysicalInputs",
    "ProbabilityIndexRow",
    "build_surface",
    "materialize_physical_inputs",
    "physical_partition_hash",
    "probability_index_rows",
    "runtime_summary_payload",
)
