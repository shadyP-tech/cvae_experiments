"""Pinned, spawn-safe workstation topology for a future authorized run."""

from __future__ import annotations

from collections.abc import Mapping

from midogpp_thesis.cvae.protocol import ProtocolError


def workstation_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v1_planned_workstation_v1",
        "profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_generation_workers": 2,
        "one_persistent_worker_per_physical_gpu": True,
        "generated_source_family_streams": 81,
        "full_source_rows_per_class": 1024,
        "exact_B_sources_per_target": 8,
        "exact_B_rows_per_source_per_class": 128,
        "prediction_store_dtype": "float32",
        "prediction_store_mode": "read_only_memmap",
        "prediction_store_materialized_once": True,
        "scientific_reduction_dtype": "float64",
        "cpu_outer_center_workers": 4,
        "blas_threads_per_worker": 1,
        "native_threads_per_worker": 1,
        "multiprocessing_start_method": "spawn",
        "top_level_spawn_pool_only": True,
        "nested_pools_allowed": False,
        "process_transport": ["paths", "hashes", "tuples", "scalars"],
        "estimator_objects_cross_process_allowed": False,
        "mappingproxy_cross_process_allowed": False,
        "execution_authorized": False,
        "output_root_resolution_allowed": False,
        "scratch_root_resolution_allowed": False,
        "output_or_scratch_creation_allowed": False,
        "cross_run_recovery_allowed": False,
    }


def validate_workstation_payload(payload: Mapping[str, object]) -> None:
    if dict(payload) != workstation_payload():
        raise ProtocolError("SCEPTRE workstation topology drifted.")


__all__ = ("validate_workstation_payload", "workstation_payload")
