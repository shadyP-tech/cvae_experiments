"""Exact workstation preflight for the endpoint-router execution topology."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.preflight import run_label_free_workstation_preflight as _shared_preflight
from .artifact_io import atomic_json, read_json
from .source_cache import source_generation_runtime


_REPORT_FIELDS: Mapping[str, object] = {
    "endpoint_router_phase_order": [
        "prelabel_input_admission",
        "two_A5000_frozen_source_generation",
        "cuda_free_four_by_three_CPU_development_predictions",
        "global_development_prediction_seal",
        "outer_H_scoped_cross_center_development_labels",
        "label_free_target_support_features_and_models",
        "target_physical_predictions_and_static_policy_plans",
        "global_target_prediction_and_policy_seal",
        "same_outer_H_terminal_evaluation_labels",
        "terminal_consumed_test_scoring",
    ],
    "persistent_a5000_gpu_worker_count": 2,
    "cpu_classifier_worker_count": 4,
    "blas_threads_per_classifier_worker": 3,
    "source_stream_count": 81,
    "development_prediction_cell_count": 5_184,
    "target_physical_action_identity_count": 90,
    "target_action_identity_count": 90,
    "target_prediction_cell_count": 810,
    "target_probability_cell_count": 810,
    "target_unique_classifier_fit_count": 810,
    "maximum_total_classifier_fit_count": 5_994,
    "array_storage_dtype": "float32",
    "scientific_reduction_dtype": "float64",
    "same_H_labels_available_preplan": False,
    "support_labels_available": False,
    "phase_disjoint_gpu_and_cpu_pools": True,
    "preflight_reprobed_before_each_compute_session": True,
}


def run_endpoint_router_workstation_preflight(
    root: Path,
    *,
    runtime: Mapping[str, object],
) -> Mapping[str, object]:
    """Probe hardware through the neutral implementation, then bind our counts."""

    _assert_endpoint_runtime(runtime)
    report_path = root / "reports/workstation_preflight.json"
    compatible = source_generation_runtime(runtime)
    # Compatibility-only topology for the neutral hardware probe.  These
    # historical counts are overwritten in the durable report below.
    compatible.update(
        {
            "target_task_count": 81,
            "target_action_identity_count": 81,
            "target_probability_cell_count": 729,
            "target_unique_classifier_fit_count": 729,
            "maximum_total_classifier_fit_count": 729,
            "scratch_preference": ["/data/local", "artifact_parent"],
        }
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".endpoint-router-preflight-", dir=root.parent
    ) as probe:
        payload = dict(
            _shared_preflight(
                Path(probe),
                runtime=compatible,
                expected_scratch_root="/data/local",
            )
        )
    payload["disk_probe_path"] = str(root.resolve())
    payload.update(_REPORT_FIELDS)
    if report_path.is_file():
        persisted = read_json(report_path)
        if persisted.get("status") != "PASS" or any(
            persisted.get(key) != value for key, value in _REPORT_FIELDS.items()
        ):
            raise ProtocolError("Persisted endpoint-router preflight drifted.")
        # The fresh neutral probe above is the admission decision for this
        # compute session.  Keep the first accepted observation durable so a
        # resumed run does not rewrite downstream reports merely because free
        # disk or VRAM changed between otherwise admissible launches.
        return persisted
    atomic_json(report_path, payload)
    return payload


def _assert_endpoint_runtime(runtime: Mapping[str, object]) -> None:
    required = {
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "array_storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "phase_order": "two_A5000_generation_then_four_by_three_CPU",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "source_stream_count": 81,
        "development_prediction_cell_count": 5_184,
        "target_physical_action_identity_count": 90,
        "target_prediction_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 5_994,
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }
    if any(runtime.get(key) != value for key, value in required.items()):
        raise ProtocolError("Endpoint-router workstation runtime contract drifted.")


__all__ = ("run_endpoint_router_workstation_preflight",)
