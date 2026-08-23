from __future__ import annotations

import os
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.execution.outer_worker import (
    WORKER_DEPTH_ENV,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.config import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2 import (
    workstation,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json
from midogpp_thesis.cvae.runtime.preflight import REQUIRED_THREAD_ENVIRONMENT


def _neutral_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_label_free_workstation_preflight_v1",
        "status": "PASS",
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_classifier_worker": 3,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "gpu_then_cpu_phase_order": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "parent_cuda_initialized": False,
        "tf32_enabled": False,
        "amp_enabled": False,
        "scratch_preference": ["discarded", "artifact_parent"],
        "available_cpu_affinity_count": 24,
        "physical_ram_bytes": 125 * 1024**3,
        "disk_probe_path": "/discarded",
        "disk_free_bytes_at_launch": 100 * 1024**3,
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "cuda_visible_devices": "0,1",
        "package_versions": {},
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24564,
                "memory_free_mib": 24000,
            },
            {
                "index": 1,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24564,
                "memory_free_mib": 24000,
            },
        ],
    }


def test_runtime_is_exact_two_gpu_four_spawn_no_nested() -> None:
    runtime = canonical_runtime_payload()
    workstation.assert_runtime(runtime)
    assert runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert runtime["persistent_generation_worker_count"] == 2
    assert runtime["outer_process_workers"] == 4
    assert runtime["classifier_threads_per_worker"] == 3
    assert runtime["calibration_threads_per_worker"] == 1
    assert runtime["multiprocessing_start_method"] == "spawn"
    assert runtime["nested_process_pools_forbidden"] is True

    drifted = dict(runtime)
    drifted["nested_process_pools_forbidden"] = False
    with pytest.raises(ProtocolError, match="runtime contract"):
        workstation.assert_runtime(drifted)


def test_extended_preflight_is_atomic_and_exactly_reloadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    (root / "reports").mkdir(parents=True)
    runtime = canonical_runtime_payload()
    calls: list[tuple[Path, dict[str, object]]] = []

    def neutral(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append((path, dict(kwargs)))
        assert path != root
        return _neutral_payload()

    monkeypatch.setattr(workstation, "_neutral", neutral)
    monkeypatch.setattr(
        workstation,
        "probe_scratch",
        lambda _root, _runtime: {
            "scratch_root_id": "scratch",
            "scratch_role": "artifact_parent",
            "scratch_absent_at_launch": True,
            "scratch_parent_writable": True,
            "scratch_free_bytes_at_launch": 100 * 1024**3,
            "scratch_recovery_used": False,
            "v1_scratch_or_checkpoint_used": False,
        },
    )
    written = workstation.run_workstation_preflight(root, runtime=runtime)
    assert written["schema_version"] == workstation.PREFLIGHT_SCHEMA
    assert written["gpu_phase"]["persistent_worker_count"] == 2
    assert written["cpu_phase"]["outer_process_workers"] == 4
    assert written["cpu_phase"]["blas_threads_per_worker"] == 1
    assert written["outer_process_blas_threads"] == 1
    assert written["cpu_phase"]["nested_process_pools"] is False
    assert len(calls) == 1
    assert calls[0][1]["expected_target_probability_cell_count"] == 810
    persisted = read_json(root / workstation.PREFLIGHT_MEMBER)
    assert persisted == dict(written)
    assert dict(
        workstation.load_validated_workstation_preflight(root, runtime=runtime)
    ) == persisted
    assert not tuple((root / "reports").glob("*.tmp"))

    persisted["nested_process_pools"] = True
    atomic_json(root / workstation.PREFLIGHT_MEMBER, persisted)
    with pytest.raises(ProtocolError, match="persisted workstation preflight"):
        workstation.load_validated_workstation_preflight(root, runtime=runtime)


def test_surface_estimate_matches_frozen_workload() -> None:
    estimate = workstation.estimate_workstation_surface()
    assert estimate.outer_centers == 9
    assert estimate.pseudo_routes == 1744
    assert estimate.attempted_action_cells == 10464
    assert estimate.maximum_prefix_cells == 381936
    assert estimate.action_ridge_fits == 999
    assert estimate.policy_ridge_fits == 999


def test_cpu_environment_rejects_nested_pool_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(WORKER_DEPTH_ENV, "1")
    with pytest.raises(ProtocolError, match="worker environment"):
        with workstation.cpu_phase_environment():
            pass
    monkeypatch.delenv(WORKER_DEPTH_ENV)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    with workstation.cpu_phase_environment():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == previous
