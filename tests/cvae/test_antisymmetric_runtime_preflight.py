from __future__ import annotations

from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router import (
    runtime_preflight,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _runtime() -> dict[str, object]:
    return {
        "generation_devices": ["cuda:0", "cuda:1"],
        "kernel_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "tf32_disabled_in_gpu_workers": True,
        "dependency_version_policy": "presence_gate_versions_report_only",
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 100 * 1024**3,
        "minimum_artifact_disk_free_bytes": 8 * 1024**3,
        "minimum_gpu_free_mib_per_device": 18000,
    }


def _patch_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    for key, value in runtime_preflight.REQUIRED_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(runtime_preflight.os, "cpu_count", lambda: 24)
    monkeypatch.delattr(runtime_preflight.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(
        runtime_preflight,
        "_physical_ram_bytes",
        lambda: 125 * 1024**3,
    )
    monkeypatch.setattr(
        runtime_preflight.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=20 * 1024**3),
    )
    monkeypatch.setattr(
        runtime_preflight,
        "_package_versions",
        lambda: {name: "test" for name in runtime_preflight.REQUIRED_DISTRIBUTIONS},
    )
    monkeypatch.setattr(
        runtime_preflight,
        "_nvidia_smi_rows",
        lambda: (
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
        ),
    )
    import torch

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


def test_preflight_accepts_the_frozen_workstation_without_parent_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_healthy(monkeypatch)
    report = runtime_preflight.run_workstation_preflight(
        tmp_path,
        runtime=_runtime(),
    )
    assert report["status"] == "PASS"
    assert report["classifier_worker_thread_product"] == 12
    assert report["parent_cuda_context_initialized"] is False
    assert [row["index"] for row in report["gpus"]] == [0, 1]


def test_preflight_fails_before_jobs_when_vram_or_thread_env_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_healthy(monkeypatch)
    monkeypatch.setattr(
        runtime_preflight,
        "_nvidia_smi_rows",
        lambda: (
            {
                "index": 0,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24564,
                "memory_free_mib": 1000,
            },
            {
                "index": 1,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24564,
                "memory_free_mib": 24000,
            },
        ),
    )
    with pytest.raises(ProtocolError, match="insufficient free VRAM"):
        runtime_preflight.run_workstation_preflight(tmp_path, runtime=_runtime())

    _patch_healthy(monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(ProtocolError, match="thread environment"):
        runtime_preflight.run_workstation_preflight(tmp_path, runtime=_runtime())
