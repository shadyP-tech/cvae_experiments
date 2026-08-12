"""CUDA-free parent and workstation-topology checks."""

from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

from ..protocol import ProtocolError
from .artifact_io import atomic_json


REQUIRED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "threadpoolctl",
    "torch",
)


def run_label_free_workstation_preflight(
    root: Path,
    *,
    runtime: Mapping[str, object],
    expected_scratch_root: str = (
        "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1"
    ),
    expected_target_action_identity_count: int = 81,
    expected_target_probability_cell_count: int = 729,
    expected_unique_classifier_fit_count: int = 729,
) -> Mapping[str, object]:
    devices = tuple(str(value) for value in runtime.get("generation_devices", ()))
    scratch = tuple(str(value) for value in runtime.get("scratch_preference", ()))
    if (
        devices != ("cuda:0", "cuda:1")
        or runtime.get("cuda_visible_devices") != "0,1"
        or int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or runtime.get("persistent_source_workers") is not True
        or int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or int(runtime.get("launch_blas_threads", -1)) != 1
        or runtime.get("tf32_enabled") is not False
        or runtime.get("amp_enabled") is not False
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_forbidden") is not True
        or runtime.get("generated_cache_format") != "float32_npy_memmap"
        or runtime.get("scientific_reductions_dtype") != "float64"
        or int(runtime.get("source_job_count", -1)) != 27
        or int(runtime.get("source_stream_count", -1)) != 81
        or int(runtime.get("source_prefix_rows_per_class", -1)) != 270
        or int(runtime.get("target_task_count", -1)) != 81
        or int(runtime.get("target_action_identity_count", -1))
        != expected_target_action_identity_count
        or int(runtime.get("target_probability_cell_count", -1))
        != expected_target_probability_cell_count
        or int(runtime.get("target_unique_classifier_fit_count", -1))
        != expected_unique_classifier_fit_count
        or int(runtime.get("maximum_total_classifier_fit_count", -1))
        != expected_unique_classifier_fit_count
        or runtime.get("resume_policy")
        != "hash_validated_atomic_phase_and_task_checkpoints"
        or scratch != (str(expected_scratch_root), "artifact_parent")
    ):
        raise ProtocolError("Label-free workstation topology drifted.")
    if "spawn" not in mp.get_all_start_methods():
        raise ProtocolError("Label-free workstation requires multiprocessing spawn.")
    mismatched = {
        key: os.environ.get(key)
        for key, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatched or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise ProtocolError(
            "Deterministic workstation environment is absent; launch through workspace run."
        )
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Runtime parent process must remain CUDA-free.")
    cpu_count = _available_cpu_count()
    ram_bytes = _physical_ram_bytes()
    disk_probe = _nearest_existing_parent(root)
    disk_bytes = int(shutil.disk_usage(disk_probe).free)
    if cpu_count < int(runtime["minimum_logical_cpu_count"]):
        raise ProtocolError("Label-free workstation exposes too few CPUs.")
    if ram_bytes < int(runtime["minimum_physical_ram_bytes"]):
        raise ProtocolError("Label-free workstation exposes too little RAM.")
    if disk_bytes < int(runtime["minimum_artifact_disk_free_bytes"]):
        raise ProtocolError("Label-free artifact filesystem reserve is too low.")
    gpu_rows = _nvidia_smi_rows()
    by_index = {int(row["index"]): row for row in gpu_rows}
    if tuple(sorted(by_index)) != (0, 1):
        raise ProtocolError("Label-free runtime requires exactly CUDA devices 0 and 1.")
    for index in (0, 1):
        row = by_index[index]
        if "RTX A5000" not in str(row["name"]):
            raise ProtocolError("Label-free runtime requires two RTX A5000 GPUs.")
        if int(row["memory_free_mib"]) < int(runtime["minimum_gpu_free_mib_per_device"]):
            raise ProtocolError(f"CUDA device {index} has insufficient free VRAM.")
    payload = {
        "schema_version": "midogpp_label_free_workstation_preflight_v1",
        "status": "PASS",
        "generation_devices": list(devices),
        "persistent_gpu_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_classifier_worker": 3,
        "target_action_identity_count": expected_target_action_identity_count,
        "target_probability_cell_count": expected_target_probability_cell_count,
        "target_unique_classifier_fit_count": expected_unique_classifier_fit_count,
        "maximum_total_classifier_fit_count": expected_unique_classifier_fit_count,
        "gpu_then_cpu_phase_order": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "parent_cuda_initialized": False,
        "tf32_enabled": False,
        "amp_enabled": False,
        "scratch_preference": list(scratch),
        "available_cpu_affinity_count": cpu_count,
        "physical_ram_bytes": ram_bytes,
        "disk_probe_path": str(disk_probe.resolve()),
        "disk_free_bytes_at_launch": disk_bytes,
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "cuda_visible_devices": "0,1",
        "package_versions": _package_versions(),
        "gpus": [by_index[index] for index in (0, 1)],
    }
    atomic_json(root / "reports/workstation_preflight.json", payload)
    return payload


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    if not current.exists():
        raise ProtocolError("Cannot locate label-free artifact filesystem.")
    return current


def _available_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if callable(affinity) else int(os.cpu_count() or 0)


def _physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as exc:
        raise ProtocolError("Cannot determine label-free workstation RAM.") from exc


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProtocolError(f"Missing label-free runtime dependency: {name}.") from exc
    return versions


def _nvidia_smi_rows() -> tuple[dict[str, object], ...]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtocolError("Cannot query label-free CUDA devices.") from exc
    if completed.returncode != 0:
        raise ProtocolError("nvidia-smi failed during label-free preflight.")
    rows: list[dict[str, object]] = []
    try:
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != 4:
                raise ValueError
            rows.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "memory_total_mib": int(values[2]),
                    "memory_free_mib": int(values[3]),
                }
            )
    except ValueError as exc:
        raise ProtocolError("nvidia-smi returned malformed label-free rows.") from exc
    return tuple(rows)


__all__ = ("run_label_free_workstation_preflight",)
