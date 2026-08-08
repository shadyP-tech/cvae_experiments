"""Fail-fast checks for the frozen dual-A5000 execution schedule."""

from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from ...protocol import ProtocolError


CLASSIFIER_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3
MINIMUM_LOGICAL_CPUS = 12
MINIMUM_RAM_BYTES = 107374182400
MINIMUM_DISK_FREE_BYTES = 8589934592
MINIMUM_GPU_FREE_MIB = 18000
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


def run_workstation_preflight(
    artifact_root: Path, *, runtime: Mapping[str, object]
) -> dict[str, object]:
    if (
        tuple(runtime.get("generation_devices", ())) != ("cuda:0", "cuda:1")
        or runtime.get("cuda_visible_devices") != "0,1"
        or int(runtime.get("classifier_workers", -1)) != CLASSIFIER_WORKERS
        or int(runtime.get("classifier_threads_per_worker", -1))
        != CLASSIFIER_THREADS_PER_WORKER
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("tf32_disabled_in_gpu_workers") is not True
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("generated_cache_format") != "float32_npy_memmap"
    ):
        raise ProtocolError("Case-OOF workstation contract drifted.")
    if "spawn" not in mp.get_all_start_methods():
        raise ProtocolError("Case-OOF runtime requires multiprocessing spawn.")
    mismatched = {
        key: os.environ.get(key)
        for key, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatched or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise ProtocolError(
            "Case-OOF deterministic environment is not active; launch through "
            "`workspace run`."
        )
    cpu_count = _available_cpu_count()
    ram_bytes = _physical_ram_bytes()
    disk_bytes = int(shutil.disk_usage(artifact_root).free)
    if cpu_count < MINIMUM_LOGICAL_CPUS:
        raise ProtocolError("Case-OOF workstation exposes fewer than 12 CPUs.")
    if ram_bytes < MINIMUM_RAM_BYTES:
        raise ProtocolError("Case-OOF workstation exposes less than 100 GiB RAM.")
    if disk_bytes < MINIMUM_DISK_FREE_BYTES:
        raise ProtocolError("Case-OOF artifact filesystem has less than 8 GiB free.")
    versions = _package_versions()
    rows = _nvidia_smi_rows()
    by_index = {int(row["index"]): row for row in rows}
    if tuple(sorted(by_index)) != (0, 1):
        raise ProtocolError("Case-OOF runtime requires exactly CUDA 0 and 1.")
    for index in (0, 1):
        row = by_index[index]
        if "RTX A5000" not in str(row["name"]):
            raise ProtocolError("Case-OOF runtime requires RTX A5000 GPUs.")
        if int(row["memory_free_mib"]) < MINIMUM_GPU_FREE_MIB:
            raise ProtocolError(f"CUDA device {index} has insufficient free VRAM.")
    import torch

    if torch.cuda.is_initialized():
        raise ProtocolError("Parent CUDA context exists before case-OOF spawn.")
    return {
        "status": "PASS",
        "probe_method": "nvidia_smi_without_parent_cuda_context",
        "available_cpu_affinity_count": cpu_count,
        "physical_ram_bytes": ram_bytes,
        "disk_free_bytes_at_launch": disk_bytes,
        "classifier_worker_thread_product": CLASSIFIER_WORKERS
        * CLASSIFIER_THREADS_PER_WORKER,
        "multiprocessing_start_method": "spawn",
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "cuda_visible_devices": "0,1",
        "package_versions": versions,
        "dependency_version_policy": "presence_gate_versions_report_only",
        "tf32_disabled_in_gpu_workers": True,
        "gpus": [by_index[index] for index in (0, 1)],
        "parent_cuda_context_initialized": False,
    }


def _available_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if callable(affinity) else int(os.cpu_count() or 0)


def _physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as exc:
        raise ProtocolError("Cannot determine workstation RAM.") from exc


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProtocolError(f"Missing case-OOF dependency: {name}.") from exc
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
        raise ProtocolError("Cannot query case-OOF CUDA devices.") from exc
    if completed.returncode != 0:
        raise ProtocolError("nvidia-smi failed during case-OOF preflight.")
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
        raise ProtocolError("nvidia-smi returned malformed rows.") from exc
    return tuple(rows)


__all__ = ("run_workstation_preflight",)
