"""Fail-fast workstation checks without creating a parent CUDA context."""

from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from ...protocol import ProtocolError
from .contracts import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    MINIMUM_WORKSTATION_DISK_FREE_BYTES,
    MINIMUM_WORKSTATION_GPU_FREE_MIB,
    MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT,
    MINIMUM_WORKSTATION_RAM_BYTES,
)


EXPECTED_CUDA_INDICES = (0, 1)
EXPECTED_GPU_NAME_TOKEN = "RTX A5000"
REQUIRED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
REQUIRED_DISTRIBUTIONS = ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch")


def run_workstation_preflight(
    artifact_root: Path,
    *,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Verify the frozen W-2265 / dual-A5000 execution contract early.

    GPU visibility and free memory are queried through ``nvidia-smi`` so this
    check does not create a CUDA context in the parent process.  Worker code
    remains responsible for setting its own CUDA device after ``spawn``.
    """

    if (
        tuple(str(value) for value in runtime.get("generation_devices", ()))
        != ("cuda:0", "cuda:1")
        or tuple(str(value) for value in runtime.get("kernel_devices", ()))
        != ("cuda:0", "cuda:1")
        or runtime.get("cuda_visible_devices") != "0,1"
        or int(runtime.get("classifier_workers", -1)) != CLASSIFIER_WORKERS
        or int(runtime.get("classifier_threads_per_worker", -1))
        != CLASSIFIER_THREADS_PER_WORKER
        or CLASSIFIER_WORKERS * CLASSIFIER_THREADS_PER_WORKER != 12
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("tf32_disabled_in_gpu_workers") is not True
        or runtime.get("dependency_version_policy")
        != "presence_gate_versions_report_only"
        or int(runtime.get("minimum_logical_cpu_count", -1))
        != MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT
        or int(runtime.get("minimum_physical_ram_bytes", -1))
        != MINIMUM_WORKSTATION_RAM_BYTES
        or int(runtime.get("minimum_artifact_disk_free_bytes", -1))
        != MINIMUM_WORKSTATION_DISK_FREE_BYTES
        or int(runtime.get("minimum_gpu_free_mib_per_device", -1))
        != MINIMUM_WORKSTATION_GPU_FREE_MIB
    ):
        raise ProtocolError("Antisymmetric workstation runtime contract drifted.")
    if "spawn" not in mp.get_all_start_methods():
        raise ProtocolError("Antisymmetric runtime requires multiprocessing spawn.")
    mismatched_environment = {
        key: os.environ.get(key)
        for key, expected in REQUIRED_THREAD_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatched_environment:
        raise ProtocolError(
            "Antisymmetric deterministic thread environment is not active: "
            f"{mismatched_environment}. Launch through `workspace run`."
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != runtime["cuda_visible_devices"]:
        raise ProtocolError(
            "Antisymmetric runtime requires CUDA devices 0 and 1 in canonical order; "
            "launch through `workspace run`."
        )

    logical_cpus = _available_cpu_count()
    ram_bytes = _physical_ram_bytes()
    disk_free_bytes = int(shutil.disk_usage(artifact_root).free)
    if logical_cpus < MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT:
        raise ProtocolError("Antisymmetric workstation exposes fewer than 12 CPUs.")
    if ram_bytes < MINIMUM_WORKSTATION_RAM_BYTES:
        raise ProtocolError("Antisymmetric workstation exposes less than 100 GiB RAM.")
    if disk_free_bytes < MINIMUM_WORKSTATION_DISK_FREE_BYTES:
        raise ProtocolError("Antisymmetric artifact filesystem has less than 8 GiB free.")
    _probe_atomic_rename(artifact_root)

    package_versions = _package_versions()
    gpu_rows = _nvidia_smi_rows()
    by_index = {int(row["index"]): row for row in gpu_rows}
    if tuple(sorted(by_index)) != EXPECTED_CUDA_INDICES:
        raise ProtocolError(
            "Antisymmetric workstation requires exactly visible CUDA devices 0 and 1."
        )
    for index in EXPECTED_CUDA_INDICES:
        row = by_index[index]
        if EXPECTED_GPU_NAME_TOKEN not in str(row["name"]):
            raise ProtocolError(
                f"CUDA device {index} is not the frozen RTX A5000 workstation GPU."
            )
        if int(row["memory_free_mib"]) < MINIMUM_WORKSTATION_GPU_FREE_MIB:
            raise ProtocolError(
                f"CUDA device {index} has insufficient free VRAM for this run."
            )

    # Importing torch is already unavoidable through the generation modules,
    # but this read-only predicate does not initialize CUDA.  A pre-existing
    # parent context would make spawn/hash behavior unsafe, so fail closed.
    import torch

    if torch.cuda.is_initialized():
        raise ProtocolError(
            "CUDA was initialized in the parent before worker spawn."
        )
    return {
        "status": "PASS",
        "probe_method": "nvidia_smi_without_parent_cuda_context",
        "available_cpu_affinity_count": logical_cpus,
        "physical_ram_bytes": ram_bytes,
        "disk_free_bytes_at_launch": disk_free_bytes,
        "minimum_disk_free_bytes": MINIMUM_WORKSTATION_DISK_FREE_BYTES,
        "classifier_worker_thread_product": (
            CLASSIFIER_WORKERS * CLASSIFIER_THREADS_PER_WORKER
        ),
        "multiprocessing_start_method": "spawn",
        "thread_environment": dict(REQUIRED_THREAD_ENVIRONMENT),
        "cuda_visible_devices": visible_devices,
        "package_versions": package_versions,
        "dependency_version_policy": "presence_gate_versions_report_only",
        "tf32_disabled_in_gpu_workers": True,
        "gpus": [by_index[index] for index in EXPECTED_CUDA_INDICES],
        "minimum_gpu_free_mib": MINIMUM_WORKSTATION_GPU_FREE_MIB,
        "parent_cuda_context_initialized": False,
        "atomic_rename_probe_passed": True,
    }


def _physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as exc:
        raise ProtocolError("Cannot determine workstation physical memory.") from exc


def _available_cpu_count() -> int:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            return len(get_affinity(0))
        except OSError as exc:
            raise ProtocolError("Cannot determine workstation CPU affinity.") from exc
    return int(os.cpu_count() or 0)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProtocolError(
                f"Antisymmetric runtime dependency is missing: {distribution}."
            ) from exc
    return versions


def _probe_atomic_rename(root: Path) -> None:
    source = root / f".runtime_preflight.{os.getpid()}.source.tmp"
    target = root / f".runtime_preflight.{os.getpid()}.target.tmp"
    try:
        with source.open("wb") as handle:
            handle.write(b"antisymmetric-runtime-preflight\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(source, target)
        if target.read_bytes() != b"antisymmetric-runtime-preflight\n":
            raise OSError("atomic rename probe content drifted")
    except OSError as exc:
        raise ProtocolError(
            "Antisymmetric artifact filesystem failed its atomic rename probe."
        ) from exc
    finally:
        source.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def _nvidia_smi_rows() -> tuple[dict[str, object], ...]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtocolError("Cannot query the frozen CUDA workstation.") from exc
    if completed.returncode != 0:
        raise ProtocolError(
            "nvidia-smi failed during antisymmetric workstation preflight."
        )
    rows: list[dict[str, object]] = []
    try:
        for raw_line in completed.stdout.splitlines():
            values = [value.strip() for value in raw_line.split(",")]
            if len(values) != 4:
                raise ValueError("wrong column count")
            rows.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "memory_total_mib": int(values[2]),
                    "memory_free_mib": int(values[3]),
                }
            )
    except ValueError as exc:
        raise ProtocolError("nvidia-smi returned malformed GPU rows.") from exc
    if not rows:
        raise ProtocolError("nvidia-smi returned no visible GPUs.")
    return tuple(rows)


__all__ = ("run_workstation_preflight",)
