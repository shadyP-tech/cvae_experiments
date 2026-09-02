"""Mutation-free admission checks for the dedicated HARP v9 workstation."""

from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

from ...protocol import ProtocolError


_EXPECTED_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    # The parent and the small nested-LODO jobs remain single threaded.  The
    # physical classifier initializer raises its own local limit to three.
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONHASHSEED": "0",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}
_REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "threadpoolctl",
    "torch",
)
_MINIMUM_CPU_AFFINITY = 12
_MINIMUM_RAM_BYTES = 100 * 1024**3
_MINIMUM_SCRATCH_FREE_BYTES = 100 * 1024**3
_MINIMUM_GPU_FREE_MIB = 20_000
_SCRATCH_ROOT = Path("/data/local/fixed_bank_harp_router_v9")


def inspect_harp_v9_workstation(runtime: Mapping[str, object]) -> Mapping[str, object]:
    """Validate live resources without creating output, scratch, or CUDA state."""

    if (
        runtime.get("profile") != "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
        or runtime.get("gpu_devices") != ["cuda:0", "cuda:1"]
        or runtime.get("persistent_gpu_workers") != 2
        or runtime.get("global_parent_blas_threads") != 1
        or runtime.get("classifier_workers") != 4
        or runtime.get("classifier_blas_threads_per_worker") != 3
        or runtime.get("science_workers") != 4
        or runtime.get("science_blas_threads_per_worker") != 1
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("parent_cuda_context_created") is not False
        or runtime.get("late_torch_interop_setter_used") is not False
        or runtime.get("probability_transport_dtype") != "float32"
        or runtime.get("scientific_reduction_dtype") != "float64"
        or runtime.get("memory_mapped_surfaces") is not True
        or runtime.get("bounded_inflight_batches_per_gpu") != 2
        or runtime.get("bounded_inflight_classifier_tasks_per_worker") != 2
        or runtime.get("bounded_inflight_science_tasks_per_worker") != 1
        or Path(str(runtime.get("scratch_root"))) != _SCRATCH_ROOT
    ):
        raise ProtocolError("HARP v9 live workstation profile drifted.")
    if "spawn" not in mp.get_all_start_methods():
        raise ProtocolError("HARP v9 workstation requires multiprocessing spawn.")

    mismatched = {
        key: os.environ.get(key)
        for key, expected in _EXPECTED_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatched:
        raise ProtocolError(
            "HARP v9 deterministic environment is absent; launch through workspace run."
        )

    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("HARP v9 parent process already owns a CUDA context.")

    cpu_count = _available_cpu_count()
    ram_bytes = _physical_ram_bytes()
    scratch_probe = _safe_nearest_existing_parent(_SCRATCH_ROOT)
    scratch_free_bytes = int(shutil.disk_usage(scratch_probe).free)
    if cpu_count < _MINIMUM_CPU_AFFINITY:
        raise ProtocolError("HARP v9 workstation exposes too few CPU threads.")
    if ram_bytes < _MINIMUM_RAM_BYTES:
        raise ProtocolError("HARP v9 workstation exposes too little physical RAM.")
    if not os.access(scratch_probe, os.W_OK | os.X_OK):
        raise ProtocolError("HARP v9 local scratch parent is not writable.")
    if scratch_free_bytes < _MINIMUM_SCRATCH_FREE_BYTES:
        raise ProtocolError("HARP v9 local scratch reserve is too low.")

    gpu_rows = _nvidia_smi_rows()
    by_index = {int(row["index"]): row for row in gpu_rows}
    if len(gpu_rows) != 2 or len(by_index) != 2 or tuple(sorted(by_index)) != (0, 1):
        raise ProtocolError("HARP v9 requires exactly CUDA devices 0 and 1.")
    for index in (0, 1):
        row = by_index[index]
        if "RTX A5000" not in str(row["name"]):
            raise ProtocolError("HARP v9 requires two RTX A5000 GPUs.")
        if int(row["memory_free_mib"]) < _MINIMUM_GPU_FREE_MIB:
            raise ProtocolError(f"HARP v9 CUDA device {index} has insufficient free VRAM.")

    return {
        "schema_version": "midogpp_harp_v9_live_workstation_preflight_v1",
        "status": "PASS",
        "persistent_gpu_workers": 2,
        "gpu_devices": ["cuda:0", "cuda:1"],
        "global_parent_blas_threads": 1,
        "classifier_workers": 4,
        "classifier_blas_threads_per_worker": 3,
        "science_workers": 4,
        "science_blas_threads_per_worker": 1,
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "physical_expert_weight": 1.0,
        "tf32_enabled": False,
        "amp_enabled": False,
        "parent_cuda_context_created": False,
        "shared_validated_menu_index": True,
        "labels_consumed": False,
        "available_cpu_affinity_count": cpu_count,
        "physical_ram_bytes": ram_bytes,
        "scratch_probe_path": str(scratch_probe),
        "scratch_free_bytes": scratch_free_bytes,
        "minimum_scratch_free_bytes": _MINIMUM_SCRATCH_FREE_BYTES,
        "thread_environment": dict(_EXPECTED_ENVIRONMENT),
        "package_versions": _package_versions(),
        "gpus": [by_index[index] for index in (0, 1)],
        "filesystem_mutations": 0,
    }


def _available_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if callable(affinity) else int(os.cpu_count() or 0)


def _physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as exc:
        raise ProtocolError("Cannot determine HARP v9 workstation RAM.") from exc


def _safe_nearest_existing_parent(path: Path) -> Path:
    if not path.is_absolute() or path != _SCRATCH_ROOT:
        raise ProtocolError("HARP v9 scratch root is not the dedicated local path.")
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    if not current.is_dir() or current.is_symlink():
        raise ProtocolError("HARP v9 scratch parent is absent or unsafe.")
    for parent in (current, *current.parents):
        if parent.exists() and parent.is_symlink():
            raise ProtocolError("HARP v9 scratch path traverses a symbolic link.")
        if parent == Path("/"):
            break
    return current.resolve()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProtocolError(f"Missing HARP v9 runtime dependency: {name}.") from exc
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
        raise ProtocolError("Cannot query HARP v9 CUDA devices.") from exc
    if completed.returncode != 0:
        raise ProtocolError("nvidia-smi failed during HARP v9 preflight.")
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
        raise ProtocolError("nvidia-smi returned malformed HARP v9 rows.") from exc
    return tuple(rows)


__all__ = ("inspect_harp_v9_workstation",)
