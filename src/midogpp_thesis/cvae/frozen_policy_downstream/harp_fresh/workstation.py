"""HARP-owned Stage-70 workstation admission and scratch controls.

The hardware probes are intentionally generic, while runtime validation is
bound to the HARP contract.  In particular, this module must not inherit the
source-block or cache-format semantics of another Stage-70 experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import SOURCE_ROWS_PER_CLASS
from ..fresh_runtime_contract import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    MINIMUM_ARTIFACT_DISK_FREE_BYTES,
    MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    MINIMUM_LOGICAL_CPU_COUNT,
    MINIMUM_PHYSICAL_RAM_BYTES,
    OPTIONAL_LOCAL_SCRATCH_ROOT,
)
from .config import HARP_SOURCE_CACHE_FORMAT, canonical_harp_runtime_payload


EXPECTED_GPU_INDICES = (0, 1)
EXPECTED_GPU_NAME_TOKEN = "RTX A5000"
REQUIRED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CANONICAL_PUBLICATION_MODE = (
    "sha256_validated_copy_to_canonical_sibling_then_atomic_replace"
)


@dataclass(frozen=True)
class WorkstationSnapshot:
    """Hardware state collected without creating a parent CUDA context."""

    available_cpu_count: int
    physical_ram_bytes: int
    artifact_disk_free_bytes: int
    gpu_rows: tuple[Mapping[str, object], ...]
    spawn_available: bool
    parent_cuda_context_initialized: bool


@dataclass(frozen=True)
class WorkstationProbes:
    """Injectable probes keep admission independently unit-testable."""

    available_cpu_count: Callable[[], int]
    physical_ram_bytes: Callable[[], int]
    disk_free_bytes: Callable[[Path], int]
    gpu_rows: Callable[[], Sequence[Mapping[str, object]]]
    spawn_available: Callable[[], bool]
    parent_cuda_context_initialized: Callable[[], bool]
    directory_writable: Callable[[Path], bool]
    atomic_replace_supported: Callable[[Path], bool]


def default_workstation_probes() -> WorkstationProbes:
    """Return production probes that do not initialize CUDA in the parent."""

    return WorkstationProbes(
        available_cpu_count=_available_cpu_count,
        physical_ram_bytes=_physical_ram_bytes,
        disk_free_bytes=lambda path: int(shutil.disk_usage(path).free),
        gpu_rows=_nvidia_smi_rows,
        spawn_available=lambda: "spawn" in mp.get_all_start_methods(),
        parent_cuda_context_initialized=_parent_cuda_context_initialized,
        directory_writable=_directory_writable,
        atomic_replace_supported=_probe_atomic_replace,
    )


def collect_workstation_snapshot(
    artifact_root: str | Path,
    *,
    probes: WorkstationProbes | None = None,
) -> WorkstationSnapshot:
    """Collect resource state through the supplied label-free probes."""

    active = probes or default_workstation_probes()
    root = Path(artifact_root)
    try:
        return WorkstationSnapshot(
            available_cpu_count=int(active.available_cpu_count()),
            physical_ram_bytes=int(active.physical_ram_bytes()),
            artifact_disk_free_bytes=int(active.disk_free_bytes(root)),
            gpu_rows=tuple(dict(row) for row in active.gpu_rows()),
            spawn_available=bool(active.spawn_available()),
            parent_cuda_context_initialized=bool(
                active.parent_cuda_context_initialized()
            ),
        )
    except ProtocolError:
        raise
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Fresh HARP workstation probing failed.") from exc


def validate_workstation_snapshot(
    snapshot: WorkstationSnapshot,
    *,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Validate the dual-A5000 resources against the exact HARP runtime."""

    _validate_harp_runtime(runtime)
    if snapshot.available_cpu_count < MINIMUM_LOGICAL_CPU_COUNT:
        raise ProtocolError("Fresh HARP workstation exposes fewer than 12 CPUs.")
    if snapshot.physical_ram_bytes < MINIMUM_PHYSICAL_RAM_BYTES:
        raise ProtocolError("Fresh HARP workstation exposes less than 100 GiB RAM.")
    if snapshot.artifact_disk_free_bytes < MINIMUM_ARTIFACT_DISK_FREE_BYTES:
        raise ProtocolError(
            "Fresh HARP canonical artifact filesystem has less than 8 GiB free."
        )
    if snapshot.spawn_available is not True:
        raise ProtocolError("Fresh HARP requires multiprocessing spawn.")
    if snapshot.parent_cuda_context_initialized is not False:
        raise ProtocolError(
            "Fresh HARP parent CUDA context exists before worker spawn."
        )

    gpu_rows = tuple(snapshot.gpu_rows)
    try:
        by_index = {int(row["index"]): row for row in gpu_rows}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Fresh HARP GPU probe rows are malformed.") from exc
    if (
        len(gpu_rows) != len(EXPECTED_GPU_INDICES)
        or len(by_index) != len(EXPECTED_GPU_INDICES)
        or tuple(sorted(by_index)) != EXPECTED_GPU_INDICES
    ):
        raise ProtocolError("Fresh HARP requires exactly CUDA devices 0 and 1.")

    normalized_gpu_rows: list[dict[str, object]] = []
    for index in EXPECTED_GPU_INDICES:
        row = by_index[index]
        try:
            name = str(row["name"])
            free_mib = int(row["memory_free_mib"])
            total_mib = int(row["memory_total_mib"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Fresh HARP GPU probe rows are malformed.") from exc
        if EXPECTED_GPU_NAME_TOKEN not in name:
            raise ProtocolError(
                f"Fresh HARP CUDA device {index} is not an RTX A5000."
            )
        if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
            raise ProtocolError("Fresh HARP GPU memory probe is malformed.")
        if free_mib < MINIMUM_GPU_FREE_MIB_PER_DEVICE:
            raise ProtocolError(
                f"Fresh HARP CUDA device {index} has less than 18 GiB free."
            )
        normalized_gpu_rows.append(
            {
                "index": index,
                "name": name,
                "memory_total_mib": total_mib,
                "memory_free_mib": free_mib,
            }
        )

    return {
        "available_cpu_count": snapshot.available_cpu_count,
        "physical_ram_bytes": snapshot.physical_ram_bytes,
        "artifact_disk_free_bytes": snapshot.artifact_disk_free_bytes,
        "gpus": normalized_gpu_rows,
        "minimum_gpu_free_mib_per_device": MINIMUM_GPU_FREE_MIB_PER_DEVICE,
        "spawn_available": True,
        "parent_cuda_context_initialized": False,
    }


def run_workstation_preflight(
    artifact_root: str | Path,
    *,
    runtime: Mapping[str, object],
    probes: WorkstationProbes | None = None,
    environment: Mapping[str, str] | None = None,
    enable_optional_local_scratch: bool = False,
) -> dict[str, object]:
    """Admit HARP's spawn topology without making scratch authoritative."""

    _validate_harp_runtime(runtime)
    if not isinstance(enable_optional_local_scratch, bool):
        raise ProtocolError("Fresh HARP scratch opt-in must be boolean.")
    active = probes or default_workstation_probes()
    observed_environment = os.environ if environment is None else environment
    mismatched = {
        key: observed_environment.get(key)
        for key, expected in REQUIRED_ENVIRONMENT.items()
        if observed_environment.get(key) != expected
    }
    if mismatched:
        raise ProtocolError(
            "Fresh HARP deterministic environment is not active: "
            f"{mismatched}. Launch through `workspace run`."
        )

    root = Path(artifact_root).expanduser().resolve()
    if not active.directory_writable(root):
        raise ProtocolError(
            "Fresh HARP canonical artifact root must exist and be writable."
        )
    snapshot = collect_workstation_snapshot(root, probes=active)
    hardware = validate_workstation_snapshot(snapshot, runtime=runtime)
    if not active.atomic_replace_supported(root):
        raise ProtocolError(
            "Fresh HARP canonical artifact root lacks atomic publication."
        )

    scratch_root: Path | None = None
    if enable_optional_local_scratch:
        scratch_root = Path(str(runtime["optional_local_scratch_root"]))
        if str(scratch_root) != OPTIONAL_LOCAL_SCRATCH_ROOT:
            raise ProtocolError("Fresh HARP optional scratch root drifted.")
        if not active.directory_writable(scratch_root):
            raise ProtocolError(
                "Fresh HARP /data/local scratch was requested but is not writable."
            )
        if not active.atomic_replace_supported(scratch_root):
            raise ProtocolError(
                "Fresh HARP /data/local scratch lacks atomic checkpoints."
            )

    return {
        "status": "PASS",
        "schema_version": "midogpp_harp_fresh_workstation_preflight_v1",
        "probe_method": "injected_or_nvidia_smi_without_parent_cuda_context",
        **hardware,
        "environment": dict(REQUIRED_ENVIRONMENT),
        "generation_devices": list(runtime["generation_devices"]),
        "generation_workers_per_device": int(
            runtime["generation_workers_per_device"]
        ),
        "generation_worker_count": (
            len(tuple(runtime["generation_devices"]))
            * int(runtime["generation_workers_per_device"])
        ),
        "source_block_per_class": int(runtime["source_block_per_class"]),
        "source_cache_format": str(runtime["source_cache_format"]),
        "prediction_cache_format": str(runtime["prediction_cache_format"]),
        "classifier_workers": int(runtime["classifier_workers"]),
        "classifier_threads_per_worker": int(
            runtime["classifier_threads_per_worker"]
        ),
        "classifier_worker_thread_product": (
            int(runtime["classifier_workers"])
            * int(runtime["classifier_threads_per_worker"])
        ),
        "multiprocessing_start_method": str(
            runtime["multiprocessing_start_method"]
        ),
        "tf32_enabled": False,
        "amp_enabled": False,
        "gpu_and_cpu_phases_disjoint": True,
        "phase_order": ["gpu_source_generation", "cpu_classifier_prediction"],
        "canonical_artifact_root": str(root),
        "canonical_atomic_replace_probe_passed": True,
        "optional_local_scratch_enabled": scratch_root is not None,
        "optional_local_scratch_root": (
            None if scratch_root is None else str(scratch_root)
        ),
        "scratch_authoritative": False,
        "canonical_publication_required": True,
        "canonical_publication_mode": CANONICAL_PUBLICATION_MODE,
    }


def _validate_harp_runtime(runtime: Mapping[str, object]) -> None:
    if not isinstance(runtime, Mapping):
        raise ProtocolError("Fresh HARP workstation runtime must be a mapping.")
    if runtime.get("source_block_per_class") != SOURCE_ROWS_PER_CLASS:
        raise ProtocolError("Fresh HARP source block must contain 270 rows per class.")
    if runtime.get("source_cache_format") != HARP_SOURCE_CACHE_FORMAT:
        raise ProtocolError("Fresh HARP frozen source-cache format drifted.")
    if not _strict_equal(runtime, canonical_harp_runtime_payload()):
        raise ProtocolError("Fresh HARP workstation runtime contract drifted.")


def _strict_equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    if isinstance(expected, float):
        return isinstance(observed, float) and observed == expected
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(observed) == set(expected)
            and all(
                _strict_equal(observed[key], expected_value)
                for key, expected_value in expected.items()
            )
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and len(observed) == len(expected)
            and all(
                _strict_equal(left, right)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    return observed == expected


def _available_cpu_count() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            return len(affinity(0))
        except OSError as exc:
            raise ProtocolError("Cannot determine fresh HARP CPU affinity.") from exc
    return int(os.cpu_count() or 0)


def _physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot determine fresh HARP physical RAM.") from exc


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
        raise ProtocolError("Cannot query fresh HARP CUDA devices.") from exc
    if completed.returncode != 0:
        raise ProtocolError("nvidia-smi failed during fresh HARP preflight.")
    rows: list[dict[str, object]] = []
    try:
        for line in completed.stdout.splitlines():
            values = [value.strip() for value in line.split(",")]
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
        raise ProtocolError("nvidia-smi returned malformed HARP rows.") from exc
    if not rows:
        raise ProtocolError("nvidia-smi returned no HARP GPUs.")
    return tuple(rows)


def _parent_cuda_context_initialized() -> bool:
    import torch

    return bool(torch.cuda.is_initialized())


def _directory_writable(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def _probe_atomic_replace(root: Path) -> bool:
    source: Path | None = None
    target: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".harp-fresh-stage70-preflight.",
            suffix=".source.tmp",
            dir=root,
            delete=False,
        ) as handle:
            source = Path(handle.name)
            handle.write(b"harp-fresh-stage70-atomic-publication\n")
            handle.flush()
            os.fsync(handle.fileno())
        target = source.with_suffix(".target.tmp")
        os.replace(source, target)
        source = None
        return target.read_bytes() == b"harp-fresh-stage70-atomic-publication\n"
    except OSError:
        return False
    finally:
        if source is not None:
            source.unlink(missing_ok=True)
        if target is not None:
            target.unlink(missing_ok=True)


__all__ = (
    "CANONICAL_PUBLICATION_MODE",
    "EXPECTED_GPU_INDICES",
    "EXPECTED_GPU_NAME_TOKEN",
    "REQUIRED_ENVIRONMENT",
    "WorkstationProbes",
    "WorkstationSnapshot",
    "collect_workstation_snapshot",
    "default_workstation_probes",
    "run_workstation_preflight",
    "validate_workstation_snapshot",
)
