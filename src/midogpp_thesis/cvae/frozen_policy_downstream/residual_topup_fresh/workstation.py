"""Workstation admission and non-authoritative scratch publication controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence

from ...protocol import ProtocolError
from .config import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    MINIMUM_ARTIFACT_DISK_FREE_BYTES,
    MINIMUM_GPU_FREE_MIB_PER_DEVICE,
    MINIMUM_LOGICAL_CPU_COUNT,
    MINIMUM_PHYSICAL_RAM_BYTES,
    OPTIONAL_LOCAL_SCRATCH_ROOT,
    SOURCE_BLOCK_PER_CLASS,
    canonical_runtime_payload,
)


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
    """Probe results kept separate from pure admission validation."""

    available_cpu_count: int
    physical_ram_bytes: int
    artifact_disk_free_bytes: int
    gpu_rows: tuple[Mapping[str, object], ...]
    spawn_available: bool
    parent_cuda_context_initialized: bool


@dataclass(frozen=True)
class WorkstationProbes:
    """Injectable hardware/filesystem probes for GPU-free unit tests."""

    available_cpu_count: Callable[[], int]
    physical_ram_bytes: Callable[[], int]
    disk_free_bytes: Callable[[Path], int]
    gpu_rows: Callable[[], Sequence[Mapping[str, object]]]
    spawn_available: Callable[[], bool]
    parent_cuda_context_initialized: Callable[[], bool]
    directory_writable: Callable[[Path], bool]
    atomic_replace_supported: Callable[[Path], bool]


def default_workstation_probes() -> WorkstationProbes:
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
    """Collect hardware state without creating a CUDA context in the parent."""

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
        raise ProtocolError("Fresh Stage-70 workstation probing failed.") from exc


def validate_workstation_snapshot(
    snapshot: WorkstationSnapshot,
    *,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Purely validate the frozen dual-A5000 resource contract."""

    _validate_runtime(runtime)
    if snapshot.available_cpu_count < MINIMUM_LOGICAL_CPU_COUNT:
        raise ProtocolError("Fresh Stage-70 workstation exposes fewer than 12 CPUs.")
    if snapshot.physical_ram_bytes < MINIMUM_PHYSICAL_RAM_BYTES:
        raise ProtocolError("Fresh Stage-70 workstation exposes less than 100 GiB RAM.")
    if snapshot.artifact_disk_free_bytes < MINIMUM_ARTIFACT_DISK_FREE_BYTES:
        raise ProtocolError(
            "Fresh Stage-70 canonical artifact filesystem has less than 8 GiB free."
        )
    if snapshot.spawn_available is not True:
        raise ProtocolError("Fresh Stage-70 requires multiprocessing spawn.")
    if snapshot.parent_cuda_context_initialized is not False:
        raise ProtocolError(
            "Fresh Stage-70 parent CUDA context exists before worker spawn."
        )

    gpu_rows = tuple(snapshot.gpu_rows)
    try:
        by_index = {int(row["index"]): row for row in gpu_rows}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Fresh Stage-70 GPU probe rows are malformed.") from exc
    if len(gpu_rows) != 2 or len(by_index) != 2 or tuple(sorted(by_index)) != (0, 1):
        raise ProtocolError(
            "Fresh Stage-70 requires exactly CUDA devices 0 and 1."
        )
    normalized_gpu_rows: list[dict[str, object]] = []
    for index in EXPECTED_GPU_INDICES:
        row = by_index[index]
        try:
            name = str(row["name"])
            free_mib = int(row["memory_free_mib"])
            total_mib = int(row["memory_total_mib"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Fresh Stage-70 GPU probe rows are malformed.") from exc
        if EXPECTED_GPU_NAME_TOKEN not in name:
            raise ProtocolError(
                f"Fresh Stage-70 CUDA device {index} is not an RTX A5000."
            )
        if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
            raise ProtocolError("Fresh Stage-70 GPU memory probe is malformed.")
        if free_mib < MINIMUM_GPU_FREE_MIB_PER_DEVICE:
            raise ProtocolError(
                f"Fresh Stage-70 CUDA device {index} has less than 18 GiB free."
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
    """Admit the frozen schedule without silently enabling local scratch."""

    _validate_runtime(runtime)
    if not isinstance(enable_optional_local_scratch, bool):
        raise ProtocolError("Fresh Stage-70 scratch opt-in must be boolean.")
    active = probes or default_workstation_probes()
    observed_environment = os.environ if environment is None else environment
    mismatched = {
        key: observed_environment.get(key)
        for key, expected in REQUIRED_ENVIRONMENT.items()
        if observed_environment.get(key) != expected
    }
    if mismatched:
        raise ProtocolError(
            "Fresh Stage-70 deterministic environment is not active: "
            f"{mismatched}. Launch through `workspace run`."
        )

    root = Path(artifact_root).expanduser().resolve()
    if not active.directory_writable(root):
        raise ProtocolError(
            "Fresh Stage-70 canonical artifact root must exist and be writable."
        )
    snapshot = collect_workstation_snapshot(root, probes=active)
    hardware = validate_workstation_snapshot(snapshot, runtime=runtime)
    if not active.atomic_replace_supported(root):
        raise ProtocolError(
            "Fresh Stage-70 canonical artifact root lacks atomic publication."
        )

    scratch_root: Path | None = None
    if enable_optional_local_scratch:
        scratch_root = Path(str(runtime["optional_local_scratch_root"]))
        if str(scratch_root) != OPTIONAL_LOCAL_SCRATCH_ROOT:
            raise ProtocolError("Fresh Stage-70 optional scratch root drifted.")
        if not active.directory_writable(scratch_root):
            raise ProtocolError(
                "Fresh Stage-70 /data/local scratch was requested but is not writable."
            )
        if not active.atomic_replace_supported(scratch_root):
            raise ProtocolError(
                "Fresh Stage-70 /data/local scratch lacks atomic checkpoints."
            )

    return {
        "status": "PASS",
        "probe_method": "injected_or_nvidia_smi_without_parent_cuda_context",
        **hardware,
        "environment": dict(REQUIRED_ENVIRONMENT),
        "generation_devices": ["cuda:0", "cuda:1"],
        "generation_workers_per_device": 1,
        "generation_worker_count": 2,
        "source_block_per_class": SOURCE_BLOCK_PER_CLASS,
        "classifier_workers": CLASSIFIER_WORKERS,
        "classifier_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
        "classifier_worker_thread_product": (
            CLASSIFIER_WORKERS * CLASSIFIER_THREADS_PER_WORKER
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


def publish_validated_scratch_file(
    scratch_file: str | Path,
    canonical_file: str | Path,
    *,
    expected_sha256: str,
    scratch_root: str | Path,
) -> Path:
    """Validate a scratch file and atomically publish a canonical copy.

    Scratch is input-only and never becomes the returned authoritative path.
    An existing canonical file is accepted only when its digest already
    matches; a conflicting canonical result fails closed.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ProtocolError("Fresh Stage-70 publication SHA-256 is invalid.")
    scratch_boundary = Path(scratch_root).expanduser().resolve()
    source = Path(scratch_file).expanduser().resolve()
    target = Path(canonical_file).expanduser().resolve()
    if not source.is_file() or not source.is_relative_to(scratch_boundary):
        raise ProtocolError(
            "Fresh Stage-70 publication source is outside explicit scratch."
        )
    if target.is_relative_to(scratch_boundary):
        raise ProtocolError(
            "Fresh Stage-70 canonical publication cannot target scratch."
        )
    if _sha256_file(source) != expected_sha256:
        raise ProtocolError("Fresh Stage-70 scratch publication hash mismatched.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or _sha256_file(target) != expected_sha256:
            raise ProtocolError(
                "Fresh Stage-70 canonical publication conflicts with existing output."
            )
        return target

    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".publish.tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            staged_path = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if _sha256_file(staged_path) != expected_sha256:
            raise ProtocolError(
                "Fresh Stage-70 canonical staged-copy validation failed."
            )
        os.replace(staged_path, target)
        staged_path = None
        _fsync_directory(target.parent)
        if _sha256_file(target) != expected_sha256:
            raise ProtocolError(
                "Fresh Stage-70 canonical publication validation failed."
            )
        return target
    except ProtocolError:
        raise
    except OSError as exc:
        raise ProtocolError("Fresh Stage-70 canonical publication failed.") from exc
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)


def _validate_runtime(runtime: Mapping[str, object]) -> None:
    if not isinstance(runtime, Mapping) or not _strict_equal(
        runtime, canonical_runtime_payload()
    ):
        raise ProtocolError("Fresh Stage-70 workstation runtime contract drifted.")


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
            raise ProtocolError(
                "Cannot determine fresh Stage-70 CPU affinity."
            ) from exc
    return int(os.cpu_count() or 0)


def _physical_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot determine fresh Stage-70 physical RAM.") from exc


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
        raise ProtocolError("Cannot query fresh Stage-70 CUDA devices.") from exc
    if completed.returncode != 0:
        raise ProtocolError("nvidia-smi failed during fresh Stage-70 preflight.")
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
        raise ProtocolError("nvidia-smi returned malformed Stage-70 rows.") from exc
    if not rows:
        raise ProtocolError("nvidia-smi returned no Stage-70 GPUs.")
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
            prefix=".fresh-stage70-preflight.",
            suffix=".source.tmp",
            dir=root,
            delete=False,
        ) as handle:
            source = Path(handle.name)
            handle.write(b"fresh-stage70-atomic-publication\n")
            handle.flush()
            os.fsync(handle.fileno())
        target = source.with_suffix(".target.tmp")
        os.replace(source, target)
        source = None
        return target.read_bytes() == b"fresh-stage70-atomic-publication\n"
    except OSError:
        return False
    finally:
        if source is not None:
            source.unlink(missing_ok=True)
        if target is not None:
            target.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "CANONICAL_PUBLICATION_MODE",
    "EXPECTED_GPU_INDICES",
    "EXPECTED_GPU_NAME_TOKEN",
    "REQUIRED_ENVIRONMENT",
    "WorkstationProbes",
    "WorkstationSnapshot",
    "collect_workstation_snapshot",
    "default_workstation_probes",
    "publish_validated_scratch_file",
    "run_workstation_preflight",
    "validate_workstation_snapshot",
)
