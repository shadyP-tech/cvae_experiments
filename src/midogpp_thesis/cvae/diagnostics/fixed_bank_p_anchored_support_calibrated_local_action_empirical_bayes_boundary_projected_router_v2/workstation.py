"""Deterministic two-GPU/four-CPU workstation topology for SCALE-BP v2."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .hashing import canonical_hash
from .identity import (
    EXPECTED_CASE_COUNT,
    EXPECTED_CENTER_COUNT,
    EXPECTED_PHYSICAL_CELL_COUNT,
    GovernanceError,
    SUPPORT_FOLD_COUNT,
)


WORKSTATION_PLAN_SCHEMA = "scale_bp_v2_workstation_plan_v1"
WORKSTATION_PREFLIGHT_SCHEMA = "scale_bp_v2_workstation_preflight_v1"
CPU_WORKER_ENV = "MIDOGPP_SCALE_BP_V2_CPU_OUTER_WORKER"
GPU_WORKER_ENV = "MIDOGPP_SCALE_BP_V2_GPU_WORKER"
BLAS_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
DETERMINISTIC_PARENT_ENVIRONMENT = {
    **{name: "1" for name in BLAS_ENVIRONMENT_NAMES},
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
}
_CPU_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True, slots=True)
class WorkstationPlan:
    profile: str
    generation_devices: tuple[str, str]
    persistent_generation_workers: int
    physical_cells_materialized_once: int
    cpu_outer_workers: int
    outer_task_count: int
    outer_task_unit: str
    support_folds_per_outer_task: int
    blas_threads_per_outer_worker: int
    storage_dtype: str
    reduction_dtype: str
    multiprocessing_start_method: str
    nested_process_pools_allowed: bool
    phase_disjoint_gpu_and_cpu_pools: bool
    cross_run_recovery_allowed: bool
    terminal_recovery_allowed: bool
    minimum_logical_cpu_count: int
    minimum_physical_ram_bytes: int
    minimum_gpu_free_mib_per_device: int
    minimum_artifact_disk_free_bytes: int
    minimum_scratch_disk_free_bytes: int
    execution_authorized: bool

    def to_payload(self) -> dict[str, object]:
        body = {
            "schema_version": WORKSTATION_PLAN_SCHEMA,
            "profile": self.profile,
            "generation_devices": list(self.generation_devices),
            "cuda_visible_devices": "0,1",
            "source_workers_per_device": 1,
            "persistent_generation_workers": self.persistent_generation_workers,
            "generation_workers_per_device": 1,
            "persistent_source_workers": True,
            "source_job_count": 27,
            "source_stream_count": 81,
            "source_prefix_rows_per_class": 270,
            "physical_cells_materialized_once": self.physical_cells_materialized_once,
            "physical_store": "FLOAT32_READ_ONLY_MEMMAP",
            "cpu_outer_workers": self.cpu_outer_workers,
            "classifier_workers": self.cpu_outer_workers,
            "classifier_threads_per_worker": 3,
            "launch_blas_threads": 1,
            "outer_task_count": self.outer_task_count,
            "outer_task_unit": self.outer_task_unit,
            "support_folds_per_outer_task": self.support_folds_per_outer_task,
            "support_folds_inside_outer_worker": "SEQUENTIAL",
            "blas_threads_per_outer_worker": self.blas_threads_per_outer_worker,
            "storage_dtype": self.storage_dtype,
            "reduction_dtype": self.reduction_dtype,
            "scientific_reductions_dtype": self.reduction_dtype,
            "generated_cache_format": "float32_npy_memmap",
            "target_task_count": 81,
            "target_action_identity_count": 90,
            "target_probability_cell_count": EXPECTED_PHYSICAL_CELL_COUNT,
            "target_unique_classifier_fit_count": EXPECTED_PHYSICAL_CELL_COUNT,
            "maximum_total_classifier_fit_count": EXPECTED_PHYSICAL_CELL_COUNT,
            "tf32_enabled": False,
            "amp_enabled": False,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "nested_process_pools_allowed": self.nested_process_pools_allowed,
            "phase_disjoint_gpu_and_cpu_pools": (
                self.phase_disjoint_gpu_and_cpu_pools
            ),
            "parent_cuda_context_forbidden": True,
            "cpu_phase_cuda_visible_devices": "",
            "worker_payload": "PRIMITIVE_FROZEN_DTOS_HASHES_AND_OFFSETS_ONLY",
            "mappingproxy_estimator_handle_or_memmap_cross_process_forbidden": True,
            "atomic_outer_center_chunks": True,
            "cross_run_recovery_allowed": self.cross_run_recovery_allowed,
            "terminal_recovery_allowed": self.terminal_recovery_allowed,
            "minimum_logical_cpu_count": self.minimum_logical_cpu_count,
            "minimum_physical_ram_bytes": self.minimum_physical_ram_bytes,
            "minimum_gpu_free_mib_per_device": (
                self.minimum_gpu_free_mib_per_device
            ),
            "minimum_artifact_disk_free_bytes": (
                self.minimum_artifact_disk_free_bytes
            ),
            "minimum_scratch_disk_free_bytes": self.minimum_scratch_disk_free_bytes,
            "execution_authorized": self.execution_authorized,
        }
        return {**body, "plan_hash": canonical_hash(body)}


@dataclass(frozen=True, slots=True)
class GPUProbe:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int

    def to_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_free_mib": self.memory_free_mib,
        }


@dataclass(frozen=True, slots=True)
class WorkstationProbe:
    logical_cpu_count: int
    physical_ram_bytes: int
    artifact_disk_free_bytes: int
    scratch_disk_free_bytes: int
    spawn_available: bool
    parent_cuda_initialized: bool
    gpus: tuple[GPUProbe, ...]
    package_versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class WorkstationPreflightReceipt:
    payload: dict[str, object]

    @property
    def receipt_hash(self) -> str:
        return str(self.payload["receipt_hash"])

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def canonical_workstation_plan() -> WorkstationPlan:
    """Return the frozen topology without probing or mutating the host."""

    return WorkstationPlan(
        profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        generation_devices=("cuda:0", "cuda:1"),
        persistent_generation_workers=2,
        physical_cells_materialized_once=EXPECTED_PHYSICAL_CELL_COUNT,
        cpu_outer_workers=4,
        outer_task_count=EXPECTED_CENTER_COUNT,
        outer_task_unit="ONE_COMPLETE_OUTER_CENTER_H",
        support_folds_per_outer_task=SUPPORT_FOLD_COUNT,
        blas_threads_per_outer_worker=1,
        storage_dtype="float32",
        reduction_dtype="float64",
        multiprocessing_start_method="spawn",
        nested_process_pools_allowed=False,
        phase_disjoint_gpu_and_cpu_pools=True,
        cross_run_recovery_allowed=False,
        terminal_recovery_allowed=False,
        minimum_logical_cpu_count=12,
        minimum_physical_ram_bytes=100 * 1024**3,
        minimum_gpu_free_mib_per_device=18_000,
        minimum_artifact_disk_free_bytes=20 * 1024**3,
        minimum_scratch_disk_free_bytes=50 * 1024**3,
        execution_authorized=True,
    )


def canonical_workstation_payload() -> dict[str, object]:
    return canonical_workstation_plan().to_payload()


def validate_workstation_plan(payload: dict[str, object]) -> None:
    if dict(payload) != canonical_workstation_payload():
        raise GovernanceError("SCALE-BP v2 workstation topology drifted.")


def preflight_workstation(
    artifact_root: str | Path,
    scratch_root: str | Path,
    *,
    probe: WorkstationProbe | None = None,
) -> WorkstationPreflightReceipt:
    """Validate the intended host without creating output, scratch, or locks."""

    plan = canonical_workstation_plan()
    artifact = _absolute_path(artifact_root, "artifact root")
    scratch = _absolute_path(scratch_root, "scratch root")
    observed = probe if probe is not None else _probe_host(artifact, scratch)
    if (
        observed.logical_cpu_count < plan.minimum_logical_cpu_count
        or observed.physical_ram_bytes < plan.minimum_physical_ram_bytes
        or observed.artifact_disk_free_bytes
        < plan.minimum_artifact_disk_free_bytes
        or observed.scratch_disk_free_bytes < plan.minimum_scratch_disk_free_bytes
        or not observed.spawn_available
        or observed.parent_cuda_initialized
        or tuple(gpu.index for gpu in observed.gpus) != (0, 1)
        or any("RTX A5000" not in gpu.name for gpu in observed.gpus)
        or any(
            gpu.memory_free_mib < plan.minimum_gpu_free_mib_per_device
            for gpu in observed.gpus
        )
    ):
        raise GovernanceError("SCALE-BP v2 workstation preflight failed.")
    package_versions = dict(observed.package_versions)
    if any(
        dependency not in package_versions
        for dependency in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch")
    ):
        raise GovernanceError("SCALE-BP v2 workstation dependency is absent.")
    body = {
        "schema_version": WORKSTATION_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "plan_hash": plan.to_payload()["plan_hash"],
        "artifact_root": str(artifact),
        "scratch_root": str(scratch),
        "logical_cpu_count": observed.logical_cpu_count,
        "physical_ram_bytes": observed.physical_ram_bytes,
        "artifact_disk_free_bytes": observed.artifact_disk_free_bytes,
        "scratch_disk_free_bytes": observed.scratch_disk_free_bytes,
        "spawn_available": observed.spawn_available,
        "parent_cuda_initialized": observed.parent_cuda_initialized,
        "gpus": [gpu.to_payload() for gpu in observed.gpus],
        "package_versions": package_versions,
        "gpu_then_cpu_phase_order": True,
        "persistent_gpu_worker_count": 2,
        "cpu_outer_worker_count": 4,
        "outer_task_count": EXPECTED_CENTER_COUNT,
        "route_count": EXPECTED_CASE_COUNT,
        "nested_process_pools": False,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "predecessor_scratch_or_checkpoint_used": False,
        "mutation_performed": False,
    }
    return WorkstationPreflightReceipt(
        {**body, "receipt_hash": canonical_hash(body)}
    )


run_workstation_preflight = preflight_workstation


def initialize_cpu_outer_worker() -> None:
    """Hide CUDA and cap native pools before importing numerical modules."""

    if os.environ.get(CPU_WORKER_ENV) == "1" or os.environ.get(GPU_WORKER_ENV):
        raise GovernanceError("SCALE-BP v2 nested/phase-mixed CPU worker refused.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    os.environ[CPU_WORKER_ENV] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - workstation dependency
        raise GovernanceError("SCALE-BP v2 requires threadpoolctl.") from exc
    global _CPU_THREADPOOL_LIMITER
    _CPU_THREADPOOL_LIMITER = threadpool_limits(limits=1)


initialize_cpu_worker = initialize_cpu_outer_worker


def initialize_gpu_worker(device: str) -> None:
    """Bind one persistent generation worker to exactly one A5000 device."""

    if os.environ.get(CPU_WORKER_ENV) == "1" or os.environ.get(GPU_WORKER_ENV):
        raise GovernanceError("SCALE-BP v2 nested/phase-mixed GPU worker refused.")
    if device not in {"cuda:0", "cuda:1"}:
        raise GovernanceError("SCALE-BP v2 GPU worker device drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    os.environ[GPU_WORKER_ENV] = device


def assert_coordinator_process() -> None:
    if os.environ.get(CPU_WORKER_ENV) == "1" or os.environ.get(GPU_WORKER_ENV):
        raise GovernanceError("SCALE-BP v2 nested process pools are forbidden.")


def _probe_host(artifact_root: Path, scratch_root: Path) -> WorkstationProbe:
    package_versions: list[tuple[str, str]] = []
    for dependency in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "torch"):
        try:
            package_versions.append(
                (dependency, importlib.metadata.version(dependency))
            )
        except importlib.metadata.PackageNotFoundError as exc:
            raise GovernanceError(
                f"SCALE-BP v2 missing workstation dependency: {dependency}."
            ) from exc
    affinity = getattr(os, "sched_getaffinity", None)
    cpu_count = len(affinity(0)) if callable(affinity) else int(os.cpu_count() or 0)
    try:
        ram_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as exc:
        raise GovernanceError("SCALE-BP v2 cannot determine workstation RAM.") from exc
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    parent_cuda_initialized = bool(
        cuda is not None and callable(getattr(cuda, "is_initialized", None))
        and cuda.is_initialized()
    )
    return WorkstationProbe(
        logical_cpu_count=cpu_count,
        physical_ram_bytes=ram_bytes,
        artifact_disk_free_bytes=int(
            shutil.disk_usage(_nearest_existing_parent(artifact_root)).free
        ),
        scratch_disk_free_bytes=int(
            shutil.disk_usage(_nearest_existing_parent(scratch_root)).free
        ),
        spawn_available="spawn" in mp.get_all_start_methods(),
        parent_cuda_initialized=parent_cuda_initialized,
        gpus=_probe_gpus(),
        package_versions=tuple(package_versions),
    )


def _probe_gpus() -> tuple[GPUProbe, ...]:
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
        raise GovernanceError("SCALE-BP v2 cannot query CUDA devices.") from exc
    if completed.returncode != 0:
        raise GovernanceError("SCALE-BP v2 nvidia-smi preflight failed.")
    rows: list[GPUProbe] = []
    try:
        for line in completed.stdout.splitlines():
            values = tuple(value.strip() for value in line.split(","))
            if len(values) != 4:
                raise ValueError
            rows.append(
                GPUProbe(
                    index=int(values[0]),
                    name=values[1],
                    memory_total_mib=int(values[2]),
                    memory_free_mib=int(values[3]),
                )
            )
    except ValueError as exc:
        raise GovernanceError("SCALE-BP v2 CUDA probe output drifted.") from exc
    return tuple(rows)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    if not current.exists():
        raise GovernanceError("SCALE-BP v2 cannot locate a filesystem parent.")
    return current


def _absolute_path(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor):
        raise GovernanceError(f"SCALE-BP v2 {role} is unsafe.")
    return path.resolve(strict=False)


__all__ = (
    "BLAS_ENVIRONMENT_NAMES",
    "CPU_WORKER_ENV",
    "DETERMINISTIC_PARENT_ENVIRONMENT",
    "GPUProbe",
    "GPU_WORKER_ENV",
    "WORKSTATION_PLAN_SCHEMA",
    "WORKSTATION_PREFLIGHT_SCHEMA",
    "WorkstationPlan",
    "WorkstationPreflightReceipt",
    "WorkstationProbe",
    "assert_coordinator_process",
    "canonical_workstation_payload",
    "canonical_workstation_plan",
    "initialize_cpu_outer_worker",
    "initialize_cpu_worker",
    "initialize_gpu_worker",
    "preflight_workstation",
    "run_workstation_preflight",
    "validate_workstation_plan",
)
