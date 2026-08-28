"""Pinned workstation topology and injectable, allocation-free preflight."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType
from typing import Callable, Mapping

from ....protocol import ProtocolError
from ..experiment_contracts import CANONICAL_SCRATCH_ROOT


MINIMUM_LOGICAL_CPUS = 12
MINIMUM_RAM_BYTES = 100 * 1024**3
MINIMUM_ARTIFACT_FREE_BYTES = 16 * 1024**3
MINIMUM_SCRATCH_FREE_BYTES = 32 * 1024**3
MINIMUM_GPU_FREE_MIB = 18_000


def workstation_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v4_workstation_runtime_v1",
        "profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_generation_workers": 2,
        "one_persistent_worker_per_physical_gpu": True,
        "gpu_generation_precedes_cpu_prediction": True,
        "generated_source_family_streams": 81,
        "full_source_rows_per_class": 1024,
        "exact_B_sources_per_target": 8,
        "exact_B_rows_per_source_per_class": 128,
        "prediction_store_dtype": "float32",
        "prediction_store_mode": "read_only_memmap",
        "prediction_store_materialized_once": True,
        "scientific_reduction_dtype": "float64",
        "cpu_prediction_workers": 4,
        "cpu_outer_center_workers": 4,
        "cpu_worker_task_unit": "one_complete_training_generation_seed_cell",
        "cpu_worker_task_count": 9,
        "blas_threads_per_worker": 1,
        "native_threads_per_worker": 1,
        "multiprocessing_start_method": "spawn",
        "top_level_spawn_pool_only": True,
        "nested_pools_allowed": False,
        "prelease_worker_runtime_smoke_required": True,
        "worker_initializer_configures_torch_threads_once": True,
        "parent_cuda_context_forbidden": True,
        "process_transport": ["paths", "hashes", "tuples", "scalars"],
        "estimator_objects_cross_process_allowed": False,
        "mappingproxy_cross_process_allowed": False,
        "minimum_logical_cpu_count": MINIMUM_LOGICAL_CPUS,
        "minimum_physical_ram_bytes": MINIMUM_RAM_BYTES,
        "minimum_artifact_disk_free_bytes": MINIMUM_ARTIFACT_FREE_BYTES,
        "minimum_scratch_disk_free_bytes": MINIMUM_SCRATCH_FREE_BYTES,
        "minimum_gpu_free_mib_per_device": MINIMUM_GPU_FREE_MIB,
        "scratch_preference": [CANONICAL_SCRATCH_ROOT, "artifact_parent"],
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "admission_allocates_gpu_memory": False,
    }


def validate_workstation_payload(payload: Mapping[str, object]) -> None:
    if dict(payload) != workstation_payload():
        raise ProtocolError("SCEPTRE v4 workstation topology drifted.")


@dataclass(frozen=True, slots=True)
class GpuFact:
    index: int
    name: str
    free_mib: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or self.index < 0
            or not self.name
            or isinstance(self.free_mib, bool)
            or self.free_mib < 0
        ):
            raise ProtocolError("SCEPTRE v4 GPU preflight fact drifted.")


@dataclass(frozen=True, slots=True)
class WorkstationFacts:
    logical_cpu_count: int
    physical_ram_bytes: int
    artifact_free_bytes: int
    scratch_free_bytes: int
    gpus: tuple[GpuFact, ...]

    def __post_init__(self) -> None:
        scalars = (
            self.logical_cpu_count,
            self.physical_ram_bytes,
            self.artifact_free_bytes,
            self.scratch_free_bytes,
        )
        if any(isinstance(value, bool) or int(value) < 0 for value in scalars):
            raise ProtocolError("SCEPTRE v4 workstation fact drifted.")


def probe_workstation_facts(artifact_root: Path, scratch_parent: Path) -> WorkstationFacts:
    """Read host facts without importing torch or allocating a CUDA context."""

    return WorkstationFacts(
        logical_cpu_count=len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else int(os.cpu_count() or 0),
        physical_ram_bytes=_physical_ram_bytes(),
        artifact_free_bytes=int(shutil.disk_usage(Path(artifact_root).parent).free),
        scratch_free_bytes=int(shutil.disk_usage(scratch_parent).free),
        gpus=_nvidia_smi_facts(),
    )


def run_workstation_preflight(
    artifact_root: Path,
    scratch_parent: Path,
    *,
    runtime: Mapping[str, object],
    probe: Callable[[Path, Path], WorkstationFacts] = probe_workstation_facts,
) -> Mapping[str, object]:
    """Validate production resources or an injected test probe, without writes."""

    validate_workstation_payload(runtime)
    facts = probe(Path(artifact_root), Path(scratch_parent))
    if (
        facts.logical_cpu_count < MINIMUM_LOGICAL_CPUS
        or facts.physical_ram_bytes < MINIMUM_RAM_BYTES
        or facts.artifact_free_bytes < MINIMUM_ARTIFACT_FREE_BYTES
        or facts.scratch_free_bytes < MINIMUM_SCRATCH_FREE_BYTES
        or len(facts.gpus) != 2
        or tuple(gpu.index for gpu in facts.gpus) != (0, 1)
        or any("RTX A5000" not in gpu.name for gpu in facts.gpus)
        or any(gpu.free_mib < MINIMUM_GPU_FREE_MIB for gpu in facts.gpus)
    ):
        raise ProtocolError("SCEPTRE v4 workstation preflight failed.")
    return MappingProxyType(
        {
            "schema_version": "sceptre_v4_workstation_preflight_v1",
            "status": "PASS",
            "logical_cpu_count": facts.logical_cpu_count,
            "physical_ram_bytes": facts.physical_ram_bytes,
            "artifact_free_bytes": facts.artifact_free_bytes,
            "scratch_free_bytes": facts.scratch_free_bytes,
            "gpus": [
                {"index": gpu.index, "name": gpu.name, "free_mib": gpu.free_mib}
                for gpu in facts.gpus
            ],
            "gpu_memory_allocated": False,
            "persistent_gpu_workers": 2,
            "spawn_cpu_workers": 4,
            "blas_threads_per_worker": 1,
            "cross_run_recovery_allowed": False,
        }
    )


def _physical_ram_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0
    return page_size * pages


def _nvidia_smi_facts() -> tuple[GpuFact, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    rows: list[GpuFact] = []
    try:
        for line in completed.stdout.splitlines():
            index, name, free = (part.strip() for part in line.split(",", 2))
            rows.append(GpuFact(int(index), name, int(free)))
    except (TypeError, ValueError):
        return ()
    return tuple(rows)


__all__ = (
    "GpuFact",
    "WorkstationFacts",
    "probe_workstation_facts",
    "run_workstation_preflight",
    "validate_workstation_payload",
    "workstation_payload",
)
