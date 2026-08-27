"""Read-only workstation topology admission for OE-PPUR v2."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .run_paths import validate_requested_run_root


GPU_WORKER_COUNT = 2
CPU_OUTER_WORKER_COUNT = 4
BLAS_THREADS_PER_CPU_WORKER = 1
MINIMUM_SCRATCH_FREE_BYTES = 80 * 1024**3


@dataclass(frozen=True, slots=True)
class WorkstationPreflightReceipt:
    gpu_count: int
    gpu_names: tuple[str, ...]
    cpu_count: int
    scratch_free_bytes: int
    artifact_parent: str
    scratch_parent: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.gpu_names)
        if (
            type(self.gpu_count) is not int
            or self.gpu_count < GPU_WORKER_COUNT
            or len(names) != self.gpu_count
            or not all("A5000" in value.upper() for value in names[:2])
            or type(self.cpu_count) is not int
            or self.cpu_count < CPU_OUTER_WORKER_COUNT
            or type(self.scratch_free_bytes) is not int
            or self.scratch_free_bytes < MINIMUM_SCRATCH_FREE_BYTES
            or not Path(self.artifact_parent).is_absolute()
            or not Path(self.scratch_parent).is_absolute()
        ):
            raise ProtocolError("OE-PPUR v2 workstation topology drifted.")
        object.__setattr__(self, "gpu_names", names)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_workstation_preflight_v1",
            "gpu_count": self.gpu_count,
            "gpu_names": list(self.gpu_names),
            "persistent_gpu_worker_count": GPU_WORKER_COUNT,
            "gpu_assignment": ["cuda:0", "cuda:1"],
            "cpu_count": self.cpu_count,
            "cpu_outer_worker_count": CPU_OUTER_WORKER_COUNT,
            "blas_threads_per_cpu_worker": BLAS_THREADS_PER_CPU_WORKER,
            "scratch_free_bytes": self.scratch_free_bytes,
            "minimum_scratch_free_bytes": MINIMUM_SCRATCH_FREE_BYTES,
            "artifact_parent": self.artifact_parent,
            "scratch_parent": self.scratch_parent,
            "prediction_storage": "ROW_SHARDED_RAW_LITTLE_ENDIAN_FLOAT32",
            "prediction_access": "READ_ONLY_MEMMAP_AFTER_PARSE",
            "reductions_dtype": "float64",
            "multiprocessing_start_method": "spawn",
            "nested_process_pools_allowed": False,
            "cuda_visible_to_cpu_outer_workers": False,
            "host_probe_only": True,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def preflight_workstation(
    artifact_root: str | Path,
    scratch_root: str | Path,
    *,
    observed: dict[str, object] | None = None,
) -> WorkstationPreflightReceipt:
    """Probe resources without creating output, scratch, or lease state."""

    artifact = validate_requested_run_root(
        artifact_root,
        role="artifact root",
        allow_workspace_envelope=True,
    )
    scratch = validate_requested_run_root(
        scratch_root,
        role="scratch root",
        allow_workspace_envelope=False,
    )
    if artifact == scratch or artifact in scratch.parents or scratch in artifact.parents:
        raise ProtocolError("OE-PPUR v2 output and scratch roots overlap.")
    artifact_parent = artifact.parent
    scratch_parent = scratch.parent
    if (
        not artifact_parent.is_dir()
        or artifact_parent.is_symlink()
        or not scratch_parent.is_dir()
        or scratch_parent.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v2 run-root parent is absent or unsafe.")
    probe = _live_probe(scratch_parent) if observed is None else dict(observed)
    names_value = probe.get("gpu_names")
    if not isinstance(names_value, (list, tuple)):
        raise ProtocolError("OE-PPUR v2 GPU probe is malformed.")
    return WorkstationPreflightReceipt(
        gpu_count=int(probe.get("gpu_count", -1)),
        gpu_names=tuple(str(value) for value in names_value),
        cpu_count=int(probe.get("cpu_count", -1)),
        scratch_free_bytes=int(probe.get("scratch_free_bytes", -1)),
        artifact_parent=str(artifact_parent),
        scratch_parent=str(scratch_parent),
    )


def workstation_plan_payload() -> dict[str, object]:
    body = {
        "schema_version": "oe_ppur_v2_workstation_plan_v1",
        "persistent_gpu_workers": GPU_WORKER_COUNT,
        "gpu_devices": ["cuda:0", "cuda:1"],
        "cpu_outer_workers": CPU_OUTER_WORKER_COUNT,
        "blas_threads_per_cpu_worker": BLAS_THREADS_PER_CPU_WORKER,
        "prediction_matrix_dtype": "<f4",
        "prediction_matrix_memory_order": "C",
        "prediction_matrix_access": "read_only_memmap_after_parse",
        "reduction_dtype": "float64",
        "multiprocessing_start_method": "spawn",
        "process_transport": ["paths", "hashes", "tuples", "scalars"],
        "nested_process_pools_allowed": False,
        "cross_run_recovery_allowed": False,
    }
    return {**body, "plan_hash": canonical_hash(body)}


def _live_probe(scratch_parent: Path) -> dict[str, object]:
    try:
        import torch

        count = int(torch.cuda.device_count())
        names = tuple(str(torch.cuda.get_device_name(index)) for index in range(count))
    except (ImportError, RuntimeError, AssertionError) as exc:
        raise ProtocolError("OE-PPUR v2 CUDA workstation probe failed.") from exc
    disk = os.statvfs(scratch_parent)
    return {
        "gpu_count": count,
        "gpu_names": names,
        "cpu_count": int(os.cpu_count() or 0),
        "scratch_free_bytes": int(disk.f_bavail * disk.f_frsize),
    }


__all__ = (
    "BLAS_THREADS_PER_CPU_WORKER",
    "CPU_OUTER_WORKER_COUNT",
    "GPU_WORKER_COUNT",
    "MINIMUM_SCRATCH_FREE_BYTES",
    "WorkstationPreflightReceipt",
    "preflight_workstation",
    "workstation_plan_payload",
)
