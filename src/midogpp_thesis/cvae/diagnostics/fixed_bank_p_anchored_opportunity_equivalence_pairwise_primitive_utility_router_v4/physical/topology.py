"""Spawn-safe workstation topology used by the OE-PPUR v4 physical adapter.

The execution coordinator may provide a richer sealed host receipt.  The
scientific/physical layer consumes only this immutable, path-free projection so
worker processes never receive authority objects or host mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256


GPU_NAMES = ("NVIDIA RTX A5000", "NVIDIA RTX A5000")
GPU_WORKER_COUNT = 2
CPU_SPAWN_WORKER_COUNT = 4
BLAS_THREADS_PER_CPU_WORKER = 1
MULTIPROCESSING_START_METHOD = "spawn"
CPU_WORKER_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class WorkstationTopologyReceipt:
    """Minimal structural receipt accepted by label-free science services."""

    upstream_receipt_hash: str
    hostname: str
    gpu_names: tuple[str, ...] = GPU_NAMES
    cpu_count: int = 32
    start_method: str = MULTIPROCESSING_START_METHOD
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        upstream = require_sha256(
            self.upstream_receipt_hash, "workstation upstream receipt hash"
        )
        names = tuple(str(value) for value in self.gpu_names)
        if (
            not self.hostname
            or names != GPU_NAMES
            or type(self.cpu_count) is not int
            or self.cpu_count < CPU_SPAWN_WORKER_COUNT
            or self.start_method != MULTIPROCESSING_START_METHOD
        ):
            raise ProtocolError("OE-PPUR v4 workstation topology drifted.")
        object.__setattr__(self, "upstream_receipt_hash", upstream)
        object.__setattr__(self, "gpu_names", names)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        body = {
            "schema_version": "oe_ppur_v4_science_workstation_topology_v1",
            "upstream_receipt_hash": self.upstream_receipt_hash,
            "hostname": self.hostname,
            "gpu_names": list(self.gpu_names),
            "persistent_gpu_worker_count": GPU_WORKER_COUNT,
            "cpu_count": self.cpu_count,
            "spawn_cpu_worker_count": CPU_SPAWN_WORKER_COUNT,
            "blas_threads_per_cpu_worker": BLAS_THREADS_PER_CPU_WORKER,
            "multiprocessing_start_method": MULTIPROCESSING_START_METHOD,
            "cuda_visible_to_cpu_workers": False,
            "cpu_worker_environment": dict(CPU_WORKER_ENVIRONMENT),
        }
        return {**body, "receipt_hash": self.receipt_hash} if include_hash else body


def project_workstation_topology(value: object) -> WorkstationTopologyReceipt:
    """Project a sealed host receipt without importing its concrete type."""

    if isinstance(value, WorkstationTopologyReceipt):
        return value
    receipt_hash = getattr(value, "receipt_hash", None)
    to_payload = getattr(value, "to_payload", None)
    if not isinstance(receipt_hash, str) or not callable(to_payload):
        raise ProtocolError("OE-PPUR v4 workstation receipt is untyped.")
    payload = to_payload()
    if not isinstance(payload, Mapping):
        raise ProtocolError("OE-PPUR v4 workstation payload is untyped.")
    gpu_names = payload.get("gpu_names", payload.get("gpu_model_names"))
    if gpu_names is None:
        gpu_rows = payload.get("gpu_rows")
        if not isinstance(gpu_rows, (list, tuple)) or any(
            not isinstance(row, Mapping) for row in gpu_rows
        ):
            raise ProtocolError("OE-PPUR v4 workstation GPU inventory is absent.")
        if tuple(str(row.get("index", "")) for row in gpu_rows) != ("0", "1"):
            raise ProtocolError("OE-PPUR v4 workstation GPU order drifted.")
        gpu_names = tuple(str(row.get("name", "")) for row in gpu_rows)
    if not isinstance(gpu_names, (list, tuple)):
        raise ProtocolError("OE-PPUR v4 workstation GPU inventory is absent.")
    hostname = payload.get("hostname", payload.get("host_id"))
    cpu_count = payload.get("cpu_count", payload.get("logical_cpu_count"))
    start_method = payload.get(
        "multiprocessing_start_method", payload.get("start_method")
    )
    if start_method is None:
        if (
            payload.get("cpu_worker_count") != CPU_SPAWN_WORKER_COUNT
            or payload.get("blas_threads_per_cpu_worker")
            != BLAS_THREADS_PER_CPU_WORKER
        ):
            raise ProtocolError("OE-PPUR v4 workstation CPU topology is absent.")
        start_method = MULTIPROCESSING_START_METHOD
    try:
        projected = WorkstationTopologyReceipt(
            upstream_receipt_hash=receipt_hash,
            hostname=str(hostname or ""),
            gpu_names=tuple(str(item) for item in gpu_names),
            cpu_count=int(cpu_count),
            start_method=str(start_method),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 workstation payload is malformed.") from exc
    return projected


__all__ = (
    "BLAS_THREADS_PER_CPU_WORKER",
    "CPU_SPAWN_WORKER_COUNT",
    "CPU_WORKER_ENVIRONMENT",
    "GPU_NAMES",
    "GPU_WORKER_COUNT",
    "MULTIPROCESSING_START_METHOD",
    "WorkstationTopologyReceipt",
    "project_workstation_topology",
)
