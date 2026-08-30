"""Read-only OE-PPUR v4 workstation topology attestation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import subprocess
import sys

from ...protocol import ProtocolError
from .hashing import payload_sha256


EXPECTED_PYTHON = Path("/home/stud/spark/.venvs/cvae-breakhis/bin/python")


@dataclass(frozen=True, slots=True)
class WorkstationTopologyReceipt:
    hostname: str
    system: str
    machine: str
    python_executable: Path
    artifact_filesystem_type: str
    scratch_filesystem_type: str
    cpu_count: int
    memory_kib: int
    gpu_rows: tuple[tuple[str, str, int], ...]
    fuse_active_for_artifact_parent: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not self.hostname
            or self.system != "Linux"
            or self.machine not in {"x86_64", "amd64"}
            or not isinstance(self.python_executable, Path)
            or not self.python_executable.is_absolute()
            or self.python_executable != EXPECTED_PYTHON
            or self.artifact_filesystem_type not in {"nfs", "nfs4"}
            or not self.scratch_filesystem_type
            or type(self.cpu_count) is not int
            or self.cpu_count < 16
            or type(self.memory_kib) is not int
            or self.memory_kib < 100_000_000
            or type(self.gpu_rows) is not tuple
            or len(self.gpu_rows) != 2
            or tuple(row[0] for row in self.gpu_rows) != ("0", "1")
            or any("RTX A5000" not in row[1] or row[2] < 23_000 for row in self.gpu_rows)
            or self.fuse_active_for_artifact_parent is not False
        ):
            raise ProtocolError("OE-PPUR v4 workstation topology drifted.")
        object.__setattr__(self, "receipt_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_workstation_topology_receipt_v1",
            "hostname": self.hostname,
            "system": self.system,
            "machine": self.machine,
            "python_executable": self.python_executable.as_posix(),
            "artifact_filesystem_type": self.artifact_filesystem_type,
            "scratch_filesystem_type": self.scratch_filesystem_type,
            "cpu_count": self.cpu_count,
            "memory_kib": self.memory_kib,
            "gpu_rows": [
                {"index": index, "name": name, "memory_total_mib": memory}
                for index, name, memory in self.gpu_rows
            ],
            "fuse_active_for_artifact_parent": False,
            "gpu_worker_count": 2,
            "cpu_worker_count": 4,
            "blas_threads_per_cpu_worker": 1,
            "nested_process_pools_allowed": False,
            "topology_probe_mutated_filesystem": False,
        }


def capture_workstation_topology(
    *,
    artifact_parent: Path,
    scratch_root: Path,
) -> WorkstationTopologyReceipt:
    """Probe the exact xai-master/delli2 execution host without mutation."""

    artifact_fs = _filesystem_type(artifact_parent)
    scratch_parent = next(
        (parent for parent in (scratch_root, *scratch_root.parents) if parent.exists()),
        None,
    )
    if scratch_parent is None:
        raise ProtocolError("OE-PPUR v4 scratch filesystem is unavailable.")
    scratch_fs = _filesystem_type(scratch_parent)
    gpu_rows = _gpu_rows()
    return WorkstationTopologyReceipt(
        hostname=platform.node(),
        system=platform.system(),
        machine=platform.machine().lower(),
        python_executable=Path(sys.executable).resolve(),
        artifact_filesystem_type=artifact_fs,
        scratch_filesystem_type=scratch_fs,
        cpu_count=os.cpu_count() or 0,
        memory_kib=_memory_kib(),
        gpu_rows=gpu_rows,
        fuse_active_for_artifact_parent=artifact_fs.startswith("fuse"),
    )


def _filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ("findmnt", "-n", "-o", "FSTYPE", "--target", str(path)),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("OE-PPUR v4 filesystem topology probe failed.") from exc
    effective_rows = tuple(
        row.strip().lower()
        for row in result.splitlines()
        if row.strip() and row.strip().lower() != "autofs"
    )
    effective_types = tuple(dict.fromkeys(effective_rows))
    if len(effective_types) != 1:
        raise ProtocolError("OE-PPUR v4 filesystem topology is ambiguous.")
    return effective_types[0]


def _gpu_rows() -> tuple[tuple[str, str, int], ...]:
    try:
        raw = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("OE-PPUR v4 GPU topology probe failed.") from exc
    rows: list[tuple[str, str, int]] = []
    for line in raw.splitlines():
        parts = tuple(part.strip() for part in line.split(","))
        if len(parts) != 3:
            raise ProtocolError("OE-PPUR v4 GPU topology payload is malformed.")
        try:
            memory = int(parts[2])
        except ValueError as exc:
            raise ProtocolError("OE-PPUR v4 GPU memory payload is malformed.") from exc
        rows.append((parts[0], parts[1], memory))
    return tuple(rows)


def _memory_kib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 memory topology probe failed.") from exc
    raise ProtocolError("OE-PPUR v4 memory topology is unavailable.")


__all__ = (
    "EXPECTED_PYTHON",
    "WorkstationTopologyReceipt",
    "capture_workstation_topology",
)
