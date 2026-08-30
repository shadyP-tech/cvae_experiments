"""Read-only resource-capacity admission before the irreversible v4 lease."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .run_paths import assert_no_symlink_chain, validate_absolute_path


MINIMUM_GPU_FREE_MIB = 18_000
MINIMUM_RAM_AVAILABLE_BYTES = 32 * 1024**3
MINIMUM_ARTIFACT_FREE_BYTES = 16 * 1024**3
MINIMUM_SCRATCH_FREE_BYTES = 32 * 1024**3


@dataclass(frozen=True, slots=True)
class GpuCapacity:
    index: int
    name: str
    total_mib: int
    free_mib: int

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or not str(self.name)
            or type(self.total_mib) is not int
            or type(self.free_mib) is not int
            or not 0 <= self.free_mib <= self.total_mib
        ):
            raise ProtocolError("OE-PPUR v4 GPU capacity observation drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
        }


@dataclass(frozen=True, slots=True)
class ResourceCapacityReceipt:
    gpus: tuple[GpuCapacity, ...]
    ram_available_bytes: int
    artifact_free_bytes: int
    scratch_free_bytes: int
    artifact_device: int
    scratch_device: int
    filesystem_mutation_performed: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        gpus = tuple(self.gpus)
        shared = self.artifact_device == self.scratch_device
        disk_ok = (
            self.artifact_free_bytes
            >= MINIMUM_ARTIFACT_FREE_BYTES + MINIMUM_SCRATCH_FREE_BYTES
            if shared
            else self.artifact_free_bytes >= MINIMUM_ARTIFACT_FREE_BYTES
            and self.scratch_free_bytes >= MINIMUM_SCRATCH_FREE_BYTES
        )
        if (
            len(gpus) < 2
            or tuple(row.index for row in gpus[:2]) != (0, 1)
            or any("A5000" not in row.name.upper() for row in gpus[:2])
            or any(row.free_mib < MINIMUM_GPU_FREE_MIB for row in gpus[:2])
            or type(self.ram_available_bytes) is not int
            or self.ram_available_bytes < MINIMUM_RAM_AVAILABLE_BYTES
            or type(self.artifact_free_bytes) is not int
            or type(self.scratch_free_bytes) is not int
            or type(self.artifact_device) is not int
            or type(self.scratch_device) is not int
            or not disk_ok
            or self.filesystem_mutation_performed is not False
        ):
            raise ProtocolError("OE-PPUR v4 workstation capacity is insufficient.")
        object.__setattr__(self, "gpus", gpus)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_resource_capacity_receipt_v1",
            "gpus": [row.to_payload() for row in self.gpus],
            "minimum_gpu_free_mib": MINIMUM_GPU_FREE_MIB,
            "ram_available_bytes": self.ram_available_bytes,
            "minimum_ram_available_bytes": MINIMUM_RAM_AVAILABLE_BYTES,
            "artifact_free_bytes": self.artifact_free_bytes,
            "minimum_artifact_free_bytes": MINIMUM_ARTIFACT_FREE_BYTES,
            "scratch_free_bytes": self.scratch_free_bytes,
            "minimum_scratch_free_bytes": MINIMUM_SCRATCH_FREE_BYTES,
            "artifact_device": self.artifact_device,
            "scratch_device": self.scratch_device,
            "shared_artifact_scratch_filesystem": (
                self.artifact_device == self.scratch_device
            ),
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def validate_capacity_observation(
    observed: Mapping[str, object],
) -> ResourceCapacityReceipt:
    """Validate a primitive observation without permitting caller-side bypass."""

    rows = observed.get("gpus") if isinstance(observed, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ProtocolError("OE-PPUR v4 capacity observation is malformed.")
    try:
        gpus = tuple(
            GpuCapacity(
                index=int(row["index"]),
                name=str(row["name"]),
                total_mib=int(row["total_mib"]),
                free_mib=int(row["free_mib"]),
            )
            for row in rows
            if isinstance(row, Mapping)
        )
        if len(gpus) != len(rows):
            raise ValueError("non-mapping GPU row")
        return ResourceCapacityReceipt(
            gpus=gpus,
            ram_available_bytes=int(observed["ram_available_bytes"]),
            artifact_free_bytes=int(observed["artifact_free_bytes"]),
            scratch_free_bytes=int(observed["scratch_free_bytes"]),
            artifact_device=int(observed["artifact_device"]),
            scratch_device=int(observed["scratch_device"]),
            filesystem_mutation_performed=False,
        )
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 capacity observation is malformed.") from exc


def preflight_resource_capacity(
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> ResourceCapacityReceipt:
    """Probe GPU, RAM, and both filesystems without creating any path."""

    artifact = validate_absolute_path(artifact_root, role="capacity artifact root")
    scratch = validate_absolute_path(scratch_root, role="capacity scratch root")
    assert_no_symlink_chain(artifact, allow_missing_leaf=True)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    artifact_parent = _nearest_existing_directory(artifact)
    scratch_parent = _nearest_existing_directory(scratch)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        gpu_rows = _parse_gpu_rows(completed.stdout)
        artifact_usage = shutil.disk_usage(artifact_parent)
        scratch_usage = shutil.disk_usage(scratch_parent)
        ram_available = _read_mem_available_bytes()
        artifact_device = os.stat(artifact_parent).st_dev
        scratch_device = os.stat(scratch_parent).st_dev
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("OE-PPUR v4 capacity probe failed.") from exc
    return validate_capacity_observation(
        {
            "gpus": gpu_rows,
            "ram_available_bytes": ram_available,
            "artifact_free_bytes": int(artifact_usage.free),
            "scratch_free_bytes": int(scratch_usage.free),
            "artifact_device": int(artifact_device),
            "scratch_device": int(scratch_device),
        }
    )


def _parse_gpu_rows(value: str) -> tuple[dict[str, object], ...]:
    rows = []
    for raw in value.splitlines():
        parts = tuple(part.strip() for part in raw.split(","))
        if len(parts) != 4:
            raise ProtocolError("OE-PPUR v4 GPU capacity output is malformed.")
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mib": int(parts[2]),
                    "free_mib": int(parts[3]),
                }
            )
        except ValueError as exc:
            raise ProtocolError(
                "OE-PPUR v4 GPU capacity output is malformed."
            ) from exc
    return tuple(rows)


def _read_mem_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    return int(parts[1]) * 1024
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v4 RAM capacity probe failed.") from exc
    raise ProtocolError("OE-PPUR v4 RAM capacity is unavailable.")


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while not current.exists():
        if current == current.parent:
            raise ProtocolError("OE-PPUR v4 scratch filesystem is unavailable.")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ProtocolError("OE-PPUR v4 scratch filesystem parent is unsafe.")
    return current


__all__ = (
    "GpuCapacity",
    "MINIMUM_ARTIFACT_FREE_BYTES",
    "MINIMUM_GPU_FREE_MIB",
    "MINIMUM_RAM_AVAILABLE_BYTES",
    "MINIMUM_SCRATCH_FREE_BYTES",
    "ResourceCapacityReceipt",
    "preflight_resource_capacity",
    "validate_capacity_observation",
)
