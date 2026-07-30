"""Hardware-aware runtime controls outside the scientific config identity."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

from ...protocol import ProtocolError
from .config import UniformBTaskGeometryConfig


SCORING_WORKERS_ENV = "MIDOGPP_UNIFORM_B_SCORING_WORKERS"
TRAINING_DEVICES_ENV = "MIDOGPP_UNIFORM_B_TRAINING_DEVICES"
_CUDA_DEVICE = re.compile(r"cuda:(\d+)")


@dataclass(frozen=True)
class RuntimePlan:
    scoring_workers: int
    training_devices: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_runtime_plan_v1",
            "scoring_workers": self.scoring_workers,
            "training_devices": list(self.training_devices),
            "scientific_contract_unchanged": True,
            "deterministic_result_order": True,
            "one_training_process_per_device": len(self.training_devices) > 1,
            "mixed_precision": False,
            "tf32": False,
        }


def resolve_runtime_plan(config: UniformBTaskGeometryConfig) -> RuntimePlan:
    """Resolve execution-only controls without changing ``config.contract_hash``."""

    raw_workers = os.environ.get(SCORING_WORKERS_ENV, "1").strip()
    try:
        scoring_workers = int(raw_workers)
    except ValueError as exc:
        raise ProtocolError(
            f"{SCORING_WORKERS_ENV} must be an integer."
        ) from exc
    if not 1 <= scoring_workers <= 32:
        raise ProtocolError(
            f"{SCORING_WORKERS_ENV} must be in [1, 32]."
        )

    raw_devices = os.environ.get(TRAINING_DEVICES_ENV, "").strip()
    devices = (
        tuple(part.strip() for part in raw_devices.split(",") if part.strip())
        if raw_devices
        else (str(config.device),)
    )
    if not devices or len(set(devices)) != len(devices):
        raise ProtocolError("Uniform-B runtime devices must be nonempty and unique.")
    if len(devices) > 1 and not all(_CUDA_DEVICE.fullmatch(item) for item in devices):
        raise ProtocolError("Multiple Uniform-B training devices must be explicit CUDA devices.")
    if any(item != "cpu" and _CUDA_DEVICE.fullmatch(item) is None for item in devices):
        raise ProtocolError(
            "Uniform-B training devices must be 'cpu' or explicit 'cuda:N' values."
        )
    if any(_CUDA_DEVICE.fullmatch(item) for item in devices):
        _validate_cuda_devices(devices)
    return RuntimePlan(
        scoring_workers=scoring_workers,
        training_devices=devices,
    )


def partition_panel_tasks(
    tasks: tuple[tuple[str, int], ...],
    devices: tuple[str, ...],
) -> dict[str, tuple[tuple[str, int], ...]]:
    """Assign stable round-robin panel tasks to device-bound workers."""

    if not tasks or not devices:
        raise ProtocolError("Panel task partition requires tasks and devices.")
    partitions: dict[str, list[tuple[str, int]]] = {
        device: [] for device in devices
    }
    for index, task in enumerate(tasks):
        partitions[devices[index % len(devices)]].append(task)
    flattened = [
        task
        for device in devices
        for task in partitions[device]
    ]
    if sorted(flattened) != sorted(tasks) or len(flattened) != len(tasks):
        raise ProtocolError("Panel runtime partition lost or duplicated a task.")
    return {
        device: tuple(partitions[device])
        for device in devices
    }


def _validate_cuda_devices(devices: tuple[str, ...]) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ProtocolError("CUDA runtime planning requires torch.") from exc
    if not torch.cuda.is_available():
        raise ProtocolError("Uniform-B runtime requested CUDA but CUDA is unavailable.")
    available = int(torch.cuda.device_count())
    indices = [
        int(match.group(1))
        for item in devices
        if (match := _CUDA_DEVICE.fullmatch(item)) is not None
    ]
    if any(index >= available for index in indices):
        raise ProtocolError(
            f"Uniform-B runtime requested CUDA indices {indices}, available={available}."
        )


__all__ = (
    "RuntimePlan",
    "SCORING_WORKERS_ENV",
    "TRAINING_DEVICES_ENV",
    "partition_panel_tasks",
    "resolve_runtime_plan",
)
