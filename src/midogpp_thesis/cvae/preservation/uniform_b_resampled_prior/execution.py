"""Workstation-aware runtime controls outside the scientific contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

from ...protocol import ProtocolError
from .config import UniformBResampledPriorConfig


SCORING_WORKERS_ENV = "MIDOGPP_RESAMPLED_PRIOR_SCORING_WORKERS"
TRAINING_DEVICES_ENV = "MIDOGPP_RESAMPLED_PRIOR_TRAINING_DEVICES"
_CUDA_DEVICE = re.compile(r"cuda:(\d+)")


@dataclass(frozen=True)
class RuntimePlan:
    scoring_workers: int
    training_devices: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_resampled_prior_runtime_plan_v1",
            "scoring_workers": self.scoring_workers,
            "training_devices": list(self.training_devices),
            "scientific_contract_unchanged": True,
            "deterministic_result_order": True,
            "one_training_process_per_device": len(self.training_devices) > 1,
            "unique_score_reuse": True,
            "mixed_precision": False,
            "tf32": False,
        }


def resolve_runtime_plan(config: UniformBResampledPriorConfig) -> RuntimePlan:
    try:
        workers = int(os.environ.get(SCORING_WORKERS_ENV, "1"))
    except ValueError as exc:
        raise ProtocolError(f"{SCORING_WORKERS_ENV} must be an integer.") from exc
    if not 1 <= workers <= 24:
        raise ProtocolError(f"{SCORING_WORKERS_ENV} must be in [1, 24].")
    raw_devices = os.environ.get(TRAINING_DEVICES_ENV, "").strip()
    devices = (
        tuple(item.strip() for item in raw_devices.split(",") if item.strip())
        if raw_devices
        else (str(config.device),)
    )
    if not devices or len(devices) != len(set(devices)):
        raise ProtocolError("Training devices must be nonempty and unique.")
    if len(devices) > 1 and not all(_CUDA_DEVICE.fullmatch(item) for item in devices):
        raise ProtocolError("Multi-device training requires explicit CUDA devices.")
    if any(item != "cpu" and _CUDA_DEVICE.fullmatch(item) is None for item in devices):
        raise ProtocolError("Training devices must be 'cpu' or 'cuda:N'.")
    if any(_CUDA_DEVICE.fullmatch(item) for item in devices):
        import torch

        if not torch.cuda.is_available():
            raise ProtocolError("CUDA runtime requested but unavailable.")
        available = torch.cuda.device_count()
        if any(int(item.split(":", 1)[1]) >= available for item in devices):
            raise ProtocolError("Requested CUDA device index is unavailable.")
    return RuntimePlan(workers, devices)


def partition_panel_tasks(
    tasks: tuple[tuple[str, int], ...],
    devices: tuple[str, ...],
) -> dict[str, tuple[tuple[str, int], ...]]:
    if not tasks or not devices:
        raise ProtocolError("Panel partition requires tasks and devices.")
    result = {
        device: tuple(tasks[index::len(devices)])
        for index, device in enumerate(devices)
    }
    flattened = [task for device in devices for task in result[device]]
    if sorted(flattened) != sorted(tasks) or len(flattened) != len(tasks):
        raise ProtocolError("Panel partition lost or duplicated a task.")
    return result


__all__ = (
    "RuntimePlan",
    "SCORING_WORKERS_ENV",
    "TRAINING_DEVICES_ENV",
    "partition_panel_tasks",
    "resolve_runtime_plan",
)
