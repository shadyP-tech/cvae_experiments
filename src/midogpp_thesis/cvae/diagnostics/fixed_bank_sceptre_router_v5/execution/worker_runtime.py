"""Once-per-process Torch/CUDA setup and pre-lease worker lifecycle smoke.

The smoke uses the exact production GPU initializer, but never loads an
expert, opens target data, generates embeddings, writes artifacts, or opens
labels.  It exists to reject a broken persistent-worker lifecycle before the
single-use v5 authorization lease can be claimed.
"""

from __future__ import annotations

from concurrent.futures import Future, as_completed
import sys
from types import MappingProxyType
from typing import Mapping

from ....protocol import ProtocolError
from ...sceptre_runtime.worker_lifecycle import (
    SPAWN_START_METHOD,
    single_worker_spawn_executor,
)
from ..identity import canonical_hash, require_sha256
from ..physical.worker_runtime import (
    GPU_DEVICES,
    assert_gpu_worker_ready,
    initialize_gpu_worker,
)


SMOKE_TASKS_PER_DEVICE = 2
WORKER_RUNTIME_SMOKE_SCHEMA = "sceptre_v5_gpu_worker_runtime_smoke_v1"


def assert_parent_cuda_uninitialized() -> None:
    """Fail closed if this process already owns a CUDA context."""

    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    if not callable(is_initialized):
        raise ProtocolError("SCEPTRE v5 cannot authenticate parent CUDA state.")
    if bool(is_initialized()):
        raise ProtocolError("SCEPTRE v5 parent CUDA context was initialized.")


def run_gpu_worker_runtime_smoke(
    devices: tuple[str, ...] = GPU_DEVICES,
) -> Mapping[str, object]:
    """Submit two no-op probes to each exact production GPU initializer."""

    normalized = tuple(str(device) for device in devices)
    if normalized != GPU_DEVICES:
        raise ProtocolError("SCEPTRE v5 worker smoke topology drifted.")
    assert_parent_cuda_uninitialized()
    executors = []
    futures: dict[Future[Mapping[str, object]], tuple[str, int]] = {}
    try:
        for device in normalized:
            executors.append(
                single_worker_spawn_executor(
                    initializer=initialize_gpu_worker,
                    initargs=(device,),
                )
            )
        for device, executor in zip(normalized, executors, strict=True):
            for task_ordinal in range(SMOKE_TASKS_PER_DEVICE):
                future = executor.submit(_runtime_probe_task, device, task_ordinal)
                futures[future] = (device, task_ordinal)
        rows = tuple(future.result() for future in as_completed(futures))
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    assert_parent_cuda_uninitialized()

    ordered = tuple(
        sorted(
            (dict(row) for row in rows),
            key=lambda row: (str(row["device"]), int(row["task_ordinal"])),
        )
    )
    if len(ordered) != len(normalized) * SMOKE_TASKS_PER_DEVICE:
        raise ProtocolError("SCEPTRE v5 worker smoke coverage drifted.")
    process_ids: list[int] = []
    for device in normalized:
        device_rows = tuple(row for row in ordered if row["device"] == device)
        pids = {int(row["process_id"]) for row in device_rows}
        if (
            len(device_rows) != SMOKE_TASKS_PER_DEVICE
            or {int(row["task_ordinal"]) for row in device_rows} != {0, 1}
            or len(pids) != 1
            or any(row["initializer_invocation_count"] != 1 for row in device_rows)
        ):
            raise ProtocolError("SCEPTRE v5 persistent-worker smoke failed.")
        process_ids.extend(pids)
    if len(set(process_ids)) != len(normalized):
        raise ProtocolError("SCEPTRE v5 GPU pools unexpectedly share a process.")

    base = {
        "schema_version": WORKER_RUNTIME_SMOKE_SCHEMA,
        "status": "PASS",
        "execution_mode": SPAWN_START_METHOD,
        "gpu_devices": list(normalized),
        "persistent_worker_count": len(normalized),
        "max_workers_per_pool": 1,
        "task_count_per_worker": SMOKE_TASKS_PER_DEVICE,
        "initializer_invocation_count_per_worker": 1,
        "same_process_for_repeated_tasks": True,
        "distinct_process_per_gpu": True,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "child_cuda_context_initialized": True,
        "parent_cuda_context_initialized": False,
        "parent_cuda_state_checked_before_and_after_smoke": True,
        "scientific_gpu_work_performed": False,
        "experts_loaded": False,
        "embeddings_generated": False,
        "target_cache_opened": False,
        "target_manifest_opened": False,
        "target_labels_opened": False,
        "filesystem_mutations": 0,
        "probes": list(ordered),
    }
    return MappingProxyType(
        {**base, "worker_runtime_smoke_hash": canonical_hash(base)}
    )


def validate_worker_runtime_smoke(
    receipt: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate an in-memory smoke receipt before it is admission-bound."""

    if not isinstance(receipt, Mapping):
        raise ProtocolError("SCEPTRE v5 worker smoke receipt is malformed.")
    payload = dict(receipt)
    base = {
        key: value
        for key, value in payload.items()
        if key != "worker_runtime_smoke_hash"
    }
    if (
        payload.get("schema_version") != WORKER_RUNTIME_SMOKE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("gpu_devices") != list(GPU_DEVICES)
        or payload.get("persistent_worker_count") != 2
        or payload.get("task_count_per_worker") != 2
        or payload.get("initializer_invocation_count_per_worker") != 1
        or payload.get("same_process_for_repeated_tasks") is not True
        or payload.get("distinct_process_per_gpu") is not True
        or payload.get("child_cuda_context_initialized") is not True
        or payload.get("parent_cuda_context_initialized") is not False
        or payload.get("parent_cuda_state_checked_before_and_after_smoke") is not True
        or payload.get("scientific_gpu_work_performed") is not False
        or payload.get("experts_loaded") is not False
        or payload.get("embeddings_generated") is not False
        or payload.get("target_cache_opened") is not False
        or payload.get("target_manifest_opened") is not False
        or payload.get("target_labels_opened") is not False
        or payload.get("filesystem_mutations") != 0
        or payload.get("worker_runtime_smoke_hash") != canonical_hash(base)
    ):
        raise ProtocolError("SCEPTRE v5 worker smoke receipt failed validation.")
    require_sha256(
        payload["worker_runtime_smoke_hash"], "worker runtime smoke hash"
    )
    return MappingProxyType(payload)


def _runtime_probe_task(device: str, task_ordinal: int) -> Mapping[str, object]:
    binding = dict(assert_gpu_worker_ready(device))
    if task_ordinal not in (0, 1):
        raise ProtocolError("SCEPTRE v5 worker smoke task drifted.")
    return {
        "device": binding["device"],
        "device_index": binding["device_index"],
        "process_id": binding["process_id"],
        "initializer_invocation_count": binding["initializer_invocation_count"],
        "torch_intraop_threads": binding["torch_intraop_threads"],
        "torch_interop_threads": binding["torch_interop_threads"],
        "tf32_enabled": binding["tf32_enabled"],
        "amp_enabled": binding["amp_enabled"],
        "task_ordinal": task_ordinal,
    }


__all__ = (
    "GPU_DEVICES",
    "SMOKE_TASKS_PER_DEVICE",
    "WORKER_RUNTIME_SMOKE_SCHEMA",
    "assert_parent_cuda_uninitialized",
    "assert_gpu_worker_ready",
    "initialize_gpu_worker",
    "run_gpu_worker_runtime_smoke",
    "validate_worker_runtime_smoke",
)
