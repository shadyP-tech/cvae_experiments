"""Once-per-process Torch setup for SCEPTRE v5 physical GPU workers.

The initializer is deliberately v5-owned and is the only place that mutates
Torch's process-global thread/runtime state.  It runs in a freshly spawned
child before any expert is loaded or CUDA work begins.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Mapping

from midogpp_thesis.cvae.protocol import ProtocolError


GPU_DEVICES = ("cuda:0", "cuda:1")

_WORKER_BINDING: Mapping[str, object] | None = None


def initialize_gpu_worker(device: str) -> None:
    """Bind one fresh spawned process to one GPU exactly once.

    ``set_num_interop_threads`` must be the first Torch runtime mutation.  In
    particular, it must not run inside the per-task function after parallel
    work has begun; doing so caused the original workstation failure.
    """

    global _WORKER_BINDING
    if _WORKER_BINDING is not None:
        raise ProtocolError("SCEPTRE v5 GPU worker was initialized twice.")
    normalized = str(device)
    if normalized not in GPU_DEVICES:
        raise ProtocolError("SCEPTRE v5 GPU worker device drifted.")

    import torch

    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(normalized)

    expected_index = int(normalized.split(":", 1)[1])
    if (
        torch.get_num_interop_threads() != 1
        or torch.get_num_threads() != 1
        or int(torch.cuda.current_device()) != expected_index
    ):
        raise ProtocolError("SCEPTRE v5 GPU worker initialization drifted.")
    _WORKER_BINDING = MappingProxyType(
        {
            "device": normalized,
            "device_index": expected_index,
            "process_id": os.getpid(),
            "initializer_invocation_count": 1,
            "torch_intraop_threads": 1,
            "torch_interop_threads": 1,
            "tf32_enabled": False,
            "amp_enabled": False,
        }
    )


def assert_gpu_worker_ready(device: str) -> Mapping[str, object]:
    """Authenticate the initializer binding without reconfiguring Torch."""

    if _WORKER_BINDING is None:
        raise ProtocolError("SCEPTRE v5 GPU worker initializer was not run.")
    binding = dict(_WORKER_BINDING)
    normalized = str(device)
    if (
        binding.get("device") != normalized
        or binding.get("process_id") != os.getpid()
        or binding.get("initializer_invocation_count") != 1
    ):
        raise ProtocolError("SCEPTRE v5 GPU worker binding drifted.")

    import torch

    if (
        torch.get_num_interop_threads() != 1
        or torch.get_num_threads() != 1
        or int(torch.cuda.current_device()) != binding["device_index"]
        or torch.backends.cuda.matmul.allow_tf32 is not False
        or torch.backends.cudnn.allow_tf32 is not False
    ):
        raise ProtocolError("SCEPTRE v5 GPU worker runtime drifted.")
    return MappingProxyType(binding)


__all__ = (
    "GPU_DEVICES",
    "assert_gpu_worker_ready",
    "initialize_gpu_worker",
)
