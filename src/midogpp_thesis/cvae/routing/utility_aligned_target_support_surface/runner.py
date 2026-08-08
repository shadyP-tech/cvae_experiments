"""Fail-closed workstation runner for the target-support surface."""

from __future__ import annotations

import os
import json
import subprocess
import sys
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .config import TargetSupportSurfaceConfig, require_target_support_inputs_ready
from .contracts import REQUIRED_FILES
from .production import materialize_target_support_surface, validate_target_support_surface_bundle
from .workspace_binding import validate_production_workspace_binding


def run_utility_aligned_target_support_surface(
    config: TargetSupportSurfaceConfig,
    *,
    workspace_validator: Callable[[TargetSupportSurfaceConfig], None] = validate_production_workspace_binding,
) -> dict[str, object]:
    workspace_validator(config)
    if all((config.artifact_root / member).is_file() for member in REQUIRED_FILES):
        return dict(validate_target_support_surface_bundle(config.artifact_root))
    state_path = config.artifact_root / "reports/run_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("Target-support partial run-state is unreadable.") from exc
        if isinstance(state, Mapping) and state.get("status") == "COMPLETE":
            raise ProtocolError(
                "Target-support COMPLETE artifact is incomplete; refusing silent regeneration."
            )
    require_target_support_inputs_ready(config)
    _validate_workstation()
    root = materialize_target_support_surface(config)
    return dict(validate_target_support_surface_bundle(root))


def _validate_workstation() -> None:
    torch_module = sys.modules.get("torch")
    if torch_module is not None and getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_initialized():
        raise ProtocolError("Target-support parent process must remain CUDA-free.")
    if int(os.cpu_count() or 0) < 24:
        raise ProtocolError("Target-support workstation requires 24 logical CPUs.")
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE")); pages = int(os.sysconf("SC_PHYS_PAGES"))
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        values = [tuple(int(item.strip()) for item in line.split(",")) for line in completed.stdout.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ProtocolError("Cannot validate target-support workstation resources.") from exc
    if page_size * pages < 100 * 1024**3 or len(values) != 2 or any(total < 24_000 or free < 18_000 for total, free in values):
        raise ProtocolError("Target-support workstation capacity/free-GPU gate failed.")


__all__ = ("run_utility_aligned_target_support_surface",)
