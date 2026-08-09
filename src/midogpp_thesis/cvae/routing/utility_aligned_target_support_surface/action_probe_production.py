"""Thin composition facade for target-support action-shift production."""

from __future__ import annotations

from pathlib import Path

from ..exact_tail_utility_surface.source_contracts import GeneratedDevelopmentCache
from .action_probe_contracts import ActionProbeRuntime, TargetSupportActionShiftRow
from .action_probe_execution import execute_or_resume_action_probes
from .action_probe_planning import build_action_probe_tasks
from .action_probe_surface import build_action_shift_rows
from .inputs import TargetSupportInputs


def materialize_target_action_shifts(
    inputs: TargetSupportInputs,
    generated: GeneratedDevelopmentCache,
    *,
    execution_root: Path,
    runtime: ActionProbeRuntime,
) -> tuple[TargetSupportActionShiftRow, ...]:
    tasks = build_action_probe_tasks(
        inputs,
        generated,
        checkpoint_root=execution_root / "action_probe_checkpoints",
        runtime=runtime,
    )
    checkpoints = execute_or_resume_action_probes(tasks, runtime=runtime)
    return build_action_shift_rows(tasks, checkpoints)


__all__ = ("materialize_target_action_shifts",)
