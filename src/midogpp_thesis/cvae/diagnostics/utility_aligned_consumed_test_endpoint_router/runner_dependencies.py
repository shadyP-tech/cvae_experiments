"""Typed dependency surface for the endpoint-router orchestration layer.

The runner deliberately depends on phase-sized functions.  This keeps the
state machine testable without CUDA while the production defaults remain the
same concrete, hash-valid runtime used on the workstation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ConsumedTestEndpointRouterRunnerDependencies:
    """Science-compute substitutions only; protocol gates are not injectable."""

    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    produce_candidate_feature_rows: Callable[..., object] | None = None
    stage_target_embeddings: Callable[..., object] | None = None
    build_development_plan: Callable[..., object] | None = None
    materialize_development: Callable[..., object] | None = None
    build_target_plan: Callable[..., object] | None = None
    materialize_target: Callable[..., object] | None = None
    produce_support_shifts: Callable[..., object] | None = None
    run_prelabel_science: Callable[..., object] | None = None
    run_terminal_science: Callable[..., object] | None = None
    phase_observer: Callable[[str], None] | None = None


# Compatibility alias retained for the already-landed lazy package export.
ConsumedTestEndpointRouterDependencies = (
    ConsumedTestEndpointRouterRunnerDependencies
)


__all__ = (
    "ConsumedTestEndpointRouterDependencies",
    "ConsumedTestEndpointRouterRunnerDependencies",
)
