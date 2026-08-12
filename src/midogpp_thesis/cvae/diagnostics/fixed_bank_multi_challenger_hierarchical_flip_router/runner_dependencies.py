"""Dependency-injection seams for orchestration and leakage tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MultiChallengerRouterDependencies:
    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_donor_models: Callable[..., object] | None = None
    build_fold_decisions: Callable[..., object] | None = None
    evaluate_terminal: Callable[..., object] | None = None
    cleanup_staging: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


# Compatibility-shaped name for an orchestration module cloned from the prior
# diagnostic.  Both names denote the new experiment-local dependency contract.
FlipRouterDependencies = MultiChallengerRouterDependencies


__all__ = ("FlipRouterDependencies", "MultiChallengerRouterDependencies")
