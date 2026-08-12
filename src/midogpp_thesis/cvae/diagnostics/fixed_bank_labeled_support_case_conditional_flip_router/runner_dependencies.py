"""Dependency injection seams for phase-order and leakage tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FlipRouterDependencies:
    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_donor_models: Callable[..., object] | None = None
    build_fold_decisions: Callable[..., object] | None = None
    evaluate_terminal: Callable[..., object] | None = None
    cleanup_staging: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


__all__ = ("FlipRouterDependencies",)
