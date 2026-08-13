"""Dependency-injection seams for S4 phase-order and leakage tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SupportStaticRouterDependencies:
    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_global_static: Callable[..., object] | None = None
    build_route_products: Callable[..., object] | None = None
    cleanup_staging: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


__all__ = ("SupportStaticRouterDependencies",)
