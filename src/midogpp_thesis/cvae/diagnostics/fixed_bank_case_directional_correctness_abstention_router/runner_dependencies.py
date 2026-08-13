"""Narrow dependency-injection seams for phase and failure tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CaseDirectionalRunnerDependencies:
    validate_inputs: Callable[..., object] | None = None
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_actions: Callable[..., object] | None = None
    persist_initial: Callable[..., object] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_probability_surface: Callable[..., object] | None = None
    build_probability_index: Callable[..., object] | None = None
    persist_prelabel: Callable[..., object] | None = None
    build_features: Callable[..., object] | None = None
    build_plans: Callable[..., object] | None = None
    build_label_firewall: Callable[..., object] | None = None
    compute_priors: Callable[..., object] | None = None
    execute_route_jobs: Callable[..., object] | None = None
    persist_science: Callable[..., object] | None = None
    open_terminal_labels: Callable[..., object] | None = None
    evaluate_terminal: Callable[..., object] | None = None
    persist_terminal: Callable[..., object] | None = None
    write_content_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., object] | None = None
    write_state: Callable[..., object] | None = None
    phase_observer: Callable[[str], None] | None = None


RunnerDependencies = CaseDirectionalRunnerDependencies


__all__ = ("CaseDirectionalRunnerDependencies", "RunnerDependencies")
