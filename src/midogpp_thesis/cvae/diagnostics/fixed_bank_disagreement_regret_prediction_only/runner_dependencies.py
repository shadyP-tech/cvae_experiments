"""Dependency-injection seams for phase-order and no-label tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PredictionOnlyDependencies:
    validate_input_fence: Callable[..., object] | None = None
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_source_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    persist_initial: Callable[..., object] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source_streams: Callable[..., object] | None = None
    stage_source_streams: Callable[..., object] | None = None
    materialize_target_classifier_bank: Callable[..., object] | None = None
    materialize_source_oof_predictions: Callable[..., object] | None = None
    materialize_prelabel_prediction_seal: Callable[..., object] | None = None
    aggregate_source_probabilities: Callable[..., object] | None = None
    build_contexts: Callable[..., object] | None = None
    build_prelabel: Callable[..., object] | None = None
    persist_prelabel: Callable[..., object] | None = None
    build_source_label_capability: Callable[..., object] | None = None
    persist_source_capability: Callable[..., object] | None = None
    fit_development: Callable[..., object] | None = None
    persist_development: Callable[..., object] | None = None
    issue_test_admission: Callable[..., object] | None = None
    load_test_frame: Callable[..., object] | None = None
    materialize_test_predictions: Callable[..., object] | None = None
    aggregate_test_probabilities: Callable[..., object] | None = None
    build_inference: Callable[..., object] | None = None
    persist_inference: Callable[..., object] | None = None
    build_runtime_summary: Callable[..., object] | None = None
    persist_reports: Callable[..., object] | None = None
    write_content_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., object] | None = None
    load_post_test_recovery: Callable[..., object] | None = None
    validate_recovery_checkout: Callable[..., object] | None = None
    validate_terminal_completion: Callable[..., object] | None = None
    write_state: Callable[..., object] | None = None
    cleanup_staging: Callable[..., object] | None = None
    phase_observer: Callable[[str], None] | None = None


__all__ = ("PredictionOnlyDependencies",)
