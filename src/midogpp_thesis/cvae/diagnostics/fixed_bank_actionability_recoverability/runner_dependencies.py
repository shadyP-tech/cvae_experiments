"""Dependency-injection seams for runner phase-order tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FixedBankActionabilityRecoverabilityDependencies:
    validate_inputs: Callable[..., object] | None = None
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_partition: Callable[..., object] | None = None
    persist_initial: Callable[..., None] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_seed_rows: Callable[..., object] | None = None
    aggregate_probabilities: Callable[..., object] | None = None
    build_prelabel: Callable[..., object] | None = None
    persist_prelabel: Callable[..., object] | None = None
    build_label_manager: Callable[..., object] | None = None
    build_utility: Callable[..., object] | None = None
    fit_target_model: Callable[..., object] | None = None
    combine_models: Callable[..., object] | None = None
    persist_models: Callable[..., object] | None = None
    record_models: Callable[..., None] | None = None
    build_pre_support: Callable[..., object] | None = None
    persist_pre_support: Callable[..., object] | None = None
    record_pre_support: Callable[..., None] | None = None
    build_support_fold: Callable[..., object] | None = None
    combine_decisions: Callable[..., object] | None = None
    persist_decisions: Callable[..., object] | None = None
    record_preevaluation: Callable[..., None] | None = None
    persist_capability: Callable[..., None] | None = None
    evaluate: Callable[..., object] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    cleanup_staging: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


__all__ = ("FixedBankActionabilityRecoverabilityDependencies",)
