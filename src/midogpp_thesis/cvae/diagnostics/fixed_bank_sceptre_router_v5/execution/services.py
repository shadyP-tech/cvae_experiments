"""Dependency boundary for the SCEPTRE v5 production orchestrator.

The default services are the real workstation implementation.  A single
frozen bundle keeps the runner thin and lets focused tests replace physical
execution without weakening any production contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..development import fit_and_freeze_development
from ..physical import materialize_prediction_surface, materialize_source_streams
from ..phase_orchestrator import run_routing_phases
from ..terminal_evaluation import evaluate_terminal_surfaces
from .admission import admit_execution
from .authorization_lease import (
    claim_authorization_lease,
    mark_authorization_complete,
    mark_authorization_failed,
)
from .fresh_validation import (
    require_two_fresh_final_validations,
    require_two_fresh_preterminal_validations,
)
from .scratch import cleanup_scratch, create_scratch
from .source_inner_surfaces import load_development_surfaces
from ...fixed_bank_sceptre_router.partitions import build_three_role_partition


@dataclass(frozen=True, slots=True)
class ProductionServices:
    admit: Callable[..., object] = admit_execution
    claim_lease: Callable[..., object] = claim_authorization_lease
    complete_lease: Callable[..., object] = mark_authorization_complete
    fail_lease: Callable[..., object] = mark_authorization_failed
    create_scratch: Callable[..., object] = create_scratch
    cleanup_scratch: Callable[..., object] = cleanup_scratch
    build_partition: Callable[..., object] = build_three_role_partition
    load_development: Callable[..., object] = load_development_surfaces
    fit_development: Callable[..., object] = fit_and_freeze_development
    materialize_sources: Callable[..., object] = materialize_source_streams
    materialize_predictions: Callable[..., object] = materialize_prediction_surface
    route: Callable[..., object] = run_routing_phases
    validate_preterminal: Callable[..., object] = (
        require_two_fresh_preterminal_validations
    )
    evaluate_terminal: Callable[..., object] = evaluate_terminal_surfaces
    validate_final: Callable[..., object] = require_two_fresh_final_validations


DEFAULT_PRODUCTION_SERVICES = ProductionServices()


__all__ = ("DEFAULT_PRODUCTION_SERVICES", "ProductionServices")
