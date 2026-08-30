"""Capability-ordered orchestration shared by all HARP Stage-60 surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...protocol import ProtocolError
from .config import HarpInputReadiness, HarpStage60Config, validate_harp_inputs_ready
from .constants import ACTION_SURFACE
from .execution_contracts import (
    HarpBuiltProduct,
    HarpDurablePrelabelSeal,
    HarpRunReceipt,
    HarpStage60ExecutionAdapter,
)
from .workspace_binding import validate_harp_production_workspace_binding


def run_harp_stage60_surface(
    config: HarpStage60Config,
    *,
    adapter: HarpStage60ExecutionAdapter | None = None,
    workspace_validator: Callable[[HarpStage60Config], None] = (
        validate_harp_production_workspace_binding
    ),
) -> HarpRunReceipt:
    """Execute one HARP surface with labels opened only after a durable seal."""

    # This must precede completed-fast-path reads, mkdir, host probing, CUDA,
    # scratch selection, and label capability construction.
    workspace_validator(config)
    complete_marker = config.artifact_root / "reports/run_state.json"
    if complete_marker.is_file():
        if adapter is None:
            adapter = _production_adapter()
        return _validate_receipt(config, adapter.validate_completed_bundle(config))

    # Planned or incomplete reservations fail before adapter construction so a
    # direct invocation cannot initialize hardware or mutate the output tree.
    readiness = validate_harp_inputs_ready(config)
    if adapter is None:
        adapter = _production_adapter()
    adapter.preflight(config, readiness)
    seal = adapter.materialize_and_seal_label_free_menu(config, readiness)
    if seal.surface != config.contract.surface:
        raise ProtocolError("HARP adapter sealed another surface.")
    seal.verify_durable()

    source_labels: object | None = None
    if config.contract == ACTION_SURFACE:
        # The adapter receives the verified seal capability, never only a path.
        source_labels = adapter.open_source_development_labels(config, seal)
    product = adapter.build_product(config, seal, source_labels)
    if (
        product.surface != config.contract.surface
        or product.source_development_labels_used_for_scoring_only
        is not (config.contract == ACTION_SURFACE)
    ):
        raise ProtocolError("HARP product crossed a surface label boundary.")
    root = Path(adapter.persist_product(config, seal, product))
    if root.resolve() != config.artifact_root.resolve():
        raise ProtocolError("HARP adapter published outside the canonical workspace output.")
    return _validate_receipt(config, adapter.validate_completed_bundle(config))


def _validate_receipt(
    config: HarpStage60Config, receipt: HarpRunReceipt
) -> HarpRunReceipt:
    if (
        receipt.surface != config.contract.surface
        or Path(receipt.artifact_root).resolve() != config.artifact_root.resolve()
    ):
        raise ProtocolError("HARP completion receipt escaped its workspace binding.")
    return receipt


def _production_adapter() -> HarpStage60ExecutionAdapter:
    # Lazy import is essential: planned/direct calls must reject before touching
    # workstation-only modules.
    from .production import ProductionHarpStage60Adapter

    return ProductionHarpStage60Adapter()


__all__ = (
    "HarpBuiltProduct",
    "HarpDurablePrelabelSeal",
    "HarpRunReceipt",
    "HarpStage60ExecutionAdapter",
    "run_harp_stage60_surface",
)
