"""Production adapter dispatch for HARP Stage-60 surfaces.

The heavy producers live with their scientific surfaces.  Keeping this file as
the sole dispatch edge prevents the workspace runner from importing GPU code
before authorization and readiness have passed.
"""

from __future__ import annotations

from ...protocol import ProtocolError
from .config import HarpInputReadiness, HarpStage60Config
from .constants import ACTION_SURFACE, POLICY_LOCK, TARGET_SUPPORT_SURFACE
from .execution_contracts import (
    HarpBuiltProduct,
    HarpDurablePrelabelSeal,
    HarpRunReceipt,
)


class ProductionHarpStage60Adapter:
    """Lazy role dispatcher with no protocol decisions of its own."""

    def __init__(self) -> None:
        self._delegate: object | None = None

    def _for(self, config: HarpStage60Config) -> object:
        if self._delegate is not None:
            return self._delegate
        if config.contract == ACTION_SURFACE:
            from ..harp_action_surface.production import ProductionActionSurfaceAdapter

            delegate: object = ProductionActionSurfaceAdapter()
        elif config.contract == TARGET_SUPPORT_SURFACE:
            from ..harp_action_surface.production import ProductionTargetSupportAdapter

            delegate = ProductionTargetSupportAdapter()
        elif config.contract == POLICY_LOCK:
            from ..harp_portfolio.production import ProductionPolicyLockAdapter

            delegate = ProductionPolicyLockAdapter()
        else:  # pragma: no cover
            raise ProtocolError("Unknown HARP production surface.")
        self._delegate = delegate
        return delegate

    def validate_completed_bundle(self, config: HarpStage60Config) -> HarpRunReceipt:
        return self._for(config).validate_completed_bundle(config)  # type: ignore[attr-defined,no-any-return]

    def preflight(self, config: HarpStage60Config, readiness: HarpInputReadiness) -> None:
        self._for(config).preflight(config, readiness)  # type: ignore[attr-defined]

    def materialize_and_seal_label_free_menu(
        self, config: HarpStage60Config, readiness: HarpInputReadiness
    ) -> HarpDurablePrelabelSeal:
        return self._for(config).materialize_and_seal_label_free_menu(  # type: ignore[attr-defined,no-any-return]
            config, readiness
        )

    def open_source_development_labels(
        self, config: HarpStage60Config, seal: HarpDurablePrelabelSeal
    ) -> object:
        return self._for(config).open_source_development_labels(config, seal)  # type: ignore[attr-defined,no-any-return]

    def build_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        source_development_labels: object | None,
    ) -> HarpBuiltProduct:
        return self._for(config).build_product(  # type: ignore[attr-defined,no-any-return]
            config, seal, source_development_labels
        )

    def persist_product(
        self,
        config: HarpStage60Config,
        seal: HarpDurablePrelabelSeal,
        product: HarpBuiltProduct,
    ):
        return self._for(config).persist_product(config, seal, product)  # type: ignore[attr-defined,no-any-return]


__all__ = ("ProductionHarpStage60Adapter",)
