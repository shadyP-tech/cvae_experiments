"""Independent public validator for the conditional contrast-MMD bundle."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ..mmd_kmm_router.profiles import CONDITIONAL_ROUTER_MODE
from ..mmd_kmm_router.validation import validate_mmd_kmm_router_bundle
from .config import ConditionalContrastMMDRouterDiagnosticConfig


def validate_conditional_contrast_mmd_router_bundle(
    root: str | Path,
    *,
    config: ConditionalContrastMMDRouterDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    if config.router_mode != CONDITIONAL_ROUTER_MODE:
        raise ProtocolError("Conditional contrast-MMD validator profile drifted.")
    return validate_mmd_kmm_router_bundle(
        root,
        config=config,
        allow_pending=allow_pending,
    )


__all__ = ("validate_conditional_contrast_mmd_router_bundle",)
