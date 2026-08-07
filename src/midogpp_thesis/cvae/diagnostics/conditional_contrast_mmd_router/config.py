"""Public experiment-fenced configuration entry point."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ..mmd_kmm_router.config import (
    MMDKMMRouterDiagnosticConfig,
    load_mmd_kmm_router_config,
)
from ..mmd_kmm_router.profiles import CONDITIONAL_ROUTER_MODE


ConditionalContrastMMDRouterDiagnosticConfig = MMDKMMRouterDiagnosticConfig


def load_conditional_contrast_mmd_router_config(
    path: str | Path,
) -> ConditionalContrastMMDRouterDiagnosticConfig:
    config = load_mmd_kmm_router_config(path)
    if config.router_mode != CONDITIONAL_ROUTER_MODE:
        raise ProtocolError(
            "Conditional contrast-MMD entry point received a different profile."
        )
    return config


__all__ = (
    "ConditionalContrastMMDRouterDiagnosticConfig",
    "load_conditional_contrast_mmd_router_config",
)
