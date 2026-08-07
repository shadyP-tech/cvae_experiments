"""Run the separately fenced conditional contrast-MMD diagnostic."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ..mmd_kmm_router.profiles import CONDITIONAL_ROUTER_MODE
from ..mmd_kmm_router.runner import run_mmd_kmm_router_diagnostic
from .config import ConditionalContrastMMDRouterDiagnosticConfig


def run_conditional_contrast_mmd_router_diagnostic(
    config: ConditionalContrastMMDRouterDiagnosticConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    if config.router_mode != CONDITIONAL_ROUTER_MODE:
        raise ProtocolError("Conditional contrast-MMD runner profile drifted.")
    return run_mmd_kmm_router_diagnostic(config, artifact_root=artifact_root)


__all__ = ("run_conditional_contrast_mmd_router_diagnostic",)
