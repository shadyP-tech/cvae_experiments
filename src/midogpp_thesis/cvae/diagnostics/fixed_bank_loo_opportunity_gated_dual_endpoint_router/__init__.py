"""Public facade for the opportunity-gated dual-endpoint diagnostic."""

from __future__ import annotations

from pathlib import Path

from .config import (
    FixedBankLooOpportunityGatedDualEndpointRouterConfig,
    load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config,
)


def run_fixed_bank_loo_opportunity_gated_dual_endpoint_router(
    config: FixedBankLooOpportunityGatedDualEndpointRouterConfig,
    *,
    artifact_root: str | Path | None = None,
    **kwargs: object,
) -> Path:
    from .runner import run_fixed_bank_loo_opportunity_gated_dual_endpoint_router as run

    return run(config, artifact_root=artifact_root, **kwargs)


__all__ = (
    "FixedBankLooOpportunityGatedDualEndpointRouterConfig",
    "load_fixed_bank_loo_opportunity_gated_dual_endpoint_router_config",
    "run_fixed_bank_loo_opportunity_gated_dual_endpoint_router",
)
