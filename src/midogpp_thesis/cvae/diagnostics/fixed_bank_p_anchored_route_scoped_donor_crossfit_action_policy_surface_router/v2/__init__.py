"""Authorized P-DCAPS v2 identity and integration surface."""

from __future__ import annotations

from pathlib import Path

from .config import (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig,
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config,
)


def run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2(
    config: PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config,
    *,
    artifact_root: str | Path,
) -> Path:
    """Late-bind the scientific runner supplied by the execution layer."""

    from .runner import (  # type: ignore[import-not-found]
        run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2 as run,
    )

    return run(config, artifact_root=artifact_root)


run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router = (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2
)


__all__ = (
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV2Config",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2_config",
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router",
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v2",
)
