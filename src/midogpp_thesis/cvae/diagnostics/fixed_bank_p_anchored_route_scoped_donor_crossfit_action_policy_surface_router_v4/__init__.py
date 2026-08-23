"""Executable P-DCAPS v4 terminal consumed-test diagnostic."""

from .config import (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config,
)
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4,
)


__all__ = (
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4_config",
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4",
)
