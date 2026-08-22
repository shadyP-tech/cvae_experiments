"""Terminal-only consumed-test CBPUPR diagnostic."""

from .config import (
    PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig,
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from .runner import (
    run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router,
)
from .validation import (
    validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle,
)


__all__ = (
    "PAnchoredRouteScopedCenterBalancedPosteriorUtilityPrefixRouterConfig",
    "load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config",
    "run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router",
    "validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle",
)
