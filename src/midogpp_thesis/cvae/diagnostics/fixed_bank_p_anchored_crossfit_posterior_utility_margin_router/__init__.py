"""Terminal MIDOG++ P-anchored posterior-utility margin diagnostic."""

from .config import (
    PAnchoredCrossfitPosteriorUtilityMarginRouterConfig,
    load_p_anchored_crossfit_posterior_utility_margin_router_config,
)
from .runner import run_p_anchored_crossfit_posterior_utility_margin_router
from .validation import validate_p_anchored_crossfit_posterior_utility_margin_router_bundle


__all__ = (
    "PAnchoredCrossfitPosteriorUtilityMarginRouterConfig",
    "load_p_anchored_crossfit_posterior_utility_margin_router_config",
    "run_p_anchored_crossfit_posterior_utility_margin_router",
    "validate_p_anchored_crossfit_posterior_utility_margin_router_bundle",
)
