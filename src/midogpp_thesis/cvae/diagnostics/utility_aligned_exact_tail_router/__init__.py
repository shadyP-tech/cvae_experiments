"""Public facade for the consumed utility-aligned exact-tail diagnostic."""

from .config import (
    UtilityAlignedExactTailRouterConfig,
    load_utility_aligned_exact_tail_router_config,
)


def run_utility_aligned_exact_tail_router_diagnostic(*args: object, **kwargs: object):
    from .runner import run_utility_aligned_exact_tail_router_diagnostic as run

    return run(*args, **kwargs)


__all__ = (
    "UtilityAlignedExactTailRouterConfig",
    "load_utility_aligned_exact_tail_router_config",
    "run_utility_aligned_exact_tail_router_diagnostic",
)
