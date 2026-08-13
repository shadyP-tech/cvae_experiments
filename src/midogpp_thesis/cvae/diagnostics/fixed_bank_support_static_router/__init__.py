"""Quarantined fixed-bank support-static S4 consumed-test diagnostic."""

from .config import (
    FixedBankSupportStaticRouterConfig,
    load_fixed_bank_support_static_router_config,
)


def run_fixed_bank_support_static_router(*args: object, **kwargs: object):
    from .runner import run_fixed_bank_support_static_router as implementation

    return implementation(*args, **kwargs)


def validate_fixed_bank_support_static_router_bundle(
    *args: object, **kwargs: object
):
    from .validation import (
        validate_fixed_bank_support_static_router_bundle as implementation,
    )

    return implementation(*args, **kwargs)


__all__ = (
    "FixedBankSupportStaticRouterConfig",
    "load_fixed_bank_support_static_router_config",
    "run_fixed_bank_support_static_router",
    "validate_fixed_bank_support_static_router_bundle",
)
