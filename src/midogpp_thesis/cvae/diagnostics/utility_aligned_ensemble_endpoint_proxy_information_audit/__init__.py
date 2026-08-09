"""Consumed-data Stage-90 proxy-information audit.

The public facade deliberately exposes only configuration and orchestration.
Scientific proxy/model contracts live in the package's narrow leaf modules.
"""

from .config import (
    ProxyInformationAuditConfig,
    load_utility_aligned_ensemble_endpoint_proxy_information_audit_config,
)


def run_utility_aligned_ensemble_endpoint_proxy_information_audit(
    *args: object, **kwargs: object
):
    from .runner import (
        run_utility_aligned_ensemble_endpoint_proxy_information_audit as run,
    )

    return run(*args, **kwargs)


__all__ = (
    "ProxyInformationAuditConfig",
    "load_utility_aligned_ensemble_endpoint_proxy_information_audit_config",
    "run_utility_aligned_ensemble_endpoint_proxy_information_audit",
)
