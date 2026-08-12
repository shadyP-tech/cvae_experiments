"""Public API for the consumed-test target-static endpoint router."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    ConsumedTestEndpointRouterConfig,
    load_utility_aligned_consumed_test_endpoint_router_config,
)
from .protocol import (
    ConsumedTestEndpointRouterProtocol,
    assert_consumed_test_diagnostic_only,
    canonical_consumed_test_protocol,
)
from .validation import (
    validate_utility_aligned_consumed_test_endpoint_router_bundle,
)

if TYPE_CHECKING:
    from .runner import ConsumedTestEndpointRouterDependencies


def run_utility_aligned_consumed_test_endpoint_router(
    config: ConsumedTestEndpointRouterConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: ConsumedTestEndpointRouterDependencies | None = None,
) -> Path:
    """Load the runtime lazily so registration/config imports stay lightweight."""

    from .runner import run_utility_aligned_consumed_test_endpoint_router as _run

    return _run(
        config,
        artifact_root=artifact_root,
        dependencies=dependencies,
    )


def __getattr__(name: str) -> Any:
    if name == "ConsumedTestEndpointRouterDependencies":
        from .runner import ConsumedTestEndpointRouterDependencies

        return ConsumedTestEndpointRouterDependencies
    raise AttributeError(name)


__all__ = (
    "ConsumedTestEndpointRouterConfig",
    "ConsumedTestEndpointRouterDependencies",
    "ConsumedTestEndpointRouterProtocol",
    "assert_consumed_test_diagnostic_only",
    "canonical_consumed_test_protocol",
    "load_utility_aligned_consumed_test_endpoint_router_config",
    "run_utility_aligned_consumed_test_endpoint_router",
    "validate_utility_aligned_consumed_test_endpoint_router_bundle",
)
