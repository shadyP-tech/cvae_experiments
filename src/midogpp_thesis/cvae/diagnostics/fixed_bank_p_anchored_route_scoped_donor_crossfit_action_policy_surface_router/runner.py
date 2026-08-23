"""Canonical fail-closed entrypoint for the planned P-DCAPS identity."""

from __future__ import annotations

from pathlib import Path

from .config import PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig
from .execution_admission import assert_execution_authorized


def run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router(
    config: PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig,
    *,
    artifact_root: str | Path,
) -> Path:
    """Reject the frozen non-authorizing v1 identity before any side effect.

    The scientific implementation is exposed through package-local pure
    engines.  A future execution, if explicitly authorized, requires a new
    hashed config/ledger identity and its own runner admission branch; mutating
    this planned identity in place is intentionally impossible.
    """

    assert_execution_authorized(
        config,
        artifact_root=artifact_root,
        scratch_root=None,
    )
    raise AssertionError("P-DCAPS planned admission returned unexpectedly.")


__all__ = (
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router",
)
