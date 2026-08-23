"""Canonical rejecting runner for the planned P-DCAPS v3 identity."""

from __future__ import annotations

from pathlib import Path

from .execution_admission import assert_execution_authorized


def run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3(
    config: object,
    *,
    artifact_root: str | Path,
    scratch_root: str | Path | None = None,
) -> Path:
    """Seal source scopes, then reject v3 before any run-path capability."""

    assert_execution_authorized(
        config,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
    )
    raise AssertionError("P-DCAPS v3 planned admission returned unexpectedly.")


__all__ = (
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3",
)
