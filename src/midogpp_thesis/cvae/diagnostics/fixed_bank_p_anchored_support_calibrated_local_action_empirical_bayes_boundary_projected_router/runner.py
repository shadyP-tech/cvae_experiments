"""Canonical rejecting runner for the inactive SCALE-BP v1 identity."""

from __future__ import annotations

from pathlib import Path

from .execution_admission import assert_execution_authorized


def run_support_calibrated_local_action_empirical_bayes_boundary_projected_router(
    config: object,
    *,
    artifact_root: str | Path,
    scratch_root: str | Path | None = None,
) -> Path:
    """Reject before output, scratch, lock, run-state, or label mutation."""

    assert_execution_authorized(
        config,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
    )
    raise AssertionError("SCALE-BP planned admission returned unexpectedly.")


__all__ = (
    "run_support_calibrated_local_action_empirical_bayes_boundary_projected_router",
)
