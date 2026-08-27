"""Canonical rejecting runner for the planned SCEPTRE identity."""

from __future__ import annotations

from pathlib import Path

from .execution_admission import assert_execution_authorized


def run_planned_sceptre_router(
    config: object,
    *,
    artifact_root: object,
    scratch_root: object | None = None,
) -> Path:
    """Reject before output/scratch path resolution, inspection, or mutation."""

    assert_execution_authorized(
        config,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
    )
    raise AssertionError("SCEPTRE planned admission returned unexpectedly.")


__all__ = ("run_planned_sceptre_router",)
