"""Fail-closed v1 entrypoint and path-free executable-runner blueprint."""

from __future__ import annotations

from pathlib import Path

from .execution_admission import (
    assert_execution_authorized,
    validate_planned_execution_contract,
)
from .runner_blueprint import RunnerBlueprint, build_runner_blueprint


def inspect_runner_blueprint(config: object) -> RunnerBlueprint:
    """Validate and seal the future lifecycle without claiming implementation."""

    source_receipt = validate_planned_execution_contract(config)
    return build_runner_blueprint(config, source_receipt)


def run_planned_router(
    config: object,
    *,
    artifact_root: object,
    scratch_root: object | None = None,
) -> Path:
    """Reject before output/scratch resolution, inspection, or mutation."""

    assert_execution_authorized(
        config,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
    )
    raise AssertionError("OE-PPUR planned admission returned unexpectedly.")


def run_opportunity_equivalence_pairwise_primitive_utility_router_v1(
    config: object,
    *,
    artifact_root: object,
    scratch_root: object | None = None,
) -> Path:
    return run_planned_router(
        config,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
    )


__all__ = (
    "inspect_runner_blueprint",
    "run_opportunity_equivalence_pairwise_primitive_utility_router_v1",
    "run_planned_router",
)
