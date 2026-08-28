"""Dedicated launch edge for the prepared OE-PPUR v3 execution."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    load_resolved_config,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.runner import (
    run_oe_ppur_v3,
)
from .paths import DEFAULT_SCRATCH_ROOT, resolve_canonical_preparation_paths


def run_authorized_experiment(
    repository_root: str | Path,
    *,
    scratch_root: str | Path = DEFAULT_SCRATCH_ROOT,
) -> Path:
    """Launch only a previously issued and read-back-validated envelope."""

    paths = resolve_canonical_preparation_paths(
        repository_root,
        scratch_root=scratch_root,
        require_source=True,
        require_amendment=True,
    )
    resolved_path = paths.artifact_root / "config.resolved.yaml"
    if not resolved_path.is_file() or resolved_path.is_symlink():
        raise ProtocolError("OE-PPUR v3 authorization-ready envelope is absent.")
    bundle = load_resolved_config(resolved_path)
    if bundle.input_bindings != paths.input_bindings:
        raise ProtocolError("OE-PPUR v3 prepared input bindings drifted.")
    return run_oe_ppur_v3(
        bundle,
        artifact_root=paths.artifact_root,
        scratch_root=paths.scratch_root,
    )


__all__ = ("run_authorized_experiment",)
