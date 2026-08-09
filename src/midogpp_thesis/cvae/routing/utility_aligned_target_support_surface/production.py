"""Thin executable facade for target-support materialization."""

from pathlib import Path

from .action_probe_production import materialize_target_action_shifts
from .artifact_writer import persist_target_support_artifact
from .bundle_validation import validate_target_support_surface_bundle
from .config import TargetSupportSurfaceConfig
from .feature_production import build_all_target_features, execution_root_for


def materialize_target_support_surface(config: TargetSupportSurfaceConfig) -> Path:
    inputs, generated, productions = build_all_target_features(config)
    action_shift_rows = materialize_target_action_shifts(
        inputs,
        generated,
        execution_root=execution_root_for(config),
        runtime=config.action_probe_runtime,
    )
    root = persist_target_support_artifact(
        config,
        inputs,
        generated,
        productions,
        action_shift_rows,
    )
    validate_target_support_surface_bundle(root)
    return root


__all__ = ("materialize_target_support_surface", "validate_target_support_surface_bundle")
