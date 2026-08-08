"""Thin executable facade for target-support materialization."""

from pathlib import Path

from .artifact_writer import persist_target_support_artifact
from .bundle_validation import validate_target_support_surface_bundle
from .config import TargetSupportSurfaceConfig
from .feature_production import build_all_target_features


def materialize_target_support_surface(config: TargetSupportSurfaceConfig) -> Path:
    inputs, generated, productions = build_all_target_features(config)
    root = persist_target_support_artifact(config, inputs, generated, productions)
    validate_target_support_surface_bundle(root)
    return root


__all__ = ("materialize_target_support_surface", "validate_target_support_surface_bundle")
