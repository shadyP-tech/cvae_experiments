"""Compatibility facade for neutral utility-aligned target feature logic."""

from ..utility_aligned.target_features import (
    TargetCandidateComponents,
    TargetFeatureProduction,
    build_target_feature_production,
)


__all__ = (
    "TargetCandidateComponents",
    "TargetFeatureProduction",
    "build_target_feature_production",
)
