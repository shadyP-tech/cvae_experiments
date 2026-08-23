"""Compatibility facade for the modular P-DCAPS legacy control subsystem."""

from .legacy import (
    LegacyControlDecision,
    LegacyControlSeal,
    LegacyControlSurface,
    LegacyPseudoReference,
    LegacyTargetPolicyDecision,
    build_legacy_control_decision,
    build_legacy_control_surface,
    resolve_legacy_control,
    seal_legacy_control,
)


__all__ = (
    "LegacyControlDecision",
    "LegacyControlSeal",
    "LegacyControlSurface",
    "LegacyPseudoReference",
    "LegacyTargetPolicyDecision",
    "build_legacy_control_decision",
    "build_legacy_control_surface",
    "resolve_legacy_control",
    "seal_legacy_control",
)
