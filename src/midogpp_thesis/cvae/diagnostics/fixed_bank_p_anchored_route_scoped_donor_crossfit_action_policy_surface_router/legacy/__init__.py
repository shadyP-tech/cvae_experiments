"""Modular same-run legacy center-pooled control subsystem."""

from .contracts import (
    LegacyControlDecision,
    LegacyControlSurface,
    LegacyPseudoReference,
    LegacyTargetPolicyDecision,
)
from .sealing import LegacyControlSeal, resolve_legacy_control, seal_legacy_control
from .selection import (
    build_legacy_control_decision,
    build_legacy_control_surface,
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
