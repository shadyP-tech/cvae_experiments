"""Execution boundary for the HARP v15 support-adapted diagnostic.

Only modules on the support/target production path are re-exported.  The
retained H/q/r migration helpers are intentionally not imported and cannot
become implicit scientific dependencies.
"""

from .action_capacity import (
    ActionCapacityRequirement,
    build_action_capacity_certificate,
    enumerate_complete_action_capacity,
    validate_action_capacity,
    validate_action_capacity_certificate,
)
from .contracts import (
    ActionKind,
    ArtifactValue,
    FrozenRouteReceipt,
    HarpV15Pipeline,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
    TerminalEvaluation,
)
from .physical_actions import (
    SUPPORT_SURFACE,
    TARGET_SURFACE,
    HarpActionSpec,
    build_support_action_menu,
    build_target_action_menu,
)

__all__ = (
    "ActionCapacityRequirement",
    "ActionKind",
    "ArtifactValue",
    "FrozenRouteReceipt",
    "HarpV15Pipeline",
    "HarpActionSpec",
    "LabelFreeActionBlock",
    "LabelFreeOuterMenu",
    "PrelabelRouteSet",
    "RoutedCase",
    "TerminalEvaluation",
    "SUPPORT_SURFACE",
    "TARGET_SURFACE",
    "build_action_capacity_certificate",
    "enumerate_complete_action_capacity",
    "build_support_action_menu",
    "build_target_action_menu",
    "validate_action_capacity",
    "validate_action_capacity_certificate",
)
