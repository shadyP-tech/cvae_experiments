"""Public facade for the label-free target-support feature producer."""

from .action_probe_contracts import (
    ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS,
    ACTION_SHIFT_LOCK_MEMBER,
    ACTION_SHIFT_LOCK_SCHEMA,
    ACTION_SHIFT_ROW_SCALAR_SEMANTICS,
    ACTION_SHIFT_ROW_SCHEMA,
    ACTION_SHIFT_TABLE_MEMBER,
    TargetSupportActionShiftRow,
)
from .config import TargetSupportSurfaceConfig, load_utility_aligned_target_support_surface_config
from .production import validate_target_support_surface_bundle
from .runner import run_utility_aligned_target_support_surface


__all__ = (
    "ACTION_SHIFT_LOCK_MEMBER",
    "ACTION_SHIFT_LOCK_SCHEMA",
    "ACTION_SHIFT_AGGREGATE_SCALAR_SEMANTICS",
    "ACTION_SHIFT_ROW_SCALAR_SEMANTICS",
    "ACTION_SHIFT_ROW_SCHEMA",
    "ACTION_SHIFT_TABLE_MEMBER",
    "TargetSupportActionShiftRow",
    "TargetSupportSurfaceConfig",
    "load_utility_aligned_target_support_surface_config",
    "run_utility_aligned_target_support_surface",
    "validate_target_support_surface_bundle",
)
