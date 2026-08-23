"""Compatibility facade for the modular P-DCAPS physical bank runtime."""

from __future__ import annotations

from .physical_actions import (
    A1_ACTION_PREFIX,
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    B_ACTION_ID,
    B_ROWS_PER_SOURCE_CLASS,
    U_ACTION_ID,
    U_ROWS_PER_SOURCE_CLASS,
    PhysicalActionSpec,
    a1_action_id,
    action_library_by_target,
    candidate_sources,
)
from .physical_contracts import (
    CenterPhysicalSurface,
    MaterializedPhysicalBank,
    PhysicalSurface,
)
from .physical_materializer import (
    build_physical_surface,
    materialize_physical_bank,
    physical_partition_hash,
)


__all__ = (
    "A1_ACTION_PREFIX",
    "B_ACTION_ID",
    "CenterPhysicalSurface",
    "MaterializedPhysicalBank",
    "PhysicalActionSpec",
    "PhysicalSurface",
    "U_ACTION_ID",
    "a1_action_id",
    "action_library_by_target",
    "build_physical_surface",
    "candidate_sources",
    "materialize_physical_bank",
    "physical_partition_hash",
)
