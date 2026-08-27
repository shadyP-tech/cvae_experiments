"""Public workstation execution contracts for SCALE-BP v2."""

from .coordinator import (
    OuterCenterWorker,
    execute_outer_center_task,
    run_outer_center_tasks,
    run_support_folds_sequentially,
)
from .dtos import (
    CANONICAL_CENTERS,
    DEFAULT_SUPPORT_FOLD_IDS,
    MemmapRef,
    MemmapReference,
    OuterCenterResult,
    OuterCenterTask,
)
from .memmaps import (
    open_memmap_bundle,
    open_readonly_memmap,
    row_index_hash,
    validate_memmap_reference,
    validate_memmap_references,
    validate_row_index,
)
from .workstation import WorkstationPlan, build_workstation_plan
from ..artifacts.chunks import ChunkRef


__all__ = (
    "CANONICAL_CENTERS",
    "ChunkRef",
    "DEFAULT_SUPPORT_FOLD_IDS",
    "MemmapRef",
    "MemmapReference",
    "OuterCenterResult",
    "OuterCenterTask",
    "OuterCenterWorker",
    "WorkstationPlan",
    "build_workstation_plan",
    "execute_outer_center_task",
    "open_memmap_bundle",
    "open_readonly_memmap",
    "row_index_hash",
    "run_outer_center_tasks",
    "run_support_folds_sequentially",
    "validate_memmap_reference",
    "validate_memmap_references",
    "validate_row_index",
)
