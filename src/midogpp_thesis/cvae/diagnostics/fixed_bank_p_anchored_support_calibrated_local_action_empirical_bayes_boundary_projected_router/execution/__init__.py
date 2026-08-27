"""Spawn-safe workstation primitives for planned SCALE-BP execution."""

from .coordinator import execute_outer_center_task, run_outer_center_tasks
from .dtos import MemmapReference, OuterCenterResult, OuterCenterTask
from .memmaps import open_readonly_memmap, row_index_hash, validate_row_index
from .physical_bank import (
    PhysicalBankCellSpec,
    PhysicalBankReceipt,
    build_physical_bank_receipt,
)
from .workstation import WorkstationPlan, build_workstation_plan

__all__ = (
    "MemmapReference",
    "OuterCenterResult",
    "OuterCenterTask",
    "PhysicalBankCellSpec",
    "PhysicalBankReceipt",
    "WorkstationPlan",
    "build_workstation_plan",
    "build_physical_bank_receipt",
    "execute_outer_center_task",
    "open_readonly_memmap",
    "row_index_hash",
    "run_outer_center_tasks",
    "validate_row_index",
)
