"""Source-bound read-only inputs for an explicitly authorized successor."""

from .manifest import ManifestIdentityReceipt, load_manifest_identity_receipt
from .physical_memmap import load_physical_cell_surface
from ..execution.physical_bank import (
    PhysicalBankCellSpec,
    PhysicalBankReceipt,
    build_physical_bank_receipt,
)


__all__ = (
    "ManifestIdentityReceipt",
    "PhysicalBankCellSpec",
    "PhysicalBankReceipt",
    "build_physical_bank_receipt",
    "load_manifest_identity_receipt",
    "load_physical_cell_surface",
)
