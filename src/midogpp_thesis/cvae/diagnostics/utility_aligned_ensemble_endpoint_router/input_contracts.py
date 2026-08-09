"""Label-free validation contracts shared by consumed Stage-90 diagnostics."""

from ..utility_aligned_exact_tail_router.input_contracts import (
    FixedPartitionSurface,
    LabelFreeValidationFrame,
    ValidationRowIdentity,
    row_identity_hash,
)

__all__ = (
    "FixedPartitionSurface",
    "LabelFreeValidationFrame",
    "ValidationRowIdentity",
    "row_identity_hash",
)
