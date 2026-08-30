"""Terminal-only capability surfaces for OE-PPUR v4."""

from .contracts import (
    ALLOWED_AGGREGATE_METRICS,
    ArtifactOnlyPreterminalAttestationReceipt,
    GuardedPreterminalBoundary,
    seal_guarded_preterminal_boundary,
)
from .evaluator import issue_terminal_aggregate_capability
from .label_reader import (
    build_physical_manifest_label_reader,
    validate_resolved_terminal_authority,
)

__all__ = (
    "ALLOWED_AGGREGATE_METRICS",
    "ArtifactOnlyPreterminalAttestationReceipt",
    "GuardedPreterminalBoundary",
    "build_physical_manifest_label_reader",
    "issue_terminal_aggregate_capability",
    "seal_guarded_preterminal_boundary",
    "validate_resolved_terminal_authority",
)
