"""Terminal-only capability surfaces for OE-PPUR v3."""

from .contracts import (
    ALLOWED_AGGREGATE_METRICS,
    AggregateOnlyTerminalReceipt,
    AggregateTerminalScoreRequest,
    ArtifactOnlyPreterminalAttestationReceipt,
    GuardedPreterminalBoundary,
    assert_aggregate_only_payload,
    issue_artifact_only_preterminal_attestation,
    seal_guarded_preterminal_boundary,
)
from .evaluator import (
    AggregateOnlyTerminalEvaluator,
    TerminalAggregateCapability,
    build_manager_owned_terminal_evaluator,
    issue_terminal_aggregate_capability,
)
from .label_reader import (
    AggregateOnlyLabelReader,
    CaseRoutingDiagnostic,
    ManagerOwnedManifestLabelReader,
    build_manager_owned_manifest_label_reader,
    build_physical_manifest_label_reader,
    seal_manager_owned_terminal_input,
)

__all__ = (
    "ALLOWED_AGGREGATE_METRICS",
    "AggregateOnlyLabelReader",
    "AggregateOnlyTerminalEvaluator",
    "AggregateOnlyTerminalReceipt",
    "AggregateTerminalScoreRequest",
    "ArtifactOnlyPreterminalAttestationReceipt",
    "CaseRoutingDiagnostic",
    "GuardedPreterminalBoundary",
    "ManagerOwnedManifestLabelReader",
    "TerminalAggregateCapability",
    "assert_aggregate_only_payload",
    "build_manager_owned_manifest_label_reader",
    "build_manager_owned_terminal_evaluator",
    "build_physical_manifest_label_reader",
    "issue_artifact_only_preterminal_attestation",
    "issue_terminal_aggregate_capability",
    "seal_guarded_preterminal_boundary",
    "seal_manager_owned_terminal_input",
)
