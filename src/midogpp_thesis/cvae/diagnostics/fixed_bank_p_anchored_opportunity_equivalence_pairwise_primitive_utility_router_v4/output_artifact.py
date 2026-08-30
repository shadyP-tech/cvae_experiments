"""Typed façade for OE-PPUR v4 final aggregate output lifecycle.

Durable member creation lives in :mod:`.output_persistence`; whole-artifact
semantic and inventory validation lives in :mod:`.output_validation`.  This
module intentionally remains the sole public composition surface used by the
runner, run-state machine, authorization lease, and tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifact.contracts import (
    CLAIM_BOUNDARY_MEMBER,
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    COMPLETE_CATALOG_MEMBERS,
    COMPLETE_INTERNAL_MEMBERS,
    CONTENT_INDEX_MEMBER,
    DIAGNOSTIC_SUMMARY_MEMBER,
    FINAL_ATTESTATION_MEMBER,
    FINAL_BINDING_MEMBER,
    FINAL_INDEXED_MEMBERS,
    FINAL_PAYLOAD_MEMBERS,
    LEAKAGE_REPORT_MEMBER,
    PUBLICATION_DECISION_MEMBER,
    RUNTIME_SUMMARY_MEMBER,
    TERMINAL_METRICS_MEMBER,
    TERMINAL_RESULT_MEMBER,
    VALIDATION_INDEX_MEMBER,
    VALIDATION_REPORT_MEMBER,
)
from .config import RouterV4Config
from .execution.inputs import SevenInputContractReceipt
from .execution.preterminal_artifact import FinalAggregateAttestationReceipt
from .output_persistence import _persist_final_aggregate_members
from .complete_artifact_validation import (
    CompleteArtifactSealReceipt,
    build_complete_artifact_seal,
    validate_complete_artifact_seal,
)
from .output_validation import (
    FinalAggregateBundleReceipt,
    _issue_final_aggregate_bundle,
    validate_complete_artifact_inventory,
    validate_final_aggregate_bundle,
)
from .source_seal import SourceSealReceipt
from .source_supervision import SourceTrainingSurface
from .run_admission import SevenInputRunAdmission
from .terminal.contracts import (
    AggregateOnlyTerminalReceipt,
    ArtifactOnlyPreterminalAttestationReceipt,
    GuardedPreterminalBoundary,
)


def assemble_final_aggregate_bundle(
    root: str | Path,
    *,
    config: RouterV4Config,
    seven_input_contract: SevenInputContractReceipt,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
    preterminal_boundary: GuardedPreterminalBoundary,
    preterminal_attestations: Sequence[ArtifactOnlyPreterminalAttestationReceipt],
    terminal_receipt: AggregateOnlyTerminalReceipt,
    final_attestation: FinalAggregateAttestationReceipt,
    runtime_summary: Mapping[str, object],
    run_admission: SevenInputRunAdmission,
) -> FinalAggregateBundleReceipt:
    """Persist final members, then issue a receipt only after full validation."""

    destination = _persist_final_aggregate_members(
        root,
        config=config,
        seven_input_contract=seven_input_contract,
        source_seal=source_seal,
        source_surface=source_surface,
        preterminal_boundary=preterminal_boundary,
        preterminal_attestations=preterminal_attestations,
        terminal_receipt=terminal_receipt,
        final_attestation=final_attestation,
        runtime_summary=runtime_summary,
        run_admission=run_admission,
    )
    return _issue_final_aggregate_bundle(destination)


__all__ = (
    "CLAIM_BOUNDARY_MEMBER",
    "COMPLETE_ARTIFACT_INDEX_MEMBER",
    "COMPLETE_CATALOG_MEMBERS",
    "COMPLETE_INTERNAL_MEMBERS",
    "CONTENT_INDEX_MEMBER",
    "DIAGNOSTIC_SUMMARY_MEMBER",
    "FINAL_ATTESTATION_MEMBER",
    "FINAL_BINDING_MEMBER",
    "FINAL_INDEXED_MEMBERS",
    "FINAL_PAYLOAD_MEMBERS",
    "FinalAggregateBundleReceipt",
    "CompleteArtifactSealReceipt",
    "LEAKAGE_REPORT_MEMBER",
    "PUBLICATION_DECISION_MEMBER",
    "RUNTIME_SUMMARY_MEMBER",
    "TERMINAL_RESULT_MEMBER",
    "TERMINAL_METRICS_MEMBER",
    "VALIDATION_INDEX_MEMBER",
    "VALIDATION_REPORT_MEMBER",
    "assemble_final_aggregate_bundle",
    "build_complete_artifact_seal",
    "validate_complete_artifact_inventory",
    "validate_complete_artifact_seal",
    "validate_final_aggregate_bundle",
)
