"""Dependency-light receipt and member contracts for v4 artifacts.

This module is deliberately below every persistence writer and semantic
validator.  It owns immutable schemas and inventory names only; factories are
package-private so callers cannot mint lifecycle authority directly.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256


FINAL_BINDING_MEMBER = "provenance/final_aggregate_binding.json"
TERMINAL_RESULT_MEMBER = "tables/terminal_result.json"
DIAGNOSTIC_SUMMARY_MEMBER = "reports/diagnostic_summary.json"
LEAKAGE_REPORT_MEMBER = "reports/leakage_report.json"
PUBLICATION_DECISION_MEMBER = "reports/publication_decision.json"
RUNTIME_SUMMARY_MEMBER = "reports/runtime_summary.json"
CLAIM_BOUNDARY_MEMBER = "reports/claim_boundary.json"
FINAL_ATTESTATION_MEMBER = "reports/final_fresh_process_attestation.json"
CONTENT_INDEX_MEMBER = "manifests/content_index.json"
VALIDATION_REPORT_MEMBER = "reports/validation_report.json"
VALIDATION_INDEX_MEMBER = "manifests/validation_index.json"
COMPLETE_ARTIFACT_INDEX_MEMBER = "manifests/complete_artifact_index.json"
TERMINAL_METRICS_MEMBER = "reports/terminal_metrics.json"

FINAL_PAYLOAD_MEMBERS = (
    FINAL_BINDING_MEMBER,
    TERMINAL_RESULT_MEMBER,
    DIAGNOSTIC_SUMMARY_MEMBER,
    LEAKAGE_REPORT_MEMBER,
    PUBLICATION_DECISION_MEMBER,
    RUNTIME_SUMMARY_MEMBER,
    CLAIM_BOUNDARY_MEMBER,
)
FINAL_INDEXED_MEMBERS = (*FINAL_PAYLOAD_MEMBERS, FINAL_ATTESTATION_MEMBER)

# Exact catalog-required file inventory for a scientifically COMPLETE output.
# The run lock is immutable same-run state but not a catalog member.
COMPLETE_CATALOG_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "provenance/execution_admission.json",
    "provenance/authorization_consumption_lease.json",
    FINAL_BINDING_MEMBER,
    "physical/source_streams/arrays/frozen_source_streams.npy",
    "physical/source_streams/manifests/frozen_source_stream_index.json",
    "physical/source_streams/manifests/frozen_source_stream_lock.json",
    "physical/predictions/arrays/fixed_bank_a1_action_probabilities.npz",
    "physical/predictions/manifests/fixed_bank_a1_prediction_index.json",
    "physical/predictions/manifests/fixed_bank_a1_prediction_seal.json",
    "arrays/preterminal_probability_matrix.npy",
    "manifests/preterminal_result.json",
    "reports/launch_receipts.json",
    "reports/preterminal_fresh_process_attestation.json",
    TERMINAL_METRICS_MEMBER,
    TERMINAL_RESULT_MEMBER,
    DIAGNOSTIC_SUMMARY_MEMBER,
    LEAKAGE_REPORT_MEMBER,
    PUBLICATION_DECISION_MEMBER,
    RUNTIME_SUMMARY_MEMBER,
    CLAIM_BOUNDARY_MEMBER,
    CONTENT_INDEX_MEMBER,
    FINAL_ATTESTATION_MEMBER,
    VALIDATION_REPORT_MEMBER,
    VALIDATION_INDEX_MEMBER,
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    "reports/run_state.json",
    "preparation/final_authorization_envelope.json",
    "preparation/execution_launch_authority.json",
    "preparation/sealed_execution_replay.json",
    "COMMITTED",
)
COMPLETE_INTERNAL_MEMBERS = (".run.lock",)

COMPLETION_COMMIT_MEMBER = "completion_commit.json"
COMPLETION_ABORT_MEMBER = "completion_abort.json"

_COMPLETE_ARTIFACT_SEAL_TOKEN = object()
_COMPLETION_COMMIT_TOKEN = object()
_INTERRUPTED_COMPLETION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class CompleteArtifactSealReceipt:
    """Authority issued only after durable whole-artifact validation."""

    artifact_root: Path
    prepared_state_hash: str
    prepared_state_receipt_hash: str
    final_bundle_receipt_hash: str
    artifact_inventory_hash: str
    complete_artifact_index_hash: str
    complete_artifact_index_file_sha256: str
    semantic_validation_hash: str
    source_seal_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _COMPLETE_ARTIFACT_SEAL_TOKEN:
            raise ProtocolError(
                "OE-PPUR v4 complete artifact seal bypassed durable validation."
            )
        root = Path(self.artifact_root)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
            or root == Path(root.anchor)
        ):
            raise ProtocolError("OE-PPUR v4 complete artifact seal root is unsafe.")
        object.__setattr__(self, "artifact_root", root)
        for role in (
            "prepared_state_hash",
            "prepared_state_receipt_hash",
            "final_bundle_receipt_hash",
            "artifact_inventory_hash",
            "complete_artifact_index_hash",
            "complete_artifact_index_file_sha256",
            "semantic_validation_hash",
            "source_seal_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_complete_artifact_seal_receipt_v1",
            "artifact_root": self.artifact_root.as_posix(),
            "prepared_state_hash": self.prepared_state_hash,
            "prepared_state_receipt_hash": self.prepared_state_receipt_hash,
            "final_bundle_receipt_hash": self.final_bundle_receipt_hash,
            "artifact_inventory_hash": self.artifact_inventory_hash,
            "complete_artifact_index_hash": self.complete_artifact_index_hash,
            "complete_artifact_index_file_sha256": (
                self.complete_artifact_index_file_sha256
            ),
            "semantic_validation_hash": self.semantic_validation_hash,
            "source_seal_hash": self.source_seal_hash,
        }


@dataclass(frozen=True, slots=True)
class CompletionCommitReceipt:
    lease_path: Path
    claim_hash: str
    prepared_state_receipt_hash: str
    prepared_state_hash: str
    final_bundle_receipt_hash: str
    complete_artifact_seal_receipt_hash: str
    artifact_inventory_hash: str
    journal_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _COMPLETION_COMMIT_TOKEN:
            raise ProtocolError("OE-PPUR v4 completion journal bypassed validation.")
        path = Path(self.lease_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ProtocolError("OE-PPUR v4 completion journal path drifted.")
        object.__setattr__(self, "lease_path", path)
        for role in (
            "claim_hash",
            "prepared_state_receipt_hash",
            "prepared_state_hash",
            "final_bundle_receipt_hash",
            "complete_artifact_seal_receipt_hash",
            "artifact_inventory_hash",
            "journal_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_completion_commit_receipt_v1",
            "lease_path": self.lease_path.as_posix(),
            "claim_hash": self.claim_hash,
            "prepared_state_receipt_hash": self.prepared_state_receipt_hash,
            "prepared_state_hash": self.prepared_state_hash,
            "final_bundle_receipt_hash": self.final_bundle_receipt_hash,
            "complete_artifact_seal_receipt_hash": (
                self.complete_artifact_seal_receipt_hash
            ),
            "artifact_inventory_hash": self.artifact_inventory_hash,
            "journal_hash": self.journal_hash,
        }


@dataclass(frozen=True, slots=True)
class InterruptedCompletionReceipt:
    lease_path: Path
    evidence_hash: str
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _INTERRUPTED_COMPLETION_TOKEN:
            raise ProtocolError("OE-PPUR v4 interrupted completion bypassed discovery.")
        object.__setattr__(self, "lease_path", Path(self.lease_path))
        object.__setattr__(
            self,
            "evidence_hash",
            require_sha256(self.evidence_hash, "completion interruption evidence"),
        )


def _issue_complete_artifact_seal(
    *,
    artifact_root: Path,
    prepared_state_hash: str,
    prepared_state_receipt_hash: str,
    final_bundle_receipt_hash: str,
    artifact_inventory_hash: str,
    complete_artifact_index_hash: str,
    complete_artifact_index_file_sha256: str,
    semantic_validation_hash: str,
    source_seal_hash: str,
) -> CompleteArtifactSealReceipt:
    return CompleteArtifactSealReceipt(
        artifact_root=artifact_root,
        prepared_state_hash=prepared_state_hash,
        prepared_state_receipt_hash=prepared_state_receipt_hash,
        final_bundle_receipt_hash=final_bundle_receipt_hash,
        artifact_inventory_hash=artifact_inventory_hash,
        complete_artifact_index_hash=complete_artifact_index_hash,
        complete_artifact_index_file_sha256=complete_artifact_index_file_sha256,
        semantic_validation_hash=semantic_validation_hash,
        source_seal_hash=source_seal_hash,
        _factory_token=_COMPLETE_ARTIFACT_SEAL_TOKEN,
    )


def _issue_completion_commit_receipt(
    *,
    lease_path: Path,
    claim_hash: str,
    prepared_state_receipt_hash: str,
    prepared_state_hash: str,
    final_bundle_receipt_hash: str,
    complete_artifact_seal_receipt_hash: str,
    artifact_inventory_hash: str,
    journal_hash: str,
) -> CompletionCommitReceipt:
    return CompletionCommitReceipt(
        lease_path=lease_path,
        claim_hash=claim_hash,
        prepared_state_receipt_hash=prepared_state_receipt_hash,
        prepared_state_hash=prepared_state_hash,
        final_bundle_receipt_hash=final_bundle_receipt_hash,
        complete_artifact_seal_receipt_hash=complete_artifact_seal_receipt_hash,
        artifact_inventory_hash=artifact_inventory_hash,
        journal_hash=journal_hash,
        _factory_token=_COMPLETION_COMMIT_TOKEN,
    )


def _issue_interrupted_completion_receipt(
    *, lease_path: Path, evidence_hash: str
) -> InterruptedCompletionReceipt:
    return InterruptedCompletionReceipt(
        lease_path=lease_path,
        evidence_hash=evidence_hash,
        _factory_token=_INTERRUPTED_COMPLETION_TOKEN,
    )


__all__ = (
    "CLAIM_BOUNDARY_MEMBER",
    "COMPLETE_ARTIFACT_INDEX_MEMBER",
    "COMPLETE_CATALOG_MEMBERS",
    "COMPLETE_INTERNAL_MEMBERS",
    "COMPLETION_ABORT_MEMBER",
    "COMPLETION_COMMIT_MEMBER",
    "CONTENT_INDEX_MEMBER",
    "CompleteArtifactSealReceipt",
    "CompletionCommitReceipt",
    "DIAGNOSTIC_SUMMARY_MEMBER",
    "FINAL_ATTESTATION_MEMBER",
    "FINAL_BINDING_MEMBER",
    "FINAL_INDEXED_MEMBERS",
    "FINAL_PAYLOAD_MEMBERS",
    "InterruptedCompletionReceipt",
    "LEAKAGE_REPORT_MEMBER",
    "PUBLICATION_DECISION_MEMBER",
    "RUNTIME_SUMMARY_MEMBER",
    "TERMINAL_METRICS_MEMBER",
    "TERMINAL_RESULT_MEMBER",
    "VALIDATION_INDEX_MEMBER",
    "VALIDATION_REPORT_MEMBER",
)
