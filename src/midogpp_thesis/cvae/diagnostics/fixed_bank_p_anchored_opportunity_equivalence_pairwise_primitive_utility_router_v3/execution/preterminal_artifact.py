"""Compatibility facade for OE-PPUR v3 preterminal persistence and attestation."""

from __future__ import annotations

# Keep this alias for focused syscall-observation tests.  It is the same module
# object used by the persistence implementation, so monkeypatches remain valid.
import os
from pathlib import Path

from .fresh_attestation import (
    FinalAggregateAttestationReceipt,
    _reconstruct_final_aggregate_attestation,
    _validate_preterminal_files,
    attest_preterminal_artifact_twice,
    attest_terminal_aggregate_twice,
)
from .preterminal_persistence import (
    ATTESTATION_MEMBERS,
    FINAL_ATTESTATION_MEMBER,
    MANIFEST_MEMBER,
    MATRIX_MEMBER,
    PRETERMINAL_ATTESTATION_MEMBER,
    PersistedPreterminalArtifact,
    _array_sha256,
    _fsync_preterminal_tree,
    _write_json_exclusive,
    _write_npy_exclusive,
    persist_attestation_json_exclusive,
    persist_preterminal_files,
)
from .services import CanonicalPreterminalResult, CanonicalRouterExecutionRequest


def persist_preterminal_artifact(
    root: Path,
    result: CanonicalPreterminalResult,
    request: CanonicalRouterExecutionRequest,
) -> PersistedPreterminalArtifact:
    """Compose durable file persistence with independent lineage validation."""

    artifact_root, matrix_path, manifest_path = persist_preterminal_files(
        root,
        result,
        request,
    )
    validated = _validate_preterminal_files(
        manifest_path,
        matrix_path,
        expected_ledger_hash=result.decision_ledger.ledger_hash,
        expected_result_hash=result.result_hash,
    )
    return PersistedPreterminalArtifact(
        root=artifact_root,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        artifact_file_sha256=str(validated["artifact_file_sha256"]),
        artifact_file_identity_sha256=str(
            validated["artifact_file_identity_sha256"]
        ),
        decision_ledger_hash=result.decision_ledger.ledger_hash,
        result_hash=result.result_hash,
    )


__all__ = (
    "ATTESTATION_MEMBERS",
    "FINAL_ATTESTATION_MEMBER",
    "FinalAggregateAttestationReceipt",
    "MANIFEST_MEMBER",
    "MATRIX_MEMBER",
    "PRETERMINAL_ATTESTATION_MEMBER",
    "PersistedPreterminalArtifact",
    "attest_preterminal_artifact_twice",
    "attest_terminal_aggregate_twice",
    "persist_attestation_json_exclusive",
    "persist_preterminal_artifact",
)
