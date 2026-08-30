"""Read-only completion-journal discovery and schema reconstruction.

The completion writer depends on this reader contract.  Whole-artifact
semantic reopening also depends on it, so neither needs to import the other.
"""

from __future__ import annotations

from pathlib import Path

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from ..lease_io import pending_publications, read_json_regular
from .contracts import (
    COMPLETION_ABORT_MEMBER,
    COMPLETION_COMMIT_MEMBER,
    CompletionCommitReceipt,
    InterruptedCompletionReceipt,
    _issue_completion_commit_receipt,
    _issue_interrupted_completion_receipt,
)


def discover_completion_commit(
    claim: AuthorizationLeaseClaim,
) -> CompletionCommitReceipt | InterruptedCompletionReceipt | None:
    validated = validate_authorization_lease(claim)
    abort_pending = pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
    abort_path = validated.path / COMPLETION_ABORT_MEMBER
    if abort_pending or abort_path.exists() or abort_path.is_symlink():
        members = abort_pending or (abort_path,)
        return _issue_interrupted_completion_receipt(
            lease_path=validated.path,
            evidence_hash=canonical_hash(
                {
                    "schema_version": "oe_ppur_v4_aborted_completion_discovery_v1",
                    "claim_hash": validated.claim_hash,
                    "members": [path.name for path in members],
                }
            ),
        )
    pending = pending_publications(validated.path, COMPLETION_COMMIT_MEMBER)
    if pending:
        return _issue_interrupted_completion_receipt(
            lease_path=validated.path,
            evidence_hash=canonical_hash(
                {
                    "schema_version": "oe_ppur_v4_interrupted_completion_v1",
                    "claim_hash": validated.claim_hash,
                    "pending_members": [path.name for path in pending],
                }
            ),
        )
    path = validated.path / COMPLETION_COMMIT_MEMBER
    if not path.exists() and not path.is_symlink():
        return None
    payload = read_json_regular(path, role="completion commit journal")
    receipt = _completion_commit_receipt(validated.path, payload)
    if receipt.claim_hash != validated.claim_hash:
        raise ProtocolError("OE-PPUR v4 completion journal/claim binding drifted.")
    return receipt


def _completion_commit_receipt(
    lease_path: Path,
    payload: dict[str, object],
) -> CompletionCommitReceipt:
    expected_keys = {
        "schema_version",
        "status",
        "claim_hash",
        "run_identity_hash",
        "artifact_root",
        "prepared_state_receipt_hash",
        "prepared_state_hash",
        "pending_state_hash",
        "final_bundle_receipt_hash",
        "complete_artifact_seal_receipt_hash",
        "artifact_inventory_hash",
        "committed_at_utc",
        "authorization_exhausted",
        "authorization_restored",
        "recovery_allowed",
        "journal_hash",
    }
    body = {key: item for key, item in payload.items() if key != "journal_hash"}
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != "oe_ppur_v4_completion_commit_v1"
        or payload.get("status") != "PREPARED_COMPLETE_DURABLE"
        or payload.get("journal_hash") != canonical_hash(body)
        or not isinstance(payload.get("committed_at_utc"), str)
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_restored") is not False
        or payload.get("recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v4 completion journal drifted.")
    for role in (
        "claim_hash",
        "run_identity_hash",
        "prepared_state_receipt_hash",
        "prepared_state_hash",
        "pending_state_hash",
        "final_bundle_receipt_hash",
        "complete_artifact_seal_receipt_hash",
        "artifact_inventory_hash",
        "journal_hash",
    ):
        require_sha256(payload.get(role), role.replace("_", " "))
    return _issue_completion_commit_receipt(
        lease_path=Path(lease_path),
        claim_hash=str(payload["claim_hash"]),
        prepared_state_receipt_hash=str(payload["prepared_state_receipt_hash"]),
        prepared_state_hash=str(payload["prepared_state_hash"]),
        final_bundle_receipt_hash=str(payload["final_bundle_receipt_hash"]),
        complete_artifact_seal_receipt_hash=str(
            payload["complete_artifact_seal_receipt_hash"]
        ),
        artifact_inventory_hash=str(payload["artifact_inventory_hash"]),
        journal_hash=str(payload["journal_hash"]),
    )


__all__ = ("discover_completion_commit",)
