"""Typed terminal-state to authorization-outcome composition."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ...protocol import ProtocolError
from .authorization_outcome_contracts import (
    AuthorizationOutcomeReceipt,
    build_authorization_outcome_payload,
    safe_text,
)
from .authorization_outcome_store import persist_authorization_outcome
from .complete_artifact_validation import CompleteArtifactSealReceipt
from .complete_run_validation import reopen_complete_run_evidence
from .completion_transaction import CompletionCommitReceipt
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .output_validation import FinalAggregateBundleReceipt
from .run_state import (
    PreparedCompleteRunState,
    TerminalRunStateReceipt,
    read_run_state,
    validate_terminal_run_state,
)


def record_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    terminal_state: object,
    final_bundle: object | None = None,
    prepared_state: object | None = None,
    completion_commit: object | None = None,
    complete_artifact_seal: object | None = None,
) -> AuthorizationOutcomeReceipt:
    validated = validate_authorization_lease(claim)
    if type(terminal_state) is not TerminalRunStateReceipt:
        raise ProtocolError("OE-PPUR v4 authorization outcome requires typed state.")
    state = validate_terminal_run_state(terminal_state)
    artifact_root = Path(str(validated.payload["artifact_root"]))
    if (
        state.artifact_root != artifact_root
        or state.authorization_lease_claim_hash != validated.claim_hash
        or state.run_identity_hash != validated.payload.get("run_identity_hash")
    ):
        raise ProtocolError("OE-PPUR v4 authorization outcome/state binding drifted.")
    status = state.status
    final_bundle_hash = inventory_hash = lifecycle_hash = None
    commit_hash = complete_seal_hash = None
    if status == "COMPLETE":
        if (
            type(final_bundle) is not FinalAggregateBundleReceipt
            or type(prepared_state) is not PreparedCompleteRunState
            or type(completion_commit) is not CompletionCommitReceipt
            or type(complete_artifact_seal) is not CompleteArtifactSealReceipt
        ):
            raise ProtocolError(
                "OE-PPUR v4 complete outcome requires typed transaction evidence."
            )
        evidence = reopen_complete_run_evidence(
            validated,
            terminal_state=state,
            final_bundle=final_bundle,
            prepared_state=prepared_state,
            completion_commit=completion_commit,
            complete_artifact_seal=complete_artifact_seal,
        )
        final_bundle_hash = evidence.final_bundle.receipt_hash
        inventory_hash = evidence.complete_artifact_seal.artifact_inventory_hash
        lifecycle_hash = evidence.lifecycle_lineage_hash
        commit_hash = evidence.completion_commit.journal_hash
        complete_seal_hash = evidence.complete_artifact_seal.receipt_hash
        error_class = None
    elif status == "FAILED_EXHAUSTED":
        if any(
            value is not None
            for value in (
                final_bundle,
                prepared_state,
                completion_commit,
                complete_artifact_seal,
            )
        ):
            raise ProtocolError("OE-PPUR v4 failed outcome accepted final state.")
        raw_state = read_run_state(artifact_root)
        error_class = safe_text(raw_state.get("error_class"))
        if not error_class:
            raise ProtocolError("OE-PPUR v4 failed outcome lacks its error class.")
    else:  # pragma: no cover
        raise ProtocolError("OE-PPUR v4 authorization outcome status drifted.")
    payload = build_authorization_outcome_payload(
        status=status,
        claim_hash=validated.claim_hash,
        evidence_hash=state.state_hash,
        terminal_run_state_receipt_hash=state.receipt_hash,
        final_bundle_receipt_hash=final_bundle_hash,
        artifact_inventory_hash=inventory_hash,
        lifecycle_lineage_hash=lifecycle_hash,
        completion_commit_hash=commit_hash,
        complete_artifact_seal_receipt_hash=complete_seal_hash,
        error_class=error_class,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return persist_authorization_outcome(validated, payload)


__all__ = ("record_authorization_outcome",)
