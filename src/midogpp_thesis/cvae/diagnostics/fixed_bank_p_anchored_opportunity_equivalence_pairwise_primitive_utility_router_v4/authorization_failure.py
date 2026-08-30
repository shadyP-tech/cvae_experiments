"""Irrecoverable post-claim failure finalization for OE-PPUR v4."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ...protocol import ProtocolError
from .authorization_outcome_contracts import (
    AuthorizationOutcomeReceipt,
    build_authorization_outcome_payload,
    safe_text,
)
from .authorization_outcome_recording import record_authorization_outcome
from .authorization_outcome_store import persist_authorization_outcome
from .completion_transaction import (
    discover_completion_commit,
    record_completion_abort,
)
from .hashing import canonical_hash, require_sha256
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease


def finalize_failed_authorization(
    claim: AuthorizationLeaseClaim,
    *,
    artifact_root: Path,
    original_error: BaseException,
) -> AuthorizationOutcomeReceipt:
    if not isinstance(original_error, BaseException):
        raise ProtocolError("OE-PPUR v4 failure finalization requires original error.")
    validated = validate_authorization_lease(claim)
    root = Path(artifact_root)
    if root != Path(str(validated.payload["artifact_root"])):
        raise ProtocolError("OE-PPUR v4 failure finalization root drifted.")
    try:
        from .run_state import (
            mark_failed_exhausted,
            read_run_state,
            read_terminal_run_state,
        )

        completion = discover_completion_commit(validated)
        completion_abort_hash = None
        if completion is not None:
            completion_abort_hash = record_completion_abort(
                validated,
                completion=completion,
                original_error=original_error,
                artifact_root=root,
            )
        state_path = root / "reports/run_state.json"
        terminal_state = None
        if state_path.exists() or state_path.is_symlink():
            state = read_run_state(root)
            if state.get("status") == "RUNNING":
                failure_hash = canonical_hash(
                    {
                        "schema_version": "oe_ppur_v4_runner_failure_v4",
                        "claim_hash": validated.claim_hash,
                        "run_identity_hash": validated.payload["run_identity_hash"],
                        "error_class": type(original_error).__name__,
                        "authorization_exhausted": True,
                    }
                )
                terminal_state = mark_failed_exhausted(
                    root,
                    error_class=type(original_error).__name__,
                    evidence_hash=failure_hash,
                )
            elif state.get("status") == "FAILED_EXHAUSTED":
                terminal_state = read_terminal_run_state(root)
            elif state.get("status") == "COMPLETE" and completion_abort_hash:
                return _record_fail_closed_outcome(
                    validated,
                    evidence_hash=completion_abort_hash,
                    error_class=type(original_error).__name__,
                )
            else:
                raise ProtocolError(
                    "OE-PPUR v4 failure bookkeeping encountered COMPLETE state."
                )
        if terminal_state is not None:
            return record_authorization_outcome(
                validated,
                terminal_state=terminal_state,
            )
        evidence_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v4_prestate_failure_v2",
                "claim_hash": validated.claim_hash,
                "run_identity_hash": validated.payload["run_identity_hash"],
                "error_class": type(original_error).__name__,
                "run_state_created": False,
                "authorization_exhausted": True,
            }
        )
        return _record_fail_closed_outcome(
            validated,
            evidence_hash=evidence_hash,
            error_class=type(original_error).__name__,
        )
    except BaseException as bookkeeping_error:
        bookkeeping_error.add_note(
            "Original OE-PPUR v4 execution failure: "
            f"{type(original_error).__name__}: {safe_text(original_error)}"
        )
        raise bookkeeping_error from original_error


def _record_fail_closed_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    evidence_hash: str,
    error_class: str,
) -> AuthorizationOutcomeReceipt:
    payload = build_authorization_outcome_payload(
        status="FAILED_EXHAUSTED",
        claim_hash=claim.claim_hash,
        evidence_hash=require_sha256(evidence_hash, "failure evidence hash"),
        terminal_run_state_receipt_hash=None,
        final_bundle_receipt_hash=None,
        artifact_inventory_hash=None,
        lifecycle_lineage_hash=None,
        completion_commit_hash=None,
        complete_artifact_seal_receipt_hash=None,
        error_class=safe_text(error_class),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return persist_authorization_outcome(claim, payload)


__all__ = ("finalize_failed_authorization",)
