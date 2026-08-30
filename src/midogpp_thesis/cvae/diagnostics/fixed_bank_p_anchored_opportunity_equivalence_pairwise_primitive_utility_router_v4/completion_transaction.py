"""Durable pre-COMPLETE journal and abort transaction for OE-PPUR v4."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ...protocol import ProtocolError
from .artifact.completion import (
    _completion_commit_receipt,
    discover_completion_commit,
)
from .artifact.contracts import (
    COMPLETION_ABORT_MEMBER,
    COMPLETION_COMMIT_MEMBER,
    CompleteArtifactSealReceipt,
    CompletionCommitReceipt,
    InterruptedCompletionReceipt,
)
from .hashing import canonical_hash
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .lease_io import (
    fsync_directory,
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)

def record_completion_commit(
    claim: AuthorizationLeaseClaim,
    *,
    prepared_state: object,
    final_bundle: object,
    complete_artifact_seal: object,
) -> CompletionCommitReceipt:
    from .complete_artifact_validation import validate_complete_artifact_seal
    from .output_validation import (
        FinalAggregateBundleReceipt,
        validate_final_aggregate_bundle,
    )
    from .run_state import (
        PreparedCompleteRunState,
        validate_prepared_complete_run_state,
    )

    validated_claim = validate_authorization_lease(claim)
    if (
        (validated_claim.path / COMPLETION_ABORT_MEMBER).exists()
        or (validated_claim.path / COMPLETION_ABORT_MEMBER).is_symlink()
        or pending_publications(validated_claim.path, COMPLETION_ABORT_MEMBER)
    ):
        raise ProtocolError("OE-PPUR v4 completion transaction was already aborted.")
    if (
        type(prepared_state) is not PreparedCompleteRunState
        or type(final_bundle) is not FinalAggregateBundleReceipt
        or type(complete_artifact_seal) is not CompleteArtifactSealReceipt
    ):
        raise ProtocolError("OE-PPUR v4 completion journal inputs are untyped.")
    prepared = validate_prepared_complete_run_state(prepared_state)
    root = Path(str(validated_claim.payload["artifact_root"]))
    bundle = validate_final_aggregate_bundle(root, expected_receipt=final_bundle)
    complete_seal = validate_complete_artifact_seal(
        root,
        expected=complete_artifact_seal,
        expected_complete_state=prepared,
    )
    if (
        prepared.artifact_root != root
        or prepared.authorization_lease_claim_hash != validated_claim.claim_hash
        or prepared.run_identity_hash
        != validated_claim.payload.get("run_identity_hash")
        or prepared.final_bundle_receipt_hash != bundle.receipt_hash
        or complete_seal.artifact_root != root
        or complete_seal.prepared_state_hash != prepared.state_hash
        or complete_seal.prepared_state_receipt_hash != prepared.receipt_hash
        or complete_seal.final_bundle_receipt_hash != bundle.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 completion journal lineage drifted.")
    body = {
        "schema_version": "oe_ppur_v4_completion_commit_v1",
        "status": "PREPARED_COMPLETE_DURABLE",
        "claim_hash": validated_claim.claim_hash,
        "run_identity_hash": prepared.run_identity_hash,
        "artifact_root": root.as_posix(),
        "prepared_state_receipt_hash": prepared.receipt_hash,
        "prepared_state_hash": prepared.state_hash,
        "pending_state_hash": prepared.pending_state_hash,
        "final_bundle_receipt_hash": bundle.receipt_hash,
        "complete_artifact_seal_receipt_hash": complete_seal.receipt_hash,
        "artifact_inventory_hash": complete_seal.artifact_inventory_hash,
        "committed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "journal_hash": canonical_hash(body)}
    publish_json_no_overwrite(
        validated_claim.path / COMPLETION_COMMIT_MEMBER,
        payload,
        role="completion commit journal",
    )
    fsync_directory(validated_claim.path)
    receipt = _completion_commit_receipt(validated_claim.path, payload)
    return validate_completion_commit(
        receipt,
        expected_prepared_state=prepared,
    )


def validate_completion_commit(
    value: CompletionCommitReceipt,
    *,
    expected_prepared_state: object,
) -> CompletionCommitReceipt:
    from .run_state import PreparedCompleteRunState

    if (
        type(value) is not CompletionCommitReceipt
        or type(expected_prepared_state) is not PreparedCompleteRunState
    ):
        raise ProtocolError("OE-PPUR v4 completion journal validation is untyped.")
    if (
        pending_publications(value.lease_path, COMPLETION_COMMIT_MEMBER)
        or pending_publications(value.lease_path, COMPLETION_ABORT_MEMBER)
        or (value.lease_path / COMPLETION_ABORT_MEMBER).exists()
        or (value.lease_path / COMPLETION_ABORT_MEMBER).is_symlink()
    ):
        raise ProtocolError("OE-PPUR v4 completion journal is interrupted.")
    payload = read_json_regular(
        value.lease_path / COMPLETION_COMMIT_MEMBER,
        role="completion commit journal",
    )
    observed = _completion_commit_receipt(value.lease_path, payload)
    prepared = expected_prepared_state
    claim_payload = read_json_regular(
        value.lease_path / "claim.json",
        role="authorization claim",
    )
    claim_hash = str(claim_payload.get("claim_hash", ""))
    validate_authorization_lease(
        AuthorizationLeaseClaim(value.lease_path, claim_payload, claim_hash)
    )
    if (
        observed != value
        or observed.claim_hash != claim_hash
        or payload.get("artifact_root") != prepared.artifact_root.as_posix()
        or payload.get("run_identity_hash") != prepared.run_identity_hash
        or payload.get("pending_state_hash") != prepared.pending_state_hash
        or observed.prepared_state_receipt_hash != prepared.receipt_hash
        or observed.prepared_state_hash != prepared.state_hash
        or observed.final_bundle_receipt_hash
        != prepared.final_bundle_receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 completion journal changed after issuance.")
    return observed


def record_completion_abort(
    claim: AuthorizationLeaseClaim,
    *,
    completion: CompletionCommitReceipt | InterruptedCompletionReceipt,
    original_error: BaseException,
    artifact_root: Path,
) -> str:
    validated_claim = validate_authorization_lease(claim)
    if (
        type(completion) not in {CompletionCommitReceipt, InterruptedCompletionReceipt}
        or not isinstance(original_error, BaseException)
        or Path(artifact_root)
        != Path(str(validated_claim.payload["artifact_root"]))
    ):
        raise ProtocolError("OE-PPUR v4 completion abort evidence is untyped.")
    state_hash: str | None = None
    state_phase: str | None = None
    state_status: str | None = None
    state_path = artifact_root / "reports/run_state.json"
    if state_path.exists() and not state_path.is_symlink():
        from .run_state import read_run_state

        state = read_run_state(artifact_root)
        state_hash = str(state["state_hash"])
        state_phase = str(state["phase"])
        state_status = str(state["status"])
    commit_hash = (
        completion.journal_hash
        if type(completion) is CompletionCommitReceipt
        else completion.evidence_hash
    )
    body = {
        "schema_version": "oe_ppur_v4_completion_abort_v2",
        "status": "FAILED_EXHAUSTED_AFTER_COMPLETION_TRANSACTION",
        "claim_hash": validated_claim.claim_hash,
        "completion_transaction_hash": commit_hash,
        "prepared_state_hash": getattr(completion, "prepared_state_hash", None),
        "complete_artifact_seal_receipt_hash": getattr(
            completion,
            "complete_artifact_seal_receipt_hash",
            None,
        ),
        "observed_run_state_hash": state_hash,
        "observed_run_state_phase": state_phase,
        "observed_run_state_status": state_status,
        "error_class": _safe_text(type(original_error).__name__),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "abort_hash": canonical_hash(body)}
    publish_json_no_overwrite(
        validated_claim.path / COMPLETION_ABORT_MEMBER,
        payload,
        role="completion abort journal",
    )
    fsync_directory(validated_claim.path)
    observed = read_json_regular(
        validated_claim.path / COMPLETION_ABORT_MEMBER,
        role="completion abort journal",
    )
    if observed != payload:
        raise ProtocolError("OE-PPUR v4 completion abort read-back drifted.")
    return str(payload["abort_hash"])


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


__all__ = (
    "COMPLETION_ABORT_MEMBER",
    "COMPLETION_COMMIT_MEMBER",
    "CompletionCommitReceipt",
    "InterruptedCompletionReceipt",
    "discover_completion_commit",
    "record_completion_abort",
    "record_completion_commit",
    "validate_completion_commit",
)
