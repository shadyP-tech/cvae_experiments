"""Dependency-inverted COMPLETE transition composition for OE-PPUR v4."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from ...protocol import ProtocolError
from .artifact.completion import discover_completion_commit
from .artifact.contracts import (
    COMPLETION_ABORT_MEMBER,
    COMPLETION_COMMIT_MEMBER,
    CompletionCommitReceipt,
)
from .hashing import canonical_hash
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .lease_io import pending_publications, read_json_regular
from .output_validation import (
    FinalAggregateBundleReceipt,
    validate_final_aggregate_bundle,
)
from .run_state_contracts import (
    PreparedCompleteRunState,
    TerminalRunStateReceipt,
    _issue_prepared_complete_run_state,
    _thaw_json,
)


ReadState = Callable[[Path], dict[str, object]]
ReplaceState = Callable[..., dict[str, object]]
WriteState = Callable[[Path, Mapping[str, object]], None]
IssueTerminal = Callable[[Path, Mapping[str, object]], TerminalRunStateReceipt]


def prepare_complete_run_state(
    artifact_root: Path,
    *,
    final_bundle: object,
    read_state: ReadState,
    replace_state: ReplaceState,
) -> PreparedCompleteRunState:
    if type(final_bundle) is not FinalAggregateBundleReceipt:
        raise ProtocolError("OE-PPUR v4 completion requires a typed final bundle.")
    validated_bundle = validate_final_aggregate_bundle(
        artifact_root,
        expected_receipt=final_bundle,
    )
    state = read_state(Path(artifact_root))
    transitions = state.get("transitions")
    if (
        state.get("status") != "RUNNING"
        or state.get("phase") != "COMPLETION_PENDING"
        or not isinstance(transitions, list)
        or not transitions
        or transitions[-1].get("evidence_hash")
        != validated_bundle.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 completion preparation requires pending state.")
    prepared_payload = replace_state(
        state,
        status="COMPLETE",
        phase="COMPLETE",
        evidence_hash=validated_bundle.receipt_hash,
        error_class=None,
    )
    prepared_body = {
        **{
            key: value
            for key, value in prepared_payload.items()
            if key != "state_hash"
        },
        "updated_at_utc": state["updated_at_utc"],
    }
    prepared_payload = {
        **prepared_body,
        "state_hash": canonical_hash(prepared_body),
    }
    return _issue_prepared_complete_run_state(
        artifact_root=Path(artifact_root),
        payload=prepared_payload,
        state_hash=str(prepared_payload["state_hash"]),
        pending_state_hash=str(state["state_hash"]),
        final_bundle_receipt_hash=validated_bundle.receipt_hash,
        run_identity_hash=str(state["run_identity_hash"]),
        authorization_lease_claim_hash=str(
            state["authorization_lease_claim_hash"]
        ),
    )


def validate_prepared_complete_run_state(
    prepared: PreparedCompleteRunState,
    *,
    read_state: ReadState,
) -> PreparedCompleteRunState:
    if type(prepared) is not PreparedCompleteRunState:
        raise ProtocolError("OE-PPUR v4 prepared COMPLETE state is untyped.")
    observed = read_state(prepared.artifact_root)
    if (
        observed.get("status") != "RUNNING"
        or observed.get("phase") != "COMPLETION_PENDING"
        or observed.get("state_hash") != prepared.pending_state_hash
        or observed.get("run_identity_hash") != prepared.run_identity_hash
        or observed.get("authorization_lease_claim_hash")
        != prepared.authorization_lease_claim_hash
        or prepared.receipt_hash != canonical_hash(prepared.to_payload())
    ):
        raise ProtocolError("OE-PPUR v4 prepared COMPLETE state changed before commit.")
    return prepared


def commit_complete_run_state(
    prepared: PreparedCompleteRunState,
    *,
    completion_commit: object,
    read_state: ReadState,
    write_state: WriteState,
    issue_terminal: IssueTerminal,
) -> TerminalRunStateReceipt:
    validated_prepared = validate_prepared_complete_run_state(
        prepared,
        read_state=read_state,
    )
    validated_commit = _validate_completion_commit(
        completion_commit,
        expected_prepared_state=validated_prepared,
    )
    if (
        validated_commit.prepared_state_receipt_hash
        != validated_prepared.receipt_hash
        or validated_commit.prepared_state_hash != validated_prepared.state_hash
    ):
        raise ProtocolError("OE-PPUR v4 completion journal/state binding drifted.")
    payload = _thaw_json(validated_prepared.payload)
    if not isinstance(payload, dict):  # pragma: no cover - constructor guarded
        raise ProtocolError("OE-PPUR v4 prepared COMPLETE payload is malformed.")
    write_state(
        validated_prepared.artifact_root / "reports/run_state.json",
        payload,
    )
    observed = issue_terminal(
        validated_prepared.artifact_root,
        read_state(validated_prepared.artifact_root),
    )
    if observed.state_hash != validated_prepared.state_hash:
        raise ProtocolError("OE-PPUR v4 committed COMPLETE bytes drifted.")
    return observed


def _validate_completion_commit(
    value: object,
    *,
    expected_prepared_state: PreparedCompleteRunState,
) -> CompletionCommitReceipt:
    if type(value) is not CompletionCommitReceipt:
        raise ProtocolError("OE-PPUR v4 COMPLETE commit requires typed journal.")
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
    claim_payload = read_json_regular(
        value.lease_path / "claim.json",
        role="authorization claim",
    )
    claim_hash = str(claim_payload.get("claim_hash", ""))
    claim = validate_authorization_lease(
        AuthorizationLeaseClaim(value.lease_path, claim_payload, claim_hash)
    )
    observed = discover_completion_commit(claim)
    prepared = expected_prepared_state
    if (
        type(observed) is not CompletionCommitReceipt
        or observed != value
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


__all__ = ()
