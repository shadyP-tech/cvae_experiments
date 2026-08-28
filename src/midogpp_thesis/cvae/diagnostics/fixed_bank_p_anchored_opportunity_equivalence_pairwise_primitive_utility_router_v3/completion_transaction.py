"""Durable pre-COMPLETE journal and abort transaction for OE-PPUR v3."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .lease_io import (
    fsync_directory,
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)


COMPLETION_COMMIT_MEMBER = "completion_commit.json"
COMPLETION_ABORT_MEMBER = "completion_abort.json"
_COMPLETION_COMMIT_TOKEN = object()
_INTERRUPTED_COMPLETION_TOKEN = object()


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
            raise ProtocolError("OE-PPUR v3 completion journal bypassed validation.")
        path = Path(self.lease_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ProtocolError("OE-PPUR v3 completion journal path drifted.")
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
            "schema_version": "oe_ppur_v3_completion_commit_receipt_v1",
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
            raise ProtocolError("OE-PPUR v3 interrupted completion bypassed discovery.")
        object.__setattr__(self, "lease_path", Path(self.lease_path))
        object.__setattr__(
            self,
            "evidence_hash",
            require_sha256(self.evidence_hash, "completion interruption evidence"),
        )


def record_completion_commit(
    claim: AuthorizationLeaseClaim,
    *,
    prepared_state: object,
    final_bundle: object,
    complete_artifact_seal: object,
) -> CompletionCommitReceipt:
    from .complete_artifact_validation import (
        CompleteArtifactSealReceipt,
        validate_complete_artifact_seal,
    )
    from .output_artifact import (
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
        raise ProtocolError("OE-PPUR v3 completion transaction was already aborted.")
    if (
        type(prepared_state) is not PreparedCompleteRunState
        or type(final_bundle) is not FinalAggregateBundleReceipt
        or type(complete_artifact_seal) is not CompleteArtifactSealReceipt
    ):
        raise ProtocolError("OE-PPUR v3 completion journal inputs are untyped.")
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
        raise ProtocolError("OE-PPUR v3 completion journal lineage drifted.")
    body = {
        "schema_version": "oe_ppur_v3_completion_commit_v1",
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
        raise ProtocolError("OE-PPUR v3 completion journal validation is untyped.")
    if (
        pending_publications(value.lease_path, COMPLETION_COMMIT_MEMBER)
        or pending_publications(value.lease_path, COMPLETION_ABORT_MEMBER)
        or (value.lease_path / COMPLETION_ABORT_MEMBER).exists()
        or (value.lease_path / COMPLETION_ABORT_MEMBER).is_symlink()
    ):
        raise ProtocolError("OE-PPUR v3 completion journal is interrupted.")
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
        raise ProtocolError("OE-PPUR v3 completion journal changed after issuance.")
    return observed


def discover_completion_commit(
    claim: AuthorizationLeaseClaim,
) -> CompletionCommitReceipt | InterruptedCompletionReceipt | None:
    validated = validate_authorization_lease(claim)
    abort_pending = pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
    abort_path = validated.path / COMPLETION_ABORT_MEMBER
    if abort_pending or abort_path.exists() or abort_path.is_symlink():
        members = abort_pending or (abort_path,)
        return InterruptedCompletionReceipt(
            lease_path=validated.path,
            evidence_hash=canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_aborted_completion_discovery_v1",
                    "claim_hash": validated.claim_hash,
                    "members": [path.name for path in members],
                }
            ),
            _factory_token=_INTERRUPTED_COMPLETION_TOKEN,
        )
    pending = pending_publications(validated.path, COMPLETION_COMMIT_MEMBER)
    if pending:
        return InterruptedCompletionReceipt(
            lease_path=validated.path,
            evidence_hash=canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_interrupted_completion_v1",
                    "claim_hash": validated.claim_hash,
                    "pending_members": [path.name for path in pending],
                }
            ),
            _factory_token=_INTERRUPTED_COMPLETION_TOKEN,
        )
    path = validated.path / COMPLETION_COMMIT_MEMBER
    if not path.exists() and not path.is_symlink():
        return None
    payload = read_json_regular(path, role="completion commit journal")
    receipt = _completion_commit_receipt(validated.path, payload)
    if receipt.claim_hash != validated.claim_hash:
        raise ProtocolError("OE-PPUR v3 completion journal/claim binding drifted.")
    return receipt


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
        raise ProtocolError("OE-PPUR v3 completion abort evidence is untyped.")
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
        "schema_version": "oe_ppur_v3_completion_abort_v2",
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
        raise ProtocolError("OE-PPUR v3 completion abort read-back drifted.")
    return str(payload["abort_hash"])


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
        or payload.get("schema_version") != "oe_ppur_v3_completion_commit_v1"
        or payload.get("status") != "PREPARED_COMPLETE_DURABLE"
        or payload.get("journal_hash") != canonical_hash(body)
        or not isinstance(payload.get("committed_at_utc"), str)
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_restored") is not False
        or payload.get("recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v3 completion journal drifted.")
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
    return CompletionCommitReceipt(
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
        _factory_token=_COMPLETION_COMMIT_TOKEN,
    )


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
