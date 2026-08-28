"""Terminal authorization outcomes and whole-run validation for OE-PPUR v3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ...protocol import ProtocolError
from .completion_transaction import (
    COMPLETION_ABORT_MEMBER,
    CompletionCommitReceipt,
    discover_completion_commit,
    record_completion_abort,
    validate_completion_commit,
)
from .hashing import canonical_hash, require_sha256
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .lease_io import (
    fsync_directory,
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)
from .lifecycle_lineage import validate_complete_lifecycle_evidence


OUTCOME_MEMBER = "outcome.json"
_OUTCOME_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AuthorizationOutcomeReceipt:
    lease_path: Path
    status: str
    claim_hash: str
    evidence_hash: str
    terminal_run_state_receipt_hash: str | None
    final_bundle_receipt_hash: str | None
    artifact_inventory_hash: str | None
    lifecycle_lineage_hash: str | None
    outcome_hash: str
    completion_commit_hash: str | None = None
    complete_artifact_seal_receipt_hash: str | None = None
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _OUTCOME_TOKEN:
            raise ProtocolError("OE-PPUR v3 authorization outcome bypassed validation.")
        path = Path(self.lease_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_dir()
            or self.status not in {"COMPLETE", "FAILED_EXHAUSTED"}
        ):
            raise ProtocolError("OE-PPUR v3 authorization outcome receipt drifted.")
        object.__setattr__(self, "lease_path", path)
        for role in ("claim_hash", "evidence_hash", "outcome_hash"):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        for role in (
            "terminal_run_state_receipt_hash",
            "final_bundle_receipt_hash",
            "artifact_inventory_hash",
            "lifecycle_lineage_hash",
            "completion_commit_hash",
            "complete_artifact_seal_receipt_hash",
        ):
            value = getattr(self, role)
            if value is not None:
                object.__setattr__(
                    self,
                    role,
                    require_sha256(value, role.replace("_", " ")),
                )
        if self.status == "COMPLETE" and any(
            getattr(self, role) is None
            for role in (
                "terminal_run_state_receipt_hash",
                "final_bundle_receipt_hash",
                "artifact_inventory_hash",
                "lifecycle_lineage_hash",
                "completion_commit_hash",
                "complete_artifact_seal_receipt_hash",
            )
        ):
            raise ProtocolError("OE-PPUR v3 complete outcome lacks whole-run evidence.")
        if self.status == "FAILED_EXHAUSTED" and any(
            getattr(self, role) is not None
            for role in (
                "final_bundle_receipt_hash",
                "artifact_inventory_hash",
                "lifecycle_lineage_hash",
                "completion_commit_hash",
                "complete_artifact_seal_receipt_hash",
            )
        ):
            raise ProtocolError("OE-PPUR v3 failed outcome carries success evidence.")
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_authorization_outcome_receipt_v1",
            "lease_path": self.lease_path.as_posix(),
            "status": self.status,
            "claim_hash": self.claim_hash,
            "evidence_hash": self.evidence_hash,
            "terminal_run_state_receipt_hash": self.terminal_run_state_receipt_hash,
            "final_bundle_receipt_hash": self.final_bundle_receipt_hash,
            "artifact_inventory_hash": self.artifact_inventory_hash,
            "lifecycle_lineage_hash": self.lifecycle_lineage_hash,
            "completion_commit_hash": self.completion_commit_hash,
            "complete_artifact_seal_receipt_hash": (
                self.complete_artifact_seal_receipt_hash
            ),
            "outcome_hash": self.outcome_hash,
        }


def record_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    terminal_state: object,
    final_bundle: object | None = None,
    prepared_state: object | None = None,
    completion_commit: object | None = None,
    complete_artifact_seal: object | None = None,
) -> AuthorizationOutcomeReceipt:
    from .output_artifact import (
        FinalAggregateBundleReceipt,
        validate_final_aggregate_bundle,
    )
    from .run_state import (
        PreparedCompleteRunState,
        TerminalRunStateReceipt,
        validate_terminal_run_state,
    )

    validated = validate_authorization_lease(claim)
    if type(terminal_state) is not TerminalRunStateReceipt:
        raise ProtocolError("OE-PPUR v3 authorization outcome requires typed state.")
    state = validate_terminal_run_state(terminal_state)
    artifact_root = Path(str(validated.payload["artifact_root"]))
    if (
        state.artifact_root != artifact_root
        or state.authorization_lease_claim_hash != validated.claim_hash
        or state.run_identity_hash != validated.payload.get("run_identity_hash")
    ):
        raise ProtocolError("OE-PPUR v3 authorization outcome/state binding drifted.")
    status = state.status
    final_bundle_hash = inventory_hash = lifecycle_hash = None
    commit_hash = complete_seal_hash = None
    if status == "COMPLETE":
        from .complete_artifact_validation import (
            CompleteArtifactSealReceipt,
            validate_complete_artifact_seal,
        )

        if (
            type(final_bundle) is not FinalAggregateBundleReceipt
            or type(prepared_state) is not PreparedCompleteRunState
            or type(completion_commit) is not CompletionCommitReceipt
            or type(complete_artifact_seal) is not CompleteArtifactSealReceipt
        ):
            raise ProtocolError(
                "OE-PPUR v3 complete outcome requires typed transaction evidence."
            )
        bundle = validate_final_aggregate_bundle(
            artifact_root,
            expected_receipt=final_bundle,
        )
        commit = validate_completion_commit(
            completion_commit,
            expected_prepared_state=prepared_state,
        )
        complete_seal = validate_complete_artifact_seal(
            artifact_root,
            expected=complete_artifact_seal,
        )
        if (
            state.evidence_hash != bundle.receipt_hash
            or state.state_hash != prepared_state.state_hash
            or commit.prepared_state_hash != state.state_hash
            or commit.final_bundle_receipt_hash != bundle.receipt_hash
            or commit.complete_artifact_seal_receipt_hash
            != complete_seal.receipt_hash
            or commit.artifact_inventory_hash
            != complete_seal.artifact_inventory_hash
            or (validated.path / COMPLETION_ABORT_MEMBER).exists()
            or (validated.path / COMPLETION_ABORT_MEMBER).is_symlink()
            or pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
        ):
            raise ProtocolError("OE-PPUR v3 complete transaction lineage drifted.")
        inventory_hash = complete_seal.artifact_inventory_hash
        lifecycle_hash = _validate_complete_lifecycle_lineage(
            artifact_root,
            claim=validated,
            terminal_state=state,
            final_bundle=bundle,
        )
        final_bundle_hash = bundle.receipt_hash
        commit_hash = commit.journal_hash
        complete_seal_hash = complete_seal.receipt_hash
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
            raise ProtocolError("OE-PPUR v3 failed outcome accepted final state.")
        from .run_state import read_run_state

        raw_state = read_run_state(artifact_root)
        error_class = _safe_text(raw_state.get("error_class"))
        if not error_class:
            raise ProtocolError("OE-PPUR v3 failed outcome lacks its error class.")
    else:  # pragma: no cover
        raise ProtocolError("OE-PPUR v3 authorization outcome status drifted.")
    body = {
        "schema_version": "oe_ppur_v3_single_use_authorization_outcome_v1",
        "status": status,
        "claim_hash": validated.claim_hash,
        "evidence_hash": state.state_hash,
        "terminal_run_state_receipt_hash": state.receipt_hash,
        "final_bundle_receipt_hash": final_bundle_hash,
        "artifact_inventory_hash": inventory_hash,
        "lifecycle_lineage_hash": lifecycle_hash,
        "completion_commit_hash": commit_hash,
        "complete_artifact_seal_receipt_hash": complete_seal_hash,
        "error_class": error_class,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "outcome_hash": canonical_hash(body)}
    publish_json_no_overwrite(
        validated.path / OUTCOME_MEMBER,
        payload,
        role="authorization outcome",
    )
    fsync_directory(validated.path)
    receipt = _outcome_receipt(validated.path, payload)
    return validate_authorization_outcome(validated, expected=receipt)


def validate_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    expected: AuthorizationOutcomeReceipt,
) -> AuthorizationOutcomeReceipt:
    validated = validate_authorization_lease(claim)
    if type(expected) is not AuthorizationOutcomeReceipt:
        raise ProtocolError("OE-PPUR v3 authorization outcome receipt is untyped.")
    if pending_publications(validated.path, OUTCOME_MEMBER):
        raise ProtocolError("OE-PPUR v3 authorization outcome is interrupted.")
    payload = read_json_regular(
        validated.path / OUTCOME_MEMBER,
        role="authorization outcome",
    )
    receipt = _outcome_receipt(validated.path, payload)
    abort_path = validated.path / COMPLETION_ABORT_MEMBER
    if receipt.status == "COMPLETE" and (
        abort_path.exists()
        or abort_path.is_symlink()
        or pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
    ):
        raise ProtocolError(
            "OE-PPUR v3 complete authorization outcome was aborted."
        )
    if receipt != expected or receipt.claim_hash != validated.claim_hash:
        raise ProtocolError("OE-PPUR v3 authorization outcome changed after issuance.")
    return receipt


def validate_complete_run_bundle(
    claim: AuthorizationLeaseClaim,
    *,
    terminal_state: object,
    final_bundle: object,
    prepared_state: object,
    completion_commit: object,
    complete_artifact_seal: object,
    outcome: AuthorizationOutcomeReceipt,
) -> AuthorizationOutcomeReceipt:
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
        TerminalRunStateReceipt,
        validate_terminal_run_state,
    )

    validated = validate_authorization_lease(claim)
    if (
        type(terminal_state) is not TerminalRunStateReceipt
        or type(final_bundle) is not FinalAggregateBundleReceipt
        or type(prepared_state) is not PreparedCompleteRunState
        or type(completion_commit) is not CompletionCommitReceipt
        or type(complete_artifact_seal) is not CompleteArtifactSealReceipt
        or type(outcome) is not AuthorizationOutcomeReceipt
    ):
        raise ProtocolError("OE-PPUR v3 complete-run validation inputs are untyped.")
    state = validate_terminal_run_state(terminal_state)
    root = Path(str(validated.payload["artifact_root"]))
    bundle = validate_final_aggregate_bundle(root, expected_receipt=final_bundle)
    commit = validate_completion_commit(
        completion_commit,
        expected_prepared_state=prepared_state,
    )
    complete_seal = validate_complete_artifact_seal(
        root,
        expected=complete_artifact_seal,
    )
    inventory_hash = complete_seal.artifact_inventory_hash
    lifecycle_hash = _validate_complete_lifecycle_lineage(
        root,
        claim=validated,
        terminal_state=state,
        final_bundle=bundle,
    )
    persisted = validate_authorization_outcome(validated, expected=outcome)
    if (
        state.status != "COMPLETE"
        or state.phase != "COMPLETE"
        or state.artifact_root != root
        or state.authorization_lease_claim_hash != validated.claim_hash
        or state.run_identity_hash != validated.payload.get("run_identity_hash")
        or state.evidence_hash != bundle.receipt_hash
        or state.state_hash != prepared_state.state_hash
        or complete_seal.prepared_state_hash != prepared_state.state_hash
        or complete_seal.prepared_state_receipt_hash != prepared_state.receipt_hash
        or complete_seal.final_bundle_receipt_hash != bundle.receipt_hash
        or commit.prepared_state_hash != state.state_hash
        or commit.final_bundle_receipt_hash != bundle.receipt_hash
        or commit.complete_artifact_seal_receipt_hash != complete_seal.receipt_hash
        or commit.artifact_inventory_hash != inventory_hash
        or (validated.path / COMPLETION_ABORT_MEMBER).exists()
        or (validated.path / COMPLETION_ABORT_MEMBER).is_symlink()
        or pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
        or persisted.status != "COMPLETE"
        or persisted.evidence_hash != state.state_hash
        or persisted.terminal_run_state_receipt_hash != state.receipt_hash
        or persisted.final_bundle_receipt_hash != bundle.receipt_hash
        or persisted.artifact_inventory_hash != inventory_hash
        or persisted.lifecycle_lineage_hash != lifecycle_hash
        or persisted.completion_commit_hash != commit.journal_hash
        or persisted.complete_artifact_seal_receipt_hash
        != complete_seal.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v3 complete-run lifecycle binding drifted.")
    return persisted


def finalize_failed_authorization(
    claim: AuthorizationLeaseClaim,
    *,
    artifact_root: Path,
    original_error: BaseException,
) -> AuthorizationOutcomeReceipt:
    if not isinstance(original_error, BaseException):
        raise ProtocolError("OE-PPUR v3 failure finalization requires original error.")
    validated = validate_authorization_lease(claim)
    root = Path(artifact_root)
    if root != Path(str(validated.payload["artifact_root"])):
        raise ProtocolError("OE-PPUR v3 failure finalization root drifted.")
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
                        "schema_version": "oe_ppur_v3_runner_failure_v3",
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
                    "OE-PPUR v3 failure bookkeeping encountered COMPLETE state."
                )
        if terminal_state is not None:
            return record_authorization_outcome(
                validated,
                terminal_state=terminal_state,
            )
        evidence_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v3_prestate_failure_v2",
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
            "Original OE-PPUR v3 execution failure: "
            f"{type(original_error).__name__}: {_safe_text(original_error)}"
        )
        raise bookkeeping_error from original_error


def _record_fail_closed_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    evidence_hash: str,
    error_class: str,
) -> AuthorizationOutcomeReceipt:
    body = {
        "schema_version": "oe_ppur_v3_single_use_authorization_outcome_v1",
        "status": "FAILED_EXHAUSTED",
        "claim_hash": claim.claim_hash,
        "evidence_hash": require_sha256(evidence_hash, "failure evidence hash"),
        "terminal_run_state_receipt_hash": None,
        "final_bundle_receipt_hash": None,
        "artifact_inventory_hash": None,
        "lifecycle_lineage_hash": None,
        "completion_commit_hash": None,
        "complete_artifact_seal_receipt_hash": None,
        "error_class": _safe_text(error_class),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "outcome_hash": canonical_hash(body)}
    publish_json_no_overwrite(
        claim.path / OUTCOME_MEMBER,
        payload,
        role="authorization outcome",
    )
    fsync_directory(claim.path)
    receipt = _outcome_receipt(claim.path, payload)
    return validate_authorization_outcome(claim, expected=receipt)


def _outcome_receipt(
    lease_path: Path,
    payload: Mapping[str, object],
) -> AuthorizationOutcomeReceipt:
    expected_keys = {
        "schema_version",
        "status",
        "claim_hash",
        "evidence_hash",
        "terminal_run_state_receipt_hash",
        "final_bundle_receipt_hash",
        "artifact_inventory_hash",
        "lifecycle_lineage_hash",
        "completion_commit_hash",
        "complete_artifact_seal_receipt_hash",
        "error_class",
        "recorded_at_utc",
        "authorization_exhausted",
        "authorization_restored",
        "recovery_allowed",
        "outcome_hash",
    }
    body = {key: value for key, value in payload.items() if key != "outcome_hash"}
    status = payload.get("status")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != "oe_ppur_v3_single_use_authorization_outcome_v1"
        or status not in {"COMPLETE", "FAILED_EXHAUSTED"}
        or payload.get("outcome_hash") != canonical_hash(body)
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_restored") is not False
        or payload.get("recovery_allowed") is not False
        or not isinstance(payload.get("recorded_at_utc"), str)
        or (status == "COMPLETE") != (payload.get("error_class") is None)
        or (
            status == "FAILED_EXHAUSTED"
            and not isinstance(payload.get("error_class"), str)
        )
        or (
            status == "FAILED_EXHAUSTED"
            and any(
                payload.get(role) is not None
                for role in (
                    "final_bundle_receipt_hash",
                    "artifact_inventory_hash",
                    "lifecycle_lineage_hash",
                    "completion_commit_hash",
                    "complete_artifact_seal_receipt_hash",
                )
            )
        )
    ):
        raise ProtocolError("OE-PPUR v3 authorization outcome drifted.")
    return AuthorizationOutcomeReceipt(
        lease_path=Path(lease_path),
        status=str(status),
        claim_hash=str(payload["claim_hash"]),
        evidence_hash=str(payload["evidence_hash"]),
        terminal_run_state_receipt_hash=_optional_text(
            payload["terminal_run_state_receipt_hash"]
        ),
        final_bundle_receipt_hash=_optional_text(payload["final_bundle_receipt_hash"]),
        artifact_inventory_hash=_optional_text(payload["artifact_inventory_hash"]),
        lifecycle_lineage_hash=_optional_text(payload["lifecycle_lineage_hash"]),
        completion_commit_hash=_optional_text(payload["completion_commit_hash"]),
        complete_artifact_seal_receipt_hash=_optional_text(
            payload["complete_artifact_seal_receipt_hash"]
        ),
        outcome_hash=str(payload["outcome_hash"]),
        _factory_token=_OUTCOME_TOKEN,
    )


def _validate_complete_lifecycle_lineage(
    root: Path,
    *,
    claim: AuthorizationLeaseClaim,
    terminal_state: object,
    final_bundle: object,
) -> str:
    from .config import load_resolved_config
    from .output_artifact import (
        FINAL_ATTESTATION_MEMBER,
        FINAL_BINDING_MEMBER,
        TERMINAL_METRICS_MEMBER,
    )
    from .run_state import read_run_state

    state = read_run_state(root)
    resolved = load_resolved_config(root / "config.resolved.yaml")
    admission = read_json_regular(
        root / "provenance/execution_admission.json",
        role="persisted execution admission",
    )
    admission_body = {
        key: value for key, value in admission.items() if key != "receipt_hash"
    }
    lease_copy = read_json_regular(
        root / "provenance/authorization_consumption_lease.json",
        role="persisted authorization lease",
    )
    run_lock = read_json_regular(root / ".run.lock", role="run lock")
    lock_body = {key: value for key, value in run_lock.items() if key != "lock_hash"}
    launch = read_json_regular(
        root / "reports/launch_receipts.json",
        role="launch receipt",
    )
    prediction = read_json_regular(
        root / "physical/predictions/manifests/fixed_bank_a1_prediction_seal.json",
        role="prediction seal",
    )
    preterminal = read_json_regular(
        root / "manifests/preterminal_result.json",
        role="preterminal result",
    )
    preterminal_attestation = read_json_regular(
        root / "reports/preterminal_fresh_process_attestation.json",
        role="preterminal attestation",
    )
    guarded_boundary = preterminal_attestation.get("guarded_boundary")
    terminal = read_json_regular(root / TERMINAL_METRICS_MEMBER, role="terminal metrics")
    final_attestation = read_json_regular(
        root / FINAL_ATTESTATION_MEMBER,
        role="final attestation",
    )
    final_binding = read_json_regular(
        root / FINAL_BINDING_MEMBER,
        role="final binding",
    )
    if not isinstance(guarded_boundary, Mapping):
        raise ProtocolError("OE-PPUR v3 guarded boundary is absent at completion.")
    boundary_body = {
        key: value for key, value in guarded_boundary.items() if key != "receipt_hash"
    }
    lifecycle = validate_complete_lifecycle_evidence(
        state["transitions"],
        inputs_sealed_hash=canonical_hash(launch),
        prediction_seal_hash=prediction.get("global_prediction_seal_hash"),
        preterminal_result_hash=preterminal.get("result_hash"),
        preterminal_boundary_hash=guarded_boundary.get("receipt_hash"),
        terminal_receipt_hash=terminal.get("receipt_hash"),
        final_attestation_hash=final_attestation.get("receipt_hash"),
        final_bundle_receipt_hash=getattr(final_bundle, "receipt_hash", None),
    )
    if (
        resolved.artifact_root != root
        or resolved.config.contract_hash != state["config_contract_hash"]
        or resolved.config.protocol_hash != state["protocol_hash"]
        or admission.get("schema_version")
        != "oe_ppur_v3_seven_input_run_admission_v1"
        or admission.get("receipt_hash") != canonical_hash(admission_body)
        or admission.get("receipt_hash") != state["seven_input_admission_hash"]
        or admission.get("artifact_root") != root.as_posix()
        or admission.get("config_contract_hash") != state["config_contract_hash"]
        or admission.get("protocol_hash") != state["protocol_hash"]
        or admission.get("source_seal_hash") != state["source_seal_hash"]
        or admission.get("target_labels_opened") is not False
        or admission.get("mutation_performed") is not False
        or lease_copy != claim.to_payload()
        or run_lock.get("schema_version") != "oe_ppur_v3_run_lock_v1"
        or run_lock.get("lock_hash") != canonical_hash(lock_body)
        or run_lock.get("run_identity_hash") != state["run_identity_hash"]
        or run_lock.get("authorization_lease_claim_hash") != claim.claim_hash
        or run_lock.get("authorization_exhausted") is not True
        or run_lock.get("recovery_allowed") is not False
        or launch.get("schema_version") != "oe_ppur_v3_launch_receipts_v1"
        or launch.get("seven_input_admission") != admission
        or launch.get("target_labels_opened") is not False
        or not isinstance(launch.get("source_seal"), Mapping)
        or launch["source_seal"].get("combined_source_sha256")
        != state["source_seal_hash"]
        or prediction.get("labels_opened") is not False
        or prediction.get("target_expert_used") is not False
        or preterminal.get("schema_version")
        != "oe_ppur_v3_persisted_preterminal_result_v1"
        or guarded_boundary.get("receipt_hash") != canonical_hash(boundary_body)
        or guarded_boundary.get("decision_ledger_receipt_hash")
        != preterminal.get("decision_ledger_hash")
        or getattr(terminal_state, "state_hash", None) != state["state_hash"]
        or final_binding.get("config_contract_hash") != state["config_contract_hash"]
        or final_binding.get("protocol_hash") != state["protocol_hash"]
        or final_binding.get("source_seal_hash") != state["source_seal_hash"]
        or final_binding.get("seven_input_contract_hash")
        != admission.get("seven_input_contract_hash")
        or final_binding.get("source_training_surface_receipt_hash")
        != admission.get("source_training_surface_receipt_hash")
        or final_binding.get("preterminal_boundary_receipt_hash")
        != guarded_boundary.get("receipt_hash")
        or final_binding.get("preterminal_ledger_receipt_hash")
        != preterminal.get("decision_ledger_hash")
        or final_binding.get("terminal_receipt_hash") != terminal.get("receipt_hash")
        or final_binding.get("final_attestation_hash")
        != final_attestation.get("receipt_hash")
    ):
        raise ProtocolError("OE-PPUR v3 complete lifecycle lineage drifted.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_complete_lifecycle_lineage_v2",
            "run_identity_hash": state["run_identity_hash"],
            "authorization_lease_claim_hash": claim.claim_hash,
            "seven_input_admission_hash": admission["receipt_hash"],
            "phase_evidence": lifecycle.to_payload(),
            "final_bundle_receipt_hash": getattr(final_bundle, "receipt_hash"),
            "terminal_run_state_receipt_hash": getattr(
                terminal_state,
                "receipt_hash",
            ),
        }
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


__all__ = (
    "AuthorizationOutcomeReceipt",
    "OUTCOME_MEMBER",
    "finalize_failed_authorization",
    "record_authorization_outcome",
    "validate_authorization_outcome",
    "validate_complete_run_bundle",
)
