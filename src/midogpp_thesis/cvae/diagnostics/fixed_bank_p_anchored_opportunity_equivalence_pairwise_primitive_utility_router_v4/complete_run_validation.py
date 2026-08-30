"""Whole-run reopening and COMPLETE authorization validation for v4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .artifact.contracts import (
    COMPLETION_ABORT_MEMBER,
    FINAL_ATTESTATION_MEMBER,
    FINAL_BINDING_MEMBER,
    TERMINAL_METRICS_MEMBER,
    CompleteArtifactSealReceipt,
)
from .authorization_outcome_contracts import AuthorizationOutcomeReceipt
from .authorization_outcome_store import validate_authorization_outcome
from .complete_artifact_validation import validate_complete_artifact_seal
from .completion_transaction import (
    CompletionCommitReceipt,
    validate_completion_commit,
)
from .config import build_workspace_sealed_config
from .hashing import canonical_hash
from .lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from .lease_io import pending_publications, read_json_regular
from .lifecycle_lineage import validate_complete_lifecycle_evidence
from .output_persistence import _read_regular_bytes_nofollow
from .output_validation import (
    FinalAggregateBundleReceipt,
    validate_final_aggregate_bundle,
)
from .run_state import (
    PreparedCompleteRunState,
    TerminalRunStateReceipt,
    read_run_state,
    validate_terminal_run_state,
)


@dataclass(frozen=True, slots=True)
class CompleteRunEvidence:
    terminal_state: TerminalRunStateReceipt
    final_bundle: FinalAggregateBundleReceipt
    completion_commit: CompletionCommitReceipt
    complete_artifact_seal: CompleteArtifactSealReceipt
    lifecycle_lineage_hash: str


def reopen_complete_run_evidence(
    claim: AuthorizationLeaseClaim,
    *,
    terminal_state: TerminalRunStateReceipt,
    final_bundle: FinalAggregateBundleReceipt,
    prepared_state: PreparedCompleteRunState,
    completion_commit: CompletionCommitReceipt,
    complete_artifact_seal: CompleteArtifactSealReceipt,
) -> CompleteRunEvidence:
    validated = validate_authorization_lease(claim)
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
        or commit.artifact_inventory_hash != complete_seal.artifact_inventory_hash
        or (validated.path / COMPLETION_ABORT_MEMBER).exists()
        or (validated.path / COMPLETION_ABORT_MEMBER).is_symlink()
        or pending_publications(validated.path, COMPLETION_ABORT_MEMBER)
    ):
        raise ProtocolError("OE-PPUR v4 complete transaction lineage drifted.")
    lifecycle_hash = _validate_complete_lifecycle_lineage(
        root,
        claim=validated,
        terminal_state=state,
        final_bundle=bundle,
    )
    return CompleteRunEvidence(
        terminal_state=state,
        final_bundle=bundle,
        completion_commit=commit,
        complete_artifact_seal=complete_seal,
        lifecycle_lineage_hash=lifecycle_hash,
    )


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
    if (
        type(terminal_state) is not TerminalRunStateReceipt
        or type(final_bundle) is not FinalAggregateBundleReceipt
        or type(prepared_state) is not PreparedCompleteRunState
        or type(completion_commit) is not CompletionCommitReceipt
        or type(complete_artifact_seal) is not CompleteArtifactSealReceipt
        or type(outcome) is not AuthorizationOutcomeReceipt
    ):
        raise ProtocolError("OE-PPUR v4 complete-run validation inputs are untyped.")
    validated = validate_authorization_lease(claim)
    evidence = reopen_complete_run_evidence(
        validated,
        terminal_state=terminal_state,
        final_bundle=final_bundle,
        prepared_state=prepared_state,
        completion_commit=completion_commit,
        complete_artifact_seal=complete_artifact_seal,
    )
    state = evidence.terminal_state
    bundle = evidence.final_bundle
    commit = evidence.completion_commit
    complete_seal = evidence.complete_artifact_seal
    persisted = validate_authorization_outcome(validated, expected=outcome)
    if (
        persisted.status != "COMPLETE"
        or persisted.evidence_hash != state.state_hash
        or persisted.terminal_run_state_receipt_hash != state.receipt_hash
        or persisted.final_bundle_receipt_hash != bundle.receipt_hash
        or persisted.artifact_inventory_hash
        != complete_seal.artifact_inventory_hash
        or persisted.lifecycle_lineage_hash != evidence.lifecycle_lineage_hash
        or persisted.completion_commit_hash != commit.journal_hash
        or persisted.complete_artifact_seal_receipt_hash
        != complete_seal.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v4 complete-run lifecycle binding drifted.")
    return persisted


def _validate_complete_lifecycle_lineage(
    root: Path,
    *,
    claim: AuthorizationLeaseClaim,
    terminal_state: object,
    final_bundle: object,
) -> str:
    state = read_run_state(root)
    resolved = read_json_regular(
        root / "config.resolved.yaml",
        role="persisted resolved config",
    )
    resolved_config = build_workspace_sealed_config(
        workspace_plan_sha256=str(resolved.get("pre_amendment_plan_sha256", "")),
        authorization_amendment_sha256=str(
            resolved.get("authorization_amendment_sha256", "")
        ),
    )
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
    launch_authority = read_json_regular(
        root / "preparation/execution_launch_authority.json",
        role="persisted execution launch authority",
    )
    sealed_replay = read_json_regular(
        root / "preparation/sealed_execution_replay.json",
        role="persisted sealed replay",
    )
    preparation_commit = read_json_regular(
        root / "COMMITTED",
        role="preparation commit marker",
    )
    replay_body = {
        key: value for key, value in sealed_replay.items() if key != "receipt_hash"
    }
    if not isinstance(guarded_boundary, Mapping):
        raise ProtocolError("OE-PPUR v4 guarded boundary is absent at completion.")
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
        set(resolved)
        != {
            "schema_version",
            "experiment_id",
            "output_artifact_id",
            "authorization_state",
            "pre_amendment_plan_sha256",
            "authorization_amendment_sha256",
            "workspace_snapshot_sha256",
            "existing_input_inventory_sha256",
            "amendment_input_template_sha256",
            "execution_topology_sha256",
            "scientific_seals_sha256",
            "predecessor_preservation_witness_sha256",
            "artifact_root",
            "scratch_root",
            "launch_authorized",
            "authorization_consumed",
            "target_labels_opened",
        }
        or resolved.get("schema_version")
        != "oe_ppur_v4_workspace_sealed_resolved_config_v1"
        or resolved.get("artifact_root") != root.as_posix()
        or resolved.get("pre_amendment_plan_sha256")
        != state["workspace_plan_sha256"]
        or resolved.get("workspace_snapshot_sha256")
        != state["workspace_snapshot_sha256"]
        or resolved.get("authorization_amendment_sha256")
        != admission.get("authorization_amendment_sha256")
        or resolved.get("launch_authorized") is not False
        or resolved.get("authorization_consumed") is not False
        or resolved.get("target_labels_opened") is not False
        or resolved_config.contract_hash != state["config_contract_hash"]
        or resolved_config.protocol_hash != state["protocol_hash"]
        or admission.get("schema_version")
        != "oe_ppur_v4_seven_input_run_admission_v1"
        or admission.get("receipt_hash") != canonical_hash(admission_body)
        or admission.get("receipt_hash") != state["seven_input_admission_hash"]
        or admission.get("artifact_root") != root.as_posix()
        or admission.get("config_contract_hash") != state["config_contract_hash"]
        or admission.get("protocol_hash") != state["protocol_hash"]
        or admission.get("source_seal_hash") != state["source_seal_hash"]
        or admission.get("workspace_snapshot_sha256")
        != state["workspace_snapshot_sha256"]
        or admission.get("workspace_plan_sha256") != state["workspace_plan_sha256"]
        or admission.get("final_envelope_sha256") != state["final_envelope_sha256"]
        or admission.get("execution_launch_authority_sha256")
        != state["execution_launch_authority_sha256"]
        or admission.get("target_labels_opened") is not False
        or admission.get("mutation_performed") is not False
        or lease_copy != claim.to_payload()
        or run_lock.get("schema_version") != "oe_ppur_v4_run_lock_v1"
        or run_lock.get("lock_hash") != canonical_hash(lock_body)
        or run_lock.get("run_identity_hash") != state["run_identity_hash"]
        or run_lock.get("authorization_lease_claim_hash") != claim.claim_hash
        or run_lock.get("authorization_exhausted") is not True
        or run_lock.get("recovery_allowed") is not False
        or launch.get("schema_version") != "oe_ppur_v4_launch_receipts_v1"
        or launch.get("seven_input_admission") != admission
        or launch.get("target_labels_opened") is not False
        or not isinstance(launch.get("source_seal"), Mapping)
        or launch["source_seal"].get("combined_source_sha256")
        != state["source_seal_hash"]
        or prediction.get("labels_opened") is not False
        or prediction.get("target_expert_used") is not False
        or preterminal.get("schema_version")
        != "oe_ppur_v4_persisted_preterminal_result_v1"
        or guarded_boundary.get("receipt_hash") != canonical_hash(boundary_body)
        or guarded_boundary.get("decision_ledger_receipt_hash")
        != preterminal.get("decision_ledger_hash")
        or getattr(terminal_state, "state_hash", None) != state["state_hash"]
        or final_binding.get("config_contract_hash") != state["config_contract_hash"]
        or final_binding.get("protocol_hash") != state["protocol_hash"]
        or final_binding.get("source_seal_hash") != state["source_seal_hash"]
        or final_binding.get("seven_input_admission_hash")
        != admission.get("receipt_hash")
        or final_binding.get("execution_launch_authority_sha256")
        != admission.get("execution_launch_authority_sha256")
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
        or launch_authority.get("authority_hash") is None
        or admission.get("execution_launch_authority_sha256")
        != _sha256_regular_file(
            root / "preparation/execution_launch_authority.json"
        )
        or sealed_replay.get("schema_version")
        != "oe_ppur_v4_sealed_execution_replay_v1"
        or sealed_replay.get("receipt_hash") != canonical_hash(replay_body)
        or sealed_replay.get("receipt_hash")
        != admission.get("sealed_replay_receipt_hash")
        or sealed_replay.get("workspace_snapshot_sha256")
        != state["workspace_snapshot_sha256"]
        or sealed_replay.get("workspace_plan_sha256")
        != state["workspace_plan_sha256"]
        or sealed_replay.get("final_envelope_sha256")
        != state["final_envelope_sha256"]
        or sealed_replay.get("target_labels_opened") is not False
        or sealed_replay.get("filesystem_mutation_performed") is not False
        or _sha256_regular_file(
            root / "preparation/final_authorization_envelope.json"
        )
        != state["final_envelope_sha256"]
        or preparation_commit.get("schema_version")
        != "oe_ppur_v4_preparation_commit_marker_v1"
        or preparation_commit.get("status") != "COMMITTED"
        or preparation_commit.get("final_envelope_sha256")
        != state["final_envelope_sha256"]
        or preparation_commit.get("pre_amendment_plan_sha256")
        != state["workspace_plan_sha256"]
        or preparation_commit.get("authorization_amendment_sha256")
        != admission.get("authorization_amendment_sha256")
        or preparation_commit.get("member_writes_used_o_excl") is not True
        or preparation_commit.get("commit_marker_written_last") is not True
        or preparation_commit.get("authorization_consumed") is not True
        or preparation_commit.get("authorization_exhausted") is not True
        or preparation_commit.get("preparation_commit_is_scientific_complete")
        is not False
        or preparation_commit.get("target_labels_opened") is not False
        or preparation_commit.get("experiment_launched") is not False
    ):
        raise ProtocolError("OE-PPUR v4 complete lifecycle lineage drifted.")
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v4_complete_lifecycle_lineage_v2",
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


def _sha256_regular_file(path: Path) -> str:
    raw, _metadata = _read_regular_bytes_nofollow(path)
    return hashlib.sha256(raw).hexdigest()


__all__ = ("validate_complete_run_bundle",)
