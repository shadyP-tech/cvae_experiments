from __future__ import annotations

import json
from pathlib import Path

import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_outcome as outcome_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.completion_transaction as completion_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_claim as claim_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_io as lease_io_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.complete_artifact_validation as seal_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.output_artifact as output_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_state as state_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_lease import (
    AuthorizationLeaseClaim,
    COMPLETION_ABORT_MEMBER,
    COMPLETION_COMMIT_MEMBER,
    InterruptedCompletionReceipt,
    LEASE_DIRECTORY_NAME,
    discover_completion_commit,
    finalize_failed_authorization,
    record_completion_abort,
    record_authorization_outcome,
    record_completion_commit,
    validate_authorization_outcome,
    validate_completion_commit,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.complete_artifact_validation import (
    CompleteArtifactSealReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.output_artifact import (
    FinalAggregateBundleReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_state import (
    PHASE_ORDER,
    commit_complete_run_state,
    prepare_complete_run_state,
    read_run_state,
    transition_run,
    write_exclusive_json,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _write_final_attested_state(root: Path, claim_hash: str) -> None:
    history: list[dict[str, object]] = []
    previous_hash: str | None = None
    current = "ADMITTED"
    for sequence, target in enumerate(PHASE_ORDER[1:PHASE_ORDER.index("FINAL_ATTESTED") + 1]):
        body = {
            "sequence": sequence,
            "from_phase": current,
            "to_phase": target,
            "status": "RUNNING",
            "evidence_hash": "1" * 64,
            "previous_transition_hash": previous_hash,
        }
        row = {**body, "transition_hash": canonical_hash(body)}
        history.append(row)
        previous_hash = str(row["transition_hash"])
        current = target
    body = {
        "schema_version": "oe_ppur_v3_single_use_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": "2" * 64,
        "config_contract_hash": "3" * 64,
        "protocol_hash": "4" * 64,
        "source_seal_hash": "5" * 64,
        "seven_input_admission_hash": "6" * 64,
        "authorization_lease_claim_hash": claim_hash,
        "status": "RUNNING",
        "phase": "FINAL_ATTESTED",
        "transition_count": len(history),
        "transitions": history,
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "scratch_recovery_allowed": False,
        "raw_labels_persisted": False,
        "updated_at_utc": "2026-08-28T00:00:00+00:00",
        "error_class": None,
    }
    state_module.atomic_json(
        root / "reports/run_state.json",
        {**body, "state_hash": canonical_hash(body)},
    )


def _claim(root: Path, monkeypatch: pytest.MonkeyPatch) -> AuthorizationLeaseClaim:
    scratch = root.parent / "scratch"
    lease = root.parent / LEASE_DIRECTORY_NAME
    lease.mkdir()
    monkeypatch.setattr(
        claim_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )
    body = {
        "schema_version": "oe_ppur_v3_single_use_authorization_claim_v1",
        "status": "CONSUMED_EXHAUSTED",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "artifact_root": root.as_posix(),
        "scratch_root": scratch.as_posix(),
        "lease_path": lease.as_posix(),
        "run_identity_hash": "2" * 64,
        "seven_input_admission_hash": "6" * 64,
        "config_contract_hash": "3" * 64,
        "protocol_hash": "4" * 64,
        "source_seal_hash": "5" * 64,
        "authorization_amendment_sha256": "8" * 64,
        "consumed_at_utc": "2026-08-28T00:00:00+00:00",
        "process_id_at_claim": 100,
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "authorization_restored": False,
        "cross_run_recovery_allowed": False,
    }
    payload = {**body, "claim_hash": canonical_hash(body)}
    write_exclusive_json(lease / "claim.json", payload)
    return AuthorizationLeaseClaim(lease, payload, str(payload["claim_hash"]))


def _unchecked_bundle(root: Path) -> FinalAggregateBundleReceipt:
    value = object.__new__(FinalAggregateBundleReceipt)
    object.__setattr__(value, "artifact_root", root.as_posix())
    object.__setattr__(value, "receipt_hash", "9" * 64)
    return value


def _unchecked_seal(root: Path, prepared, bundle) -> CompleteArtifactSealReceipt:
    value = object.__new__(CompleteArtifactSealReceipt)
    object.__setattr__(value, "artifact_root", root)
    object.__setattr__(value, "prepared_state_hash", prepared.state_hash)
    object.__setattr__(value, "prepared_state_receipt_hash", prepared.receipt_hash)
    object.__setattr__(value, "final_bundle_receipt_hash", bundle.receipt_hash)
    object.__setattr__(value, "artifact_inventory_hash", "a" * 64)
    object.__setattr__(value, "receipt_hash", "b" * 64)
    return value


def _prepared_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "artifact"
    root.mkdir()
    claim = _claim(root, monkeypatch)
    _write_final_attested_state(root, claim.claim_hash)
    bundle = _unchecked_bundle(root)
    monkeypatch.setattr(
        output_module,
        "validate_final_aggregate_bundle",
        lambda _root, *, expected_receipt: expected_receipt,
    )
    transition_run(
        root,
        "COMPLETION_PENDING",
        expected_phase="FINAL_ATTESTED",
        evidence_hash=bundle.receipt_hash,
    )
    prepared = prepare_complete_run_state(root, final_bundle=bundle)
    assert prepare_complete_run_state(root, final_bundle=bundle) == prepared
    seal = _unchecked_seal(root, prepared, bundle)
    monkeypatch.setattr(
        seal_module,
        "validate_complete_artifact_seal",
        lambda _root, *, expected, expected_complete_state=None: expected,
    )
    return root, claim, bundle, prepared, seal


def test_journal_write_failure_never_exposes_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, claim, bundle, prepared, seal = _prepared_transaction(tmp_path, monkeypatch)
    original_publish = completion_module.publish_json_no_overwrite

    def fail_journal(path: Path, payload, *, role: str) -> None:
        if path.name == COMPLETION_COMMIT_MEMBER:
            raise OSError("injected journal fsync failure")
        original_publish(path, payload, role=role)

    monkeypatch.setattr(
        completion_module,
        "publish_json_no_overwrite",
        fail_journal,
    )
    with pytest.raises(OSError, match="journal fsync"):
        record_completion_commit(
            claim,
            prepared_state=prepared,
            final_bundle=bundle,
            complete_artifact_seal=seal,
        )

    assert read_run_state(root)["phase"] == "COMPLETION_PENDING"
    assert not (claim.path / COMPLETION_COMMIT_MEMBER).exists()


def test_complete_state_write_failure_preserves_journal_and_records_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, claim, bundle, prepared, seal = _prepared_transaction(tmp_path, monkeypatch)
    commit = record_completion_commit(
        claim,
        prepared_state=prepared,
        final_bundle=bundle,
        complete_artifact_seal=seal,
    )
    journal_payload = json.loads(
        (claim.path / COMPLETION_COMMIT_MEMBER).read_text(encoding="utf-8")
    )
    assert journal_payload["claim_hash"] == claim.claim_hash
    assert journal_payload["prepared_state_hash"] == prepared.state_hash
    assert journal_payload["final_bundle_receipt_hash"] == bundle.receipt_hash
    assert (
        journal_payload["complete_artifact_seal_receipt_hash"]
        == seal.receipt_hash
    )
    original_atomic = state_module.atomic_json

    def fail_complete(path: Path, payload) -> None:
        if path.name == "run_state.json" and payload.get("status") == "COMPLETE":
            raise OSError("injected COMPLETE state fsync failure")
        original_atomic(path, payload)

    monkeypatch.setattr(state_module, "atomic_json", fail_complete)
    failure = OSError("injected COMPLETE state fsync failure")
    with pytest.raises(OSError, match="COMPLETE state fsync"):
        commit_complete_run_state(prepared, completion_commit=commit)
    monkeypatch.setattr(state_module, "atomic_json", original_atomic)

    outcome = finalize_failed_authorization(
        claim,
        artifact_root=root,
        original_error=failure,
    )
    assert outcome.status == "FAILED_EXHAUSTED"
    assert read_run_state(root)["status"] == "FAILED_EXHAUSTED"
    assert (claim.path / COMPLETION_COMMIT_MEMBER).is_file()
    assert (claim.path / COMPLETION_ABORT_MEMBER).is_file()


def test_final_outcome_write_failure_leaves_abort_and_no_success_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, claim, bundle, prepared, seal = _prepared_transaction(tmp_path, monkeypatch)
    commit = record_completion_commit(
        claim,
        prepared_state=prepared,
        final_bundle=bundle,
        complete_artifact_seal=seal,
    )
    completed = commit_complete_run_state(prepared, completion_commit=commit)
    monkeypatch.setattr(
        outcome_module,
        "_validate_complete_lifecycle_lineage",
        lambda *args, **kwargs: "c" * 64,
    )
    original_publish = outcome_module.publish_json_no_overwrite

    def fail_outcome(path: Path, payload, *, role: str) -> None:
        if path.name == "outcome.json":
            raise OSError("injected final outcome fsync failure")
        original_publish(path, payload, role=role)

    monkeypatch.setattr(
        outcome_module,
        "publish_json_no_overwrite",
        fail_outcome,
    )
    failure = OSError("injected final outcome fsync failure")
    with pytest.raises(OSError, match="final outcome fsync"):
        record_authorization_outcome(
            claim,
            terminal_state=completed,
            final_bundle=bundle,
            prepared_state=prepared,
            completion_commit=commit,
            complete_artifact_seal=seal,
        )
    with pytest.raises(OSError, match="final outcome fsync"):
        finalize_failed_authorization(
            claim,
            artifact_root=root,
            original_error=failure,
        )

    assert read_run_state(root)["status"] == "COMPLETE"
    assert (claim.path / COMPLETION_COMMIT_MEMBER).is_file()
    assert (claim.path / COMPLETION_ABORT_MEMBER).is_file()
    assert not (claim.path / "outcome.json").exists()


def test_interrupted_commit_publication_is_discovered_and_aborted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, claim, bundle, prepared, seal = _prepared_transaction(tmp_path, monkeypatch)
    original_link = lease_io_module.os.link

    def interrupt_after_publish(
        source,
        destination,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(source, destination, follow_symlinks=follow_symlinks)
        if Path(destination).name == COMPLETION_COMMIT_MEMBER:
            raise OSError("injected interruption after commit publication")

    monkeypatch.setattr(lease_io_module.os, "link", interrupt_after_publish)
    with pytest.raises(OSError, match="after commit publication"):
        record_completion_commit(
            claim,
            prepared_state=prepared,
            final_bundle=bundle,
            complete_artifact_seal=seal,
        )

    commit_path = claim.path / COMPLETION_COMMIT_MEMBER
    assert commit_path.is_file()
    assert commit_path.stat().st_nlink == 2
    interrupted = discover_completion_commit(claim)
    assert type(interrupted) is InterruptedCompletionReceipt

    outcome = finalize_failed_authorization(
        claim,
        artifact_root=root,
        original_error=OSError("injected interruption after commit publication"),
    )
    assert outcome.status == "FAILED_EXHAUSTED"
    assert read_run_state(root)["status"] == "FAILED_EXHAUSTED"
    assert (claim.path / COMPLETION_ABORT_MEMBER).is_file()


def test_complete_outcome_is_invalidated_by_later_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, claim, bundle, prepared, seal = _prepared_transaction(tmp_path, monkeypatch)
    commit = record_completion_commit(
        claim,
        prepared_state=prepared,
        final_bundle=bundle,
        complete_artifact_seal=seal,
    )
    completed = commit_complete_run_state(prepared, completion_commit=commit)
    monkeypatch.setattr(
        outcome_module,
        "_validate_complete_lifecycle_lineage",
        lambda *args, **kwargs: "c" * 64,
    )
    outcome = record_authorization_outcome(
        claim,
        terminal_state=completed,
        final_bundle=bundle,
        prepared_state=prepared,
        completion_commit=commit,
        complete_artifact_seal=seal,
    )

    record_completion_abort(
        claim,
        completion=commit,
        original_error=OSError("post-outcome validation failed"),
        artifact_root=root,
    )

    with pytest.raises(ProtocolError, match="journal is interrupted"):
        validate_completion_commit(commit, expected_prepared_state=prepared)
    with pytest.raises(ProtocolError, match="outcome was aborted"):
        validate_authorization_outcome(claim, expected=outcome)


def test_abort_before_complete_permanently_blocks_complete_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, claim, bundle, prepared, seal = _prepared_transaction(tmp_path, monkeypatch)
    commit = record_completion_commit(
        claim,
        prepared_state=prepared,
        final_bundle=bundle,
        complete_artifact_seal=seal,
    )
    record_completion_abort(
        claim,
        completion=commit,
        original_error=OSError("pre-COMPLETE validation failed"),
        artifact_root=root,
    )

    with pytest.raises(ProtocolError, match="journal is interrupted"):
        validate_completion_commit(commit, expected_prepared_state=prepared)
    with pytest.raises(ProtocolError, match="journal is interrupted"):
        commit_complete_run_state(prepared, completion_commit=commit)
    assert read_run_state(root)["phase"] == "COMPLETION_PENDING"
