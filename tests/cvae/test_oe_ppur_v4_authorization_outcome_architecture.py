from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.authorization_outcome as outcome_facade
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.authorization_outcome_contracts as outcome_contracts
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.authorization_outcome_store as outcome_store
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.authorization_outcome_recording as outcome_recording
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.authorization_failure as outcome_failure
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.complete_run_validation as complete_validation
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lease_claim as claim_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lease_claim import (
    claim_authorization_lease,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.run_admission import (
    SevenInputRunAdmission,
    _ADMISSION_TOKEN,
)
from midogpp_thesis.cvae.protocol import ProtocolError


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4"
)

OUTCOME_MODULES = {
    "authorization_failure",
    "authorization_outcome",
    "authorization_outcome_contracts",
    "authorization_outcome_recording",
    "authorization_outcome_store",
    "complete_run_validation",
}


def _outcome_import_graph() -> dict[str, set[str]]:
    graph = {module: set() for module in OUTCOME_MODULES}
    for module in graph:
        path = PACKAGE_ROOT / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 1
                and node.module in graph
            ):
                graph[module].add(node.module)
    return graph


def test_authorization_outcome_layers_are_acyclic_and_inverted() -> None:
    graph = _outcome_import_graph()
    assert graph["authorization_outcome_contracts"] == set()
    assert graph["authorization_outcome_store"] == {
        "authorization_outcome_contracts"
    }
    assert "authorization_outcome" not in graph["complete_run_validation"]
    assert "authorization_failure" not in graph["complete_run_validation"]
    assert "authorization_outcome" not in graph["authorization_failure"]

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = " -> ".join((*visiting[visiting.index(module) :], module))
            raise AssertionError(f"authorization outcome import cycle: {cycle}")
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)


def test_authorization_outcome_facade_preserves_public_identity() -> None:
    assert (
        outcome_facade.AuthorizationOutcomeReceipt
        is outcome_contracts.AuthorizationOutcomeReceipt
    )
    assert outcome_facade.OUTCOME_MEMBER == outcome_contracts.OUTCOME_MEMBER
    assert (
        outcome_facade.record_authorization_outcome
        is outcome_recording.record_authorization_outcome
    )
    assert (
        outcome_facade.validate_authorization_outcome
        is outcome_store.validate_authorization_outcome
    )
    assert (
        outcome_facade.finalize_failed_authorization
        is outcome_failure.finalize_failed_authorization
    )
    assert (
        outcome_facade.validate_complete_run_bundle
        is complete_validation.validate_complete_run_bundle
    )


def _admission(tmp_path: Path) -> SevenInputRunAdmission:
    values = tuple(f"{index:x}" * 64 for index in range(1, 16))
    return SevenInputRunAdmission(
        config_contract_hash=values[0],
        protocol_hash=values[1],
        seven_input_contract_hash=values[2],
        source_seal_hash=values[3],
        source_seal_receipt_hash=values[4],
        source_training_surface_receipt_hash=values[5],
        source_training_surface_hash=values[6],
        input_location_binding_hash=values[7],
        workspace_input_manifest_sha256=values[8],
        workspace_provenance_receipt_hash=values[9],
        authorization_amendment_sha256=values[10],
        lifecycle_source_seal_sha256=values[11],
        lifecycle_source_seal_receipt_hash=values[12],
        workspace_snapshot_sha256=values[13],
        workspace_plan_sha256=values[14],
        final_envelope_sha256="a" * 64,
        execution_launch_authority_sha256="b" * 64,
        sealed_replay_receipt_hash="c" * 64,
        artifact_root=tmp_path / "artifact",
        scratch_root=tmp_path / "scratch",
        _factory_token=_ADMISSION_TOKEN,
    )


def test_post_claim_prestate_failure_is_durably_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    monkeypatch.setattr(
        claim_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )
    claim = claim_authorization_lease(admission, run_identity_hash="d" * 64)
    outcome = outcome_facade.finalize_failed_authorization(
        claim,
        artifact_root=admission.artifact_root,
        original_error=RuntimeError("injected post-claim failure"),
    )
    assert outcome.status == "FAILED_EXHAUSTED"
    assert outcome.claim_hash == claim.claim_hash
    assert outcome.terminal_run_state_receipt_hash is None
    assert outcome.final_bundle_receipt_hash is None
    assert not admission.artifact_root.exists()
    assert not admission.scratch_root.exists()
    assert (claim.path / outcome_contracts.OUTCOME_MEMBER).is_file()


def test_complete_reopening_binds_all_success_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    lease = tmp_path / "lease"
    lease.mkdir()
    claim = SimpleNamespace(
        path=lease,
        claim_hash="1" * 64,
        payload={"artifact_root": root.as_posix(), "run_identity_hash": "2" * 64},
    )
    bundle = SimpleNamespace(receipt_hash="3" * 64)
    prepared = SimpleNamespace(state_hash="4" * 64, receipt_hash="5" * 64)
    state = SimpleNamespace(
        status="COMPLETE",
        phase="COMPLETE",
        artifact_root=root,
        authorization_lease_claim_hash=claim.claim_hash,
        run_identity_hash=claim.payload["run_identity_hash"],
        evidence_hash=bundle.receipt_hash,
        state_hash=prepared.state_hash,
        receipt_hash="6" * 64,
    )
    seal = SimpleNamespace(
        prepared_state_hash=prepared.state_hash,
        prepared_state_receipt_hash=prepared.receipt_hash,
        final_bundle_receipt_hash=bundle.receipt_hash,
        receipt_hash="7" * 64,
        artifact_inventory_hash="8" * 64,
    )
    commit = SimpleNamespace(
        prepared_state_hash=state.state_hash,
        final_bundle_receipt_hash=bundle.receipt_hash,
        complete_artifact_seal_receipt_hash=seal.receipt_hash,
        artifact_inventory_hash=seal.artifact_inventory_hash,
        journal_hash="9" * 64,
    )
    monkeypatch.setattr(
        complete_validation,
        "validate_authorization_lease",
        lambda value: value,
    )
    monkeypatch.setattr(
        complete_validation,
        "validate_terminal_run_state",
        lambda value: value,
    )
    monkeypatch.setattr(
        complete_validation,
        "validate_final_aggregate_bundle",
        lambda _root, *, expected_receipt: expected_receipt,
    )
    monkeypatch.setattr(
        complete_validation,
        "validate_completion_commit",
        lambda value, *, expected_prepared_state: value,
    )
    monkeypatch.setattr(
        complete_validation,
        "validate_complete_artifact_seal",
        lambda _root, *, expected: expected,
    )
    monkeypatch.setattr(
        complete_validation,
        "pending_publications",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        complete_validation,
        "_validate_complete_lifecycle_lineage",
        lambda *_args, **_kwargs: "a" * 64,
    )
    evidence = complete_validation.reopen_complete_run_evidence(
        claim,
        terminal_state=state,
        final_bundle=bundle,
        prepared_state=prepared,
        completion_commit=commit,
        complete_artifact_seal=seal,
    )
    assert evidence.lifecycle_lineage_hash == "a" * 64
    assert evidence.complete_artifact_seal is seal

    drifted_commit = SimpleNamespace(**vars(commit))
    drifted_commit.artifact_inventory_hash = "b" * 64
    with pytest.raises(ProtocolError, match="complete transaction lineage drifted"):
        complete_validation.reopen_complete_run_evidence(
            claim,
            terminal_state=state,
            final_bundle=bundle,
            prepared_state=prepared,
            completion_commit=drifted_commit,
            complete_artifact_seal=seal,
        )
