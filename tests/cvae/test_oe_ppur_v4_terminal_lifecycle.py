from __future__ import annotations

import ast
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution import preterminal_persistence
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.capacity_preflight as capacity_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lease_claim as claim_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.capacity_preflight import (
    MINIMUM_ARTIFACT_FREE_BYTES,
    MINIMUM_RAM_AVAILABLE_BYTES,
    MINIMUM_SCRATCH_FREE_BYTES,
    preflight_resource_capacity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lease_io import (
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.lease_claim import (
    assert_authorization_unclaimed,
    claim_authorization_lease,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.run_admission import (
    SevenInputRunAdmission,
    _ADMISSION_TOKEN,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.authorization_outcome import (
    record_authorization_outcome,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.run_state import (
    build_run_identity_hash,
    create_single_use_run,
    mark_failed_exhausted,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.terminal import (
    issue_terminal_aggregate_capability,
    seal_guarded_preterminal_boundary,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.terminal.contracts import (
    _ATTESTATION_TOKEN,
    _issue_artifact_only_preterminal_attestation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.terminal.label_reader import (
    CaseRoutingDiagnostic,
    _VIEW_TOKEN,
    _build_manager_owned_manifest_label_reader,
    _seal_manager_owned_terminal_input,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _attestation(*, pid: int, ledger: str = "a" * 64):
    return _issue_artifact_only_preterminal_attestation(
        sealed_ledger_receipt_hash=ledger,
        artifact_file_sha256="b" * 64,
        artifact_file_identity_sha256="c" * 64,
        validator_runtime_sha256="d" * 64,
        process_pid=pid,
        _validator_token=_ATTESTATION_TOKEN,
    )


def _boundary():
    return seal_guarded_preterminal_boundary(
        seven_input_contract_hash="1" * 64,
        source_seal_hash="2" * 64,
        source_training_surface_receipt_hash="3" * 64,
        decision_ledger_receipt_hash="a" * 64,
        attestations=(_attestation(pid=101), _attestation(pid=102)),
        case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        case_count=218,
        exact_p_fallback_count=109,
    )


def test_terminal_boundary_requires_exact_218_and_two_distinct_attestations() -> None:
    with pytest.raises(ProtocolError, match="two distinct artifact-only"):
        row = _attestation(pid=101)
        seal_guarded_preterminal_boundary(
            seven_input_contract_hash="1" * 64,
            source_seal_hash="2" * 64,
            source_training_surface_receipt_hash="3" * 64,
            decision_ledger_receipt_hash="a" * 64,
            attestations=(row, row),
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            case_count=218,
            exact_p_fallback_count=109,
        )
    with pytest.raises(ProtocolError, match="boundary coverage drifted"):
        seal_guarded_preterminal_boundary(
            seven_input_contract_hash="1" * 64,
            source_seal_hash="2" * 64,
            source_training_surface_receipt_hash="3" * 64,
            decision_ledger_receipt_hash="a" * 64,
            attestations=(_attestation(pid=101), _attestation(pid=102)),
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            case_count=217,
            exact_p_fallback_count=109,
        )


def test_terminal_capability_is_process_local_one_shot_and_aggregate_only() -> None:
    boundary = _boundary()
    row_case_ids = tuple(
        ("0", f"case-{index % 218:03d}") for index in range(9_928)
    )
    labels = tuple(index % 2 for index in range(9_928))
    protected = tuple(0.8 if value else 0.2 for value in labels)
    diagnostics = tuple(
        CaseRoutingDiagnostic(
            center_id="0",
            case_id=f"case-{index:03d}",
            selected_action_id=(
                "P_PROTECTED" if index < 109 else "B::zero_to_one"
            ),
            oracle_action_id="B::zero_to_one",
            spearman_rank_correlation=0.25,
            normalized_oracle_gap=0.1,
        )
        for index in range(218)
    )
    view = _seal_manager_owned_terminal_input(
        boundary,
        row_case_ids=row_case_ids,
        row_labels=labels,
        selected_probabilities=protected,
        protected_probabilities=protected,
        case_diagnostics=diagnostics,
        _manager_token=_VIEW_TOKEN,
    )
    capability = issue_terminal_aggregate_capability(
        boundary,
        reader=_build_manager_owned_manifest_label_reader(view),
    )
    payload = capability.score_aggregates().to_payload()
    assert payload["evaluated_case_count"] == 218
    assert payload["raw_labels_present"] is False
    assert payload["per_row_values_present"] is False
    assert payload["per_case_values_present"] is False
    with pytest.raises(ProtocolError, match="replayed"):
        capability.score_aggregates()
    with pytest.raises(TypeError):
        pickle.dumps(capability)
    with pytest.raises(TypeError):
        copy.copy(capability)


def test_preterminal_and_journal_writes_refuse_overwrite(tmp_path: Path) -> None:
    array_path = tmp_path / "matrix.npy"
    preterminal_persistence._write_npy_exclusive(
        array_path,
        np.asarray([[0.25, 0.75]], dtype="<f4"),
    )
    with pytest.raises(FileExistsError):
        preterminal_persistence._write_npy_exclusive(
            array_path,
            np.asarray([[0.5, 0.5]], dtype="<f4"),
        )

    journal = tmp_path / "claim.json"
    publish_json_no_overwrite(journal, {"status": "CONSUMED"}, role="claim")
    assert read_json_regular(journal, role="claim") == {"status": "CONSUMED"}
    assert not pending_publications(tmp_path, journal.name)
    with pytest.raises(ProtocolError, match="target is unsafe"):
        publish_json_no_overwrite(journal, {"status": "REUSED"}, role="claim")


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


def test_lease_is_consumed_before_absent_output_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    monkeypatch.setattr(
        claim_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )
    assert_authorization_unclaimed(admission.artifact_root, admission.scratch_root)
    claim = claim_authorization_lease(admission, run_identity_hash="d" * 64)
    assert claim.path.is_dir()
    assert not admission.artifact_root.exists()
    assert not admission.scratch_root.exists()
    with pytest.raises(ProtocolError, match="authorization is exhausted"):
        assert_authorization_unclaimed(admission.artifact_root, admission.scratch_root)


def test_capacity_preflight_is_read_only_for_two_absent_launch_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(
        capacity_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                "0, NVIDIA RTX A5000, 24576, 20000\n"
                "1, NVIDIA RTX A5000, 24576, 20000\n"
            )
        ),
    )
    monkeypatch.setattr(
        capacity_module,
        "_read_mem_available_bytes",
        lambda: MINIMUM_RAM_AVAILABLE_BYTES,
    )
    monkeypatch.setattr(
        capacity_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            free=MINIMUM_ARTIFACT_FREE_BYTES + MINIMUM_SCRATCH_FREE_BYTES
        ),
    )
    receipt = preflight_resource_capacity(artifact, scratch)
    assert receipt.filesystem_mutation_performed is False
    assert not artifact.exists()
    assert not scratch.exists()


def test_every_post_claim_failure_is_durably_failed_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    monkeypatch.setattr(
        claim_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )
    run_identity = build_run_identity_hash(admission)
    claim = claim_authorization_lease(admission, run_identity_hash=run_identity)
    state = create_single_use_run(
        admission,
        claim,
        run_identity_hash=run_identity,
    )
    assert state["phase"] == "ADMITTED"
    terminal = mark_failed_exhausted(
        admission.artifact_root,
        error_class="InjectedFailure",
        evidence_hash="e" * 64,
    )
    outcome = record_authorization_outcome(claim, terminal_state=terminal)
    assert terminal.status == "FAILED_EXHAUSTED"
    assert outcome.status == "FAILED_EXHAUSTED"
    assert outcome.final_bundle_receipt_hash is None
    assert not outcome.to_payload().get("recovery_allowed", False)


def test_owned_v4_lifecycle_tree_has_no_predecessor_package_imports() -> None:
    package = (
        Path(__file__).resolve().parents[2]
        / "src/midogpp_thesis/cvae/diagnostics/"
        "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4"
    )
    predecessor = (
        "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3"
    )
    for source in package.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(predecessor not in alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert predecessor not in node.module
