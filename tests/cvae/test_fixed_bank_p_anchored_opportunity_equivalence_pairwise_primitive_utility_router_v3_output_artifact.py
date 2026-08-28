from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_lease import (
    AuthorizationLeaseClaim,
    AuthorizationOutcomeReceipt,
    LEASE_DIRECTORY_NAME,
    record_authorization_outcome,
    validate_authorization_outcome,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    build_authorization_ready_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.inputs import (
    build_authorized_seven_input_contract,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    EXPERIMENT_ID,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.output_artifact import (
    CONTENT_INDEX_MEMBER,
    FINAL_ATTESTATION_MEMBER,
    FINAL_PAYLOAD_MEMBERS,
    VALIDATION_INDEX_MEMBER,
    assemble_final_aggregate_bundle,
    validate_final_aggregate_bundle,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.preterminal_artifact import (
    FinalAggregateAttestationReceipt,
    _reconstruct_final_aggregate_attestation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_state import (
    PHASE_ORDER,
    TerminalRunStateReceipt,
    atomic_json,
    mark_complete,
    mark_failed_exhausted,
    validate_terminal_run_state,
    write_exclusive_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    build_source_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_supervision import (
    SourceTrainingSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal.contracts import (
    ALLOWED_AGGREGATE_METRICS,
    AggregateOnlyTerminalReceipt,
    _ATTESTATION_TOKEN,
    _issue_artifact_only_preterminal_attestation,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
    seal_guarded_preterminal_boundary,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _source_surface_test_double(producer_seal: str) -> SourceTrainingSurface:
    """A typed shell only; source-bundle parsing has dedicated contract tests."""

    surface = object.__new__(SourceTrainingSurface)
    receipt = SimpleNamespace(
        target_rows_present=False,
        target_labels_used=False,
        receipt_hash="a" * 64,
        row_order_sha256="b" * 64,
        compiler_recomputation_receipt_sha256="d" * 64,
        contract=SimpleNamespace(
            contract_hash="4" * 64,
            producer_source_seal_sha256=producer_seal,
        ),
    )
    object.__setattr__(surface, "receipt", receipt)
    object.__setattr__(surface, "surface_hash", "5" * 64)
    return surface


def _boundary_and_attestations(
    source_seal_hash: str,
    seven_input_contract_hash: str,
):
    ledger = "a" * 64
    attestations = tuple(
        _issue_artifact_only_preterminal_attestation(
            sealed_ledger_receipt_hash=ledger,
            artifact_file_sha256="b" * 64,
            artifact_file_identity_sha256="c" * 64,
            validator_runtime_sha256="d" * 64,
            process_pid=100 + index,
            _validator_token=_ATTESTATION_TOKEN,
        )
        for index in range(2)
    )
    return (
        seal_guarded_preterminal_boundary(
            seven_input_contract_hash=seven_input_contract_hash,
            source_seal_hash=source_seal_hash,
            source_training_surface_receipt_hash="a" * 64,
            decision_ledger_receipt_hash=ledger,
            attestations=attestations,
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            case_count=218,
            exact_p_fallback_count=109,
        ),
        attestations,
    )


def _terminal_receipt(boundary_hash: str, ledger: str) -> AggregateOnlyTerminalReceipt:
    payload = {
        "schema_version": "oe_ppur_v3_aggregate_only_terminal_receipt_v1",
        "boundary_receipt_hash": boundary_hash,
        "decision_ledger_receipt_hash": ledger,
        "evaluated_case_count": 218,
        "routed_case_count": 109,
        "exact_p_fallback_count": 109,
        "aggregate_metrics": {
            key: 0.25 for key in ALLOWED_AGGREGATE_METRICS
        },
        "raw_paths_present": False,
        "raw_labels_present": False,
        "per_row_values_present": False,
        "per_case_values_present": False,
    }
    payload["receipt_hash"] = canonical_hash(payload)
    return _reconstruct_persisted_aggregate_only_terminal_receipt(payload)


def _final_attestation(
    root: Path,
    terminal: AggregateOnlyTerminalReceipt,
    *,
    pids: tuple[int, int] = (300, 301),
) -> FinalAggregateAttestationReceipt:
    terminal_path = root / "reports/terminal_metrics.json"
    write_exclusive_json(terminal_path, terminal.to_payload())
    raw = terminal_path.read_bytes()
    metadata = terminal_path.stat()
    body = {
        "schema_version": "oe_ppur_v3_final_aggregate_fresh_process_attestation_v1",
        "terminal_receipt_hash": terminal.receipt_hash,
        "terminal_file_sha256": hashlib.sha256(raw).hexdigest(),
        "terminal_file_identity_sha256": canonical_hash(
            {
                "schema_version": "oe_ppur_v3_terminal_file_identity_v1",
                "stat": (
                    int(metadata.st_dev),
                    int(metadata.st_ino),
                    int(metadata.st_mode),
                    int(metadata.st_size),
                    int(metadata.st_mtime_ns),
                ),
            }
        ),
        "validator_runtime_sha256": "7" * 64,
        "validator_process_pids": list(pids),
        "worker_attestation_hashes": ["8" * 64, "9" * 64],
        "fresh_process_count": 2,
        "aggregate_only": True,
        "raw_labels_present": False,
    }
    body["receipt_hash"] = canonical_hash(body)
    return _reconstruct_final_aggregate_attestation(body)


def _write_final_attested_state(
    root: Path,
    final_attestation_hash: str,
    *,
    run_identity_hash: str = "2" * 64,
    lease_claim_hash: str = "7" * 64,
    config_contract_hash: str = "3" * 64,
    protocol_hash: str = "4" * 64,
    source_seal_hash: str = "5" * 64,
    seven_input_admission_hash: str = "6" * 64,
    evidence_by_phase: dict[str, str] | None = None,
) -> None:
    transitions = []
    previous_hash = None
    current = "ADMITTED"
    for sequence, target in enumerate(PHASE_ORDER[1:-2]):
        transition_body = {
            "sequence": sequence,
            "from_phase": current,
            "to_phase": target,
            "status": "RUNNING",
            "evidence_hash": (
                evidence_by_phase[target]
                if evidence_by_phase is not None
                else (
                    final_attestation_hash
                    if target == "FINAL_ATTESTED"
                    else "1" * 64
                )
            ),
            "previous_transition_hash": previous_hash,
        }
        transition = {
            **transition_body,
            "transition_hash": canonical_hash(transition_body),
        }
        transitions.append(transition)
        previous_hash = transition["transition_hash"]
        current = target
    body = {
        "schema_version": "oe_ppur_v3_single_use_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": run_identity_hash,
        "config_contract_hash": config_contract_hash,
        "protocol_hash": protocol_hash,
        "source_seal_hash": source_seal_hash,
        "seven_input_admission_hash": seven_input_admission_hash,
        "authorization_lease_claim_hash": lease_claim_hash,
        "status": "RUNNING",
        "phase": "FINAL_ATTESTED",
        "transition_count": len(transitions),
        "transitions": transitions,
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "scratch_recovery_allowed": False,
        "raw_labels_persisted": False,
        "updated_at_utc": "2026-08-28T00:00:00+00:00",
        "error_class": None,
    }
    atomic_json(root / "reports/run_state.json", {**body, "state_hash": canonical_hash(body)})


def _assemble_bundle(root: Path):
    source_seal = build_source_seal()
    seven_inputs = build_authorized_seven_input_contract()
    boundary, preterminal_attestations = _boundary_and_attestations(
        source_seal.combined_source_sha256,
        seven_inputs.receipt_hash,
    )
    terminal = _terminal_receipt(
        boundary.receipt_hash, boundary.decision_ledger_receipt_hash
    )
    final_attestation = _final_attestation(root, terminal)
    final_path = root / FINAL_ATTESTATION_MEMBER
    final_path.write_text(
        (
            json.dumps(
                final_attestation.to_payload(), sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ),
        encoding="utf-8",
    )
    config = build_authorization_ready_config(
        source_supervision_content_sha256="a" * 64,
        source_supervision_row_order_sha256="b" * 64,
        source_supervision_producer_seal_sha256=source_seal.combined_source_sha256,
        source_supervision_recomputation_receipt_sha256="d" * 64,
        authorization_amendment_sha256="e" * 64,
    )
    source_surface = _source_surface_test_double(source_seal.combined_source_sha256)
    receipt = assemble_final_aggregate_bundle(
        root,
        config=config,
        seven_input_contract=seven_inputs,
        source_seal=source_seal,
        source_surface=source_surface,
        preterminal_boundary=boundary,
        preterminal_attestations=preterminal_attestations,
        terminal_receipt=terminal,
        final_attestation=final_attestation,
        runtime_summary={"persistent_gpu_worker_count": 2, "spawn_cpu_worker_count": 4},
    )
    return (
        receipt,
        config,
        source_seal,
        source_surface,
        seven_inputs,
        boundary,
        preterminal_attestations,
        terminal,
        final_attestation,
    )


def test_final_aggregate_bundle_is_exclusive_read_back_and_label_free(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (
        receipt,
        config,
        source_seal,
        _source_surface,
        seven_inputs,
        boundary,
        preterminal_attestations,
        terminal,
        final_attestation,
    ) = _assemble_bundle(root)

    assert validate_final_aggregate_bundle(root, expected_receipt=receipt) == receipt
    assert (root / CONTENT_INDEX_MEMBER).is_file()
    assert (root / FINAL_ATTESTATION_MEMBER).is_file()
    assert (root / VALIDATION_INDEX_MEMBER).is_file()
    assert all((root / member).is_file() for member in FINAL_PAYLOAD_MEMBERS)
    assert "raw_labels_present" in (root / FINAL_ATTESTATION_MEMBER).read_text()
    assert "case_id" not in (root / CONTENT_INDEX_MEMBER).read_text()

    _write_final_attested_state(root, receipt.final_attestation_hash)
    with pytest.raises(ProtocolError, match="durable commit journal"):
        mark_complete(root, final_bundle=receipt)

    with pytest.raises(ProtocolError, match="refuses overwrite"):
        assemble_final_aggregate_bundle(
            root,
            config=config,
            seven_input_contract=seven_inputs,
            source_seal=source_seal,
            source_surface=_source_surface_test_double(
                source_seal.combined_source_sha256
            ),
            preterminal_boundary=boundary,
            preterminal_attestations=preterminal_attestations,
            terminal_receipt=terminal,
            final_attestation=final_attestation,
            runtime_summary={"persistent_gpu_worker_count": 2},
        )


def test_final_attestation_requires_two_distinct_processes() -> None:
    with pytest.raises(ProtocolError, match="bypassed fresh-process"):
        FinalAggregateAttestationReceipt(
            terminal_receipt_hash="b" * 64,
            terminal_file_sha256="c" * 64,
            terminal_file_identity_sha256="d" * 64,
            validator_runtime_sha256="c" * 64,
            validator_process_pids=(22, 22),
            worker_attestation_hashes=("d" * 64, "e" * 64),
        )

    body = {
        "schema_version": "oe_ppur_v3_final_aggregate_fresh_process_attestation_v1",
        "terminal_receipt_hash": "b" * 64,
        "terminal_file_sha256": "c" * 64,
        "terminal_file_identity_sha256": "d" * 64,
        "validator_runtime_sha256": "c" * 64,
        "validator_process_pids": [22, 22],
        "worker_attestation_hashes": ["d" * 64, "e" * 64],
        "fresh_process_count": 2,
        "aggregate_only": True,
        "raw_labels_present": False,
    }
    body["receipt_hash"] = canonical_hash(body)
    with pytest.raises(ProtocolError, match="attestation drifted"):
        _reconstruct_final_aggregate_attestation(body)

    with pytest.raises(ProtocolError, match="bypassed durable validation"):
        TerminalRunStateReceipt(
            artifact_root=Path("/tmp/not-authoritative"),
            status="COMPLETE",
            phase="COMPLETE",
            state_hash="1" * 64,
            run_identity_hash="2" * 64,
            authorization_lease_claim_hash="3" * 64,
            evidence_hash="4" * 64,
        )


def test_failed_authorization_outcome_is_derived_from_typed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_claim as claim_module

    root = tmp_path / "artifact"
    root.mkdir()
    scratch = tmp_path / "scratch"
    lease_path = root.parent / LEASE_DIRECTORY_NAME
    lease_path.mkdir()
    monkeypatch.setattr(
        claim_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )
    claim_body = {
        "schema_version": "oe_ppur_v3_single_use_authorization_claim_v1",
        "status": "CONSUMED_EXHAUSTED",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "artifact_root": root.as_posix(),
        "scratch_root": scratch.as_posix(),
        "lease_path": lease_path.as_posix(),
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
    claim_payload = {**claim_body, "claim_hash": canonical_hash(claim_body)}
    write_exclusive_json(lease_path / "claim.json", claim_payload)
    claim = AuthorizationLeaseClaim(
        path=lease_path,
        payload=claim_payload,
        claim_hash=str(claim_payload["claim_hash"]),
    )
    _write_final_attested_state(
        root,
        "9" * 64,
        lease_claim_hash=claim.claim_hash,
    )
    failed = mark_failed_exhausted(
        root,
        error_class="SyntheticFailure",
        evidence_hash="a" * 64,
    )
    outcome = record_authorization_outcome(claim, terminal_state=failed)

    assert outcome.status == "FAILED_EXHAUSTED"
    assert outcome.evidence_hash == failed.state_hash
    assert validate_authorization_outcome(claim, expected=outcome) == outcome
    with pytest.raises(ProtocolError, match="bypassed validation"):
        AuthorizationOutcomeReceipt(
            lease_path=lease_path,
            status="FAILED_EXHAUSTED",
            claim_hash=claim.claim_hash,
            evidence_hash=failed.state_hash,
            terminal_run_state_receipt_hash=failed.receipt_hash,
            final_bundle_receipt_hash=None,
            artifact_inventory_hash=None,
            lifecycle_lineage_hash=None,
            outcome_hash="b" * 64,
        )
