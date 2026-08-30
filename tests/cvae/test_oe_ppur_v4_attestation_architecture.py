from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution import (
    attestation_contracts,
    attestation_validation,
    attestation_workers,
    fresh_attestation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    EXPECTED_CASE_COUNT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.terminal.contracts import (
    ALLOWED_AGGREGATE_METRICS,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _imports(module) -> tuple[str, ...]:
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), path.as_posix())
    rows: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rows.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            rows.append(node.module or "")
    return tuple(rows)


def _terminal_receipt():
    payload = {
        "schema_version": "oe_ppur_v4_aggregate_only_terminal_receipt_v1",
        "boundary_receipt_hash": "a" * 64,
        "decision_ledger_receipt_hash": "b" * 64,
        "evaluated_case_count": EXPECTED_CASE_COUNT,
        "routed_case_count": 0,
        "exact_p_fallback_count": EXPECTED_CASE_COUNT,
        "aggregate_metrics": {name: 0.0 for name in ALLOWED_AGGREGATE_METRICS},
        "raw_paths_present": False,
        "raw_labels_present": False,
        "per_row_values_present": False,
        "per_case_values_present": False,
    }
    payload["receipt_hash"] = canonical_hash(payload)
    return _reconstruct_persisted_aggregate_only_terminal_receipt(payload)


def test_fresh_attestation_facade_preserves_api_and_strict_layering() -> None:
    assert (
        fresh_attestation.FinalAggregateAttestationReceipt
        is attestation_contracts.FinalAggregateAttestationReceipt
    )
    assert (
        fresh_attestation.attest_preterminal_artifact_twice
        is attestation_workers.attest_preterminal_artifact_twice
    )
    assert (
        fresh_attestation._validate_preterminal_files
        is attestation_validation._validate_preterminal_files
    )

    facade_imports = _imports(fresh_attestation)
    contract_imports = _imports(attestation_contracts)
    validation_imports = _imports(attestation_validation)
    worker_imports = _imports(attestation_workers)
    assert {name.rsplit(".", 1)[-1] for name in facade_imports} >= {
        "attestation_contracts",
        "attestation_validation",
        "attestation_workers",
    }
    assert not any(
        name.endswith(("attestation_workers", "attestation_validation"))
        for name in contract_imports
    )
    assert not any(
        name.endswith(("attestation_workers", "attestation_contracts"))
        for name in validation_imports
    )
    assert any(name.endswith("attestation_contracts") for name in worker_imports)
    assert any(name.endswith("attestation_validation") for name in worker_imports)


def test_attestation_modules_do_not_import_terminal_label_access() -> None:
    forbidden = ("label_reader", "manifest_scoring", "terminal.evaluator")
    for module in (
        attestation_contracts,
        attestation_validation,
        attestation_workers,
        fresh_attestation,
    ):
        assert not any(
            token in imported for imported in _imports(module) for token in forbidden
        )


def test_spawn_runner_rejects_any_label_bearing_request() -> None:
    with pytest.raises(ProtocolError, match="label firewall"):
        attestation_workers._run_two_fresh_spawn_workers(
            attestation_workers._spawn_attestation_worker,
            {"manifest_path": "/tmp/manifest", "row_labels": "forbidden"},
            timeout_seconds=1.0,
        )


def test_final_aggregate_uses_two_real_distinct_spawn_processes(
    tmp_path: Path,
) -> None:
    receipt = _terminal_receipt()
    path = tmp_path / "terminal_metrics.json"
    raw = json.dumps(
        receipt.to_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    path.write_text(raw, encoding="utf-8")

    attestation = fresh_attestation.attest_terminal_aggregate_twice(
        path,
        receipt,
        timeout_seconds=60.0,
    )

    assert len(set(attestation.validator_process_pids)) == 2
    assert os.getpid() not in attestation.validator_process_pids
    assert len(set(attestation.worker_attestation_hashes)) == 2
    assert attestation.terminal_receipt_hash == receipt.receipt_hash
    assert attestation.terminal_file_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    payload = attestation.to_payload()
    assert payload["fresh_process_count"] == 2
    assert payload["aggregate_only"] is True
    assert payload["raw_labels_present"] is False


def test_persisted_attestation_rejects_label_firewall_drift() -> None:
    issued = attestation_contracts._issue_final_aggregate_attestation(
        terminal_receipt_hash="1" * 64,
        terminal_file_sha256="2" * 64,
        terminal_file_identity_sha256="3" * 64,
        validator_runtime_sha256="4" * 64,
        validator_process_pids=(101, 102),
        worker_attestation_hashes=("5" * 64, "6" * 64),
        _validator_token=attestation_contracts._FINAL_ATTESTATION_TOKEN,
    )
    payload = issued.to_payload()
    payload["raw_labels_present"] = True
    with pytest.raises(ProtocolError, match="schema drifted"):
        fresh_attestation._reconstruct_final_aggregate_attestation(payload)
