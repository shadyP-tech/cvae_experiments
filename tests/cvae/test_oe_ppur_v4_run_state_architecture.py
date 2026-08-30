from __future__ import annotations

import ast
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4 import (
    run_state,
    run_state_completion,
    run_state_contracts,
    run_state_storage,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _imported_modules(module: object) -> set[str]:
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add("." * node.level + (node.module or ""))
    return imports


def _seed_admitted_state(root: Path) -> None:
    body = {
        "schema_version": "oe_ppur_v4_single_use_run_state_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "run_identity_hash": "1" * 64,
        "config_contract_hash": "2" * 64,
        "protocol_hash": "3" * 64,
        "source_seal_hash": "4" * 64,
        "seven_input_admission_hash": "5" * 64,
        "workspace_snapshot_sha256": "6" * 64,
        "workspace_plan_sha256": "7" * 64,
        "final_envelope_sha256": "8" * 64,
        "execution_launch_authority_sha256": "9" * 64,
        "authorization_lease_claim_hash": "a" * 64,
        "status": "RUNNING",
        "phase": "ADMITTED",
        "transition_count": 0,
        "transitions": [],
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "scratch_recovery_allowed": False,
        "raw_labels_persisted": False,
        "updated_at_utc": "2026-08-30T00:00:00+00:00",
        "error_class": None,
    }
    run_state.write_exclusive_json(
        root / "reports/run_state.json",
        {**body, "state_hash": canonical_hash(body)},
    )


def test_run_state_facade_preserves_public_contract_imports() -> None:
    assert run_state.PHASE_ORDER is run_state_contracts.PHASE_ORDER
    assert (
        run_state.PreparedCompleteRunState
        is run_state_contracts.PreparedCompleteRunState
    )
    assert (
        run_state.TerminalRunStateReceipt
        is run_state_contracts.TerminalRunStateReceipt
    )
    assert set(run_state.__all__) == {
        "PHASE_ORDER",
        "PreparedCompleteRunState",
        "TerminalRunStateReceipt",
        "atomic_json",
        "build_run_identity_hash",
        "commit_complete_run_state",
        "create_single_use_run",
        "mark_complete",
        "mark_failed_exhausted",
        "prepare_complete_run_state",
        "read_terminal_run_state",
        "read_run_state",
        "transition_run",
        "validate_prepared_complete_run_state",
        "validate_terminal_run_state",
        "write_exclusive_json",
    }


def test_run_state_contract_and_storage_layers_are_dependency_light() -> None:
    contract_imports = _imported_modules(run_state_contracts)
    storage_imports = _imported_modules(run_state_storage)
    forbidden = {
        ".run_state",
        ".run_admission",
        ".lease_claim",
        ".output_artifact",
        ".completion_transaction",
        ".authorization_outcome",
        ".terminal",
        ".science",
    }
    assert not (contract_imports & forbidden)
    assert not (storage_imports & forbidden)
    assert ".durable_io" in contract_imports
    assert "." in storage_imports  # ``from . import durable_io``

    facade_imports = _imported_modules(run_state)
    assert ".run_state_contracts" in facade_imports
    assert "." in facade_imports  # ``from . import run_state_storage``
    assert not ({"json", "os", "stat", "tempfile"} & facade_imports)


def test_run_state_completion_composition_does_not_close_artifact_cycles() -> None:
    facade_imports = _imported_modules(run_state)
    completion_imports = _imported_modules(run_state_completion)
    cycle_edges = {
        ".completion_transaction",
        ".complete_artifact_validation",
        ".artifact.semantics",
        ".output_artifact",
    }
    assert not (facade_imports & cycle_edges)
    assert not (completion_imports & cycle_edges)
    assert ".output_validation" in completion_imports
    assert ".artifact.completion" in completion_imports


@pytest.mark.parametrize("phase", run_state.PHASE_ORDER[:-1])
def test_every_post_claim_running_phase_fails_once_and_stays_exhausted(
    tmp_path: Path,
    phase: str,
) -> None:
    root = tmp_path / phase.lower()
    _seed_admitted_state(root)
    phase_index = run_state.PHASE_ORDER.index(phase)
    for index in range(1, phase_index + 1):
        previous = run_state.PHASE_ORDER[index - 1]
        target = run_state.PHASE_ORDER[index]
        run_state.transition_run(
            root,
            target,
            expected_phase=previous,
            evidence_hash=f"{index:x}" * 64,
        )

    receipt = run_state.mark_failed_exhausted(
        root,
        error_class=" Injected   post-claim failure ",
        evidence_hash="f" * 64,
    )
    observed = run_state.read_run_state(root)
    assert receipt.status == "FAILED_EXHAUSTED"
    assert receipt.phase == phase
    assert observed["status"] == "FAILED_EXHAUSTED"
    assert observed["phase"] == phase
    assert observed["authorization_exhausted"] is True
    assert observed["cross_run_recovery_allowed"] is False
    assert observed["terminal_recovery_allowed"] is False
    assert observed["scratch_recovery_allowed"] is False
    assert observed["error_class"] == "Injected post-claim failure"
    with pytest.raises(ProtocolError, match="cannot be rewritten"):
        run_state.mark_failed_exhausted(
            root,
            error_class="second failure",
            evidence_hash="e" * 64,
        )


def test_run_state_cannot_skip_a_phase(tmp_path: Path) -> None:
    root = tmp_path / "out-of-order"
    _seed_admitted_state(root)

    with pytest.raises(ProtocolError, match="out of order"):
        run_state.transition_run(
            root,
            "PHYSICAL_PROBABILITIES_MATERIALIZED",
            expected_phase="ADMITTED",
            evidence_hash="b" * 64,
        )
