from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    INDEX_EXCLUDED,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.fresh_process_validation import (
    ATTESTATION_KEY,
    ATTESTATION_SCHEMA,
    THREAD_ENVIRONMENT,
    VALIDATOR_ENTRYPOINT,
    require_two_fresh_process_validations,
    verify_attested_validation_checks,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.hashing import (
    canonical_hash,
    canonical_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.run_admission import (
    reject_existing_run_state,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


CONFIG_HASH = canonical_hash({"config": "test"})
PROTOCOL_HASH = canonical_hash({"protocol": "test"})


def _content_root(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    for index, member in enumerate(CONTENT_INDEX_MEMBERS):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}\n".encode("ascii"))
    return root


def _attestation(checks: dict[str, object]) -> dict[str, object]:
    expected_hash = canonical_hash(checks)
    pids = [41001, 41002]
    results = [
        {
            "ordinal": ordinal,
            "process_id": pid,
            "exit_code": 0,
            "result_hash": canonical_hash({"process_id": pid, "checks": checks}),
            "reconstructed_check_hash": expected_hash,
        }
        for ordinal, pid in enumerate(pids, start=1)
    ]
    unhashed = {
        "schema_version": ATTESTATION_SCHEMA,
        "status": "PASS",
        "fresh_python_process_count": 2,
        "independent_fresh_python_processes": True,
        "process_launches_sequential": True,
        "persisted_resolved_config_loaded_by_each_process": True,
        "full_scientific_reconstruction_called_by_each_process": True,
        "pending_validation_allowed": True,
        "cuda_visible_devices": "",
        "outer_blas_threads": 1,
        "fitted_reconstruction_blas_threads": 3,
        "worker_thread_environment": dict(THREAD_ENVIRONMENT),
        "parent_process_id": 41000,
        "child_process_ids": pids,
        "child_process_results": results,
        "subprocess_exit_codes": [0, 0],
        "reconstructed_check_payloads_exactly_equal": True,
        "reconstructed_check_hash": expected_hash,
        "validator_entrypoint": VALIDATOR_ENTRYPOINT,
    }
    return {**unhashed, "attestation_hash": canonical_hash(unhashed)}


def test_exact_inventory_is_49_with_45_content_index_members() -> None:
    assert len(REQUIRED_FILES) == 49
    assert len(INDEX_EXCLUDED) == 4
    assert len(CONTENT_INDEX_MEMBERS) == 45
    assert set(REQUIRED_FILES) == set(CONTENT_INDEX_MEMBERS) | set(INDEX_EXCLUDED)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", "coherently_rehashed_alternate_schema"),
        ("publication_status", "PASS"),
        ("consumed_test_diagnostic_only", False),
        ("fresh_evidence", True),
        ("promotion_eligible", True),
        ("may_feed_another_experiment", True),
        ("routing_success_claimed", True),
        ("weights_selected_on_same_evaluation_surface", False),
        ("terminal_checkpoint_persisted", True),
        ("terminal_decision", "PROMOTE"),
        ("unexpected_top_level_key", True),
    ),
)
def test_content_index_rejects_coherently_rehashed_claim_header_tamper(
    tmp_path: Path, field: str, replacement: object
) -> None:
    root = _content_root(tmp_path)
    write_content_index(
        root,
        config_contract_hash=CONFIG_HASH,
        protocol_contract_hash=PROTOCOL_HASH,
    )
    path = root / "manifests/content_index.json"
    payload = read_json(path)
    payload[field] = replacement
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    payload["content_hash"] = canonical_hash(unhashed)
    atomic_json(path, payload)
    with pytest.raises(ProtocolError, match="header drifted"):
        validate_content_index(
            root,
            config_contract_hash=CONFIG_HASH,
            protocol_contract_hash=PROTOCOL_HASH,
        )


def test_content_index_is_nonrepairing_and_closed_world_is_exact(tmp_path: Path) -> None:
    root = _content_root(tmp_path)
    index = write_content_index(
        root,
        config_contract_hash=CONFIG_HASH,
        protocol_contract_hash=PROTOCOL_HASH,
    )
    assert validate_content_index(
        root,
        config_contract_hash=CONFIG_HASH,
        protocol_contract_hash=PROTOCOL_HASH,
    ) == index
    atomic_json(root / "reports/run_state.json", {"status": "RUNNING"})
    assert_closed_world(root, allow_incomplete=False, allow_pending_validation=True)
    foreign = root / "reports/foreign.json"
    atomic_json(foreign, {"foreign": True})
    with pytest.raises(ProtocolError, match="closed-world drifted"):
        assert_closed_world(root, allow_incomplete=False, allow_pending_validation=True)
    foreign.unlink()
    nested_lock = root / "tables/.run.lock"
    nested_lock.write_bytes(b"foreign nested lock\n")
    with pytest.raises(ProtocolError, match="closed-world drifted"):
        assert_closed_world(root, allow_incomplete=False, allow_pending_validation=True)
    nested_lock.unlink()
    (root / CONTENT_INDEX_MEMBERS[0]).write_bytes(b"coherent member drift\n")
    with pytest.raises(ProtocolError, match="refuses content-index repair"):
        write_content_index(
            root,
            config_contract_hash=CONFIG_HASH,
            protocol_contract_hash=PROTOCOL_HASH,
        )


def test_content_index_rejects_coherently_rehashed_row_schema_tamper(
    tmp_path: Path,
) -> None:
    root = _content_root(tmp_path)
    write_content_index(
        root,
        config_contract_hash=CONFIG_HASH,
        protocol_contract_hash=PROTOCOL_HASH,
    )
    path = root / "manifests/content_index.json"
    payload = read_json(path)
    payload["members"][0]["size_bytes"] = float(
        payload["members"][0]["size_bytes"]
    )
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    payload["content_hash"] = canonical_hash(unhashed)
    atomic_json(path, payload)
    with pytest.raises(ProtocolError, match="row malformed"):
        validate_content_index(
            root,
            config_contract_hash=CONFIG_HASH,
            protocol_contract_hash=PROTOCOL_HASH,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", "alternate"),
        ("persisted_resolved_config_loaded_by_each_process", False),
        ("full_scientific_reconstruction_called_by_each_process", False),
        ("pending_validation_allowed", False),
        ("outer_blas_threads", 3),
        ("fitted_reconstruction_blas_threads", 1),
        ("parent_process_id", True),
        ("unexpected_top_level_key", True),
    ),
)
def test_fresh_attestation_rejects_coherent_header_tampering(
    field: str, replacement: object
) -> None:
    checks = {"schema_version": "test_validation_v1", "status": "PASS"}
    attestation = _attestation(checks)
    attestation[field] = replacement
    unhashed = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    attestation["attestation_hash"] = canonical_hash(unhashed)
    with pytest.raises(ProtocolError, match="attestation drifted"):
        verify_attested_validation_checks(
            {**checks, ATTESTATION_KEY: attestation},
            expected_reconstructed_checks=checks,
            persisted_attestation=attestation,
        )


def test_fresh_attestation_recomputes_each_child_result_hash() -> None:
    checks = {"schema_version": "test_validation_v1", "status": "PASS"}
    attestation = _attestation(checks)
    assert verify_attested_validation_checks(
        {**checks, ATTESTATION_KEY: attestation},
        expected_reconstructed_checks=checks,
        persisted_attestation=attestation,
    )[ATTESTATION_KEY] == attestation
    tampered = copy.deepcopy(attestation)
    tampered["child_process_results"][0]["result_hash"] = canonical_hash("alternate")
    unhashed = {
        key: value for key, value in tampered.items() if key != "attestation_hash"
    }
    tampered["attestation_hash"] = canonical_hash(unhashed)
    with pytest.raises(ProtocolError, match="attestation drifted"):
        verify_attested_validation_checks(
            {**checks, ATTESTATION_KEY: tampered},
            expected_reconstructed_checks=checks,
            persisted_attestation=tampered,
        )


def test_two_fresh_process_launcher_requires_two_sequential_independent_pids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.fresh_process_validation as module

    root = tmp_path / "bundle"
    (root / "reports").mkdir(parents=True)
    checks = {"schema_version": "test_validation_v1", "status": "PASS"}
    pids = iter((51001, 51002))
    calls: list[int] = []

    def run_worker(_root: Path) -> subprocess.CompletedProcess[str]:
        pid = next(pids)
        calls.append(pid)
        payload = {"process_id": pid, "checks": checks}
        return subprocess.CompletedProcess(
            args=("fresh",), returncode=0, stdout=canonical_json(payload) + "\n", stderr=""
        )

    monkeypatch.setattr(module, "_run_worker", run_worker)
    result = require_two_fresh_process_validations(root, expected_checks=checks)
    assert calls == [51001, 51002]
    assert result[ATTESTATION_KEY]["child_process_ids"] == calls
    assert result[ATTESTATION_KEY]["process_launches_sequential"] is True


def test_existing_run_state_is_never_recovered(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "reports").mkdir(parents=True)
    atomic_json(
        root / "reports/run_state.json",
        {"status": "FAILED", "phase": "TERMINAL_LABELS_METRICS_ORACLES_SENSITIVITY"},
    )
    with pytest.raises(ProtocolError, match="cross-run recovery is forbidden"):
        reject_existing_run_state(root)


def test_scientific_validator_modules_are_statically_read_only() -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router as package

    root = Path(next(iter(package.__path__)))
    members = (
        "validation.py",
        "validation_prelabel.py",
        "validation_route.py",
        "validation_endpoint.py",
        "validation_terminal.py",
    )
    forbidden_calls = {
        "atomic_json",
        "persist_json",
        "persist_rows",
        "write_bytes",
        "write_text",
        "mkdir",
        "replace",
        "unlink",
        "rmtree",
    }
    for member in members:
        tree = ast.parse((root / member).read_text(encoding="utf-8"))
        calls = {
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert not calls & forbidden_calls, (member, calls & forbidden_calls)
