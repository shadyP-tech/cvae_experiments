from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Mapping

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router import (
    fresh_process_validation,
    persistence,
    recovery,
    recovery_provenance,
    runner,
    runner_runtime,
    validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.hashing import (
    canonical_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.runner_dependencies import (
    CaseDirectionalRunnerDependencies,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


_EMPTY_STATUS_HASH = hashlib.sha256(b"").hexdigest()


def test_exact_detector_pins_failure_inventory_capability_and_report_retry(
    tmp_path: Path,
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    capability = recovery.recovery_capability(root)
    assert capability == recovery.CaseDirectionalFinalizationRecoveryCapability()
    assert capability is not None
    assert capability.validation_only is True
    assert capability.scientific_products_may_be_reconstructed_for_validation is True
    assert capability.scientific_products_may_be_persisted is False
    assert capability.terminal_products_may_be_persisted is False
    assert capability.policy_may_be_mutated is False
    assert len(recovery.FINALIZATION_RECOVERABLE_INVENTORY) == 42
    assert recovery.FINALIZATION_RECOVERABLE_INVENTORY == frozenset(
        REQUIRED_FILES
    ) - {"reports/validation_report.json"}
    assert recovery.detect_registered_case_directional_correctness_abstention_router_recovery(
        root
    )

    atomic_json(root / "reports/validation_report.json", {"hard_crash": True})
    retry = recovery.recovery_capability(root)
    assert retry == recovery.CaseDirectionalFinalizationRecoveryCapability(
        validation_report_present=True
    )
    assert len(recovery.FINALIZATION_REPORT_PRESENT_RETRY_INVENTORY) == 43


def test_stale_missing_report_capability_cannot_delete_a_new_report(
    tmp_path: Path,
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    stale = recovery.recovery_capability(root)
    assert stale == recovery.CaseDirectionalFinalizationRecoveryCapability()
    report = root / "reports/validation_report.json"
    report.write_bytes(b'{"hard_crash":true}\n')
    before = report.read_bytes()

    with pytest.raises(ProtocolError, match="boundary changed after capability"):
        recovery.recover_exact_finalization(
            root,
            config=SimpleNamespace(contract_hash="config"),
            protocol=SimpleNamespace(protocol_hash="protocol"),
            capability=stale,
        )
    assert report.read_bytes() == before
    assert read_json(root / "reports/run_state.json") == (
        recovery.FAILED_FINALIZATION_STATE
    )


def test_exact_detector_rejects_state_inventory_symlink_and_atomic_drift(
    tmp_path: Path,
) -> None:
    root = _indexed_fixture(tmp_path / "state")
    atomic_json(
        root / "reports/run_state.json",
        {**recovery.FAILED_FINALIZATION_STATE, "error": "other"},
    )
    with pytest.raises(ProtocolError, match="not the exact registered"):
        recovery.recovery_capability(root)

    root = _indexed_fixture(tmp_path / "missing")
    (root / "tables/route_model_fits.csv").unlink()
    with pytest.raises(ProtocolError, match="inventory drifted"):
        recovery.recovery_capability(root)

    root = _indexed_fixture(tmp_path / "symlink")
    member = root / "tables/route_model_fits.csv"
    member.unlink()
    member.symlink_to(root / "tables/donor_priors.csv")
    with pytest.raises(ProtocolError, match="symlink"):
        recovery.recovery_capability(root)

    root = _indexed_fixture(tmp_path / "atomic")
    (root / "tables/route_model_fits.csv.123.tmp").write_bytes(b"partial")
    with pytest.raises(ProtocolError, match="unsafe member"):
        recovery.recovery_capability(root)


def test_recovery_audit_binds_clean_changed_revision_and_every_indexed_byte(
    tmp_path: Path,
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    fresh = recovery_provenance.fresh_finalization_audit_payload()
    assert fresh["finalization_recovery_used"] is False
    assert fresh["route_models_reconstructed_for_validation"] is True
    assert fresh["route_models_persisted_during_recovery"] is False
    assert fresh["terminal_evaluation_reconstructed_for_validation"] is True
    assert fresh["terminal_evaluation_persisted_during_recovery"] is False
    audit = recovery_provenance.finalization_recovery_audit_payload(
        root, current_repository_state=_repo("b" * 40)
    )
    assert audit["finalization_recovery_used"] is True
    assert audit["original_repository_revision"] == "a" * 40
    assert audit["repair_repository_revision"] == "b" * 40
    assert set(audit["indexed_member_fingerprints"]) == set(
        CONTENT_INDEX_MEMBERS
    )
    assert audit["scientific_reconstruction_performed_for_validation"] is True
    assert audit["route_models_reconstructed_for_validation"] is True
    assert audit["route_candidate_scores_reconstructed_for_validation"] is True
    assert audit["route_decisions_reconstructed_for_validation"] is True
    assert audit["terminal_evaluation_reconstructed_for_validation"] is True
    assert audit["route_models_persisted_during_recovery"] is False
    assert audit["route_candidate_scores_persisted_during_recovery"] is False
    assert audit["route_decisions_persisted_during_recovery"] is False
    assert audit["terminal_evaluation_persisted_during_recovery"] is False
    assert audit["scientific_products_persisted_during_recovery"] is False
    assert audit["terminal_products_persisted_during_recovery"] is False
    assert audit["policy_mutated_during_validation"] is False
    assert audit["excluded_products_persisted_during_recovery"] == [
        "reports/validation_report.json",
        "reports/run_state.json",
    ]

    with pytest.raises(ProtocolError, match="revision did not change"):
        recovery_provenance.finalization_recovery_audit_payload(
            root, current_repository_state=_repo("a" * 40)
        )
    with pytest.raises(ProtocolError, match="repair repository state is invalid"):
        recovery_provenance.finalization_recovery_audit_payload(
            root,
            current_repository_state={
                **_repo("b" * 40),
                "repository_dirty": True,
                "repository_status_hash": "1" * 64,
            },
        )


def test_validation_only_recovery_runs_parent_and_two_fresh_then_writes_only_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    capability = recovery.recovery_capability(root)
    assert capability is not None
    config = SimpleNamespace(contract_hash="config")
    protocol = SimpleNamespace(protocol_hash="protocol")
    events: list[str] = []

    original_validate_index = recovery.validate_content_index

    def validate_index(*args: object, **kwargs: object) -> Mapping[str, object]:
        events.append("content_index")
        return original_validate_index(*args, **kwargs)

    monkeypatch.setattr(recovery, "validate_content_index", validate_index)
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    monkeypatch.setattr(
        runner_runtime,
        "enter_cuda_free_cpu_phase",
        lambda: events.append("cpu_validation"),
    )

    def parent_validate(_root: Path, **kwargs: object) -> Mapping[str, object]:
        assert events[:2] == ["content_index", "cpu_validation"]
        assert kwargs["config"] is config
        assert kwargs["allow_pending_validation"] is True
        audit = kwargs["finalization_recovery_audit"]
        assert isinstance(audit, Mapping)
        assert audit["finalization_recovery_used"] is True
        events.append("parent")
        return {"status": "PASS", "finalization_recovery": dict(audit)}

    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
        parent_validate,
    )

    def two_fresh(
        _root: Path, *, expected_checks: Mapping[str, object]
    ) -> Mapping[str, object]:
        assert events[-1] == "parent"
        events.append("two_fresh")
        return {**dict(expected_checks), "fixture_attestation": "two-process"}

    monkeypatch.setattr(
        fresh_process_validation,
        "require_two_fresh_process_validations",
        two_fresh,
    )

    def persist_report(_root: Path, checks: Mapping[str, object]) -> None:
        assert checks["fixture_attestation"] == "two-process"
        atomic_json(
            _root / "reports/validation_report.json",
            {"schema_version": "fixture", **dict(checks)},
        )
        events.append("validation_report")

    monkeypatch.setattr(persistence, "persist_validation_report", persist_report)
    monkeypatch.setattr(
        recovery,
        "_assert_completed_validation_binding",
        lambda *_args, **_kwargs: events.append("complete_binding"),
    )

    assert recovery.recover_exact_finalization(
        root,
        config=config,
        protocol=protocol,
        capability=capability,
    ) == root
    assert events == [
        "content_index",
        "cpu_validation",
        "parent",
        "two_fresh",
        "validation_report",
        "complete_binding",
    ]
    assert _indexed_bytes(root) == before
    assert read_json(root / "reports/run_state.json") == {
        "schema_version": "fixed_bank_cdca_run_state_v1",
        "status": "COMPLETE",
        "phase": "COMPLETE",
        "terminal_diagnostic_only": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "owned_task_checkpoint_replay_allowed": False,
        "task_checkpoints_are_intra_launch_atomicity_only": True,
    }


def test_failed_recovery_removes_only_attempt_report_and_restores_exact_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    capability = recovery.recovery_capability(root)
    assert capability is not None
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)

    def fail_after_report(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        atomic_json(root / "reports/validation_report.json", {"attempt": True})
        raise ProtocolError("parent replay failed")

    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
        fail_after_report,
    )
    with pytest.raises(ProtocolError, match="parent replay failed"):
        recovery.recover_exact_finalization(
            root,
            config=SimpleNamespace(contract_hash="config"),
            protocol=SimpleNamespace(protocol_hash="protocol"),
            capability=capability,
        )
    assert read_json(root / "reports/run_state.json") == (
        recovery.FAILED_FINALIZATION_STATE
    )
    assert not (root / "reports/validation_report.json").exists()
    assert _indexed_bytes(root) == before


def test_recovery_rejects_any_new_unindexed_member_without_hiding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    capability = recovery.recovery_capability(root)
    assert capability is not None
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)

    def create_extra(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        (root / "reports/unregistered.json").write_text("{}\n", encoding="utf-8")
        raise ProtocolError("validator created an extra")

    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
        create_extra,
    )
    with pytest.raises(ProtocolError, match="failed recovery inventory is not exact"):
        recovery.recover_exact_finalization(
            root,
            config=SimpleNamespace(contract_hash="config"),
            protocol=SimpleNamespace(protocol_hash="protocol"),
            capability=capability,
        )
    assert (root / "reports/unregistered.json").is_file()
    assert read_json(root / "reports/run_state.json") == (
        recovery.FAILED_FINALIZATION_STATE
    )


@pytest.mark.parametrize(
    "repair_state",
    (
        {
            "repository_revision": "a" * 40,
            "repository_dirty": False,
            "repository_status_hash": _EMPTY_STATUS_HASH,
        },
        {
            "repository_revision": "b" * 40,
            "repository_dirty": True,
            "repository_status_hash": "1" * 64,
        },
    ),
)
def test_unrepaired_or_dirty_checkout_aborts_before_parent_or_fresh_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repair_state: Mapping[str, object],
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    capability = recovery.recovery_capability(root)
    assert capability is not None
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: repair_state,
    )
    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
        lambda *_args, **_kwargs: pytest.fail("parent validator must not run"),
    )
    monkeypatch.setattr(
        fresh_process_validation,
        "require_two_fresh_process_validations",
        lambda *_args, **_kwargs: pytest.fail("fresh validators must not run"),
    )

    with pytest.raises(
        ProtocolError,
        match="revision did not change|repair repository state is invalid",
    ):
        recovery.recover_exact_finalization(
            root,
            config=SimpleNamespace(contract_hash="config"),
            protocol=SimpleNamespace(protocol_hash="protocol"),
            capability=capability,
        )
    assert read_json(root / "reports/run_state.json") == (
        recovery.FAILED_FINALIZATION_STATE
    )
    assert not (root / "reports/validation_report.json").exists()


def test_report_present_retry_reuses_only_exactly_attested_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    audit = recovery_provenance.finalization_recovery_audit_payload(
        root, current_repository_state=_repo("b" * 40)
    )
    parent_checks = {"status": "PASS", "finalization_recovery": dict(audit)}
    attested = _attested_checks(root, parent_checks, monkeypatch)
    persistence.persist_validation_report(root, attested)
    report_before = (root / "reports/validation_report.json").read_bytes()
    indexed_before = _indexed_bytes(root)
    capability = recovery.recovery_capability(root)
    assert capability is not None and capability.validation_report_present

    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)
    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
        lambda *_args, **_kwargs: parent_checks,
    )
    monkeypatch.setattr(
        fresh_process_validation,
        "require_two_fresh_process_validations",
        lambda *_args, **_kwargs: pytest.fail(
            "an already attested atomic report must be verified, not replaced"
        ),
    )
    monkeypatch.setattr(
        persistence,
        "persist_validation_report",
        lambda *_args, **_kwargs: pytest.fail(
            "a pre-existing verified report must not be rewritten"
        ),
    )

    assert recovery.recover_exact_finalization(
        root,
        config=SimpleNamespace(contract_hash="config"),
        protocol=SimpleNamespace(protocol_hash="protocol"),
        capability=capability,
    ) == root
    assert (root / "reports/validation_report.json").read_bytes() == report_before
    assert _indexed_bytes(root) == indexed_before
    assert read_json(root / "reports/run_state.json")["status"] == "COMPLETE"


def test_invalid_report_present_retry_is_preserved_and_remains_exactly_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    atomic_json(root / "reports/validation_report.json", {"invalid": True})
    report_before = (root / "reports/validation_report.json").read_bytes()
    indexed_before = _indexed_bytes(root)
    capability = recovery.recovery_capability(root)
    assert capability is not None and capability.validation_report_present
    audit = recovery_provenance.finalization_recovery_audit_payload(
        root, current_repository_state=_repo("b" * 40)
    )
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)
    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "finalization_recovery": dict(audit),
        },
    )

    with pytest.raises(ProtocolError, match="report header drifted"):
        recovery.recover_exact_finalization(
            root,
            config=SimpleNamespace(contract_hash="config"),
            protocol=SimpleNamespace(protocol_hash="protocol"),
            capability=capability,
        )
    assert (root / "reports/validation_report.json").read_bytes() == report_before
    assert _indexed_bytes(root) == indexed_before
    assert read_json(root / "reports/run_state.json") == (
        recovery.FAILED_FINALIZATION_STATE
    )


def test_runner_dispatches_recovery_before_any_science_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    recovered: list[Path] = []
    cleaned: list[object] = []
    monkeypatch.setattr(runner, "assert_launch_files", lambda *_args: None)
    monkeypatch.setattr(
        runner, "assert_workspace_resolved_paths", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "recover_if_possible",
        lambda path, **_kwargs: recovered.append(path) or path,
    )
    monkeypatch.setattr(
        runner,
        "cleanup_validated_scratch",
        lambda config: cleaned.append(config),
    )
    deps = CaseDirectionalRunnerDependencies(
        phase_observer=lambda _phase: pytest.fail("entered a science phase"),
        materialize_source=lambda *_args, **_kwargs: pytest.fail(
            "generated source streams"
        ),
        materialize_predictions=lambda *_args, **_kwargs: pytest.fail(
            "generated predictions"
        ),
    )
    config = SimpleNamespace(artifact_root=root)
    assert runner.run_fixed_bank_case_directional_correctness_abstention_router(
        config, artifact_root=root, dependencies=deps
    ) == root
    assert recovered == [root]
    assert cleaned == [config]


def _indexed_fixture(root: Path) -> Path:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed:{member}\n".encode("utf-8"))
    atomic_json(
        root / "provenance/input_artifacts.json",
        {
            "repository_revision": "a" * 40,
            "repository_dirty": False,
            "repository_status_hash": _EMPTY_STATUS_HASH,
        },
    )
    write_content_index(
        root,
        config_contract_hash="config",
        protocol_contract_hash="protocol",
    )
    atomic_json(root / "reports/run_state.json", recovery.FAILED_FINALIZATION_STATE)
    return root


def _indexed_bytes(root: Path) -> dict[str, tuple[int, str]]:
    return {
        member: (
            (root / member).stat().st_size,
            hashlib.sha256((root / member).read_bytes()).hexdigest(),
        )
        for member in (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json")
    }


def _repo(revision: str) -> dict[str, object]:
    return {
        "repository_revision": revision,
        "repository_dirty": False,
        "repository_status_hash": _EMPTY_STATUS_HASH,
    }


def _attested_checks(
    root: Path,
    expected: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, object]:
    parent_pid = os.getpid()
    payloads = iter(
        (
            {"process_id": parent_pid + 10_001, "checks": dict(expected)},
            {"process_id": parent_pid + 10_002, "checks": dict(expected)},
        )
    )

    def worker(_path: Path) -> subprocess.CompletedProcess[str]:
        payload = next(payloads)
        return subprocess.CompletedProcess(
            args=("fixture",),
            returncode=0,
            stdout=canonical_json(payload).decode("utf-8") + "\n",
            stderr="",
        )

    monkeypatch.setattr(fresh_process_validation, "_run_worker", worker)
    return fresh_process_validation.require_two_fresh_process_validations(
        root, expected_checks=expected
    )
