from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Mapping

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (  # noqa: E501
    fresh_process_validation,
    runner,
    runner_runtime,
    validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.bundle import (  # noqa: E501
    CONTENT_INDEX_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.recovery import (  # noqa: E501
    MultiChallengerRecoveryCapability,
    failed_finalization_schema_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.runner_dependencies import (  # noqa: E501
    MultiChallengerRouterDependencies,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


CAPABILITY = MultiChallengerRecoveryCapability(
    mode="FINALIZATION_VALIDATION",
    state_phase="FINALIZATION",
    validation_only=True,
    labels_may_be_reopened_for_validation=True,
    scientific_products_may_be_recomputed=False,
    scientific_products_may_be_persisted=False,
)


def test_finalization_recovery_is_validation_only_and_preserves_indexed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_runtime_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    checks = {"status": "PASS"}
    audit = {"finalization_recovery_used": True}
    events: list[object] = []

    monkeypatch.setattr(
        runner_runtime,
        "recovery_capability",
        lambda _root: CAPABILITY,
    )
    monkeypatch.setattr(
        runner_runtime,
        "validate_content_index",
        lambda *_args, **_kwargs: events.append("content_index_validated"),
    )
    monkeypatch.setattr(
        runner_runtime,
        "finalization_recovery_audit_payload_for_root",
        lambda *_args, **_kwargs: audit,
    )
    monkeypatch.setattr(
        runner_runtime,
        "current_repair_repository_state",
        lambda: {"repository_revision": "c" * 40},
    )
    monkeypatch.setattr(
        runner_runtime,
        "enter_cuda_free_cpu_phase",
        lambda: events.append("cpu_validation"),
    )

    def validate(_root: Path, **kwargs: object) -> Mapping[str, object]:
        assert kwargs["allow_pending_validation"] is True
        assert kwargs["finalization_recovery_audit"] == audit
        events.append("two_process_validation")
        return checks

    monkeypatch.setattr(runner_runtime, "validate_bundle", validate)

    def persist(_root: Path, observed: Mapping[str, object]) -> None:
        assert observed == checks
        atomic_json(_root / "reports/validation_report.json", observed)
        events.append("validation_report")

    monkeypatch.setattr(runner_runtime, "persist_validation_report", persist)
    monkeypatch.setattr(
        runner_runtime,
        "assert_completed_binding",
        lambda *_args, **_kwargs: events.append("complete_binding"),
    )
    monkeypatch.setattr(
        runner_runtime,
        "assert_finalization_repair_repository_state_unchanged",
        lambda observed: events.append(("checkout_unchanged", observed)),
    )

    result = runner_runtime.recover_if_possible(
        root,
        config=SimpleNamespace(contract_hash="config"),
        protocol=SimpleNamespace(contract_hash="protocol"),
    )

    assert result == root
    assert _indexed_bytes(root) == before
    assert read_json(root / "reports/run_state.json")["status"] == "COMPLETE"
    assert events == [
        "content_index_validated",
        "cpu_validation",
        "two_process_validation",
        "validation_report",
        "complete_binding",
        ("checkout_unchanged", audit),
    ]


def test_finalization_recovery_failure_restores_exact_state_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_runtime_fixture(tmp_path / "bundle", validation_report=True)
    before = _indexed_bytes(root)
    report_before = (root / "reports/validation_report.json").read_bytes()
    calls = 0

    monkeypatch.setattr(
        runner_runtime,
        "recovery_capability",
        lambda _root: CAPABILITY,
    )
    monkeypatch.setattr(
        runner_runtime,
        "validate_content_index",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner_runtime,
        "finalization_recovery_audit_payload_for_root",
        lambda *_args, **_kwargs: {"finalization_recovery_used": True},
    )
    monkeypatch.setattr(
        runner_runtime,
        "current_repair_repository_state",
        lambda: {"repository_revision": "c" * 40},
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)

    def fail_validation(*_args: object, **_kwargs: object) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        raise ProtocolError("fresh-process replay failed")

    monkeypatch.setattr(runner_runtime, "validate_bundle", fail_validation)

    for _ in range(2):
        with pytest.raises(ProtocolError, match="fresh-process replay failed"):
            runner_runtime.recover_if_possible(
                root,
                config=SimpleNamespace(contract_hash="config"),
                protocol=SimpleNamespace(contract_hash="protocol"),
            )
        assert read_json(root / "reports/run_state.json") == (
            failed_finalization_schema_state(root)
        )
        assert (root / "reports/validation_report.json").read_bytes() == report_before
        assert _indexed_bytes(root) == before

    assert calls == 2


def test_failed_attempt_removes_only_its_new_excluded_validation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_runtime_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    audit = {"finalization_recovery_used": True}

    monkeypatch.setattr(runner_runtime, "recovery_capability", lambda _root: CAPABILITY)
    monkeypatch.setattr(
        runner_runtime, "validate_content_index", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner_runtime,
        "finalization_recovery_audit_payload_for_root",
        lambda *_args, **_kwargs: audit,
    )
    monkeypatch.setattr(
        runner_runtime,
        "current_repair_repository_state",
        lambda: {"repository_revision": "c" * 40},
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)
    monkeypatch.setattr(
        runner_runtime, "validate_bundle", lambda *_args, **_kwargs: {"status": "PASS"}
    )

    def persist_then_fail(_root: Path, _checks: Mapping[str, object]) -> None:
        atomic_json(_root / "reports/validation_report.json", {"attempt": True})
        raise ProtocolError("interrupted after report write")

    monkeypatch.setattr(runner_runtime, "persist_validation_report", persist_then_fail)

    with pytest.raises(ProtocolError, match="interrupted after report write"):
        runner_runtime.recover_if_possible(
            root,
            config=SimpleNamespace(contract_hash="config"),
            protocol=SimpleNamespace(contract_hash="protocol"),
        )

    assert not (root / "reports/validation_report.json").exists()
    assert read_json(root / "reports/run_state.json") == failed_finalization_schema_state(
        root
    )
    assert _indexed_bytes(root) == before


def test_public_finalization_recovery_runs_two_fresh_validators_without_science_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise runner -> exact recovery -> parent + two fresh replay gate."""

    root = _write_runtime_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    audit = {
        "schema_version": "fixture_finalization_recovery_v1",
        "finalization_recovery_used": True,
    }
    base_checks = {
        "schema_version": "fixture_validation_v1",
        "status": "PASS",
        "terminal_finalization_recovery": audit,
    }
    launches: list[tuple[str, ...]] = []
    cleanup_calls: list[object] = []

    monkeypatch.setattr(runner, "assert_launch_files", lambda *_args: None)
    monkeypatch.setattr(
        runner, "assert_workspace_resolved_paths", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner_runtime, "validate_content_index", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner_runtime,
        "finalization_recovery_audit_payload_for_root",
        lambda *_args, **_kwargs: audit,
    )
    monkeypatch.setattr(
        runner_runtime,
        "current_repair_repository_state",
        lambda: {"repository_revision": "c" * 40},
    )
    monkeypatch.setattr(runner_runtime, "enter_cuda_free_cpu_phase", lambda: None)

    def parent_validator(_root: Path, **kwargs: object) -> Mapping[str, object]:
        assert kwargs == {
            "config": config,
            "allow_pending_validation": True,
            "finalization_recovery_audit": audit,
        }
        return base_checks

    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle",
        parent_validator,
    )

    def fresh_validator(
        command: tuple[str, ...], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        launches.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                json.dumps(
                    base_checks,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(fresh_process_validation.subprocess, "run", fresh_validator)
    monkeypatch.setattr(
        runner_runtime,
        "assert_completed_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner_runtime,
        "assert_finalization_repair_repository_state_unchanged",
        lambda observed: observed == audit
        or pytest.fail("unexpected finalization audit"),
    )

    def forbidden_science_write(*_args: object, **_kwargs: object) -> None:
        pytest.fail("validation-only recovery entered a scientific writer")

    for name in (
        "persist_initial_surfaces",
        "persist_prelabel_surfaces",
        "persist_fold_plans",
        "persist_donor_models",
        "persist_fold_decisions",
        "persist_terminal_checkpoint",
        "finalize_terminal_checkpoint",
        "write_content_index",
    ):
        monkeypatch.setattr(runner, name, forbidden_science_write)

    config = SimpleNamespace(contract_hash="config")
    deps = MultiChallengerRouterDependencies(
        cleanup_staging=lambda observed, **_kwargs: cleanup_calls.append(observed),
        phase_observer=lambda _phase: pytest.fail(
            "validation-only recovery entered an experiment phase"
        ),
    )

    assert runner._run(config, artifact_root=root, deps=deps) == root
    assert len(launches) == 2
    assert all(
        launch[1:4]
        == ("-m", fresh_process_validation.WORKER_MODULE, "--worker")
        for launch in launches
    )
    assert _indexed_bytes(root) == before
    assert read_json(root / "reports/run_state.json")["status"] == "COMPLETE"
    report = read_json(root / "reports/validation_report.json")
    assert report["fresh_process_validation_attestation"][
        "fresh_python_process_count"
    ] == 2
    assert cleanup_calls == [config]


def _write_runtime_fixture(root: Path, *, validation_report: bool = False) -> Path:
    for member in (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json"):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed:{member}\n".encode("utf-8"))
    runtime = root / "reports/runtime_summary.json"
    atomic_json(runtime, {"mappingproxy_recovery": {"recovery_used": True}})
    atomic_json(root / "reports/run_state.json", failed_finalization_schema_state(root))
    if validation_report:
        atomic_json(root / "reports/validation_report.json", {"partial": True})
    return root


def _indexed_bytes(root: Path) -> dict[str, tuple[int, str]]:
    return {
        member: (
            (root / member).stat().st_size,
            hashlib.sha256((root / member).read_bytes()).hexdigest(),
        )
        for member in (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json")
    }
