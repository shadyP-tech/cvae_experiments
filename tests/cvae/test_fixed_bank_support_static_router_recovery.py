from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router import (
    recovery,
    recovery_provenance,
    runner,
    runner_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.runner_dependencies import (
    SupportStaticRouterDependencies,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


def test_exact_recovery_detector_rejects_state_inventory_and_symlink_drift(
    tmp_path: Path,
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    assert recovery.detect_registered_support_static_router_recovery(root)

    atomic_json(root / "reports/run_state.json", {**recovery.FAILED_FINALIZATION_STATE, "error": "other"})
    with pytest.raises(ProtocolError, match="not an exact recovery boundary"):
        recovery.recovery_capability(root)

    atomic_json(root / "reports/run_state.json", recovery.FAILED_FINALIZATION_STATE)
    (root / "tables/route_decisions.csv").unlink()
    with pytest.raises(ProtocolError, match="inventory drifted"):
        recovery.recovery_capability(root)

    root = _fixture(tmp_path / "symlink")
    member = root / "tables/route_decisions.csv"
    member.unlink()
    member.symlink_to(root / "tables/five_fold_partitions.csv")
    with pytest.raises(ProtocolError, match="symlink"):
        recovery.recovery_capability(root)


def test_validation_only_recovery_preserves_indexed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    config = SimpleNamespace(contract_hash="config", source_path=root / "config.resolved.yaml")
    protocol = SimpleNamespace(contract_hash="protocol")
    checks = {"status": "PASS", "finalization_recovery": {"used": True}}
    events: list[str] = []

    monkeypatch.setattr(recovery, "validate_content_index", lambda *_args, **_kwargs: events.append("index"))
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    monkeypatch.setattr(
        recovery_provenance,
        "assert_repair_repository_state_unchanged",
        lambda _state: events.append("checkout"),
    )

    from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router import (
        fresh_process_validation,
        persistence,
        validation,
    )

    calls = 0

    def validate(*_args: object, **kwargs: object) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        assert kwargs["allow_pending_validation"] is True
        events.append("parent")
        return checks

    monkeypatch.setattr(validation, "validate_fixed_bank_support_static_router_bundle", validate)
    monkeypatch.setattr(
        fresh_process_validation,
        "run_two_fresh_process_replays",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "validation_result": checks,
        },
    )
    monkeypatch.setattr(
        persistence,
        "persist_fresh_process_report",
        lambda path, value: atomic_json(path / "reports/fresh_process_validation.json", value),
    )
    monkeypatch.setattr(
        persistence,
        "persist_validation_report",
        lambda path, value: atomic_json(path / "reports/validation_report.json", value),
    )
    monkeypatch.setattr(validation, "assert_completed_bundle_binding", lambda *_args, **_kwargs: events.append("complete"))

    capability = recovery.recovery_capability(root)
    assert capability is not None
    assert recovery.recover_exact_finalization(
        root, config=config, protocol=protocol, capability=capability
    ) == root
    assert calls == 2
    assert _indexed_bytes(root) == before
    assert read_json(root / "reports/run_state.json")["status"] == "COMPLETE"
    assert events == ["index", "parent", "parent", "complete", "checkout"]


def test_failed_recovery_rolls_back_reports_and_exact_state_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    before = _indexed_bytes(root)
    config = SimpleNamespace(contract_hash="config", source_path=root / "config.resolved.yaml")
    protocol = SimpleNamespace(contract_hash="protocol")
    monkeypatch.setattr(recovery, "validate_content_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recovery_provenance,
        "current_repair_repository_state",
        lambda: _repo("b" * 40),
    )
    from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router import validation

    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_support_static_router_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ProtocolError("replay failed")),
    )
    capability = recovery.recovery_capability(root)
    assert capability is not None
    for _ in range(2):
        with pytest.raises(ProtocolError, match="replay failed"):
            recovery.recover_exact_finalization(
                root, config=config, protocol=protocol, capability=capability
            )
        assert read_json(root / "reports/run_state.json") == recovery.FAILED_FINALIZATION_STATE
        assert not (root / "reports/fresh_process_validation.json").exists()
        assert not (root / "reports/validation_report.json").exists()
        assert _indexed_bytes(root) == before


def test_recovery_audit_binds_revisions_index_and_rejects_dirty_repair(
    tmp_path: Path,
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    audit = recovery_provenance.finalization_recovery_audit_payload(
        root, current_repository_state=_repo("b" * 40)
    )
    assert audit["original_repository_revision"] == "a" * 40
    assert audit["repair_repository_revision"] == "b" * 40
    assert set(audit["indexed_member_sha256"]) == set(CONTENT_INDEX_MEMBERS)
    assert audit["finalization_recovery_used"] is True

    with pytest.raises(ProtocolError, match="repair repository state is invalid"):
        recovery_provenance.finalization_recovery_audit_payload(
            root,
            current_repository_state={**_repo("b" * 40), "repository_dirty": True},
        )


def test_dirty_repair_checkout_rejects_before_label_replay_or_fresh_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _indexed_fixture(tmp_path / "bundle")
    config = SimpleNamespace(
        contract_hash="config", source_path=root / "config.resolved.yaml"
    )
    protocol = SimpleNamespace(contract_hash="protocol")
    dirty = {**_repo("b" * 40), "repository_dirty": True}
    monkeypatch.setattr(
        recovery_provenance, "current_repair_repository_state", lambda: dirty
    )

    from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router import (
        fresh_process_validation,
        validation,
    )

    monkeypatch.setattr(
        validation,
        "validate_fixed_bank_support_static_router_bundle",
        lambda *_args, **_kwargs: pytest.fail("label replay must not begin"),
    )
    monkeypatch.setattr(
        fresh_process_validation,
        "run_two_fresh_process_replays",
        lambda *_args, **_kwargs: pytest.fail("fresh workers must not spawn"),
    )

    capability = recovery.recovery_capability(root)
    assert capability is not None
    with pytest.raises(ProtocolError, match="repair repository state is invalid"):
        recovery.recover_exact_finalization(
            root, config=config, protocol=protocol, capability=capability
        )
    assert read_json(root / "reports/run_state.json") == recovery.FAILED_FINALIZATION_STATE
    assert not (root / "reports/fresh_process_validation.json").exists()
    assert not (root / "reports/validation_report.json").exists()


def test_runner_dispatches_recovery_before_any_science_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    recovered: list[Path] = []
    monkeypatch.setattr(runner_runtime, "assert_launch_files", lambda *_args: None)
    monkeypatch.setattr(
        runner_runtime, "assert_workspace_resolved_paths", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner_runtime,
        "recover_if_possible",
        lambda path, **_kwargs: recovered.append(path) or path,
    )
    deps = SupportStaticRouterDependencies(
        phase_observer=lambda _phase: pytest.fail("entered a science phase"),
        materialize_source=lambda *_args, **_kwargs: pytest.fail("generated sources"),
        materialize_predictions=lambda *_args, **_kwargs: pytest.fail("fit predictions"),
    )
    config = SimpleNamespace(artifact_root=root)
    assert runner._run(config, artifact_root=root, deps=deps) == root
    assert recovered == [root]


def _fixture(root: Path) -> Path:
    for member in recovery.FINALIZATION_RECOVERABLE_INVENTORY:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed:{member}\n".encode())
    atomic_json(root / "reports/run_state.json", recovery.FAILED_FINALIZATION_STATE)
    return root


def _indexed_fixture(root: Path) -> Path:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed:{member}\n".encode())
    atomic_json(
        root / "provenance/input_artifacts.json",
        {
            "repository_revision": "a" * 40,
            "repository_dirty": True,
            "repository_status_hash": "1" * 64,
        },
    )
    write_content_index(
        root,
        config_contract_hash="config",
        protocol_contract_hash="protocol",
    )
    atomic_json(root / "reports/run_state.json", recovery.FAILED_FINALIZATION_STATE)
    return root


def _indexed_bytes(root: Path) -> dict[str, str]:
    return {
        member: hashlib.sha256((root / member).read_bytes()).hexdigest()
        for member in (*CONTENT_INDEX_MEMBERS, "manifests/content_index.json")
    }


def _repo(revision: str) -> dict[str, object]:
    return {
        "repository_revision": revision,
        "repository_dirty": False,
        "repository_status_hash": "2" * 64,
    }
