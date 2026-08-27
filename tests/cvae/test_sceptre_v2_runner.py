from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.authorization_lease import (
    authorization_lease_path,
    claim_authorization_lease,
    load_authorization_lease,
    mark_authorization_failed,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.config import (
    SceptreV2Config,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.execution_admission import (
    DryRunAdmission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.identity import (
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.run_state import (
    RUN_STATE_MEMBER,
    write_run_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.scratch import (
    ScratchLease,
)
from midogpp_thesis.cvae.runtime.artifact_io import read_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v2.yaml"
)
RUNNER_MODULE = (
    "midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.runner"
)
INPUTS_MODULE = (
    "midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v2.inputs"
)


def _canonical_config(tmp_path: Path) -> SceptreV2Config:
    config = load_config(CONFIG)
    artifact_root = (tmp_path / "artifact").resolve()
    artifact_root.mkdir()
    return replace(config, artifact_root=artifact_root)


def _admission(
    config: SceptreV2Config,
    tmp_path: Path,
    *,
    authorization_path: Path | None = None,
) -> DryRunAdmission:
    return DryRunAdmission(
        artifact_root=config.artifact_root,
        scratch=ScratchLease(
            (tmp_path / "sceptre-v2-scratch").resolve(),
            "artifact_parent",
        ),
        authorization_lease_path=(
            authorization_path
            if authorization_path is not None
            else (tmp_path / "authorization-lease").resolve()
        ),
        config_hash=config.config_hash,
        source_tree_sha256=config.expected_source_snapshot_tree_sha256,
        cache_binding_hash="c" * 64,
        admission_hash="a" * 64,
    )


def _tree_inventory(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                "directory" if path.is_dir() else "file",
            )
            for path in root.rglob("*")
        )
    )


def test_dry_run_performs_admission_and_preflight_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    config = _canonical_config(tmp_path)
    admission = _admission(config, tmp_path)
    events: list[str] = []

    def fake_admission(value: object) -> DryRunAdmission:
        assert value is config
        events.append("admission")
        return admission

    def fake_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (config.artifact_root, admission.scratch.root.parent)
        assert kwargs == {"runtime": config.runtime}
        events.append("preflight")
        return {
            "status": "PASS",
            "target_labels_opened": False,
            "publication_status": PUBLICATION_STATUS,
        }

    def fake_provenance(
        root: Path,
        value: object,
    ) -> dict[str, object]:
        assert root == config.artifact_root
        assert value is config
        events.append("workspace_provenance")
        return {
            "status": "PASS",
            "target_labels_opened": False,
        }

    def forbidden_mutation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("SCEPTRE v2 dry-run attempted a mutation")

    monkeypatch.setattr(runner, "dry_run_admission", fake_admission)
    monkeypatch.setattr(runner, "run_workstation_preflight", fake_preflight)
    monkeypatch.setattr(runner, "validate_workspace_provenance", fake_provenance)
    for name in (
        "claim_authorization_lease",
        "create_scratch",
        "write_run_state",
        "atomic_json",
        "mark_authorization_complete",
        "mark_authorization_failed",
    ):
        monkeypatch.setattr(runner, name, forbidden_mutation)

    before = _tree_inventory(tmp_path)
    result = runner.dry_run_sceptre_v2(
        config,
        artifact_root=config.artifact_root,
    )

    assert events == ["admission", "preflight", "workspace_provenance"]
    assert _tree_inventory(tmp_path) == before
    assert result["status"] == "PASS"
    assert result["authorization_lease_claimed"] is False
    assert result["filesystem_mutations"] == 0
    assert result["target_labels_opened"] is False
    assert result["publication_status"] == PUBLICATION_STATUS
    assert result["terminal_decision"] == TERMINAL_DECISION
    assert result["fresh_evidence"] is False


def test_failure_immediately_after_claim_exhausts_lease_and_records_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module(RUNNER_MODULE)
    inputs = importlib.import_module(INPUTS_MODULE)
    config = _canonical_config(tmp_path)
    (config.artifact_root / "reports").mkdir()
    repository = (tmp_path / "repository").resolve()
    (
        repository / "artifacts/midogpp/90_oracles_and_diagnostics"
    ).mkdir(parents=True)
    admission = _admission(
        config,
        tmp_path,
        authorization_path=authorization_lease_path(repository),
    )
    events: list[str] = []

    class FakeValidatedInputs:
        pass

    validated = FakeValidatedInputs()

    def fake_load_inputs(value: object) -> FakeValidatedInputs:
        assert value is config
        events.append("inputs")
        return validated

    def fake_admission(
        value: object,
        *,
        input_loader: object,
    ) -> DryRunAdmission:
        assert value is config
        events.append("admission")
        assert callable(input_loader)
        assert input_loader(value) is validated
        events.append("admission_complete")
        return admission

    def fake_preflight(*args: object, **kwargs: object) -> dict[str, object]:
        assert args == (config.artifact_root, admission.scratch.root.parent)
        assert kwargs == {"runtime": config.runtime}
        events.append("preflight")
        return {"status": "PASS", "target_labels_opened": False}

    def fake_provenance(
        root: Path,
        value: object,
    ) -> dict[str, object]:
        assert root == config.artifact_root
        assert value is config
        events.append("workspace_provenance")
        return {}

    def real_claim(value: object, *, admission_hash: str):
        assert value is config
        assert admission_hash == admission.admission_hash
        assert events == [
            "admission",
            "inputs",
            "admission_complete",
            "preflight",
            "workspace_provenance",
        ]
        events.append("authorization_claim")
        return claim_authorization_lease(
            value,
            admission_hash=admission_hash,
            repository_root=repository,
        )

    def record_state(*args: object, **kwargs: object) -> dict[str, object]:
        events.append(f"state_{kwargs['status']}_{kwargs['phase']}")
        return write_run_state(*args, **kwargs)

    def fail_after_claim(*_args: object, **_kwargs: object) -> object:
        events.append("scratch_failure")
        raise RuntimeError("synthetic failure immediately after claim")

    def real_failure_finalization(*args: object, **kwargs: object):
        events.append("lease_FAILED_EXHAUSTED")
        return mark_authorization_failed(*args, **kwargs)

    monkeypatch.setattr(inputs, "ValidatedInputs", FakeValidatedInputs)
    monkeypatch.setattr(inputs, "load_validated_inputs", fake_load_inputs)
    monkeypatch.setattr(runner, "dry_run_admission", fake_admission)
    monkeypatch.setattr(runner, "run_workstation_preflight", fake_preflight)
    monkeypatch.setattr(runner, "validate_workspace_provenance", fake_provenance)
    monkeypatch.setattr(runner, "claim_authorization_lease", real_claim)
    monkeypatch.setattr(runner, "write_run_state", record_state)
    monkeypatch.setattr(runner, "create_scratch", fail_after_claim)
    monkeypatch.setattr(
        runner,
        "mark_authorization_failed",
        real_failure_finalization,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic failure immediately after claim",
    ):
        runner.run_sceptre_v2(
            config,
            artifact_root=config.artifact_root,
        )

    assert events == [
        "admission",
        "inputs",
        "admission_complete",
        "preflight",
        "workspace_provenance",
        "authorization_claim",
        "state_RUNNING_BEGIN",
        "scratch_failure",
        "lease_FAILED_EXHAUSTED",
        "state_FAILED_BEGIN",
    ]
    lease = load_authorization_lease(admission.authorization_lease_path)
    state = read_json(config.artifact_root / RUN_STATE_MEMBER)
    assert lease.status == "FAILED_EXHAUSTED"
    assert state["status"] == "FAILED"
    assert state["phase"] == "BEGIN"
    assert state["authorization_lease_hash"] == lease.lease_hash
    assert state["authorization_exhausted"] is True
    assert state["error_class"] == "RuntimeError"
    assert state["fresh_evidence"] is False
    assert state["terminal_decision"] == TERMINAL_DECISION
