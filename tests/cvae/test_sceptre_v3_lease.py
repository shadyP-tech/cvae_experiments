from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.authorization_lease import (
    assert_authorization_unclaimed,
    claim_authorization_lease,
    load_authorization_lease,
    mark_authorization_complete,
    mark_authorization_failed,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.identity import (
    EXPERIMENT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.run_state import (
    write_run_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.scratch import (
    cleanup_scratch,
    create_scratch,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.workstation import (
    workstation_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(parents=True)
    return repository


def _config() -> SimpleNamespace:
    return SimpleNamespace(experiment_id=EXPERIMENT_ID, execution_authorized=True)


def test_atomic_claim_is_irreversible_after_failure(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    lease_path = assert_authorization_unclaimed(repository)
    assert not lease_path.exists()
    lease = claim_authorization_lease(
        _config(), admission_hash="a" * 64, repository_root=repository
    )
    assert lease.root == lease_path
    with pytest.raises(ProtocolError, match="exhausted"):
        claim_authorization_lease(
            _config(), admission_hash="a" * 64, repository_root=repository
        )
    failed = mark_authorization_failed(lease, error=RuntimeError("boom"))
    assert failed.status == "FAILED_EXHAUSTED"
    assert load_authorization_lease(lease.root) == failed
    unrelated_output = tmp_path / "output"
    unrelated_output.mkdir()
    unrelated_output.rmdir()
    with pytest.raises(ProtocolError, match="exhausted"):
        assert_authorization_unclaimed(repository)


def test_partial_lease_directory_is_also_exhausted(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    path = assert_authorization_unclaimed(repository)
    path.mkdir()
    with pytest.raises(ProtocolError, match="exhausted"):
        assert_authorization_unclaimed(repository)


def test_run_state_and_scratch_require_claimed_lease(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    artifact = tmp_path / "artifact"
    (artifact / "reports").mkdir(parents=True)
    lease = claim_authorization_lease(
        _config(), admission_hash="a" * 64, repository_root=repository
    )
    state = write_run_state(
        artifact,
        authorization_lease=lease,
        config_hash="b" * 64,
        status="RUNNING",
        phase="BEGIN",
    )
    assert state["authorization_exhausted"] is False
    scratch = create_scratch(
        artifact,
        workstation_payload(),
        authorization_lease=lease,
    )
    assert scratch.root.is_dir()
    cleanup_scratch(scratch, artifact_root=artifact)
    assert not scratch.root.exists()


def test_run_state_refuses_unclaimed_authority(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    (artifact / "reports").mkdir(parents=True)
    with pytest.raises(ProtocolError, match="lease"):
        write_run_state(
            artifact,
            authorization_lease=SimpleNamespace(status="CLAIMED_IN_PROGRESS"),
            config_hash="b" * 64,
            status="RUNNING",
            phase="BEGIN",
        )


def test_complete_state_is_published_only_after_external_lease_finalizes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    artifact = tmp_path / "artifact"
    (artifact / "reports").mkdir(parents=True)
    claimed = claim_authorization_lease(
        _config(), admission_hash="a" * 64, repository_root=repository
    )
    write_run_state(
        artifact,
        authorization_lease=claimed,
        config_hash="b" * 64,
        status="RUNNING",
        phase="FINALIZING_AUTHORIZATION",
    )
    completed = mark_authorization_complete(claimed)
    state = write_run_state(
        artifact,
        authorization_lease=completed,
        config_hash="b" * 64,
        status="COMPLETE",
        phase="COMPLETE",
    )
    assert completed.status == "COMPLETE_EXHAUSTED"
    assert state["authorization_lease_hash"] == completed.lease_hash
    assert state["authorization_exhausted"] is True


def test_completed_lease_can_record_terminal_state_publication_error(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    artifact = tmp_path / "artifact"
    (artifact / "reports").mkdir(parents=True)
    claimed = claim_authorization_lease(
        _config(), admission_hash="a" * 64, repository_root=repository
    )
    write_run_state(
        artifact,
        authorization_lease=claimed,
        config_hash="b" * 64,
        status="RUNNING",
        phase="FINALIZING_AUTHORIZATION",
    )
    completed = mark_authorization_complete(claimed)
    state = write_run_state(
        artifact,
        authorization_lease=completed,
        config_hash="b" * 64,
        status="FINALIZATION_ERROR",
        phase="FINALIZING_AUTHORIZATION",
        error_class="OSError",
        error="synthetic COMPLETE publication failure",
    )
    assert completed.status == "COMPLETE_EXHAUSTED"
    assert state["status"] == "FINALIZATION_ERROR"
    assert state["authorization_lease_hash"] == completed.lease_hash
    assert state["authorization_exhausted"] is True
