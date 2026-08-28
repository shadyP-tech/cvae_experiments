from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4 import identity
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution import (
    admission,
    inputs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.authorization_lease import (
    AuthorizationLease,
    assert_authorization_unclaimed,
    claim_authorization_lease,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.run_state import (
    write_run_state,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution import (
    scratch,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.scratch import (
    ScratchLease,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.experiment_contracts import (
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _input_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=identity.EXPERIMENT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        source_inner_root=tmp_path / "protected-source-inner",
        source_inner_amendment_path=tmp_path / "source-amendment.json",
        test_cache_root=tmp_path / "protected-test-cache",
        test_manifest_path=tmp_path / "protected-manifest.csv",
        test_consumption_ledger_path=tmp_path / "parent-ledger.json",
        execution_amendment_path=tmp_path / "execution-amendment.json",
        expected_source_inner_amendment_sha256=(
            EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
        ),
    )


def test_source_inner_amendment_is_authenticated_before_surface_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _input_config(tmp_path)
    opened: list[str] = []

    monkeypatch.setattr(
        inputs,
        "_safe_file",
        lambda path, role: opened.append(role) or Path(path),
    )
    monkeypatch.setattr(
        inputs,
        "file_sha256",
        lambda _path: EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    )
    monkeypatch.setattr(inputs, "_read_json", lambda _path, _role: {})
    monkeypatch.setattr(
        inputs,
        "_safe_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("protected source-inner bytes opened before amendment")
        ),
    )

    with pytest.raises(ProtocolError, match="amendment section absent"):
        inputs.load_source_inner_inputs(config)
    assert opened == ["source-inner amendment"]


def test_execution_amendment_precedes_every_protected_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _input_config(tmp_path)
    monkeypatch.setattr(
        inputs,
        "load_ledger_chain",
        lambda _config: (_ for _ in ()).throw(ProtocolError("authority rejected")),
    )
    monkeypatch.setattr(
        inputs,
        "_safe_directory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("protected input opened before execution amendment")
        ),
    )
    with pytest.raises(ProtocolError, match="authority rejected"):
        inputs.load_validated_inputs(config)


def test_provenance_replay_follows_authorized_input_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeConfig:
        experiment_id = identity.EXPERIMENT_ID
        execution_authorized = True
        protocol = {"experiment_id": identity.EXPERIMENT_ID}
        source_provenance = {
            "source_snapshot_schema": "x",
            "source_snapshot_manifest_sha256": "a" * 64,
            "source_snapshot_tree_sha256": "b" * 64,
            "source_snapshot_member_count": 1,
            "source_snapshot_member_pattern": "x",
            "source_snapshot_excludes_bytecode_and_cache": True,
        }
        runtime = {}

    monkeypatch.setattr(admission, "SceptreV4Config", FakeConfig)
    monkeypatch.setattr(admission, "validate_protocol_payload", lambda _value: None)
    monkeypatch.setattr(admission, "validate_source_snapshot", lambda _value: None)
    monkeypatch.setattr(admission, "_assert_pristine_output", lambda _root: None)
    monkeypatch.setattr(admission, "assert_authorization_unclaimed", lambda: None)
    monkeypatch.setattr(
        admission,
        "select_scratch",
        lambda _root, _runtime: ScratchLease(tmp_path / "scratch", "artifact_parent"),
    )
    monkeypatch.setattr(admission, "assert_scratch_absent", lambda _scratch: None)

    with pytest.raises(ProtocolError, match="authority rejected"):
        admission.admit_execution(
            FakeConfig(),
            artifact_root=tmp_path,
            workspace_binding_loader=lambda _config: {"status": "PASS"},
            input_loader=lambda _config: (_ for _ in ()).throw(
                ProtocolError("authority rejected")
            ),
            workspace_provenance_loader=lambda *_args: (_ for _ in ()).throw(
                AssertionError("provenance replay ran before authority")
            ),
        )


def test_prepared_empty_prediction_directories_are_not_prior_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "prediction_store/arrays").mkdir(parents=True)
    (tmp_path / "prediction_store/manifests").mkdir(parents=True)
    admission._assert_pristine_output(tmp_path)
    stale = tmp_path / "prediction_store/arrays/stale.npy"
    stale.write_bytes(b"stale")
    with pytest.raises(ProtocolError, match="prior state"):
        admission._assert_pristine_output(tmp_path)


def test_external_authorization_lease_is_irreversible_and_one_shot(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "artifacts/midogpp/90_oracles_and_diagnostics"
    parent.mkdir(parents=True)
    config = SimpleNamespace(
        experiment_id=identity.EXPERIMENT_ID,
        execution_authorized=True,
    )
    assert_authorization_unclaimed(tmp_path)
    lease = claim_authorization_lease(
        config,
        admission_hash="c" * 64,
        repository_root=tmp_path,
    )
    assert lease.status == "CLAIMED_IN_PROGRESS"
    with pytest.raises(ProtocolError, match="already exhausted"):
        claim_authorization_lease(
            config,
            admission_hash="c" * 64,
            repository_root=tmp_path,
        )


def _typed_test_lease(tmp_path: Path) -> AuthorizationLease:
    return AuthorizationLease(
        root=(
            tmp_path
            / ".authorization_lease__midogpp_oracle_uniform_b_v2_consumed_test_"
            "fixed_bank_sceptre_router_v4"
        ),
        lease_hash="d" * 64,
        process_id=os.getpid(),
        status="CLAIMED_IN_PROGRESS",
    )


def test_scratch_creation_rejects_a_post_admission_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = (tmp_path / "artifact").resolve()
    artifact_root.mkdir()
    selected = ScratchLease(
        artifact_root.parent / ".artifact.sceptre-v4-scratch",
        "artifact_parent",
    )
    substituted = ScratchLease(
        artifact_root.parent / ".different.sceptre-v4-scratch",
        "artifact_parent",
    )
    monkeypatch.setattr(scratch, "select_scratch", lambda *_args: selected)

    with pytest.raises(ProtocolError, match="differs from read-only admission"):
        scratch.create_scratch(
            artifact_root,
            {},
            authorization_lease=_typed_test_lease(tmp_path),
            admitted=substituted,
        )
    assert not selected.root.exists()
    assert not substituted.root.exists()


def test_run_state_rejects_skipping_a_production_phase(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    lease = _typed_test_lease(tmp_path)
    write_run_state(
        artifact_root,
        authorization_lease=lease,
        config_hash="e" * 64,
        status="RUNNING",
        phase="BEGIN",
    )

    with pytest.raises(ProtocolError, match="phase successor drifted"):
        write_run_state(
            artifact_root,
            authorization_lease=lease,
            config_hash="e" * 64,
            status="RUNNING",
            phase="SOURCE_INNER_DEVELOPMENT_FREEZE",
        )
