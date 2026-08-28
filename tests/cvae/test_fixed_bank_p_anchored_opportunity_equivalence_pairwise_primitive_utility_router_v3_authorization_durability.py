from __future__ import annotations

from pathlib import Path
import stat

import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_claim as claim_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_io as lease_io_module
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.runner as runner_module
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_lease import (
    AuthorizationAcquisitionFailureReceipt,
    AuthorizationLeaseClaim,
    LEASE_DIRECTORY_NAME,
    assert_authorization_unclaimed,
    claim_authorization_lease,
    discover_authorization_acquisition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lease_io import (
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_admission import (
    SevenInputRunAdmission,
    _ADMISSION_TOKEN,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _admission(tmp_path: Path) -> SevenInputRunAdmission:
    root = tmp_path / "artifact"
    root.mkdir()
    return SevenInputRunAdmission(
        config_contract_hash="1" * 64,
        protocol_hash="2" * 64,
        seven_input_contract_hash="3" * 64,
        source_seal_hash="4" * 64,
        source_seal_receipt_hash="5" * 64,
        source_training_surface_receipt_hash="6" * 64,
        source_training_surface_hash="7" * 64,
        input_location_binding_hash="8" * 64,
        workspace_input_manifest_sha256="9" * 64,
        workspace_provenance_receipt_hash="a" * 64,
        authorization_amendment_sha256="b" * 64,
        lifecycle_source_seal_sha256="c" * 64,
        lifecycle_source_seal_receipt_hash="d" * 64,
        artifact_root=root,
        scratch_root=tmp_path / "scratch",
        _factory_token=_ADMISSION_TOKEN,
    )


def _allow_test_output_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        claim_module,
        "assert_canonical_output_root",
        lambda value: Path(value),
    )


def test_atomic_publication_partial_write_leaves_only_pending_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "claim.json"
    original_write = lease_io_module.os.write
    calls = 0

    def partial_then_interrupt(descriptor: int, raw: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, raw[:7])
        raise OSError("injected partial claim write")

    monkeypatch.setattr(lease_io_module.os, "write", partial_then_interrupt)
    with pytest.raises(OSError, match="partial claim write"):
        publish_json_no_overwrite(target, {"status": "claim"}, role="claim")

    assert not target.exists()
    pending = pending_publications(tmp_path, target.name)
    assert len(pending) == 1
    assert pending[0].stat().st_nlink == 1


def test_atomic_publication_pending_file_fsync_failure_leaves_pending_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "claim.json"
    original_fsync = lease_io_module.os.fsync

    def fail_regular_file_fsync(descriptor: int) -> None:
        if stat.S_ISREG(lease_io_module.os.fstat(descriptor).st_mode):
            raise OSError("injected pending-file fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(lease_io_module.os, "fsync", fail_regular_file_fsync)
    with pytest.raises(OSError, match="pending-file fsync failure"):
        publish_json_no_overwrite(target, {"status": "claim"}, role="claim")

    assert not target.exists()
    assert len(pending_publications(tmp_path, target.name)) == 1


def test_atomic_publication_link_failure_never_creates_final_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "completion_commit.json"
    monkeypatch.setattr(
        lease_io_module.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected link failure")
        ),
    )

    with pytest.raises(OSError, match="link failure"):
        publish_json_no_overwrite(target, {"status": "prepared"}, role="commit")

    assert not target.exists()
    assert len(pending_publications(tmp_path, target.name)) == 1


def test_atomic_publication_first_post_link_directory_fsync_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "completion_commit.json"
    original_fsync_directory = lease_io_module.fsync_directory
    calls = 0

    def fail_second_directory_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected post-link directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        lease_io_module,
        "fsync_directory",
        fail_second_directory_fsync,
    )
    with pytest.raises(OSError, match="post-link directory fsync failure"):
        publish_json_no_overwrite(target, {"status": "prepared"}, role="commit")

    assert target.is_file()
    assert target.stat().st_nlink == 2
    assert len(pending_publications(tmp_path, target.name)) == 1


def test_atomic_publication_pending_unlink_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "outcome.json"
    monkeypatch.setattr(
        lease_io_module.os,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected pending unlink failure")
        ),
    )

    with pytest.raises(OSError, match="pending unlink failure"):
        publish_json_no_overwrite(target, {"status": "complete"}, role="outcome")

    assert target.is_file()
    assert target.stat().st_nlink == 2
    assert len(pending_publications(tmp_path, target.name)) == 1


def test_atomic_publication_interruption_after_link_never_validates_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "outcome.json"
    original_link = lease_io_module.os.link

    def link_then_interrupt(
        source,
        destination,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(source, destination, follow_symlinks=follow_symlinks)
        raise OSError("injected interruption after link")

    monkeypatch.setattr(lease_io_module.os, "link", link_then_interrupt)
    with pytest.raises(OSError, match="interruption after link"):
        publish_json_no_overwrite(target, {"status": "complete"}, role="outcome")

    assert target.is_file()
    assert target.stat().st_nlink == 2
    assert len(pending_publications(tmp_path, target.name)) == 1
    with pytest.raises(ProtocolError, match="outcome is unsafe"):
        read_json_regular(target, role="outcome")


def test_lease_mkdir_failure_does_not_claim_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    _allow_test_output_root(monkeypatch)
    original_mkdir = claim_module.os.mkdir

    def fail_lease_mkdir(path, mode=0o777, *, dir_fd=None) -> None:
        if Path(path).name == LEASE_DIRECTORY_NAME:
            raise OSError("injected lease mkdir failure")
        original_mkdir(path, mode, dir_fd=dir_fd)

    with monkeypatch.context() as context:
        context.setattr(claim_module.os, "mkdir", fail_lease_mkdir)
        with pytest.raises(ProtocolError, match="lease claim failed"):
            claim_authorization_lease(admission, run_identity_hash="c" * 64)

    assert (
        discover_authorization_acquisition(
            admission.artifact_root,
            admission.scratch_root,
        )
        is None
    )


def test_parent_fsync_failure_after_mkdir_persists_typed_failure_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    _allow_test_output_root(monkeypatch)
    original_fsync = claim_module.fsync_directory
    calls = 0

    def fail_once(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected lease-parent fsync failure")
        original_fsync(path)

    monkeypatch.setattr(claim_module, "fsync_directory", fail_once)
    with pytest.raises(ProtocolError, match="failed closed after consumption"):
        claim_authorization_lease(admission, run_identity_hash="c" * 64)

    discovered = discover_authorization_acquisition(
        admission.artifact_root,
        admission.scratch_root,
    )
    assert type(discovered) is AuthorizationAcquisitionFailureReceipt
    assert discovered.marker_kind == "ACQUISITION_FAILURE"
    with pytest.raises(ProtocolError, match="authorization is exhausted"):
        assert_authorization_unclaimed(
            admission.artifact_root,
            admission.scratch_root,
        )


def test_partial_claim_is_discoverable_when_claim_call_never_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    _allow_test_output_root(monkeypatch)
    original_write = lease_io_module.os.write
    calls = 0

    def interrupt_claim_once(descriptor: int, raw: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, raw[:11])
        if calls == 2:
            raise OSError("injected interrupted claim")
        return original_write(descriptor, raw)

    monkeypatch.setattr(lease_io_module.os, "write", interrupt_claim_once)
    with pytest.raises(ProtocolError, match="failed closed after consumption"):
        claim_authorization_lease(admission, run_identity_hash="c" * 64)

    discovered = discover_authorization_acquisition(
        admission.artifact_root,
        admission.scratch_root,
    )
    assert type(discovered) is AuthorizationAcquisitionFailureReceipt
    assert discovered.marker_kind == "PENDING_PUBLICATION"
    lease = admission.artifact_root.parent / LEASE_DIRECTORY_NAME
    assert pending_publications(lease, "claim.json")
    assert (lease / "acquisition_failure.json").is_file()


def test_post_unlink_directory_fsync_failure_marks_claim_failed_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    _allow_test_output_root(monkeypatch)
    original_fsync_directory = lease_io_module.fsync_directory
    calls = 0

    def fail_third_directory_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected post-unlink directory fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        lease_io_module,
        "fsync_directory",
        fail_third_directory_fsync,
    )
    with pytest.raises(ProtocolError, match="failed closed after consumption"):
        claim_authorization_lease(admission, run_identity_hash="c" * 64)

    lease = admission.artifact_root.parent / LEASE_DIRECTORY_NAME
    assert (lease / "claim.json").is_file()
    assert (lease / "claim.json").stat().st_nlink == 1
    assert not pending_publications(lease, "claim.json")
    discovered = discover_authorization_acquisition(
        admission.artifact_root,
        admission.scratch_root,
    )
    assert type(discovered) is AuthorizationAcquisitionFailureReceipt
    assert discovered.marker_kind == "ACQUISITION_FAILURE"


def test_final_claim_readback_failure_marks_claim_failed_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = _admission(tmp_path)
    _allow_test_output_root(monkeypatch)
    original_read = lease_io_module.read_json_regular

    def fail_final_claim_readback(path: Path, *, role: str):
        if role == "authorization claim":
            raise OSError("injected final claim readback failure")
        return original_read(path, role=role)

    monkeypatch.setattr(
        lease_io_module,
        "read_json_regular",
        fail_final_claim_readback,
    )
    with pytest.raises(ProtocolError, match="failed closed after consumption"):
        claim_authorization_lease(admission, run_identity_hash="c" * 64)

    discovered = discover_authorization_acquisition(
        admission.artifact_root,
        admission.scratch_root,
    )
    assert type(discovered) is AuthorizationAcquisitionFailureReceipt
    assert discovered.marker_kind == "ACQUISITION_FAILURE"


def test_runner_finalizes_assigned_and_lost_return_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = AuthorizationLeaseClaim(tmp_path / "lease", {}, "d" * 64)
    original_error = RuntimeError("post-claim failure")
    outcome = object()
    finalized: list[AuthorizationLeaseClaim] = []

    def finalize(value, *, artifact_root: Path, original_error: BaseException):
        finalized.append(value)
        assert artifact_root == tmp_path / "artifact"
        assert str(original_error) == "post-claim failure"
        return outcome

    monkeypatch.setattr(runner_module, "finalize_failed_authorization", finalize)
    monkeypatch.setattr(
        runner_module,
        "discover_authorization_acquisition",
        lambda *_args: claim,
    )

    assert runner_module._finalize_runner_failure(
        lease=claim,
        root=tmp_path / "artifact",
        scratch_root=tmp_path / "scratch",
        original_error=original_error,
    ) is outcome
    assert runner_module._finalize_runner_failure(
        lease=None,
        root=tmp_path / "artifact",
        scratch_root=tmp_path / "scratch",
        original_error=original_error,
    ) is outcome
    assert finalized == [claim, claim]


def test_runner_accepts_only_typed_durable_acquisition_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_path = tmp_path / "lease"
    lease_path.mkdir()
    failure = AuthorizationAcquisitionFailureReceipt(
        lease_path=lease_path,
        marker_kind="EMPTY_LEASE",
        evidence_hash="e" * 64,
        _factory_token=claim_module._ACQUISITION_FAILURE_TOKEN,
    )
    monkeypatch.setattr(
        runner_module,
        "discover_authorization_acquisition",
        lambda *_args: failure,
    )
    monkeypatch.setattr(
        runner_module,
        "finalize_failed_authorization",
        lambda *_args, **_kwargs: pytest.fail("marker must not be overwritten"),
    )

    assert runner_module._finalize_runner_failure(
        lease=None,
        root=tmp_path / "artifact",
        scratch_root=tmp_path / "scratch",
        original_error=RuntimeError("claim did not return"),
    ) is failure


def test_authorization_facade_preserves_owner_object_identity() -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_lease as facade
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_outcome as outcome_module
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.completion_transaction as completion_module

    assert facade.AuthorizationLeaseClaim is claim_module.AuthorizationLeaseClaim
    assert facade.claim_authorization_lease is claim_module.claim_authorization_lease
    assert facade.CompletionCommitReceipt is completion_module.CompletionCommitReceipt
    assert facade.record_completion_commit is completion_module.record_completion_commit
    assert facade.AuthorizationOutcomeReceipt is outcome_module.AuthorizationOutcomeReceipt
    assert facade.record_authorization_outcome is outcome_module.record_authorization_outcome
