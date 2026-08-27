from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2 import (
    fresh_process_validation as fresh,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v2.persistence import (
    atomic_json,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _persisted_receipt(
    tmp_path: Path,
    **updates: object,
) -> tuple[Path, str, str]:
    body = {
        "schema_version": "oe_ppur_v2_test_sealed_receipt_v1",
        "decision_count": 218,
        "scientific_refit_performed": False,
        "labels_opened": False,
        "raw_labels_persisted": False,
        **updates,
    }
    receipt_hash = canonical_hash(body)
    path = tmp_path / "preterminal_receipt.json"
    atomic_json(path, {**body, "receipt_hash": receipt_hash})
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, receipt_hash, file_hash


def test_two_actual_spawned_artifact_only_validators_are_attested(
    tmp_path: Path,
) -> None:
    path, receipt_hash, file_hash = _persisted_receipt(tmp_path)

    attestation = fresh.require_two_fresh_artifact_attestations(
        path,
        phase="preterminal",
        expected_sealed_receipt_hash=receipt_hash,
        expected_file_sha256=file_hash,
    )

    assert attestation.parent_process_id == os.getpid()
    assert len(set(attestation.process_ids)) == 2
    assert os.getpid() not in attestation.process_ids
    assert attestation.validator_source_module == fresh.WORKER_MODULE
    assert len(set(attestation.validator_source_identity_sha256s)) == 1
    assert len(set(attestation.validator_result_hashes)) == 2
    assert attestation.to_payload()["multiprocessing_start_method"] == "spawn"
    assert attestation.to_payload()["cuda_visible_devices"] == ""
    assert attestation.to_payload()["scientific_refit_performed"] is False
    assert attestation.to_payload()["labels_opened"] is False
    assert fresh.validate_artifact_fresh_process_attestation(
        attestation,
        expected_phase="preterminal",
        expected_sealed_receipt_hash=receipt_hash,
        expected_file_sha256=file_hash,
    ) is attestation


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("file", "file hash drifted"),
        ("receipt", "receipt hash drifted"),
    ),
)
def test_expected_file_and_sealed_receipt_hashes_fail_closed(
    tmp_path: Path,
    field: str,
    match: str,
) -> None:
    path, receipt_hash, file_hash = _persisted_receipt(tmp_path)
    if field == "file":
        file_hash = "0" * 64
    else:
        receipt_hash = "0" * 64

    with pytest.raises(ProtocolError, match=match):
        fresh.require_two_fresh_artifact_attestations(
            path,
            phase="preterminal",
            expected_sealed_receipt_hash=receipt_hash,
            expected_file_sha256=file_hash,
        )


@pytest.mark.parametrize(
    "claim",
    (
        {"scientific_refit_performed": True},
        {"target_labels_read": True},
        {"nested": {"terminal_labels_opened": 1}},
    ),
)
def test_refit_or_label_access_claim_is_rejected(
    tmp_path: Path,
    claim: dict[str, object],
) -> None:
    path, receipt_hash, file_hash = _persisted_receipt(tmp_path, **claim)

    with pytest.raises(ProtocolError, match="refit or label access"):
        fresh.require_two_fresh_artifact_attestations(
            path,
            phase="preterminal",
            expected_sealed_receipt_hash=receipt_hash,
            expected_file_sha256=file_hash,
        )


def test_symlink_and_noncanonical_receipt_are_rejected(tmp_path: Path) -> None:
    path, receipt_hash, file_hash = _persisted_receipt(tmp_path)
    link = tmp_path / "linked_receipt.json"
    link.symlink_to(path)
    with pytest.raises(ProtocolError, match="symlink"):
        fresh.require_two_fresh_artifact_attestations(
            link,
            phase="preterminal",
            expected_sealed_receipt_hash=receipt_hash,
            expected_file_sha256=file_hash,
        )

    payload = path.read_text(encoding="utf-8")
    path.write_text("  " + payload, encoding="utf-8")
    noncanonical_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ProtocolError, match="not canonical JSON"):
        fresh.require_two_fresh_artifact_attestations(
            path,
            phase="preterminal",
            expected_sealed_receipt_hash=receipt_hash,
            expected_file_sha256=noncanonical_hash,
        )


def test_spawn_worker_nonzero_exit_is_rejected(
    tmp_path: Path,
) -> None:
    path, receipt_hash, file_hash = _persisted_receipt(tmp_path)
    source = fresh._validator_source_observation()
    observed = fresh._read_sealed_receipt(
        path,
        expected_sealed_receipt_hash=receipt_hash,
        expected_file_sha256=file_hash,
    )
    request = {
        "schema_version": "oe_ppur_v2_artifact_validator_request_v1",
        "phase": "preterminal",
        "receipt_path": path.as_posix(),
        "expected_sealed_receipt_hash": receipt_hash,
        "expected_file_sha256": file_hash,
        "expected_file_identity_sha256": observed.file_identity_sha256,
        "expected_validator_source_sha256": source.content_sha256,
        # A valid but incorrect source identity makes the real child fail.
        "expected_validator_source_identity_sha256": "0" * 64,
        "parent_process_id": os.getpid(),
    }

    with pytest.raises(ProtocolError, match="exited nonzero"):
        fresh._launch_spawn_validator(
            request,
            ordinal=1,
            timeout_seconds=30.0,
        )


def test_fake_or_nonfresh_pid_and_source_identity_are_rejected() -> None:
    parent_pid = os.getpid()
    request = {
        "phase": "preterminal",
        "expected_sealed_receipt_hash": "1" * 64,
        "expected_file_sha256": "2" * 64,
        "expected_file_identity_sha256": "3" * 64,
        "expected_validator_source_sha256": "4" * 64,
        "expected_validator_source_identity_sha256": "5" * 64,
    }

    def child(pid: int, *, source_identity: str = "5" * 64):
        payload = {
            "schema_version": "oe_ppur_v2_artifact_validator_result_v1",
            "phase": "preterminal",
            "process_id": pid,
            "parent_process_id": parent_pid,
            "multiprocessing_start_method": "spawn",
            "sealed_receipt_hash": "1" * 64,
            "sealed_file_sha256": "2" * 64,
            "sealed_file_identity_sha256": "3" * 64,
            "validator_source_module": fresh.WORKER_MODULE,
            "validator_source_sha256": "4" * 64,
            "validator_source_identity_sha256": source_identity,
            "environment": dict(fresh._SPAWN_ENVIRONMENT),
            "descriptor_read_only": True,
            "no_follow_used": True,
            "stable_identity_revalidated": True,
            "canonical_json_validated": True,
            "artifact_only_validation": True,
            "scientific_refit_performed": False,
            "labels_opened": False,
            "terminal_capability_opened": False,
        }
        return fresh._ChildObservation(pid, payload, canonical_hash(payload))

    with pytest.raises(ProtocolError, match="not fresh"):
        fresh._validate_children(
            (child(parent_pid), child(parent_pid)),
            request=request,
            parent_process_id=parent_pid,
        )
    with pytest.raises(ProtocolError, match="result drifted"):
        fresh._validate_children(
            (child(100_001), child(100_002, source_identity="6" * 64)),
            request=request,
            parent_process_id=parent_pid,
        )


def test_equal_byte_file_replacement_during_two_process_window_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, receipt_hash, file_hash = _persisted_receipt(tmp_path)
    original = path.read_bytes()
    calls = 0

    def replace_between_validators(
        request: dict[str, object],
        *,
        ordinal: int,
        timeout_seconds: float,
    ) -> fresh._ChildObservation:
        nonlocal calls
        del timeout_seconds
        calls += 1
        if ordinal == 1:
            path.unlink()
            path.write_bytes(original)
        pid = 200_000 + ordinal
        return fresh._ChildObservation(
            pid,
            {"process_id": pid, "ordinal": ordinal, **request},
            hashlib.sha256(f"result-{ordinal}".encode()).hexdigest(),
        )

    monkeypatch.setattr(fresh, "_launch_spawn_validator", replace_between_validators)
    monkeypatch.setattr(fresh, "_validate_children", lambda *args, **kwargs: None)
    with pytest.raises(ProtocolError, match="changed during validation"):
        fresh.require_two_fresh_artifact_attestations(
            path,
            phase="preterminal",
            expected_sealed_receipt_hash=receipt_hash,
            expected_file_sha256=file_hash,
        )
    assert calls == 2


def test_attestation_type_cannot_be_fabricated() -> None:
    with pytest.raises(ProtocolError, match="bypassed spawned validation"):
        fresh.ArtifactFreshProcessAttestationReceipt(
            phase="preterminal",
            sealed_receipt_hash="1" * 64,
            sealed_file_sha256="2" * 64,
            sealed_file_identity_sha256="3" * 64,
            parent_process_id=10,
            process_ids=(11, 12),
            validator_source_module=fresh.WORKER_MODULE,
            validator_source_sha256="4" * 64,
            validator_source_identity_sha256s=("5" * 64, "5" * 64),
            validator_result_hashes=("6" * 64, "7" * 64),
        )
