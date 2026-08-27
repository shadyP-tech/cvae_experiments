"""Irreversible single-use authorization lease for OE-PPUR v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .execution_admission import SixInputAdmissionReceipt
from .hashing import canonical_hash, canonical_json_bytes, require_sha256
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .run_paths import (
    assert_no_symlink_chain,
    paths_overlap,
    validate_absolute_path,
)


LEASE_DIRECTORY_NAME = ".oe_ppur_v2_single_use_authorization_consumed"
LEASE_CLAIM_MEMBER = "claim.json"
LEASE_OUTCOME_MEMBER = "outcome.json"
LEASE_SCHEMA = "oe_ppur_v2_single_use_authorization_claim_v1"
OUTCOME_SCHEMA = "oe_ppur_v2_single_use_authorization_outcome_v1"


@dataclass(frozen=True, slots=True)
class AuthorizationLeaseClaim:
    path: Path
    payload: Mapping[str, object]
    claim_hash: str

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def canonical_authorization_lease_path(
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> Path:
    """Return the sole legal lease path outside both run-owned roots."""

    artifact = validate_absolute_path(artifact_root, role="artifact root")
    scratch = validate_absolute_path(scratch_root, role="scratch root")
    assert_no_symlink_chain(artifact, allow_missing_leaf=True)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    artifact = artifact.resolve(strict=False)
    scratch = scratch.resolve(strict=False)
    lease = artifact.parent / LEASE_DIRECTORY_NAME
    if (
        lease == artifact
        or lease == scratch
        or paths_overlap(lease, artifact)
        or paths_overlap(lease, scratch)
        or lease.is_symlink()
        or not lease.parent.is_dir()
        or lease.parent.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v2 authorization lease topology drifted.")
    return lease


def assert_authorization_unclaimed(
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> Path:
    lease = canonical_authorization_lease_path(artifact_root, scratch_root)
    if lease.exists() or lease.is_symlink():
        raise ProtocolError(
            "OE-PPUR v2 single-use authorization is already exhausted."
        )
    return lease


def claim_authorization_lease(
    admission: SixInputAdmissionReceipt,
    *,
    run_identity_hash: str,
) -> AuthorizationLeaseClaim:
    """Atomically consume authority after the complete read-only admission."""

    if not isinstance(admission, SixInputAdmissionReceipt):
        raise ProtocolError("OE-PPUR v2 lease requires typed six-input admission.")
    payload = admission.to_payload()
    receipt_hash = require_sha256(payload.get("receipt_hash"), "admission hash")
    if receipt_hash != canonical_hash(
        {key: value for key, value in payload.items() if key != "receipt_hash"}
    ):
        raise ProtocolError("OE-PPUR v2 admission receipt hash drifted.")
    lease = assert_authorization_unclaimed(
        admission.artifact_root, admission.scratch_root
    )
    body = {
        "schema_version": LEASE_SCHEMA,
        "status": "CONSUMED_EXHAUSTED",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "artifact_root": admission.artifact_root,
        "scratch_root": admission.scratch_root,
        "lease_path": str(lease),
        "run_identity_hash": require_sha256(
            run_identity_hash, "run identity hash"
        ),
        "six_input_admission_hash": receipt_hash,
        "config_contract_hash": admission.config_contract_hash,
        "protocol_hash": admission.protocol_hash,
        "source_contract_hash": admission.source_contract_hash,
        "authorization_amendment_sha256": (
            admission.authorization_amendment_sha256
        ),
        "input_binding_hash": admission.input_binding_hash,
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id_at_claim": os.getpid(),
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "lease_outside_artifact_and_scratch": True,
        "output_or_scratch_deletion_restores_authorization": False,
        "lease_repair_removal_or_reuse_allowed": False,
        "cross_run_recovery_allowed": False,
    }
    claim = {**body, "claim_hash": canonical_hash(body)}
    try:
        os.mkdir(lease, 0o700)
    except FileExistsError as exc:
        raise ProtocolError(
            "OE-PPUR v2 single-use authorization is already exhausted."
        ) from exc
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 authorization lease claim failed.") from exc

    # mkdir is the irreversible transition.  A subsequent write failure must
    # leave the directory in place and therefore permanently exhausted.
    _fsync_directory(lease.parent)
    _write_exclusive_json(lease / LEASE_CLAIM_MEMBER, claim)
    _fsync_directory(lease)
    _fsync_directory(lease.parent)
    return AuthorizationLeaseClaim(
        path=lease,
        payload=claim,
        claim_hash=str(claim["claim_hash"]),
    )


def validate_authorization_lease(
    value: AuthorizationLeaseClaim | str | Path,
    *,
    expected_claim_hash: str | None = None,
) -> AuthorizationLeaseClaim:
    path = value.path if isinstance(value, AuthorizationLeaseClaim) else Path(value)
    assert_no_symlink_chain(path)
    if path.is_symlink() or not path.is_dir():
        raise ProtocolError("OE-PPUR v2 authorization lease is absent or unsafe.")
    claim_path = path / LEASE_CLAIM_MEMBER
    payload = _read_json_regular(claim_path)
    unhashed = {key: item for key, item in payload.items() if key != "claim_hash"}
    claim_hash = require_sha256(payload.get("claim_hash"), "lease claim hash")
    if (
        payload.get("schema_version") != LEASE_SCHEMA
        or payload.get("status") != "CONSUMED_EXHAUSTED"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("lease_path") != str(path)
        or payload.get("authorization_consumed") is not True
        or payload.get("authorization_exhausted") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or claim_hash != canonical_hash(unhashed)
        or (
            expected_claim_hash is not None
            and claim_hash
            != require_sha256(expected_claim_hash, "expected lease claim hash")
        )
    ):
        raise ProtocolError("OE-PPUR v2 authorization lease drifted.")
    for role in (
        "run_identity_hash",
        "six_input_admission_hash",
        "config_contract_hash",
        "protocol_hash",
        "source_contract_hash",
        "authorization_amendment_sha256",
        "input_binding_hash",
    ):
        require_sha256(payload.get(role), role.replace("_", " "))
    return AuthorizationLeaseClaim(path=path, payload=payload, claim_hash=claim_hash)


def record_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    status: str,
    evidence_hash: str,
    error_class: str | None = None,
) -> dict[str, object]:
    """Write one terminal outcome; it never changes or restores authority."""

    validated = validate_authorization_lease(
        claim, expected_claim_hash=claim.claim_hash
    )
    if status not in {"COMPLETE", "FAILED_EXHAUSTED"}:
        raise ProtocolError("OE-PPUR v2 authorization outcome status drifted.")
    body = {
        "schema_version": OUTCOME_SCHEMA,
        "status": status,
        "claim_hash": validated.claim_hash,
        "evidence_hash": require_sha256(evidence_hash, "outcome evidence hash"),
        "error_class": None if error_class is None else _safe_text(error_class),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "outcome_hash": canonical_hash(body)}
    _write_exclusive_json(validated.path / LEASE_OUTCOME_MEMBER, payload)
    _fsync_directory(validated.path)
    return payload


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    data = canonical_json_bytes(payload) + b"\n"
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 durable lease member write failed.") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_json_regular(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise OSError("unsafe member")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("OE-PPUR v2 lease member is unreadable.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("OE-PPUR v2 lease member is malformed.")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


__all__ = (
    "AuthorizationLeaseClaim",
    "LEASE_DIRECTORY_NAME",
    "assert_authorization_unclaimed",
    "canonical_authorization_lease_path",
    "claim_authorization_lease",
    "record_authorization_outcome",
    "validate_authorization_lease",
)
