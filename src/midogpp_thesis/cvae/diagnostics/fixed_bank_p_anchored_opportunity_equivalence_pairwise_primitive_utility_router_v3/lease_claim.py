"""Single-use authorization acquisition and crash discovery for OE-PPUR v3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .lease_io import (
    fsync_directory,
    pending_publications,
    publish_json_no_overwrite,
    read_json_regular,
)
from .run_admission import SevenInputRunAdmission
from .run_paths import assert_no_symlink_chain, paths_overlap
from .workspace_binding import assert_canonical_output_root


LEASE_DIRECTORY_NAME = ".oe_ppur_v3_single_use_authorization_consumed"
CLAIM_MEMBER = "claim.json"
ACQUISITION_FAILURE_MEMBER = "acquisition_failure.json"
_ACQUISITION_FAILURE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AuthorizationLeaseClaim:
    path: Path
    payload: Mapping[str, object]
    claim_hash: str

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class AuthorizationAcquisitionFailureReceipt:
    lease_path: Path
    marker_kind: str
    evidence_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ACQUISITION_FAILURE_TOKEN:
            raise ProtocolError(
                "OE-PPUR v3 acquisition failure bypassed durable discovery."
            )
        path = Path(self.lease_path)
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_dir()
            or self.marker_kind
            not in {"ACQUISITION_FAILURE", "PENDING_PUBLICATION", "EMPTY_LEASE"}
        ):
            raise ProtocolError("OE-PPUR v3 acquisition failure receipt drifted.")
        object.__setattr__(self, "lease_path", path)
        object.__setattr__(
            self,
            "evidence_hash",
            require_sha256(self.evidence_hash, "acquisition failure evidence hash"),
        )
        object.__setattr__(self, "receipt_hash", canonical_hash(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_authorization_acquisition_failure_receipt_v1",
            "lease_path": self.lease_path.as_posix(),
            "marker_kind": self.marker_kind,
            "evidence_hash": self.evidence_hash,
        }


def canonical_authorization_lease_path(
    artifact_root: Path,
    scratch_root: Path,
) -> Path:
    artifact = assert_canonical_output_root(Path(artifact_root))
    scratch = Path(scratch_root)
    assert_no_symlink_chain(artifact)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    lease = artifact.parent / LEASE_DIRECTORY_NAME
    assert_no_symlink_chain(lease, allow_missing_leaf=True)
    if (
        paths_overlap(lease, artifact)
        or paths_overlap(lease, scratch)
        or lease.parent.is_symlink()
        or not lease.parent.is_dir()
    ):
        raise ProtocolError("OE-PPUR v3 authorization lease topology drifted.")
    return lease


def assert_authorization_unclaimed(
    artifact_root: Path,
    scratch_root: Path,
) -> Path:
    lease = canonical_authorization_lease_path(artifact_root, scratch_root)
    if lease.exists() or lease.is_symlink():
        raise ProtocolError("OE-PPUR v3 single-use authorization is exhausted.")
    return lease


def claim_authorization_lease(
    admission: SevenInputRunAdmission,
    *,
    run_identity_hash: str,
) -> AuthorizationLeaseClaim:
    if type(admission) is not SevenInputRunAdmission:
        raise ProtocolError("OE-PPUR v3 lease requires typed admission.")
    lease = assert_authorization_unclaimed(
        admission.artifact_root,
        admission.scratch_root,
    )
    body = {
        "schema_version": "oe_ppur_v3_single_use_authorization_claim_v1",
        "status": "CONSUMED_EXHAUSTED",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "artifact_root": admission.artifact_root.as_posix(),
        "scratch_root": admission.scratch_root.as_posix(),
        "lease_path": lease.as_posix(),
        "run_identity_hash": require_sha256(run_identity_hash, "run identity hash"),
        "seven_input_admission_hash": admission.receipt_hash,
        "config_contract_hash": admission.config_contract_hash,
        "protocol_hash": admission.protocol_hash,
        "source_seal_hash": admission.source_seal_hash,
        "authorization_amendment_sha256": (
            admission.authorization_amendment_sha256
        ),
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id_at_claim": os.getpid(),
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "authorization_restored": False,
        "cross_run_recovery_allowed": False,
    }
    payload = {**body, "claim_hash": canonical_hash(body)}
    try:
        os.mkdir(lease, 0o700)
    except FileExistsError as exc:
        raise ProtocolError("OE-PPUR v3 single-use authorization is exhausted.") from exc
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 authorization lease claim failed.") from exc
    try:
        fsync_directory(lease.parent)
        publish_json_no_overwrite(
            lease / CLAIM_MEMBER,
            payload,
            role="authorization claim",
        )
        fsync_directory(lease)
        return validate_authorization_lease(
            AuthorizationLeaseClaim(lease, payload, str(payload["claim_hash"]))
        )
    except BaseException as acquisition_error:
        try:
            _persist_acquisition_failure_marker(
                lease,
                expected_claim_hash=str(payload["claim_hash"]),
                error=acquisition_error,
            )
        except BaseException as marker_error:
            marker_error.add_note(
                "Original OE-PPUR v3 acquisition failure: "
                f"{type(acquisition_error).__name__}: {_safe_text(acquisition_error)}"
            )
            raise marker_error from acquisition_error
        raise ProtocolError(
            "OE-PPUR v3 authorization acquisition failed closed after consumption."
        ) from acquisition_error


def validate_authorization_lease(
    value: AuthorizationLeaseClaim,
) -> AuthorizationLeaseClaim:
    if type(value) is not AuthorizationLeaseClaim:
        raise ProtocolError("OE-PPUR v3 authorization lease is untyped.")
    assert_no_symlink_chain(value.path)
    if (
        (value.path / ACQUISITION_FAILURE_MEMBER).exists()
        or (value.path / ACQUISITION_FAILURE_MEMBER).is_symlink()
        or pending_publications(value.path, CLAIM_MEMBER)
        or pending_publications(value.path, ACQUISITION_FAILURE_MEMBER)
    ):
        raise ProtocolError("OE-PPUR v3 authorization acquisition is fail-closed.")
    payload = read_json_regular(value.path / CLAIM_MEMBER, role="authorization claim")
    body = {key: item for key, item in payload.items() if key != "claim_hash"}
    artifact_root = Path(str(payload.get("artifact_root", "")))
    scratch_root = Path(str(payload.get("scratch_root", "")))
    if (
        set(payload)
        != {
            "schema_version",
            "status",
            "experiment_id",
            "output_artifact_id",
            "artifact_root",
            "scratch_root",
            "lease_path",
            "run_identity_hash",
            "seven_input_admission_hash",
            "config_contract_hash",
            "protocol_hash",
            "source_seal_hash",
            "authorization_amendment_sha256",
            "consumed_at_utc",
            "process_id_at_claim",
            "authorization_consumed",
            "authorization_exhausted",
            "authorization_restored",
            "cross_run_recovery_allowed",
            "claim_hash",
        }
        or payload.get("schema_version")
        != "oe_ppur_v3_single_use_authorization_claim_v1"
        or payload.get("status") != "CONSUMED_EXHAUSTED"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("lease_path") != value.path.as_posix()
        or payload.get("artifact_root") != artifact_root.as_posix()
        or payload.get("scratch_root") != scratch_root.as_posix()
        or value.path != canonical_authorization_lease_path(artifact_root, scratch_root)
        or payload.get("authorization_consumed") is not True
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_restored") is not False
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("claim_hash") != canonical_hash(body)
        or payload.get("claim_hash") != value.claim_hash
        or dict(value.payload) != payload
    ):
        raise ProtocolError("OE-PPUR v3 authorization lease drifted.")
    for role in (
        "run_identity_hash",
        "seven_input_admission_hash",
        "config_contract_hash",
        "protocol_hash",
        "source_seal_hash",
        "authorization_amendment_sha256",
        "claim_hash",
    ):
        require_sha256(payload.get(role), role.replace("_", " "))
    return AuthorizationLeaseClaim(value.path, payload, value.claim_hash)


def discover_authorization_acquisition(
    artifact_root: Path,
    scratch_root: Path,
) -> AuthorizationLeaseClaim | AuthorizationAcquisitionFailureReceipt | None:
    lease = canonical_authorization_lease_path(artifact_root, scratch_root)
    if not lease.exists() and not lease.is_symlink():
        return None
    assert_no_symlink_chain(lease)
    if not lease.is_dir():
        raise ProtocolError("OE-PPUR v3 authorization lease path is unsafe.")
    marker = lease / ACQUISITION_FAILURE_MEMBER
    pending = (
        pending_publications(lease, CLAIM_MEMBER)
        + pending_publications(lease, ACQUISITION_FAILURE_MEMBER)
    )
    if pending:
        return _issue_discovery_failure(lease, "PENDING_PUBLICATION", pending)
    if marker.exists() or marker.is_symlink():
        return _read_acquisition_failure_marker(lease)
    claim = lease / CLAIM_MEMBER
    if claim.exists() or claim.is_symlink():
        payload = read_json_regular(claim, role="authorization claim")
        return validate_authorization_lease(
            AuthorizationLeaseClaim(
                lease,
                payload,
                str(payload.get("claim_hash", "")),
            )
        )
    return _issue_discovery_failure(lease, "EMPTY_LEASE", (lease,))


def _persist_acquisition_failure_marker(
    lease: Path,
    *,
    expected_claim_hash: str,
    error: BaseException,
) -> AuthorizationAcquisitionFailureReceipt:
    body = {
        "schema_version": "oe_ppur_v3_authorization_acquisition_failure_v1",
        "status": "FAILED_CLOSED_AUTHORIZATION_EXHAUSTED",
        "lease_path": lease.as_posix(),
        "expected_claim_hash": require_sha256(
            expected_claim_hash,
            "expected claim hash",
        ),
        "error_class": _safe_text(type(error).__name__),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "authorization_restored": False,
        "recovery_allowed": False,
    }
    payload = {**body, "marker_hash": canonical_hash(body)}
    publish_json_no_overwrite(
        lease / ACQUISITION_FAILURE_MEMBER,
        payload,
        role="acquisition failure marker",
    )
    fsync_directory(lease)
    fsync_directory(lease.parent)
    return _read_acquisition_failure_marker(lease)


def _read_acquisition_failure_marker(
    lease: Path,
) -> AuthorizationAcquisitionFailureReceipt:
    payload = read_json_regular(
        lease / ACQUISITION_FAILURE_MEMBER,
        role="acquisition failure marker",
    )
    body = {key: value for key, value in payload.items() if key != "marker_hash"}
    if (
        set(payload)
        != {
            "schema_version",
            "status",
            "lease_path",
            "expected_claim_hash",
            "error_class",
            "recorded_at_utc",
            "authorization_exhausted",
            "authorization_restored",
            "recovery_allowed",
            "marker_hash",
        }
        or payload.get("schema_version")
        != "oe_ppur_v3_authorization_acquisition_failure_v1"
        or payload.get("status") != "FAILED_CLOSED_AUTHORIZATION_EXHAUSTED"
        or payload.get("lease_path") != lease.as_posix()
        or payload.get("marker_hash") != canonical_hash(body)
        or payload.get("authorization_exhausted") is not True
        or payload.get("authorization_restored") is not False
        or payload.get("recovery_allowed") is not False
    ):
        raise ProtocolError("OE-PPUR v3 acquisition failure marker drifted.")
    require_sha256(payload.get("expected_claim_hash"), "expected claim hash")
    return AuthorizationAcquisitionFailureReceipt(
        lease_path=lease,
        marker_kind="ACQUISITION_FAILURE",
        evidence_hash=str(payload["marker_hash"]),
        _factory_token=_ACQUISITION_FAILURE_TOKEN,
    )


def _issue_discovery_failure(
    lease: Path,
    marker_kind: str,
    members: tuple[Path, ...],
) -> AuthorizationAcquisitionFailureReceipt:
    evidence = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_authorization_discovery_failure_v1",
            "lease_path": lease.as_posix(),
            "marker_kind": marker_kind,
            "members": [member.name for member in members],
            "authorization_exhausted": True,
            "recovery_allowed": False,
        }
    )
    return AuthorizationAcquisitionFailureReceipt(
        lease_path=lease,
        marker_kind=marker_kind,
        evidence_hash=evidence,
        _factory_token=_ACQUISITION_FAILURE_TOKEN,
    )


def _safe_text(value: object) -> str:
    return " ".join(str(value).split())[:160]


__all__ = (
    "ACQUISITION_FAILURE_MEMBER",
    "AuthorizationAcquisitionFailureReceipt",
    "AuthorizationLeaseClaim",
    "CLAIM_MEMBER",
    "LEASE_DIRECTORY_NAME",
    "assert_authorization_unclaimed",
    "canonical_authorization_lease_path",
    "claim_authorization_lease",
    "discover_authorization_acquisition",
    "validate_authorization_lease",
)
