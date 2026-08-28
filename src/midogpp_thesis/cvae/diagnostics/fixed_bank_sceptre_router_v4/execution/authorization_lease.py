"""External, irreversible single-use authorization lease for SCEPTRE v4."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json
from ..identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    canonical_hash,
    require_sha256,
)


LEASE_MEMBER = "lease.json"
_LEASE_DIRECTORY = (
    ".authorization_lease__midogpp_oracle_uniform_b_v2_consumed_test_"
    "fixed_bank_sceptre_router_v4"
)
_FINAL_STATUSES = frozenset({"COMPLETE_EXHAUSTED", "FAILED_EXHAUSTED"})


@dataclass(frozen=True, slots=True)
class AuthorizationLease:
    root: Path
    lease_hash: str
    process_id: int
    status: str

    def __post_init__(self) -> None:
        if (
            not Path(self.root).is_absolute()
            or Path(self.root).name != _LEASE_DIRECTORY
            or isinstance(self.process_id, bool)
            or self.process_id <= 0
            or self.status not in {"CLAIMED_IN_PROGRESS", *_FINAL_STATUSES}
        ):
            raise ProtocolError("SCEPTRE v4 authorization lease drifted.")
        require_sha256(self.lease_hash, "authorization lease hash")


def authorization_lease_path(repository_root: Path | None = None) -> Path:
    repository = (
        Path(__file__).resolve().parents[6]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    return (
        repository
        / "artifacts/midogpp/90_oracles_and_diagnostics"
        / _LEASE_DIRECTORY
    )


def assert_authorization_unclaimed(repository_root: Path | None = None) -> Path:
    path = authorization_lease_path(repository_root)
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("SCEPTRE v4 authorization-lease parent is unsafe.")
    if path.exists() or path.is_symlink():
        raise ProtocolError(
            "SCEPTRE v4 single-use authorization is already exhausted."
        )
    return path


def claim_authorization_lease(
    config: object,
    *,
    admission_hash: str,
    repository_root: Path | None = None,
) -> AuthorizationLease:
    """Atomically claim authority; this mkdir is the first run mutation."""

    if (
        getattr(config, "experiment_id", None) != EXPERIMENT_ID
        or getattr(config, "execution_authorized", False) is not True
    ):
        raise ProtocolError("SCEPTRE v4 lease lacks executable authority.")
    admission = require_sha256(admission_hash, "dry-run admission hash")
    path = assert_authorization_unclaimed(repository_root)
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ProtocolError(
            "SCEPTRE v4 single-use authorization is already exhausted."
        ) from exc
    except OSError as exc:
        raise ProtocolError("Cannot claim SCEPTRE v4 authorization lease.") from exc

    base = {
        "schema_version": "sceptre_v4_authorization_lease_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "CLAIMED_IN_PROGRESS",
        "process_id": os.getpid(),
        "admission_hash": admission,
        "predecessor_lease_hash": None,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "single_use_execution_identity": True,
        "authorization_exhausted": True,
        "recovery_allowed": False,
        "output_deletion_restores_authority": False,
        "error_class": None,
        "error": None,
    }
    payload = {**base, "lease_hash": canonical_hash(base)}
    # If this write fails, the already-created directory intentionally remains
    # an exhausted partial lease and every future admission rejects it.
    atomic_json(path / LEASE_MEMBER, payload)
    return AuthorizationLease(
        root=path,
        lease_hash=str(payload["lease_hash"]),
        process_id=os.getpid(),
        status="CLAIMED_IN_PROGRESS",
    )


def mark_authorization_complete(lease: AuthorizationLease) -> AuthorizationLease:
    return _finalize(lease, status="COMPLETE_EXHAUSTED")


def mark_authorization_failed(
    lease: AuthorizationLease,
    *,
    error: BaseException | str,
) -> AuthorizationLease:
    return _finalize(
        lease,
        status="FAILED_EXHAUSTED",
        error_class=error.__class__.__name__ if isinstance(error, BaseException) else "Error",
        error=str(error),
    )


def load_authorization_lease(root: Path) -> AuthorizationLease:
    path = Path(root)
    if path.is_symlink() or not path.is_dir() or path.name != _LEASE_DIRECTORY:
        raise ProtocolError("SCEPTRE v4 authorization lease is unsafe.")
    payload = read_json(path / LEASE_MEMBER)
    _validate_payload(payload)
    return AuthorizationLease(
        root=path,
        lease_hash=str(payload["lease_hash"]),
        process_id=int(payload["process_id"]),
        status=str(payload["status"]),
    )


def validate_authorization_lease_payload(
    payload: Mapping[str, object],
    *,
    expected_status: str | None = None,
) -> Mapping[str, object]:
    """Authenticate persisted lease bytes without touching the external lease."""

    if not isinstance(payload, Mapping):
        raise ProtocolError("SCEPTRE v4 authorization lease payload is malformed.")
    _validate_payload(payload)
    if expected_status is not None and payload.get("status") != expected_status:
        raise ProtocolError("SCEPTRE v4 authorization lease status drifted.")
    return MappingProxyType(dict(payload))


def _finalize(
    lease: AuthorizationLease,
    *,
    status: str,
    error_class: str | None = None,
    error: str | None = None,
) -> AuthorizationLease:
    if (
        not isinstance(lease, AuthorizationLease)
        or lease.status != "CLAIMED_IN_PROGRESS"
        or lease.process_id != os.getpid()
        or status not in _FINAL_STATUSES
    ):
        raise ProtocolError("SCEPTRE v4 authorization finalization drifted.")
    current = read_json(lease.root / LEASE_MEMBER)
    _validate_payload(current)
    if (
        current.get("status") != "CLAIMED_IN_PROGRESS"
        or current.get("lease_hash") != lease.lease_hash
        or current.get("process_id") != os.getpid()
    ):
        raise ProtocolError("SCEPTRE v4 authorization lease was replaced.")
    base = {
        key: value
        for key, value in current.items()
        if key not in {"lease_hash", "status", "error_class", "error"}
    }
    base.update(
        {
            "status": status,
            "predecessor_lease_hash": lease.lease_hash,
            "error_class": error_class,
            "error": None if error is None else str(error)[:2000],
        }
    )
    payload = {**base, "lease_hash": canonical_hash(base)}
    atomic_json(lease.root / LEASE_MEMBER, payload)
    return AuthorizationLease(
        root=lease.root,
        lease_hash=str(payload["lease_hash"]),
        process_id=lease.process_id,
        status=status,
    )


def _validate_payload(payload: Mapping[str, object]) -> None:
    base = {key: value for key, value in payload.items() if key != "lease_hash"}
    if (
        payload.get("schema_version") != "sceptre_v4_authorization_lease_v1"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status")
        not in {"CLAIMED_IN_PROGRESS", *_FINAL_STATUSES}
        or not isinstance(payload.get("process_id"), int)
        or payload.get("authorization_basis") != AUTHORIZATION_BASIS
        or payload.get("authorization_scope") != AUTHORIZATION_SCOPE
        or payload.get("single_use_execution_identity") is not True
        or payload.get("authorization_exhausted") is not True
        or payload.get("recovery_allowed") is not False
        or payload.get("output_deletion_restores_authority") is not False
        or payload.get("lease_hash") != canonical_hash(base)
    ):
        raise ProtocolError("SCEPTRE v4 authorization lease failed authentication.")
    predecessor = payload.get("predecessor_lease_hash")
    if (
        payload.get("status") == "CLAIMED_IN_PROGRESS"
        and predecessor is not None
    ) or (
        payload.get("status") in _FINAL_STATUSES
        and require_sha256(predecessor, "lease predecessor") == payload.get("lease_hash")
    ):
        raise ProtocolError("SCEPTRE v4 authorization predecessor drifted.")
    require_sha256(payload.get("admission_hash"), "lease admission hash")


__all__ = (
    "AuthorizationLease",
    "LEASE_MEMBER",
    "assert_authorization_unclaimed",
    "authorization_lease_path",
    "claim_authorization_lease",
    "load_authorization_lease",
    "mark_authorization_complete",
    "mark_authorization_failed",
    "validate_authorization_lease_payload",
)
