"""Durable one-shot authorization consumption outside run-owned roots.

The authorization amendment is immutable issuance authority.  A successful
atomic directory creation consumes that authority permanently, even if the
process crashes before it can create the artifact or scratch roots.  Neither
output deletion nor scratch deletion can therefore make the same identity
admissible again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path

from .hashing import canonical_hash, canonical_json, require_sha256
from .identity import EXPERIMENT_ID, GovernanceError, OUTPUT_ARTIFACT_ID


LEASE_DIRECTORY_NAME = ".scale_bp_v2_single_use_authorization_consumed"
LEASE_CLAIM_MEMBER = "claim.json"
LEASE_OUTCOME_MEMBER = "outcome.json"
LEASE_SCHEMA = "scale_bp_v2_durable_authorization_claim_v1"
LEASE_OUTCOME_SCHEMA = "scale_bp_v2_durable_authorization_outcome_v1"
PERSISTED_LEASE_MEMBER = "provenance/authorization_consumption_lease.json"


@dataclass(frozen=True, slots=True)
class AuthorizationLeaseClaim:
    """Validated immutable claim issued by the exclusive lease transition."""

    path: Path
    payload: Mapping[str, object]
    claim_hash: str

    def to_payload(self) -> dict[str, object]:
        return dict(self.payload)


def canonical_authorization_lease_path(
    artifact_root: str | Path,
    scratch_root: str | Path,
    *,
    requested_path: str | Path | None = None,
    forbidden_paths: Sequence[str | Path] = (),
) -> Path:
    """Return the sole legal external lease location and reject aliases."""

    artifact = _absolute_nonsymlink_path(artifact_root, "artifact root")
    scratch = _absolute_nonsymlink_path(scratch_root, "scratch root")
    expected = artifact.parent / LEASE_DIRECTORY_NAME
    requested = expected if requested_path is None else Path(requested_path)
    if (
        not requested.is_absolute()
        or requested != expected
        or requested == artifact
        or requested == scratch
        or _is_within(requested, artifact)
        or _is_within(requested, scratch)
        or _is_within(artifact, requested)
        or _is_within(scratch, requested)
        or requested.is_symlink()
    ):
        raise GovernanceError("SCALE-BP v2 authorization lease path drifted.")
    for value in forbidden_paths:
        forbidden = _absolute_nonsymlink_path(value, "forbidden lease boundary")
        if (
            requested == forbidden
            or _is_within(requested, forbidden)
            or _is_within(forbidden, requested)
        ):
            raise GovernanceError(
                "SCALE-BP v2 authorization lease overlaps a protected path."
            )
    parent = requested.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise GovernanceError(
            "SCALE-BP v2 authorization lease parent must already be a safe directory."
        )
    return requested


def assert_authorization_unclaimed(
    artifact_root: str | Path,
    scratch_root: str | Path,
    *,
    requested_path: str | Path | None = None,
    forbidden_paths: Sequence[str | Path] = (),
) -> Path:
    """Read-only absence check; every existing leaf is permanently exhausted."""

    path = canonical_authorization_lease_path(
        artifact_root,
        scratch_root,
        requested_path=requested_path,
        forbidden_paths=forbidden_paths,
    )
    if path.exists() or path.is_symlink():
        raise GovernanceError(
            "SCALE-BP v2 single-use authorization is already exhausted."
        )
    return path


def claim_authorization_lease(
    admission_receipt: Mapping[str, object] | object,
    *,
    protocol_hash: str,
    claim_boundary_hash: str,
    authorization_amendment_sha256: str,
    run_identity_hash: str,
) -> AuthorizationLeaseClaim:
    """Atomically consume the one-shot authority after all read-only checks."""

    admission = _mapping_payload(admission_receipt, "admission receipt")
    admission_hash = require_sha256(
        admission.get("receipt_hash"), "admission receipt hash"
    )
    if admission_hash != canonical_hash(
        {key: value for key, value in admission.items() if key != "receipt_hash"}
    ):
        raise GovernanceError("SCALE-BP v2 admission receipt hash drifted.")
    artifact = Path(str(admission.get("artifact_root")))
    scratch = Path(str(admission.get("scratch_root")))
    lease = assert_authorization_unclaimed(
        artifact,
        scratch,
        requested_path=str(admission.get("authorization_lease_path")),
    )
    run_hash = require_sha256(run_identity_hash, "run identity hash")
    body = {
        "schema_version": LEASE_SCHEMA,
        "status": "CONSUMED_EXHAUSTED",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "authorization_lease_path": str(lease),
        "artifact_root": str(artifact),
        "scratch_root": str(scratch),
        "config_contract_hash": require_sha256(
            admission.get("config_contract_hash"), "config hash"
        ),
        "protocol_hash": require_sha256(protocol_hash, "protocol hash"),
        "claim_boundary_hash": require_sha256(
            claim_boundary_hash, "claim-boundary hash"
        ),
        "authorization_amendment_sha256": require_sha256(
            authorization_amendment_sha256, "authorization amendment hash"
        ),
        "source_snapshot_manifest_sha256": require_sha256(
            admission.get("source_snapshot_manifest_sha256"),
            "source snapshot manifest hash",
        ),
        "source_snapshot_tree_sha256": require_sha256(
            admission.get("source_snapshot_tree_sha256"),
            "source snapshot tree hash",
        ),
        "source_snapshot_member_count": admission.get(
            "source_snapshot_member_count"
        ),
        "direct_input_binding_hash": require_sha256(
            admission.get("direct_input_binding_hash"), "direct-input binding hash"
        ),
        "admission_receipt_hash": admission_hash,
        "run_identity_hash": run_hash,
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id_at_claim": os.getpid(),
        "authorization_consumed": True,
        "authorization_exhausted": True,
        "lease_outside_artifact_and_scratch": True,
        "output_or_scratch_deletion_restores_authorization": False,
        "lease_repair_removal_or_reuse_allowed": False,
        "cross_run_recovery_allowed": False,
    }
    count = body["source_snapshot_member_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise GovernanceError("SCALE-BP v2 lease source member count drifted.")
    payload = {**body, "claim_hash": canonical_hash(body)}

    # Directory creation is the irreversible state transition.  If anything
    # after this point fails, the empty or partial leaf still means exhausted.
    try:
        os.mkdir(lease, 0o700)
    except FileExistsError as exc:
        raise GovernanceError(
            "SCALE-BP v2 single-use authorization is already exhausted."
        ) from exc
    except OSError as exc:
        raise GovernanceError(
            "SCALE-BP v2 authorization lease could not be claimed."
        ) from exc
    # Make the irreversible directory entry durable before attempting any
    # member write.  A host crash or claim-write failure must never make this
    # authorization appear unconsumed after restart.
    _fsync_directory(lease.parent)
    try:
        _write_exclusive_json(lease / LEASE_CLAIM_MEMBER, payload)
        _fsync_directory(lease)
        _fsync_directory(lease.parent)
    except BaseException:
        # Never remove or repair the lease: the atomic mkdir already consumed it.
        raise
    return AuthorizationLeaseClaim(
        path=lease,
        payload=payload,
        claim_hash=str(payload["claim_hash"]),
    )


def validate_authorization_lease(
    value: AuthorizationLeaseClaim | str | Path,
    *,
    expected_claim_hash: str | None = None,
) -> AuthorizationLeaseClaim:
    """Validate immutable claim bytes without ever treating corruption as absence."""

    path = value.path if isinstance(value, AuthorizationLeaseClaim) else Path(value)
    if path.is_symlink() or not path.is_dir():
        raise GovernanceError("SCALE-BP v2 authorization lease is absent or unsafe.")
    claim_path = path / LEASE_CLAIM_MEMBER
    if claim_path.is_symlink() or not claim_path.is_file():
        raise GovernanceError("SCALE-BP v2 authorization lease claim is incomplete.")
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError("SCALE-BP v2 authorization lease is unreadable.") from exc
    if not isinstance(payload, dict):
        raise GovernanceError("SCALE-BP v2 authorization lease is malformed.")
    claim_hash = _validate_claim_payload(
        payload,
        expected_path=str(path),
        expected_claim_hash=expected_claim_hash,
    )
    return AuthorizationLeaseClaim(path=path, payload=payload, claim_hash=str(claim_hash))


def validate_persisted_authorization_lease(
    artifact_root: str | Path,
    *,
    expected_claim_hash: str | None = None,
) -> dict[str, object]:
    """Validate the artifact-local immutable copy without external dependence."""

    root = Path(artifact_root)
    member = root / PERSISTED_LEASE_MEMBER
    if root.is_symlink() or member.is_symlink() or not member.is_file():
        raise GovernanceError(
            "SCALE-BP v2 persisted authorization lease is absent or unsafe."
        )
    try:
        payload = json.loads(member.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(
            "SCALE-BP v2 persisted authorization lease is unreadable."
        ) from exc
    if not isinstance(payload, dict):
        raise GovernanceError("SCALE-BP v2 persisted authorization lease is malformed.")
    artifact = Path(str(payload.get("artifact_root")))
    scratch = Path(str(payload.get("scratch_root")))
    expected_path = artifact.parent / LEASE_DIRECTORY_NAME
    if (
        not artifact.is_absolute()
        or not scratch.is_absolute()
        or payload.get("authorization_lease_path") != str(expected_path)
        or expected_path == artifact
        or expected_path == scratch
        or _is_within(expected_path, artifact)
        or _is_within(expected_path, scratch)
    ):
        raise GovernanceError("SCALE-BP v2 persisted lease topology drifted.")
    _validate_claim_payload(
        payload,
        expected_path=str(expected_path),
        expected_claim_hash=expected_claim_hash,
    )
    return payload


def record_authorization_outcome(
    claim: AuthorizationLeaseClaim,
    *,
    status: str,
    evidence_hash: str,
    error_class: str | None = None,
) -> dict[str, object]:
    """Persist one external terminal outcome without changing claim authority."""

    validated = validate_authorization_lease(
        claim, expected_claim_hash=claim.claim_hash
    )
    outcome_status = str(status)
    if outcome_status not in {"COMPLETE", "FAILED_EXHAUSTED"}:
        raise GovernanceError("SCALE-BP v2 authorization outcome status drifted.")
    body = {
        "schema_version": LEASE_OUTCOME_SCHEMA,
        "status": outcome_status,
        "claim_hash": validated.claim_hash,
        "run_identity_hash": validated.payload["run_identity_hash"],
        "evidence_hash": require_sha256(evidence_hash, "lease outcome evidence hash"),
        "error_class": None if error_class is None else _safe_text(error_class),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_exhausted": True,
        "lease_reusable": False,
    }
    payload = {**body, "outcome_hash": canonical_hash(body)}
    _write_exclusive_json(validated.path / LEASE_OUTCOME_MEMBER, payload)
    _fsync_directory(validated.path)
    _fsync_directory(validated.path.parent)
    return payload


def _mapping_payload(value: Mapping[str, object] | object, role: str) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    to_payload = getattr(value, "to_payload", None)
    payload = to_payload() if callable(to_payload) else None
    if not isinstance(payload, Mapping):
        raise GovernanceError(f"SCALE-BP v2 {role} is malformed.")
    return dict(payload)


def _validate_claim_payload(
    payload: Mapping[str, object],
    *,
    expected_path: str,
    expected_claim_hash: str | None,
) -> str:
    claim_hash = payload.get("claim_hash")
    body = {key: item for key, item in payload.items() if key != "claim_hash"}
    if (
        payload.get("schema_version") != LEASE_SCHEMA
        or payload.get("status") != "CONSUMED_EXHAUSTED"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("authorization_lease_path") != expected_path
        or payload.get("authorization_consumed") is not True
        or payload.get("authorization_exhausted") is not True
        or payload.get("lease_outside_artifact_and_scratch") is not True
        or payload.get("output_or_scratch_deletion_restores_authorization") is not False
        or payload.get("lease_repair_removal_or_reuse_allowed") is not False
        or payload.get("cross_run_recovery_allowed") is not False
        or claim_hash != canonical_hash(body)
        or (
            expected_claim_hash is not None
            and claim_hash
            != require_sha256(expected_claim_hash, "expected lease claim hash")
        )
    ):
        raise GovernanceError("SCALE-BP v2 authorization lease drifted.")
    for role in (
        "config_contract_hash",
        "protocol_hash",
        "claim_boundary_hash",
        "authorization_amendment_sha256",
        "source_snapshot_manifest_sha256",
        "source_snapshot_tree_sha256",
        "direct_input_binding_hash",
        "admission_receipt_hash",
        "run_identity_hash",
    ):
        require_sha256(payload.get(role), role)
    return str(claim_hash)


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = canonical_json(dict(payload)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ELOOP}:
            raise GovernanceError(
                "SCALE-BP v2 authorization lease member already exists or is unsafe."
            ) from exc
        raise GovernanceError(
            "SCALE-BP v2 authorization lease member could not be created."
        ) from exc
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short authorization lease write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        # A partial member is still an exhausted lease and must not be repaired.
        raise
    finally:
        os.close(descriptor)


def _absolute_nonsymlink_path(value: str | Path, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor) or path.is_symlink():
        raise GovernanceError(f"SCALE-BP v2 {role} is unsafe.")
    return path


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_text(value: object) -> str:
    text = " ".join(str(value).split())[:200]
    return text or "unspecified_failure"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = (
    "AuthorizationLeaseClaim",
    "LEASE_CLAIM_MEMBER",
    "LEASE_DIRECTORY_NAME",
    "LEASE_OUTCOME_MEMBER",
    "LEASE_OUTCOME_SCHEMA",
    "LEASE_SCHEMA",
    "PERSISTED_LEASE_MEMBER",
    "assert_authorization_unclaimed",
    "canonical_authorization_lease_path",
    "claim_authorization_lease",
    "record_authorization_outcome",
    "validate_authorization_lease",
    "validate_persisted_authorization_lease",
)
