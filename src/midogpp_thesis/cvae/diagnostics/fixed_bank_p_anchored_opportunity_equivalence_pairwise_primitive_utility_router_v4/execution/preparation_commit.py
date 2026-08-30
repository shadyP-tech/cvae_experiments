"""NFS-safe post-lease materialization of the sealed v4 launch envelope."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import os
from pathlib import Path

from ....protocol import ProtocolError
from ..config import ResolvedV4ConfigBundle
from ..hashing import canonical_bytes, canonical_hash, require_sha256
from ..lease_claim import AuthorizationLeaseClaim, validate_authorization_lease
from ..lease_io import fsync_directory
from ..run_admission import SevenInputRunAdmission
from .authority import LoadedExecutionLaunchAuthority
from .sealed_replay import SealedExecutionReplay, validate_loaded_launch_authority


AUTHORITY_COPY_MEMBER = "preparation/execution_launch_authority.json"
REPLAY_RECEIPT_MEMBER = "preparation/sealed_execution_replay.json"
_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PreparedOutputReceipt:
    artifact_root: Path
    config_file_sha256: str
    input_manifest_file_sha256: str
    final_envelope_file_sha256: str
    launch_authority_file_sha256: str
    replay_receipt_file_sha256: str
    commit_marker_file_sha256: str
    authorization_lease_claim_hash: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        root = Path(self.artifact_root)
        if (
            _factory_token is not _TOKEN
            or not root.is_absolute()
            or root.is_symlink()
            or not root.is_dir()
        ):
            raise ProtocolError("OE-PPUR v4 prepared output receipt drifted.")
        for role in (
            "config_file_sha256",
            "input_manifest_file_sha256",
            "final_envelope_file_sha256",
            "launch_authority_file_sha256",
            "replay_receipt_file_sha256",
            "commit_marker_file_sha256",
            "authorization_lease_claim_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "artifact_root", root)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_prepared_output_receipt_v1",
            "status": "PREPARATION_COMMITTED",
            "artifact_root": self.artifact_root.as_posix(),
            "config_file_sha256": self.config_file_sha256,
            "input_manifest_file_sha256": self.input_manifest_file_sha256,
            "final_envelope_file_sha256": self.final_envelope_file_sha256,
            "launch_authority_file_sha256": self.launch_authority_file_sha256,
            "replay_receipt_file_sha256": self.replay_receipt_file_sha256,
            "commit_marker_file_sha256": self.commit_marker_file_sha256,
            "authorization_lease_claim_hash": self.authorization_lease_claim_hash,
            "member_writes_used_o_excl": True,
            "commit_marker_written_last": True,
            "preparation_commit_is_scientific_complete": False,
            "target_labels_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def commit_prepared_output(
    bundle: ResolvedV4ConfigBundle,
    *,
    replay: SealedExecutionReplay,
    launch_authority: LoadedExecutionLaunchAuthority,
    run_admission: SevenInputRunAdmission,
    lease: AuthorizationLeaseClaim,
) -> PreparedOutputReceipt:
    """Create the output exactly once after authority is irreversibly spent."""

    if (
        type(bundle) is not ResolvedV4ConfigBundle
        or type(replay) is not SealedExecutionReplay
        or type(launch_authority) is not LoadedExecutionLaunchAuthority
        or type(run_admission) is not SevenInputRunAdmission
        or type(lease) is not AuthorizationLeaseClaim
    ):
        raise ProtocolError("OE-PPUR v4 preparation commit is untyped.")
    validate_loaded_launch_authority(replay, launch_authority)
    claimed = validate_authorization_lease(lease)
    candidate = replay.context.candidate
    topology = candidate.plan.topology
    root = bundle.artifact_root
    if (
        root != topology.output_root
        or root != run_admission.artifact_root
        or bundle.source_path != topology.resolved_config_path
        or bundle.input_manifest_path != topology.input_manifest_path
        or bundle.final_envelope_path != topology.envelope_path
        or launch_authority.file_sha256
        != run_admission.execution_launch_authority_sha256
        or claimed.payload.get("seven_input_admission_hash")
        != run_admission.receipt_hash
        or claimed.payload.get("execution_launch_authority_sha256")
        != launch_authority.file_sha256
        or os.path.lexists(root)
    ):
        raise ProtocolError("OE-PPUR v4 preparation commit lineage drifted.")

    realized = candidate.envelope.realized_templates
    members = (
        (topology.resolved_config_path, realized.resolved_config_raw),
        (topology.input_manifest_path, realized.input_manifest_raw),
        (topology.envelope_path, candidate.envelope_raw),
        (
            root / AUTHORITY_COPY_MEMBER,
            launch_authority.authority.canonical_file_bytes(),
        ),
        (
            root / REPLAY_RECEIPT_MEMBER,
            canonical_bytes(replay.to_payload()) + b"\n",
        ),
    )
    try:
        os.mkdir(root, 0o700)
        fsync_directory(root.parent)
        for directory in (root / "provenance", root / "preparation"):
            os.mkdir(directory, 0o700)
            fsync_directory(directory.parent)
        for path, raw in members:
            _write_exclusive(path, raw)
        for path, raw in members:
            _require_exact_file(path, raw)
        _write_exclusive(topology.commit_marker_path, candidate.commit_marker_raw)
        _require_exact_file(topology.commit_marker_path, candidate.commit_marker_raw)
        fsync_directory(root)
    except BaseException as exc:
        raise ProtocolError(
            "OE-PPUR v4 preparation commit failed after authorization consumption; "
            "the run is exhausted and non-recoverable."
        ) from exc

    return PreparedOutputReceipt(
        artifact_root=root,
        config_file_sha256=hashlib.sha256(realized.resolved_config_raw).hexdigest(),
        input_manifest_file_sha256=hashlib.sha256(realized.input_manifest_raw).hexdigest(),
        final_envelope_file_sha256=hashlib.sha256(candidate.envelope_raw).hexdigest(),
        launch_authority_file_sha256=launch_authority.file_sha256,
        replay_receipt_file_sha256=hashlib.sha256(
            canonical_bytes(replay.to_payload()) + b"\n"
        ).hexdigest(),
        commit_marker_file_sha256=hashlib.sha256(
            candidate.commit_marker_raw
        ).hexdigest(),
        authorization_lease_claim_hash=claimed.claim_hash,
        _factory_token=_TOKEN,
    )


def _write_exclusive(path: Path, raw: bytes) -> None:
    if type(raw) is not bytes or not path.parent.is_dir() or path.parent.is_symlink():
        raise ProtocolError("OE-PPUR v4 preparation member target is unsafe.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short preparation write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def _require_exact_file(path: Path, expected: bytes) -> None:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v4 preparation member is unsafe.")
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 preparation member is unreadable.") from exc
    if (
        raw != expected
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ProtocolError("OE-PPUR v4 preparation member read-back drifted.")


__all__ = (
    "AUTHORITY_COPY_MEMBER",
    "PreparedOutputReceipt",
    "REPLAY_RECEIPT_MEMBER",
    "commit_prepared_output",
)
