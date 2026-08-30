"""Read-only, type-gated admission for one real OE-PPUR v4 execution."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .config import ResolvedV4ConfigBundle, validate_workspace_sealed_config
from .execution.authority import LoadedExecutionLaunchAuthority
from .execution.inputs import hash_resolved_input_locations
from .execution.replay_contract import (
    ReplayAdmissionContract,
    require_replay_admission_contract,
    validate_contract_launch_authority,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
    EXPECTED_SOURCE_RECEIPT_SHA256,
    EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256,
    EXPECTED_SOURCE_ROW_ORDER_SHA256,
    EXPECTED_SOURCE_SURFACE_SHA256,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from .lifecycle_source_seal import build_lifecycle_source_seal
from .protocol import frozen_protocol_payload
from .run_paths import assert_no_symlink_chain, paths_overlap
from .source_seal import SourceSealReceipt, validate_source_seal
from .source_supervision import SourceTrainingSurface
from .terminal.authority import validate_resolved_terminal_authority
from .workspace_binding import assert_canonical_output_root


_ADMISSION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SevenInputRunAdmission:
    config_contract_hash: str
    protocol_hash: str
    seven_input_contract_hash: str
    source_seal_hash: str
    source_seal_receipt_hash: str
    source_training_surface_receipt_hash: str
    source_training_surface_hash: str
    input_location_binding_hash: str
    workspace_input_manifest_sha256: str
    workspace_provenance_receipt_hash: str
    authorization_amendment_sha256: str
    lifecycle_source_seal_sha256: str
    lifecycle_source_seal_receipt_hash: str
    workspace_snapshot_sha256: str
    workspace_plan_sha256: str
    final_envelope_sha256: str
    execution_launch_authority_sha256: str
    sealed_replay_receipt_hash: str
    artifact_root: Path
    scratch_root: Path
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ADMISSION_TOKEN:
            raise ProtocolError("OE-PPUR v4 admission bypassed read-only validation.")
        for role in (
            "config_contract_hash",
            "protocol_hash",
            "seven_input_contract_hash",
            "source_seal_hash",
            "source_seal_receipt_hash",
            "source_training_surface_receipt_hash",
            "source_training_surface_hash",
            "input_location_binding_hash",
            "workspace_input_manifest_sha256",
            "workspace_provenance_receipt_hash",
            "authorization_amendment_sha256",
            "lifecycle_source_seal_sha256",
            "lifecycle_source_seal_receipt_hash",
            "workspace_snapshot_sha256",
            "workspace_plan_sha256",
            "final_envelope_sha256",
            "execution_launch_authority_sha256",
            "sealed_replay_receipt_hash",
        ):
            digest = require_sha256(getattr(self, role), role.replace("_", " "))
            if digest == "0" * 64:
                raise ProtocolError("OE-PPUR v4 admission contains a placeholder.")
            object.__setattr__(self, role, digest)
        artifact = Path(self.artifact_root)
        scratch = Path(self.scratch_root)
        if (
            not artifact.is_absolute()
            or not scratch.is_absolute()
            or artifact == Path(artifact.anchor)
            or scratch == Path(scratch.anchor)
            or paths_overlap(artifact, scratch)
        ):
            raise ProtocolError("OE-PPUR v4 admitted roots drifted.")
        object.__setattr__(self, "artifact_root", artifact)
        object.__setattr__(self, "scratch_root", scratch)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_seven_input_run_admission_v1",
            "status": "ADMITTED_SINGLE_USE_READ_ONLY",
            "experiment_id": EXPERIMENT_ID,
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
            "direct_input_roles": list(DIRECT_INPUT_ROLES),
            "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
            "config_contract_hash": self.config_contract_hash,
            "protocol_hash": self.protocol_hash,
            "seven_input_contract_hash": self.seven_input_contract_hash,
            "source_seal_hash": self.source_seal_hash,
            "source_seal_receipt_hash": self.source_seal_receipt_hash,
            "source_training_surface_receipt_hash": self.source_training_surface_receipt_hash,
            "source_training_surface_hash": self.source_training_surface_hash,
            "input_location_binding_hash": self.input_location_binding_hash,
            "workspace_input_manifest_sha256": self.workspace_input_manifest_sha256,
            "workspace_provenance_receipt_hash": self.workspace_provenance_receipt_hash,
            "authorization_amendment_sha256": self.authorization_amendment_sha256,
            "lifecycle_source_seal_sha256": self.lifecycle_source_seal_sha256,
            "lifecycle_source_seal_receipt_hash": self.lifecycle_source_seal_receipt_hash,
            "workspace_snapshot_sha256": self.workspace_snapshot_sha256,
            "workspace_plan_sha256": self.workspace_plan_sha256,
            "final_envelope_sha256": self.final_envelope_sha256,
            "execution_launch_authority_sha256": self.execution_launch_authority_sha256,
            "sealed_replay_receipt_hash": self.sealed_replay_receipt_hash,
            "artifact_root": self.artifact_root.as_posix(),
            "scratch_root": self.scratch_root.as_posix(),
            "source_supervision_materialized": True,
            "authorization_amendment_issued": True,
            "separate_launch_authority_validated": True,
            "execution_authorized": True,
            "launch_authority_is_scientific_input": False,
            "scientific_input_count": 7,
            "target_labels_opened": False,
            "mutation_performed": False,
            "cross_run_recovery_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def admit_seven_input_execution(
    bundle: ResolvedV4ConfigBundle,
    *,
    replay: ReplayAdmissionContract | object,
    launch_authority: LoadedExecutionLaunchAuthority,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
    scratch_root: str | Path,
) -> SevenInputRunAdmission:
    """Complete every read-only gate before the irreversible lease claim."""

    if (
        type(bundle) is not ResolvedV4ConfigBundle
        or type(launch_authority) is not LoadedExecutionLaunchAuthority
    ):
        raise ProtocolError("OE-PPUR v4 run admission is untyped.")
    replay_contract = require_replay_admission_contract(replay)
    config = validate_workspace_sealed_config(bundle.config)
    validate_contract_launch_authority(replay_contract, launch_authority)
    if (
        bundle.config != replay_contract.sealed_config
        or bundle.input_bindings != replay_contract.input_bindings
        or bundle.execution_launch_authority_sha256 != launch_authority.file_sha256
    ):
        raise ProtocolError("OE-PPUR v4 replay/bundle authority drifted.")

    artifact = assert_canonical_output_root(bundle.artifact_root)
    scratch = _validate_pristine_launch_roots(artifact, scratch_root)
    if scratch != replay_contract.paths.scratch_root:
        raise ProtocolError("OE-PPUR v4 scratch root drifted from sealed topology.")
    _validate_input_paths_and_hashes(
        bundle,
        artifact_root=artifact,
        scratch_root=scratch,
    )

    seal = validate_source_seal(source_seal)
    if type(source_surface) is not SourceTrainingSurface:
        raise ProtocolError("OE-PPUR v4 source supervision is untyped.")
    receipt = source_surface.receipt
    authority = replay_contract.authority
    lifecycle = build_lifecycle_source_seal(replay_contract.paths.repository_root)
    if (
        seal.combined_source_sha256 != authority.scientific_source_seal_sha256
        or lifecycle.lifecycle_source_seal_sha256
        != authority.lifecycle_seal_sha256
        or receipt.receipt_hash != EXPECTED_SOURCE_RECEIPT_SHA256
        or receipt.receipt_hash != config.source_supervision_content_sha256
        or source_surface.surface_hash != EXPECTED_SOURCE_SURFACE_SHA256
        or receipt.row_order_sha256 != EXPECTED_SOURCE_ROW_ORDER_SHA256
        or receipt.contract.producer_source_seal_sha256
        != EXPECTED_SOURCE_PRODUCER_SEAL_SHA256
        or receipt.compiler_recomputation_receipt_sha256
        != EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256
        or receipt.target_rows_present
        or receipt.target_labels_used
        or config.protocol_hash != frozen_protocol_payload()["protocol_hash"]
    ):
        raise ProtocolError("OE-PPUR v4 scientific admission binding drifted.")
    terminal_lifecycle = validate_resolved_terminal_authority(
        bundle,
        source_training_surface_receipt_hash=receipt.receipt_hash,
    )
    if terminal_lifecycle != lifecycle:
        raise ProtocolError("OE-PPUR v4 terminal lifecycle authority drifted.")

    return SevenInputRunAdmission(
        config_contract_hash=config.contract_hash,
        protocol_hash=config.protocol_hash,
        seven_input_contract_hash=config.seven_input_contract_hash,
        source_seal_hash=seal.combined_source_sha256,
        source_seal_receipt_hash=seal.receipt_hash,
        source_training_surface_receipt_hash=receipt.receipt_hash,
        source_training_surface_hash=source_surface.surface_hash,
        input_location_binding_hash=hash_resolved_input_locations(bundle.input_bindings),
        workspace_input_manifest_sha256=authority.input_manifest_file_sha256,
        workspace_provenance_receipt_hash=authority.seven_input_inventory_sha256,
        authorization_amendment_sha256=str(config.authorization_amendment_sha256),
        lifecycle_source_seal_sha256=lifecycle.lifecycle_source_seal_sha256,
        lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
        workspace_snapshot_sha256=authority.workspace_snapshot_sha256,
        workspace_plan_sha256=authority.workspace_plan_sha256,
        final_envelope_sha256=authority.final_envelope_sha256,
        execution_launch_authority_sha256=launch_authority.file_sha256,
        sealed_replay_receipt_hash=authority.sealed_replay_receipt_hash,
        artifact_root=artifact,
        scratch_root=scratch,
        _factory_token=_ADMISSION_TOKEN,
    )


def _validate_pristine_launch_roots(
    artifact_root: Path,
    scratch_root: str | Path,
) -> tuple[Path, Path]:
    artifact = Path(artifact_root)
    scratch = Path(scratch_root)
    if not scratch.is_absolute() or scratch != Path(os.path.abspath(scratch)):
        raise ProtocolError("OE-PPUR v4 scratch root is unsafe.")
    assert_no_symlink_chain(artifact, allow_missing_leaf=True)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    if (
        os.path.lexists(artifact)
        or os.path.lexists(scratch)
        or paths_overlap(artifact, scratch)
    ):
        raise ProtocolError("OE-PPUR v4 launch roots are not pristine.")
    return artifact, scratch


def _validate_input_paths_and_hashes(
    bundle: ResolvedV4ConfigBundle,
    *,
    artifact_root: Path,
    scratch_root: Path,
) -> None:
    scopes = tuple(
        row.path if row.kind == "directory" else row.path.parent
        for row in bundle.input_bindings
    )
    for index, scope in enumerate(scopes):
        if paths_overlap(scope, artifact_root) or paths_overlap(scope, scratch_root):
            raise ProtocolError("OE-PPUR v4 direct input overlaps a launch root.")
        for other in scopes[index + 1 :]:
            if paths_overlap(scope, other):
                raise ProtocolError("OE-PPUR v4 direct inputs overlap.")
    for row in bundle.input_bindings:
        assert_no_symlink_chain(row.path)
        try:
            metadata = row.path.lstat()
            resolved = row.path.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError("OE-PPUR v4 direct input is absent.") from exc
        kind_ok = (
            stat.S_ISDIR(metadata.st_mode)
            if row.kind == "directory"
            else stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
        )
        if resolved != row.path or not kind_ok:
            raise ProtocolError("OE-PPUR v4 direct input kind drifted.")
        for relative, expected in row.member_hashes:
            member = row.path / relative if row.kind == "directory" else row.path
            if _stable_sha256(member) != expected:
                raise ProtocolError("OE-PPUR v4 direct input bytes drifted.")


def _stable_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v4 direct input member is unsafe.")
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 direct input member is unreadable.") from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or before.st_size != len(raw)
    ):
        raise ProtocolError("OE-PPUR v4 direct input changed while read.")
    return hashlib.sha256(raw).hexdigest()


__all__ = ("SevenInputRunAdmission", "admit_seven_input_execution")
