"""Read-only seven-input admission for one authorized OE-PPUR v3 run."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path
import stat

from ...protocol import ProtocolError
from .config import ResolvedV3ConfigBundle, validate_authorization_ready_config
from .execution.inputs import ResolvedDirectInput, hash_resolved_input_locations
from .hashing import canonical_hash, require_sha256
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    INPUT_RELATIVE_MEMBERS,
    OUTPUT_ARTIFACT_ID,
)
from .protocol import frozen_protocol_payload
from .run_paths import (
    assert_no_symlink_chain,
    paths_overlap,
    validate_absolute_path,
    validate_launch_roots,
)
from .source_seal import (
    SourceSealReceipt,
    validate_live_producer_seal_binding,
    validate_source_seal,
)
from .source_supervision import SourceTrainingSurface
from .terminal.label_reader import validate_resolved_terminal_authority
from .workspace_provenance import (
    build_authorized_input_semantics,
    validate_workspace_input_provenance,
)
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
    artifact_root: Path
    scratch_root: Path
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _ADMISSION_TOKEN:
            raise ProtocolError("OE-PPUR v3 admission bypassed read-only validation.")
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
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        artifact = Path(self.artifact_root)
        scratch = Path(self.scratch_root)
        if not artifact.is_absolute() or not scratch.is_absolute():
            raise ProtocolError("OE-PPUR v3 admitted roots drifted.")
        object.__setattr__(self, "artifact_root", artifact)
        object.__setattr__(self, "scratch_root", scratch)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_seven_input_run_admission_v1",
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
            "source_training_surface_receipt_hash": (
                self.source_training_surface_receipt_hash
            ),
            "source_training_surface_hash": self.source_training_surface_hash,
            "input_location_binding_hash": self.input_location_binding_hash,
            "workspace_input_manifest_sha256": self.workspace_input_manifest_sha256,
            "workspace_provenance_receipt_hash": (
                self.workspace_provenance_receipt_hash
            ),
            "authorization_amendment_sha256": (
                self.authorization_amendment_sha256
            ),
            "lifecycle_source_seal_sha256": (
                self.lifecycle_source_seal_sha256
            ),
            "lifecycle_source_seal_receipt_hash": (
                self.lifecycle_source_seal_receipt_hash
            ),
            "artifact_root": self.artifact_root.as_posix(),
            "scratch_root": self.scratch_root.as_posix(),
            "source_supervision_materialized": True,
            "authorization_amendment_issued": True,
            "execution_authorized": True,
            "target_labels_opened": False,
            "mutation_performed": False,
            "cross_run_recovery_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def admit_seven_input_execution(
    bundle: ResolvedV3ConfigBundle,
    *,
    artifact_root: str | Path,
    scratch_root: str | Path,
    source_seal: SourceSealReceipt,
    source_surface: SourceTrainingSurface,
) -> SevenInputRunAdmission:
    """Validate all execution authority without mutating either run root."""

    if type(bundle) is not ResolvedV3ConfigBundle:
        raise ProtocolError("OE-PPUR v3 admission requires a resolved config bundle.")
    config = validate_authorization_ready_config(bundle.config)
    seal = validate_source_seal(source_seal)
    if type(source_surface) is not SourceTrainingSurface:
        raise ProtocolError("OE-PPUR v3 source supervision is untyped.")
    artifact, scratch = validate_launch_roots(artifact_root, scratch_root)
    assert_canonical_output_root(artifact)
    if artifact != bundle.artifact_root or bundle.source_path != artifact / "config.resolved.yaml":
        raise ProtocolError("OE-PPUR v3 resolved output root drifted.")
    _validate_input_paths(
        bundle,
        artifact_root=artifact,
        scratch_root=scratch,
    )
    receipt = source_surface.receipt
    validate_live_producer_seal_binding(
        configured_sha256=config.source_supervision_producer_seal_sha256,
        parsed_sha256=receipt.contract.producer_source_seal_sha256,
        source_seal=seal,
    )
    if (
        receipt.receipt_hash != config.source_supervision_content_sha256
        or receipt.row_order_sha256 != config.source_supervision_row_order_sha256
        or receipt.contract.producer_source_seal_sha256
        != config.source_supervision_producer_seal_sha256
        or receipt.compiler_recomputation_receipt_sha256
        != config.source_supervision_recomputation_receipt_sha256
        or config.protocol_hash != frozen_protocol_payload()["protocol_hash"]
    ):
        raise ProtocolError("OE-PPUR v3 source/config admission binding drifted.")
    lifecycle = validate_resolved_terminal_authority(
        bundle,
        source_training_surface_receipt_hash=receipt.receipt_hash,
    )
    authorized_semantics = build_authorized_input_semantics(
        source_contract_hash=receipt.receipt_hash,
        source_row_order_sha256=receipt.row_order_sha256,
        source_producer_seal_sha256=(
            receipt.contract.producer_source_seal_sha256
        ),
        source_recomputation_receipt_sha256=(
            receipt.compiler_recomputation_receipt_sha256
        ),
        authorization_amendment_sha256=str(
            config.authorization_amendment_sha256
        ),
        protocol_hash=config.protocol_hash,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
    )
    workspace_provenance = validate_workspace_input_provenance(
        artifact,
        bundle.input_bindings,
        expected_authorized_semantics=authorized_semantics,
    )
    return SevenInputRunAdmission(
        config_contract_hash=config.contract_hash,
        protocol_hash=config.protocol_hash,
        seven_input_contract_hash=config.seven_input_contract_hash,
        source_seal_hash=seal.combined_source_sha256,
        source_seal_receipt_hash=seal.receipt_hash,
        source_training_surface_receipt_hash=receipt.receipt_hash,
        source_training_surface_hash=source_surface.surface_hash,
        input_location_binding_hash=hash_resolved_input_locations(
            bundle.input_bindings
        ),
        workspace_input_manifest_sha256=(
            workspace_provenance.manifest_file_sha256
        ),
        workspace_provenance_receipt_hash=workspace_provenance.receipt_hash,
        authorization_amendment_sha256=str(
            config.authorization_amendment_sha256
        ),
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
        lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
        artifact_root=artifact,
        scratch_root=scratch,
        _factory_token=_ADMISSION_TOKEN,
    )


def _validate_input_paths(
    bundle: ResolvedV3ConfigBundle,
    *,
    artifact_root: Path,
    scratch_root: Path,
    allow_missing_amendment: bool = False,
) -> None:
    artifact = Path(artifact_root)
    scratch = Path(scratch_root)
    scopes = tuple(
        _direct_input_scope(row, relative_member=relative_member)
        for row, relative_member in zip(
            bundle.input_bindings,
            INPUT_RELATIVE_MEMBERS,
            strict=True,
        )
    )
    # Reject topology before opening any input member.  For file-backed direct
    # inputs, the protected scope is the artifact root rather than only the
    # leaf member (for example, the manifest CSV or ledger JSON).
    for index, scope in enumerate(scopes):
        if paths_overlap(scope, artifact):
            raise ProtocolError(
                "OE-PPUR v3 direct input overlaps the output root."
            )
        if paths_overlap(scope, scratch):
            raise ProtocolError(
                "OE-PPUR v3 direct input overlaps the scratch root."
            )
        for other in scopes[index + 1 :]:
            if paths_overlap(scope, other):
                raise ProtocolError("OE-PPUR v3 direct inputs overlap.")
    for row in bundle.input_bindings:
        path = row.path
        prospective_amendment = (
            allow_missing_amendment
            and row.artifact_id == AUTHORIZATION_AMENDMENT_ARTIFACT_ID
            and not path.exists()
            and not path.is_symlink()
        )
        assert_no_symlink_chain(
            path,
            allow_missing_leaf=prospective_amendment,
        )
        if prospective_amendment:
            continue
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ProtocolError("OE-PPUR v3 direct input is absent.") from exc
        if resolved != path:
            raise ProtocolError("OE-PPUR v3 direct input resolution drifted.")
        expected = stat.S_ISDIR(metadata.st_mode) if row.kind == "directory" else stat.S_ISREG(metadata.st_mode)
        if not expected or (row.kind == "file" and metadata.st_nlink != 1):
            raise ProtocolError("OE-PPUR v3 direct input kind drifted.")


def validate_prospective_input_paths(
    bundle: ResolvedV3ConfigBundle,
    *,
    artifact_root: str | Path,
    scratch_root: str | Path,
) -> None:
    """Validate path topology while the canonical amendment leaf is absent."""

    if type(bundle) is not ResolvedV3ConfigBundle:
        raise ProtocolError("OE-PPUR v3 prospective path bundle is untyped.")
    artifact = validate_absolute_path(artifact_root, role="artifact root")
    scratch = validate_absolute_path(scratch_root, role="scratch root")
    assert_canonical_output_root(artifact)
    assert_no_symlink_chain(artifact, allow_missing_leaf=True)
    assert_no_symlink_chain(scratch, allow_missing_leaf=True)
    if (
        artifact.exists()
        or artifact.is_symlink()
        or scratch.exists()
        or scratch.is_symlink()
        or paths_overlap(artifact, scratch)
        or bundle.artifact_root != artifact
    ):
        raise ProtocolError("OE-PPUR v3 prospective launch topology drifted.")
    _validate_input_paths(
        bundle,
        artifact_root=artifact,
        scratch_root=scratch,
        allow_missing_amendment=True,
    )


def _direct_input_scope(
    row: ResolvedDirectInput,
    *,
    relative_member: str,
) -> Path:
    scope = row.path
    for _part in Path(relative_member).parts:
        scope = scope.parent
    return scope


__all__ = (
    "SevenInputRunAdmission",
    "admit_seven_input_execution",
    "validate_prospective_input_paths",
)
