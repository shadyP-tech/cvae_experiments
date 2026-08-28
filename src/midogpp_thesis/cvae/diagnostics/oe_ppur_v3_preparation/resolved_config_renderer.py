"""Atomic, authorization-bound OE-PPUR v3 resolved-config renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import shutil
import tempfile

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    ResolvedV3ConfigBundle,
    load_resolved_config,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_source_seal import (
    LifecycleSourceSealReceipt,
    validate_lifecycle_source_seal,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_admission import (
    admit_seven_input_execution,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_paths import (
    WORKSPACE_ENVELOPE_DIRECTORIES,
    is_exact_workspace_launch_envelope,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.contracts import (
    SourceTrainingSurface,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    build_source_seal,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workspace_provenance import validate_workspace_input_provenance
from .durable_io import (
    fsync_directory,
    hash_unique_regular_file,
    read_bounded_unique_file,
    write_bytes_exclusive,
)
from .exclusive_commit import rename_directory_noreplace
from .envelope_plan import (
    build_authorization_envelope_plan,
    resolved_config_payload as _resolved_config_payload,
)
from .paths import CanonicalPreparationPaths


@dataclass(frozen=True, slots=True)
class ResolvedEnvelopeReceipt:
    artifact_root: Path
    config_contract_hash: str
    amendment_sha256: str
    workspace_provenance_receipt_hash: str
    read_only_admission_receipt_hash: str
    resolved_config_sha256: str
    lifecycle_source_seal_sha256: str
    lifecycle_source_seal_receipt_hash: str
    authorized_semantics_hash: str
    envelope_content_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.artifact_root)
        if not root.is_absolute() or not is_exact_workspace_launch_envelope(root):
            raise ProtocolError("OE-PPUR v3 resolved envelope topology drifted.")
        for role in (
            "config_contract_hash",
            "amendment_sha256",
            "workspace_provenance_receipt_hash",
            "read_only_admission_receipt_hash",
            "resolved_config_sha256",
            "lifecycle_source_seal_sha256",
            "lifecycle_source_seal_receipt_hash",
            "authorized_semantics_hash",
            "envelope_content_hash",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v3_resolved_envelope_receipt_v1",
                    "artifact_root": root.as_posix(),
                    "config_contract_hash": self.config_contract_hash,
                    "amendment_sha256": self.amendment_sha256,
                    "workspace_provenance_receipt_hash": (
                        self.workspace_provenance_receipt_hash
                    ),
                    "read_only_admission_receipt_hash": (
                        self.read_only_admission_receipt_hash
                    ),
                    "resolved_config_sha256": self.resolved_config_sha256,
                    "lifecycle_source_seal_sha256": (
                        self.lifecycle_source_seal_sha256
                    ),
                    "lifecycle_source_seal_receipt_hash": (
                        self.lifecycle_source_seal_receipt_hash
                    ),
                    "authorized_semantics_hash": self.authorized_semantics_hash,
                    "envelope_content_hash": self.envelope_content_hash,
                    "generic_workspace_prepare_used": False,
                    "authorization_consumed": False,
                    "target_labels_opened": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_resolved_envelope_receipt_v1",
            "artifact_root": self.artifact_root.as_posix(),
            "config_contract_hash": self.config_contract_hash,
            "amendment_sha256": self.amendment_sha256,
            "workspace_provenance_receipt_hash": (
                self.workspace_provenance_receipt_hash
            ),
            "read_only_admission_receipt_hash": (
                self.read_only_admission_receipt_hash
            ),
            "resolved_config_sha256": self.resolved_config_sha256,
            "lifecycle_source_seal_sha256": self.lifecycle_source_seal_sha256,
            "lifecycle_source_seal_receipt_hash": (
                self.lifecycle_source_seal_receipt_hash
            ),
            "authorized_semantics_hash": self.authorized_semantics_hash,
            "envelope_content_hash": self.envelope_content_hash,
            "generic_workspace_prepare_used": False,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "receipt_hash": self.receipt_hash,
        }


def render_authorization_ready_envelope(
    paths: CanonicalPreparationPaths,
    *,
    source_surface: SourceTrainingSurface,
    protocol_hash: str,
    lifecycle_source_seal: LifecycleSourceSealReceipt,
    allow_existing_envelope: bool = False,
) -> ResolvedEnvelopeReceipt:
    """Publish the exact two-file launch envelope without consuming the lease."""

    if not isinstance(paths, CanonicalPreparationPaths):
        raise ProtocolError("OE-PPUR v3 resolved paths are untyped.")
    if not isinstance(source_surface, SourceTrainingSurface):
        raise ProtocolError("OE-PPUR v3 resolved source surface is untyped.")
    if type(allow_existing_envelope) is not bool:
        raise ProtocolError("OE-PPUR v3 resolved recovery flag is untyped.")
    lifecycle = validate_lifecycle_source_seal(lifecycle_source_seal)
    _assert_renderable_topology(
        paths,
        allow_existing_envelope=allow_existing_envelope,
    )
    raw_amendment, observed_amendment_sha256 = read_bounded_unique_file(
        paths.amendment_path,
        maximum_bytes=1024 * 1024,
        role="authorization amendment",
    )
    plan = build_authorization_envelope_plan(
        paths,
        source_surface=source_surface,
        protocol_hash=protocol_hash,
        lifecycle_source_seal=lifecycle,
        amendment_raw=raw_amendment,
        prospective_amendment=False,
    )
    amendment_sha256 = plan.amendment_sha256
    if amendment_sha256 != observed_amendment_sha256:
        raise ProtocolError("OE-PPUR v3 amendment digest read-back drifted.")
    config = plan.config
    authorized_semantics = plan.authorized_semantics_payload()
    config_raw = plan.config_raw
    manifest_raw = plan.manifest_raw
    if paths.artifact_root.exists():
        _validate_existing_envelope(
            paths.artifact_root,
            config_raw=config_raw,
            manifest_raw=manifest_raw,
        )
    else:
        _publish_envelope(paths.artifact_root, config_raw, manifest_raw)

    bundle = load_resolved_config(paths.artifact_root / "config.resolved.yaml")
    _assert_bundle_matches_paths(bundle, paths)
    provenance = validate_workspace_input_provenance(
        paths.artifact_root,
        bundle.input_bindings,
        expected_authorized_semantics=authorized_semantics,
    )
    seal = build_source_seal(paths.repository_root)
    admission = admit_seven_input_execution(
        bundle,
        artifact_root=paths.artifact_root,
        scratch_root=paths.scratch_root,
        source_seal=seal,
        source_surface=source_surface,
    )
    observed_config_sha256, _ = hash_unique_regular_file(
        paths.artifact_root / "config.resolved.yaml",
        role="resolved config",
    )
    return ResolvedEnvelopeReceipt(
        artifact_root=paths.artifact_root,
        config_contract_hash=config.contract_hash,
        amendment_sha256=amendment_sha256,
        workspace_provenance_receipt_hash=provenance.receipt_hash,
        read_only_admission_receipt_hash=admission.receipt_hash,
        resolved_config_sha256=observed_config_sha256,
        lifecycle_source_seal_sha256=(
            lifecycle.lifecycle_source_seal_sha256
        ),
        lifecycle_source_seal_receipt_hash=lifecycle.receipt_hash,
        authorized_semantics_hash=canonical_hash(authorized_semantics),
        envelope_content_hash=plan.content_hash,
    )


def _assert_renderable_topology(
    paths: CanonicalPreparationPaths,
    *,
    allow_existing_envelope: bool,
) -> None:
    for role, candidate in (
        ("scratch", paths.scratch_root),
        ("lease", paths.lease_root),
    ):
        if candidate.exists() or candidate.is_symlink():
            raise ProtocolError(
                f"OE-PPUR v3 prior {role} state forbids resolved rendering."
            )
    output_present = paths.artifact_root.exists() or paths.artifact_root.is_symlink()
    if output_present and (
        not allow_existing_envelope
        or paths.artifact_root.is_symlink()
        or not paths.artifact_root.is_dir()
        or not is_exact_workspace_launch_envelope(paths.artifact_root)
    ):
        raise ProtocolError(
            "OE-PPUR v3 prior output state forbids resolved rendering."
        )
    if (
        not paths.amendment_root.is_dir()
        or paths.amendment_root.is_symlink()
        or not paths.amendment_path.is_file()
        or paths.amendment_path.is_symlink()
        or not paths.artifact_root.parent.is_dir()
        or paths.artifact_root.parent.is_symlink()
    ):
        raise ProtocolError("OE-PPUR v3 resolved-render topology is unsafe.")


def _validate_existing_envelope(
    artifact_root: Path,
    *,
    config_raw: bytes,
    manifest_raw: bytes,
) -> None:
    """Accept only the exact envelope from a prior post-publication failure."""

    if not is_exact_workspace_launch_envelope(artifact_root):
        raise ProtocolError("OE-PPUR v3 recovery envelope topology drifted.")
    for relative, expected, maximum in (
        ("config.resolved.yaml", config_raw, 2 * 1024 * 1024),
        ("provenance/input_artifacts.json", manifest_raw, 8 * 1024 * 1024),
    ):
        observed, observed_sha256 = read_bounded_unique_file(
            artifact_root / relative,
            maximum_bytes=maximum,
            role=f"existing {relative}",
        )
        if (
            observed != expected
            or hashlib.sha256(observed).hexdigest() != observed_sha256
        ):
            raise ProtocolError("OE-PPUR v3 recovery envelope bytes drifted.")


def _publish_envelope(
    artifact_root: Path,
    config_raw: bytes,
    manifest_raw: bytes,
) -> None:
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_root.name}.oe-ppur-v3-authorized-",
            dir=artifact_root.parent,
        )
    )
    committed = False
    try:
        for relative in WORKSPACE_ENVELOPE_DIRECTORIES:
            (stage / relative).mkdir(mode=0o700)
        write_bytes_exclusive(
            stage / "config.resolved.yaml",
            config_raw,
            role="resolved config",
        )
        write_bytes_exclusive(
            stage / "provenance/input_artifacts.json",
            manifest_raw,
            role="workspace input provenance",
        )
        for relative in reversed(WORKSPACE_ENVELOPE_DIRECTORIES):
            fsync_directory(stage / relative)
        fsync_directory(stage)
        rename_directory_noreplace(stage, artifact_root)
        committed = True
        fsync_directory(artifact_root.parent)
    except BaseException as exc:
        raise ProtocolError(
            "OE-PPUR v3 resolved envelope publication failed closed."
        ) from exc
    finally:
        if not committed and stage.exists():
            if stage.is_symlink() or stage.parent != artifact_root.parent:
                raise ProtocolError("OE-PPUR v3 owned staging root became unsafe.")
            shutil.rmtree(stage)


def _assert_bundle_matches_paths(
    bundle: ResolvedV3ConfigBundle,
    paths: CanonicalPreparationPaths,
) -> None:
    if (
        bundle.artifact_root != paths.artifact_root
        or bundle.input_bindings != paths.input_bindings
        or bundle.source_path != paths.artifact_root / "config.resolved.yaml"
    ):
        raise ProtocolError("OE-PPUR v3 resolved config read-back drifted.")


__all__ = (
    "ResolvedEnvelopeReceipt",
    "render_authorization_ready_envelope",
)
