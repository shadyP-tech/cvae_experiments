"""Fail-closed pre- and post-publication validation for OE-PPUR v4.

No function in this module writes a file, creates a directory, claims a lease,
opens labels, or launches the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from ...protocol import ProtocolError
from .amendment import (
    AuthorizationTerms,
    authorization_amendment_bytes,
    validate_authorization_amendment_bytes,
)
from .contracts import ExecutionTopologyContract, ScientificSealDescriptor
from .envelope import (
    FinalAuthorizationEnvelope,
    build_final_authorization_envelope,
    commit_marker_bytes,
    final_envelope_bytes,
    validate_final_envelope_bytes,
)
from .hashing import bytes_sha256, payload_sha256, require_sha256
from .inputs import ExistingInputInventory
from .host import WorkstationTopologyReceipt
from .plan import PreAmendmentPlan
from .predecessor import capture_predecessor_preservation
from .snapshot import WorkspaceSnapshot, validate_workspace_snapshot


@dataclass(frozen=True, slots=True)
class PublicationSurfaceObservation:
    amendment_exists: bool
    amendment_sha256: str | None
    output_root_exists: bool
    envelope_exists: bool
    envelope_sha256: str | None
    commit_marker_exists: bool
    commit_marker_sha256: str | None
    lease_exists: bool
    scratch_root_exists: bool
    scratch_receipts_exist: bool
    topology_receipt_exists: bool

    def __post_init__(self) -> None:
        bool_roles = (
            "amendment_exists",
            "output_root_exists",
            "envelope_exists",
            "commit_marker_exists",
            "lease_exists",
            "scratch_root_exists",
            "scratch_receipts_exist",
            "topology_receipt_exists",
        )
        if any(type(getattr(self, role)) is not bool for role in bool_roles):
            raise ProtocolError("OE-PPUR v4 surface observation is malformed.")
        for exists_role, digest_role in (
            ("amendment_exists", "amendment_sha256"),
            ("envelope_exists", "envelope_sha256"),
            ("commit_marker_exists", "commit_marker_sha256"),
        ):
            exists = getattr(self, exists_role)
            digest = getattr(self, digest_role)
            if exists:
                object.__setattr__(
                    self,
                    digest_role,
                    require_sha256(digest, digest_role.replace("_", " ")),
                )
            elif digest is not None:
                raise ProtocolError("OE-PPUR v4 absent surface has a digest.")
        if (self.envelope_exists or self.commit_marker_exists) and not (
            self.output_root_exists
        ):
            raise ProtocolError("OE-PPUR v4 output surface topology is malformed.")

    def to_payload(self) -> dict[str, object]:
        return {
            "amendment_exists": self.amendment_exists,
            "amendment_sha256": self.amendment_sha256,
            "output_root_exists": self.output_root_exists,
            "envelope_exists": self.envelope_exists,
            "envelope_sha256": self.envelope_sha256,
            "commit_marker_exists": self.commit_marker_exists,
            "commit_marker_sha256": self.commit_marker_sha256,
            "lease_exists": self.lease_exists,
            "scratch_root_exists": self.scratch_root_exists,
            "scratch_receipts_exist": self.scratch_receipts_exist,
            "topology_receipt_exists": self.topology_receipt_exists,
        }


@dataclass(frozen=True, slots=True)
class PreparationCandidate:
    plan: PreAmendmentPlan
    terms: AuthorizationTerms
    amendment_raw: bytes
    envelope: FinalAuthorizationEnvelope
    envelope_raw: bytes = field(init=False)
    commit_marker_raw: bytes = field(init=False)
    candidate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.plan) is not PreAmendmentPlan
            or type(self.terms) is not AuthorizationTerms
            or type(self.amendment_raw) is not bytes
            or type(self.envelope) is not FinalAuthorizationEnvelope
        ):
            raise ProtocolError("OE-PPUR v4 preparation candidate is untyped.")
        validate_authorization_amendment_bytes(
            self.amendment_raw, plan=self.plan, terms=self.terms
        )
        expected = build_final_authorization_envelope(
            self.plan, self.terms, self.amendment_raw
        )
        if expected != self.envelope:
            raise ProtocolError("OE-PPUR v4 candidate envelope drifted.")
        envelope_raw = final_envelope_bytes(self.envelope)
        marker_raw = commit_marker_bytes(self.envelope)
        object.__setattr__(self, "envelope_raw", envelope_raw)
        object.__setattr__(self, "commit_marker_raw", marker_raw)
        object.__setattr__(
            self,
            "candidate_hash",
            payload_sha256(
                {
                    "schema_version": "oe_ppur_v4_preparation_candidate_v1",
                    "pre_amendment_plan_sha256": self.plan.plan_hash,
                    "authorization_amendment_sha256": bytes_sha256(
                        self.amendment_raw
                    ),
                    "final_envelope_sha256": bytes_sha256(envelope_raw),
                    "commit_marker_sha256": bytes_sha256(marker_raw),
                    "publication_performed": False,
                    "authorization_consumed": False,
                    "target_labels_opened": False,
                    "experiment_launched": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PrepublicationValidationReceipt:
    candidate_hash: str
    workspace_snapshot_hash: str
    existing_input_inventory_hash: str
    plan_hash: str
    amendment_sha256: str
    final_envelope_sha256: str
    commit_marker_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "candidate_hash",
            "workspace_snapshot_hash",
            "existing_input_inventory_hash",
            "plan_hash",
            "amendment_sha256",
            "final_envelope_sha256",
            "commit_marker_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_prepublication_validation_v1",
            "candidate_hash": self.candidate_hash,
            "workspace_snapshot_hash": self.workspace_snapshot_hash,
            "existing_input_inventory_hash": self.existing_input_inventory_hash,
            "pre_amendment_plan_sha256": self.plan_hash,
            "prospective_amendment_sha256": self.amendment_sha256,
            "prospective_final_envelope_sha256": self.final_envelope_sha256,
            "prospective_commit_marker_sha256": self.commit_marker_sha256,
            "publication_performed": False,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }


@dataclass(frozen=True, slots=True)
class PostpublicationValidationReceipt:
    candidate_hash: str
    prepublication_receipt_hash: str
    amendment_sha256: str
    final_envelope_sha256: str
    commit_marker_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "candidate_hash",
            "prepublication_receipt_hash",
            "amendment_sha256",
            "final_envelope_sha256",
            "commit_marker_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_postpublication_validation_v1",
            "candidate_hash": self.candidate_hash,
            "prepublication_receipt_hash": self.prepublication_receipt_hash,
            "amendment_sha256": self.amendment_sha256,
            "final_envelope_sha256": self.final_envelope_sha256,
            "commit_marker_sha256": self.commit_marker_sha256,
            "commit_protocol": [
                "EXCLUSIVE_FINAL_ROOT",
                "O_EXCL_MEMBERS",
                "COMMIT_MARKER_LAST",
            ],
            "authorization_consumed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }


@dataclass(frozen=True, slots=True)
class AmendmentPublicationValidationReceipt:
    candidate_hash: str
    prepublication_receipt_hash: str
    workspace_snapshot_hash: str
    existing_input_inventory_hash: str
    amendment_sha256: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "candidate_hash",
            "prepublication_receipt_hash",
            "workspace_snapshot_hash",
            "existing_input_inventory_hash",
            "amendment_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_amendment_publication_validation_v1",
            "candidate_hash": self.candidate_hash,
            "prepublication_receipt_hash": self.prepublication_receipt_hash,
            "workspace_snapshot_hash": self.workspace_snapshot_hash,
            "existing_input_inventory_hash": self.existing_input_inventory_hash,
            "amendment_sha256": self.amendment_sha256,
            "final_envelope_rendered": False,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }


def build_preparation_candidate(
    plan: PreAmendmentPlan,
    terms: AuthorizationTerms,
) -> PreparationCandidate:
    amendment = authorization_amendment_bytes(plan, terms)
    envelope = build_final_authorization_envelope(plan, terms, amendment)
    return PreparationCandidate(plan, terms, amendment, envelope)


def validate_prepublication(
    candidate: PreparationCandidate,
    *,
    observed_workspace: WorkspaceSnapshot,
    observed_existing_inputs: ExistingInputInventory,
    observed_topology: ExecutionTopologyContract,
    observed_scientific: ScientificSealDescriptor,
    observed_workstation: WorkstationTopologyReceipt,
    observed_surfaces: PublicationSurfaceObservation,
) -> PrepublicationValidationReceipt:
    """Block every drift before a separate publisher may create a path."""

    if type(candidate) is not PreparationCandidate:
        raise ProtocolError("OE-PPUR v4 prepublication candidate is untyped.")
    _validate_live_predecessor(candidate.plan)
    validate_workspace_snapshot(candidate.plan.workspace, observed_workspace)
    if observed_topology != candidate.plan.topology:
        raise ProtocolError("OE-PPUR v4 execution topology drifted before publication.")
    if observed_scientific != candidate.plan.scientific:
        raise ProtocolError("OE-PPUR v4 scientific seals drifted before publication.")
    if observed_workstation != candidate.plan.workstation:
        raise ProtocolError("OE-PPUR v4 workstation topology drifted before publication.")
    if observed_existing_inputs != candidate.plan.existing_inputs:
        raise ProtocolError("OE-PPUR v4 existing direct inputs drifted.")
    if observed_surfaces != PublicationSurfaceObservation(
        amendment_exists=False,
        amendment_sha256=None,
        output_root_exists=False,
        envelope_exists=False,
        envelope_sha256=None,
        commit_marker_exists=False,
        commit_marker_sha256=None,
        lease_exists=False,
        scratch_root_exists=False,
        scratch_receipts_exist=False,
        topology_receipt_exists=False,
    ):
        raise ProtocolError("OE-PPUR v4 prepublication surfaces are not pristine.")
    validate_authorization_amendment_bytes(
        candidate.amendment_raw,
        plan=candidate.plan,
        terms=candidate.terms,
    )
    validate_final_envelope_bytes(candidate.envelope_raw, expected=candidate.envelope)
    return PrepublicationValidationReceipt(
        candidate_hash=candidate.candidate_hash,
        workspace_snapshot_hash=observed_workspace.snapshot_hash,
        existing_input_inventory_hash=observed_existing_inputs.inventory_hash,
        plan_hash=candidate.plan.plan_hash,
        amendment_sha256=bytes_sha256(candidate.amendment_raw),
        final_envelope_sha256=bytes_sha256(candidate.envelope_raw),
        commit_marker_sha256=bytes_sha256(candidate.commit_marker_raw),
    )


def validate_postpublication(
    candidate: PreparationCandidate,
    prepublication: PrepublicationValidationReceipt,
    *,
    observed_workspace: WorkspaceSnapshot,
    observed_existing_inputs: ExistingInputInventory,
    observed_topology: ExecutionTopologyContract,
    observed_scientific: ScientificSealDescriptor,
    observed_workstation: WorkstationTopologyReceipt,
    observed_surfaces: PublicationSurfaceObservation,
    published_amendment_raw: bytes,
    published_envelope_raw: bytes,
    published_commit_marker_raw: bytes,
) -> PostpublicationValidationReceipt:
    """Validate exact externally published bytes while the lease stays absent."""

    if (
        type(candidate) is not PreparationCandidate
        or type(prepublication) is not PrepublicationValidationReceipt
    ):
        raise ProtocolError("OE-PPUR v4 postpublication lineage is untyped.")
    _validate_live_predecessor(candidate.plan)
    validate_workspace_snapshot(candidate.plan.workspace, observed_workspace)
    if observed_topology != candidate.plan.topology:
        raise ProtocolError("OE-PPUR v4 postpublication topology drifted.")
    if observed_scientific != candidate.plan.scientific:
        raise ProtocolError("OE-PPUR v4 postpublication scientific seals drifted.")
    if observed_workstation != candidate.plan.workstation:
        raise ProtocolError("OE-PPUR v4 postpublication workstation topology drifted.")
    if observed_existing_inputs != candidate.plan.existing_inputs:
        raise ProtocolError("OE-PPUR v4 postpublication inputs drifted.")
    expected_prepublication = _prepublication_receipt(candidate)
    if prepublication != expected_prepublication:
        raise ProtocolError("OE-PPUR v4 postpublication preflight lineage drifted.")
    if (
        published_amendment_raw != candidate.amendment_raw
        or published_envelope_raw != candidate.envelope_raw
        or published_commit_marker_raw != candidate.commit_marker_raw
    ):
        raise ProtocolError("OE-PPUR v4 published preparation bytes drifted.")
    validate_authorization_amendment_bytes(
        published_amendment_raw,
        plan=candidate.plan,
        terms=candidate.terms,
    )
    validate_final_envelope_bytes(
        published_envelope_raw, expected=candidate.envelope
    )
    expected_surface = PublicationSurfaceObservation(
        amendment_exists=True,
        amendment_sha256=bytes_sha256(candidate.amendment_raw),
        output_root_exists=True,
        envelope_exists=True,
        envelope_sha256=bytes_sha256(candidate.envelope_raw),
        commit_marker_exists=True,
        commit_marker_sha256=bytes_sha256(candidate.commit_marker_raw),
        lease_exists=False,
        scratch_root_exists=False,
        scratch_receipts_exist=False,
        topology_receipt_exists=False,
    )
    if observed_surfaces != expected_surface:
        raise ProtocolError("OE-PPUR v4 postpublication surfaces drifted.")
    return PostpublicationValidationReceipt(
        candidate_hash=candidate.candidate_hash,
        prepublication_receipt_hash=prepublication.receipt_hash,
        amendment_sha256=expected_surface.amendment_sha256 or "",
        final_envelope_sha256=expected_surface.envelope_sha256 or "",
        commit_marker_sha256=expected_surface.commit_marker_sha256 or "",
    )


def validate_amendment_only_postpublication(
    candidate: PreparationCandidate,
    prepublication: PrepublicationValidationReceipt,
    *,
    observed_workspace: WorkspaceSnapshot,
    observed_existing_inputs: ExistingInputInventory,
    observed_topology: ExecutionTopologyContract,
    observed_scientific: ScientificSealDescriptor,
    observed_workstation: WorkstationTopologyReceipt,
    observed_surfaces: PublicationSurfaceObservation,
    published_amendment_raw: bytes,
) -> AmendmentPublicationValidationReceipt:
    """Validate issuance of input #7 without rendering or launching v4."""

    if (
        type(candidate) is not PreparationCandidate
        or type(prepublication) is not PrepublicationValidationReceipt
        or prepublication != _prepublication_receipt(candidate)
    ):
        raise ProtocolError("OE-PPUR v4 amendment publication lineage drifted.")
    _validate_live_predecessor(candidate.plan)
    validate_workspace_snapshot(candidate.plan.workspace, observed_workspace)
    if observed_existing_inputs != candidate.plan.existing_inputs:
        raise ProtocolError("OE-PPUR v4 inputs drifted after amendment publication.")
    if observed_topology != candidate.plan.topology:
        raise ProtocolError("OE-PPUR v4 topology drifted after amendment publication.")
    if observed_scientific != candidate.plan.scientific:
        raise ProtocolError("OE-PPUR v4 scientific seals drifted after publication.")
    if observed_workstation != candidate.plan.workstation:
        raise ProtocolError("OE-PPUR v4 workstation topology drifted after publication.")
    if published_amendment_raw != candidate.amendment_raw:
        raise ProtocolError("OE-PPUR v4 published amendment bytes drifted.")
    validate_authorization_amendment_bytes(
        published_amendment_raw,
        plan=candidate.plan,
        terms=candidate.terms,
    )
    expected = PublicationSurfaceObservation(
        amendment_exists=True,
        amendment_sha256=bytes_sha256(candidate.amendment_raw),
        output_root_exists=False,
        envelope_exists=False,
        envelope_sha256=None,
        commit_marker_exists=False,
        commit_marker_sha256=None,
        lease_exists=False,
        scratch_root_exists=False,
        scratch_receipts_exist=False,
        topology_receipt_exists=False,
    )
    if observed_surfaces != expected:
        raise ProtocolError("OE-PPUR v4 amendment-only surfaces drifted.")
    return AmendmentPublicationValidationReceipt(
        candidate_hash=candidate.candidate_hash,
        prepublication_receipt_hash=prepublication.receipt_hash,
        workspace_snapshot_hash=observed_workspace.snapshot_hash,
        existing_input_inventory_hash=observed_existing_inputs.inventory_hash,
        amendment_sha256=expected.amendment_sha256 or "",
    )


def _prepublication_receipt(
    candidate: PreparationCandidate,
) -> PrepublicationValidationReceipt:
    return PrepublicationValidationReceipt(
        candidate_hash=candidate.candidate_hash,
        workspace_snapshot_hash=candidate.plan.workspace.snapshot_hash,
        existing_input_inventory_hash=candidate.plan.existing_inputs.inventory_hash,
        plan_hash=candidate.plan.plan_hash,
        amendment_sha256=bytes_sha256(candidate.amendment_raw),
        final_envelope_sha256=bytes_sha256(candidate.envelope_raw),
        commit_marker_sha256=bytes_sha256(candidate.commit_marker_raw),
    )


def _validate_live_predecessor(plan: PreAmendmentPlan) -> None:
    expected = plan.predecessor
    observed = capture_predecessor_preservation(
        amendment_path=expected.amendment_path,
        output_root=expected.output_root,
        lease_path=expected.lease_path,
        scratch_root=expected.scratch_root,
    )
    if observed != expected:
        raise ProtocolError("OE-PPUR v3 preservation state drifted during v4 preparation.")


def observe_publication_surfaces(plan: PreAmendmentPlan) -> PublicationSurfaceObservation:
    """Observe canonical surfaces without creating or repairing them."""

    if type(plan) is not PreAmendmentPlan:
        raise ProtocolError("OE-PPUR v4 publication observation plan is untyped.")
    topology = plan.topology
    amendment = _optional_regular_file_sha256(topology.amendment_path, "amendment")
    envelope = _optional_regular_file_sha256(topology.envelope_path, "envelope")
    marker = _optional_regular_file_sha256(
        topology.commit_marker_path, "commit marker"
    )
    output_exists = _safe_exists(topology.output_root, expect_directory=True)
    return PublicationSurfaceObservation(
        amendment_exists=amendment is not None,
        amendment_sha256=amendment,
        output_root_exists=output_exists,
        envelope_exists=envelope is not None,
        envelope_sha256=envelope,
        commit_marker_exists=marker is not None,
        commit_marker_sha256=marker,
        lease_exists=_safe_exists(topology.lease_path),
        scratch_root_exists=_safe_exists(topology.scratch_root),
        scratch_receipts_exist=_safe_exists(topology.scratch_receipt_root),
        topology_receipt_exists=_safe_exists(topology.topology_receipt_path),
    )


def _safe_exists(path: Path, *, expect_directory: bool = False) -> bool:
    if not os.path.lexists(path):
        return False
    if path.is_symlink():
        raise ProtocolError("OE-PPUR v4 publication surface is a symlink.")
    if expect_directory and not path.is_dir():
        raise ProtocolError("OE-PPUR v4 output root is not a directory.")
    return True


def _optional_regular_file_sha256(path: Path, role: str) -> str | None:
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"OE-PPUR v4 published {role} is unsafe.")
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 published {role} could not be read.") from exc
    if (
        before.st_size != len(raw)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ProtocolError(f"OE-PPUR v4 published {role} changed while read.")
    return bytes_sha256(raw)


__all__ = (
    "AmendmentPublicationValidationReceipt",
    "PostpublicationValidationReceipt",
    "PreparationCandidate",
    "PrepublicationValidationReceipt",
    "PublicationSurfaceObservation",
    "build_preparation_candidate",
    "observe_publication_surfaces",
    "validate_postpublication",
    "validate_amendment_only_postpublication",
    "validate_prepublication",
)
