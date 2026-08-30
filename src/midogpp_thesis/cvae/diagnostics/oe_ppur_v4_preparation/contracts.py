"""Typed, path-bearing contracts for mutation-free OE-PPUR v4 preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import payload_sha256, require_nonempty_text, require_sha256


CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
TOPOLOGY_MODE = "NFS_SAFE_IN_PLACE_COMMIT"


def _absolute_normal_path(value: Path, role: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ProtocolError(f"OE-PPUR v4 {role} must be an absolute Path.")
    normalized = Path(os.path.normpath(value.as_posix()))
    if value != normalized or ".." in value.parts:
        raise ProtocolError(f"OE-PPUR v4 {role} is not lexically canonical.")
    return value


def _strict_descendant(child: Path, parent: Path) -> bool:
    return child != parent and child.is_relative_to(parent)


@dataclass(frozen=True, slots=True)
class ScientificSealDescriptor:
    """Caller-supplied scientific identity; no scientific module is imported."""

    experiment_id: str
    output_artifact_id: str
    amendment_artifact_id: str
    dataset_family: str
    claim_dataset_family: str
    claim_scope: str
    publication_status: str
    terminal_decision: str
    source_seal_sha256: str
    protocol_seal_sha256: str
    scientific_seal_sha256: str
    lifecycle_seal_sha256: str
    selection_uses_target_labels: bool = False
    fresh_evidence: bool = False
    may_feed_another_experiment: bool = False
    descriptor_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for role in (
            "experiment_id",
            "output_artifact_id",
            "amendment_artifact_id",
            "dataset_family",
            "claim_dataset_family",
        ):
            object.__setattr__(
                self, role, require_nonempty_text(getattr(self, role), role)
            )
        for role in (
            "source_seal_sha256",
            "protocol_seal_sha256",
            "scientific_seal_sha256",
            "lifecycle_seal_sha256",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        if (
            self.dataset_family != self.claim_dataset_family
            or self.claim_scope != CLAIM_SCOPE
            or self.publication_status != PUBLICATION_STATUS
            or self.terminal_decision != TERMINAL_DECISION
            or type(self.selection_uses_target_labels) is not bool
            or type(self.fresh_evidence) is not bool
            or type(self.may_feed_another_experiment) is not bool
            or self.selection_uses_target_labels
            or self.fresh_evidence
            or self.may_feed_another_experiment
        ):
            raise ProtocolError("OE-PPUR v4 scientific claim boundary drifted.")
        object.__setattr__(self, "descriptor_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_scientific_seals_v1",
            "experiment_id": self.experiment_id,
            "output_artifact_id": self.output_artifact_id,
            "amendment_artifact_id": self.amendment_artifact_id,
            "dataset_family": self.dataset_family,
            "claim_dataset_family": self.claim_dataset_family,
            "claim_scope": self.claim_scope,
            "publication_status": self.publication_status,
            "terminal_decision": self.terminal_decision,
            "source_seal_sha256": self.source_seal_sha256,
            "protocol_seal_sha256": self.protocol_seal_sha256,
            "scientific_seal_sha256": self.scientific_seal_sha256,
            "lifecycle_seal_sha256": self.lifecycle_seal_sha256,
            "selection_uses_target_labels": False,
            "fresh_evidence": False,
            "may_feed_another_experiment": False,
        }


@dataclass(frozen=True, slots=True)
class ExcludedWorkspaceSurface:
    role: str
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", require_nonempty_text(self.role, "surface role"))
        object.__setattr__(
            self, "path", _absolute_normal_path(self.path, f"{self.role} surface")
        )

    def to_payload(self) -> dict[str, str]:
        return {"role": self.role, "path": self.path.as_posix()}


@dataclass(frozen=True, slots=True)
class ExecutionTopologyContract:
    """Exact NFS-safe publication topology, without publication side effects."""

    host_id: str
    mode: str
    repository_root: Path
    canonical_output_parent: Path
    output_root: Path
    resolved_config_path: Path
    input_manifest_path: Path
    envelope_path: Path
    commit_marker_path: Path
    amendment_path: Path
    lease_path: Path
    scratch_root: Path
    scratch_receipt_root: Path
    topology_receipt_path: Path
    helper_path: Path
    commit_protocol: tuple[str, ...]
    contract_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_id", require_nonempty_text(self.host_id, "host id"))
        if self.mode != TOPOLOGY_MODE:
            raise ProtocolError("OE-PPUR v4 execution topology mode drifted.")
        path_roles = (
            "repository_root",
            "canonical_output_parent",
            "output_root",
            "resolved_config_path",
            "input_manifest_path",
            "envelope_path",
            "commit_marker_path",
            "amendment_path",
            "lease_path",
            "scratch_root",
            "scratch_receipt_root",
            "topology_receipt_path",
            "helper_path",
        )
        for role in path_roles:
            object.__setattr__(
                self,
                role,
                _absolute_normal_path(getattr(self, role), role.replace("_", " ")),
            )
        protocol = tuple(self.commit_protocol)
        if (
            type(self.commit_protocol) is not tuple
            or protocol
            != (
                "EXCLUSIVE_FINAL_ROOT",
                "O_EXCL_MEMBERS",
                "COMMIT_MARKER_LAST",
            )
        ):
            raise ProtocolError("OE-PPUR v4 commit protocol drifted.")
        if (
            not _strict_descendant(self.output_root, self.canonical_output_parent)
            or not _strict_descendant(self.resolved_config_path, self.output_root)
            or not _strict_descendant(self.input_manifest_path, self.output_root)
            or not _strict_descendant(self.envelope_path, self.output_root)
            or not _strict_descendant(self.commit_marker_path, self.output_root)
            or len(
                {
                    self.resolved_config_path,
                    self.input_manifest_path,
                    self.envelope_path,
                    self.commit_marker_path,
                }
            )
            != 4
            or not _strict_descendant(self.lease_path, self.canonical_output_parent)
            or self.output_root == self.lease_path
            or self.output_root.is_relative_to(self.lease_path)
            or self.lease_path.is_relative_to(self.output_root)
            or not _strict_descendant(self.scratch_receipt_root, self.scratch_root)
            or not _strict_descendant(
                self.topology_receipt_path, self.scratch_receipt_root
            )
            or self.repository_root.is_relative_to(self.scratch_root)
            or self.scratch_root.is_relative_to(self.repository_root)
            or self.canonical_output_parent.is_relative_to(self.scratch_root)
            or self.scratch_root.is_relative_to(self.canonical_output_parent)
        ):
            raise ProtocolError("OE-PPUR v4 execution topology paths drifted.")
        excluded = self.workspace_exclusions()
        if tuple(row.role for row in excluded) != (
            "amendment",
            "output",
            "lease",
            "scratch_receipts",
        ):
            raise ProtocolError("OE-PPUR v4 workspace exclusion topology drifted.")
        object.__setattr__(self, "contract_hash", payload_sha256(self.to_payload()))

    def workspace_exclusions(self) -> tuple[ExcludedWorkspaceSurface, ...]:
        """The sole surfaces omitted from the repository-status commitment."""

        return (
            ExcludedWorkspaceSurface("amendment", self.amendment_path),
            ExcludedWorkspaceSurface("output", self.output_root),
            ExcludedWorkspaceSurface("lease", self.lease_path),
            ExcludedWorkspaceSurface("scratch_receipts", self.scratch_receipt_root),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_execution_topology_v1",
            "host_id": self.host_id,
            "mode": self.mode,
            "repository_root": self.repository_root.as_posix(),
            "canonical_output_parent": self.canonical_output_parent.as_posix(),
            "output_root": self.output_root.as_posix(),
            "resolved_config_path": self.resolved_config_path.as_posix(),
            "input_manifest_path": self.input_manifest_path.as_posix(),
            "envelope_path": self.envelope_path.as_posix(),
            "commit_marker_path": self.commit_marker_path.as_posix(),
            "amendment_path": self.amendment_path.as_posix(),
            "lease_path": self.lease_path.as_posix(),
            "scratch_root": self.scratch_root.as_posix(),
            "scratch_receipt_root": self.scratch_receipt_root.as_posix(),
            "topology_receipt_path": self.topology_receipt_path.as_posix(),
            "helper_path": self.helper_path.as_posix(),
            "commit_protocol": list(self.commit_protocol),
            "workspace_exclusions": [
                row.to_payload() for row in self.workspace_exclusions()
            ],
            "publication_performed": False,
            "target_labels_opened": False,
        }


__all__ = (
    "CLAIM_SCOPE",
    "PUBLICATION_STATUS",
    "TERMINAL_DECISION",
    "TOPOLOGY_MODE",
    "ExcludedWorkspaceSurface",
    "ExecutionTopologyContract",
    "ScientificSealDescriptor",
)
