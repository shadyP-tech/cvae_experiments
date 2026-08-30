"""Sentinel-normalized resolved-config and input-manifest commitments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...protocol import ProtocolError
from .contracts import ExecutionTopologyContract, ScientificSealDescriptor
from .hashing import bytes_sha256, payload_sha256, pretty_json_bytes, require_sha256
from .inputs import AmendmentInputTemplate, ExistingInputInventory
from .predecessor import PredecessorPreservationWitness
from .snapshot import WorkspaceSnapshot


PLAN_SHA256_SENTINEL = "__OE_PPUR_V4_PRE_AMENDMENT_PLAN_SHA256__"
AMENDMENT_SHA256_SENTINEL = "__OE_PPUR_V4_AUTHORIZATION_AMENDMENT_SHA256__"


@dataclass(frozen=True, slots=True)
class TemplateDescriptor:
    role: str
    path: Path
    template_raw: bytes
    template_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.role) is not str
            or not self.role
            or not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or type(self.template_raw) is not bytes
            or self.template_raw.count(PLAN_SHA256_SENTINEL.encode()) < 1
            or self.template_raw.count(AMENDMENT_SHA256_SENTINEL.encode()) < 1
        ):
            raise ProtocolError("OE-PPUR v4 preparation template drifted.")
        object.__setattr__(self, "template_sha256", bytes_sha256(self.template_raw))

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path.as_posix(),
            "template_sha256": self.template_sha256,
            "template_size_bytes": len(self.template_raw),
            "template_utf8": self.template_raw.decode("utf-8"),
            "plan_sha256_sentinel": PLAN_SHA256_SENTINEL,
            "amendment_sha256_sentinel": AMENDMENT_SHA256_SENTINEL,
        }

    def realize(self, *, plan_sha256: str, amendment_sha256: str) -> bytes:
        plan = require_sha256(plan_sha256, "realized pre-amendment plan")
        amendment = require_sha256(amendment_sha256, "realized amendment")
        raw = self.template_raw.replace(PLAN_SHA256_SENTINEL.encode(), plan.encode())
        raw = raw.replace(AMENDMENT_SHA256_SENTINEL.encode(), amendment.encode())
        if PLAN_SHA256_SENTINEL.encode() in raw or AMENDMENT_SHA256_SENTINEL.encode() in raw:
            raise ProtocolError("OE-PPUR v4 template sentinel realization failed.")
        return raw


@dataclass(frozen=True, slots=True)
class PreparationTemplates:
    resolved_config: TemplateDescriptor
    input_manifest: TemplateDescriptor
    templates_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.resolved_config) is not TemplateDescriptor
            or type(self.input_manifest) is not TemplateDescriptor
            or self.resolved_config.role != "resolved_config"
            or self.input_manifest.role != "input_manifest"
            or self.resolved_config.path == self.input_manifest.path
        ):
            raise ProtocolError("OE-PPUR v4 preparation template topology drifted.")
        object.__setattr__(self, "templates_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_preparation_templates_v1",
            "resolved_config": self.resolved_config.to_payload(),
            "input_manifest": self.input_manifest.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class RealizedPreparationTemplates:
    resolved_config_raw: bytes
    input_manifest_raw: bytes
    resolved_config_sha256: str = field(init=False)
    input_manifest_sha256: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.resolved_config_raw) is not bytes or type(self.input_manifest_raw) is not bytes:
            raise ProtocolError("OE-PPUR v4 realized template bytes are untyped.")
        object.__setattr__(self, "resolved_config_sha256", bytes_sha256(self.resolved_config_raw))
        object.__setattr__(self, "input_manifest_sha256", bytes_sha256(self.input_manifest_raw))
        object.__setattr__(self, "receipt_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_realized_preparation_templates_v1",
            "resolved_config_sha256": self.resolved_config_sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
        }


def build_preparation_templates(
    *,
    workspace: WorkspaceSnapshot,
    existing_inputs: ExistingInputInventory,
    amendment_template: AmendmentInputTemplate,
    topology: ExecutionTopologyContract,
    scientific: ScientificSealDescriptor,
    predecessor: PredecessorPreservationWitness,
) -> PreparationTemplates:
    config = pretty_json_bytes(
        {
            "schema_version": "oe_ppur_v4_workspace_sealed_resolved_config_v1",
            "experiment_id": scientific.experiment_id,
            "output_artifact_id": scientific.output_artifact_id,
            "authorization_state": "AMENDMENT_ISSUED_NO_LAUNCH_AUTHORITY",
            "pre_amendment_plan_sha256": PLAN_SHA256_SENTINEL,
            "authorization_amendment_sha256": AMENDMENT_SHA256_SENTINEL,
            "workspace_snapshot_sha256": workspace.snapshot_hash,
            "existing_input_inventory_sha256": existing_inputs.inventory_hash,
            "amendment_input_template_sha256": amendment_template.template_hash,
            "execution_topology_sha256": topology.contract_hash,
            "scientific_seals_sha256": scientific.descriptor_hash,
            "predecessor_preservation_witness_sha256": predecessor.witness_hash,
            "artifact_root": topology.output_root.as_posix(),
            "scratch_root": topology.scratch_root.as_posix(),
            "launch_authorized": False,
            "authorization_consumed": False,
            "target_labels_opened": False,
        }
    )
    manifest = pretty_json_bytes(
        {
            "schema_version": "oe_ppur_v4_workspace_sealed_input_manifest_v1",
            "experiment_id": scientific.experiment_id,
            "pre_amendment_plan_sha256": PLAN_SHA256_SENTINEL,
            "authorization_amendment_sha256": AMENDMENT_SHA256_SENTINEL,
            "existing_inputs": existing_inputs.to_payload(),
            "amendment_input_template": amendment_template.to_payload(),
            "direct_input_count": 7,
            "v3_amendment_used_as_input": False,
            "v3_authority_inherited": False,
            "target_labels_opened": False,
        }
    )
    return PreparationTemplates(
        resolved_config=TemplateDescriptor("resolved_config", topology.resolved_config_path, config),
        input_manifest=TemplateDescriptor("input_manifest", topology.input_manifest_path, manifest),
    )


def realize_preparation_templates(
    templates: PreparationTemplates,
    *,
    plan_sha256: str,
    amendment_sha256: str,
) -> RealizedPreparationTemplates:
    if type(templates) is not PreparationTemplates:
        raise ProtocolError("OE-PPUR v4 preparation templates are untyped.")
    return RealizedPreparationTemplates(
        resolved_config_raw=templates.resolved_config.realize(
            plan_sha256=plan_sha256, amendment_sha256=amendment_sha256
        ),
        input_manifest_raw=templates.input_manifest.realize(
            plan_sha256=plan_sha256, amendment_sha256=amendment_sha256
        ),
    )


__all__ = (
    "AMENDMENT_SHA256_SENTINEL",
    "PLAN_SHA256_SENTINEL",
    "PreparationTemplates",
    "RealizedPreparationTemplates",
    "TemplateDescriptor",
    "build_preparation_templates",
    "realize_preparation_templates",
)
