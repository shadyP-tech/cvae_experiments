"""Non-circular pre-amendment plan for OE-PPUR v4."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .contracts import ExecutionTopologyContract, ScientificSealDescriptor
from .hashing import payload_sha256
from .host import WorkstationTopologyReceipt
from .inputs import AmendmentInputTemplate, ExistingInputInventory
from .predecessor import PredecessorPreservationWitness
from .snapshot import WorkspaceSnapshot
from .source_reuse import SourceContentReuseException
from .templates import PreparationTemplates


@dataclass(frozen=True, slots=True)
class PreAmendmentPlan:
    """First commitment level: all inputs except the amendment's own bytes."""

    workspace: WorkspaceSnapshot
    existing_inputs: ExistingInputInventory
    amendment_template: AmendmentInputTemplate
    topology: ExecutionTopologyContract
    scientific: ScientificSealDescriptor
    predecessor: PredecessorPreservationWitness
    templates: PreparationTemplates
    source_reuse_exception: SourceContentReuseException
    workstation: WorkstationTopologyReceipt
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.workspace) is not WorkspaceSnapshot
            or type(self.existing_inputs) is not ExistingInputInventory
            or type(self.amendment_template) is not AmendmentInputTemplate
            or type(self.topology) is not ExecutionTopologyContract
            or type(self.scientific) is not ScientificSealDescriptor
            or type(self.predecessor) is not PredecessorPreservationWitness
            or type(self.templates) is not PreparationTemplates
            or type(self.source_reuse_exception) is not SourceContentReuseException
            or type(self.workstation) is not WorkstationTopologyReceipt
        ):
            raise ProtocolError("OE-PPUR v4 pre-amendment plan is untyped.")
        if (
            self.workspace.repository_root != self.topology.repository_root
            or self.workspace.helper.path != self.topology.helper_path
            or self.workspace.exclusions != self.topology.workspace_exclusions()
            or self.amendment_template.artifact_id
            != self.scientific.amendment_artifact_id
            or self.amendment_template.member_path != self.topology.amendment_path
            or self.templates.resolved_config.path != self.topology.resolved_config_path
            or self.templates.input_manifest.path != self.topology.input_manifest_path
            or self.topology.host_id != self.workstation.hostname
            or self.scientific.output_artifact_id
            in {row.artifact_id for row in self.existing_inputs.rows}
            or self.scientific.amendment_artifact_id
            in {row.artifact_id for row in self.existing_inputs.rows}
            or any(
                "amendment" in value.lower()
                for row in self.existing_inputs.rows
                for value in (row.role, row.kind, row.artifact_id)
            )
        ):
            raise ProtocolError("OE-PPUR v4 pre-amendment lineage drifted.")
        object.__setattr__(self, "plan_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_pre_amendment_plan_v1",
            "workspace_snapshot": self.workspace.to_payload(),
            "workspace_snapshot_sha256": self.workspace.snapshot_hash,
            "existing_input_inventory": self.existing_inputs.to_payload(),
            "existing_input_inventory_sha256": self.existing_inputs.inventory_hash,
            "amendment_input_template": self.amendment_template.to_payload(),
            "amendment_input_template_sha256": self.amendment_template.template_hash,
            "execution_topology": self.topology.to_payload(),
            "execution_topology_sha256": self.topology.contract_hash,
            "scientific_seals": self.scientific.to_payload(),
            "scientific_seals_sha256": self.scientific.descriptor_hash,
            "predecessor_preservation_witness": self.predecessor.to_payload(),
            "predecessor_preservation_witness_sha256": self.predecessor.witness_hash,
            "preparation_templates": self.templates.to_payload(),
            "preparation_templates_sha256": self.templates.templates_hash,
            "source_content_reuse_exception": self.source_reuse_exception.to_payload(),
            "source_content_reuse_exception_sha256": self.source_reuse_exception.exception_hash,
            "workstation_topology": self.workstation.to_payload(),
            "workstation_topology_sha256": self.workstation.receipt_hash,
            "amendment_sha256": None,
            "amendment_issued": False,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }


def build_pre_amendment_plan(
    *,
    workspace: WorkspaceSnapshot,
    existing_inputs: ExistingInputInventory,
    amendment_template: AmendmentInputTemplate,
    topology: ExecutionTopologyContract,
    scientific: ScientificSealDescriptor,
    predecessor: PredecessorPreservationWitness,
    templates: PreparationTemplates,
    source_reuse_exception: SourceContentReuseException,
    workstation: WorkstationTopologyReceipt,
) -> PreAmendmentPlan:
    return PreAmendmentPlan(
        workspace=workspace,
        existing_inputs=existing_inputs,
        amendment_template=amendment_template,
        topology=topology,
        scientific=scientific,
        predecessor=predecessor,
        templates=templates,
        source_reuse_exception=source_reuse_exception,
        workstation=workstation,
    )


__all__ = ("PreAmendmentPlan", "build_pre_amendment_plan")
