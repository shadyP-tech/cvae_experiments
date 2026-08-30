"""Final OE-PPUR v4 envelope binding plan, amendment, and seven inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from collections.abc import Mapping

from ...protocol import ProtocolError
from .amendment import AuthorizationTerms, validate_authorization_amendment_bytes
from .hashing import bytes_sha256, payload_sha256, pretty_json_bytes
from .inputs import SevenInputInventory, build_seven_input_inventory
from .plan import PreAmendmentPlan
from .templates import RealizedPreparationTemplates, realize_preparation_templates


@dataclass(frozen=True, slots=True)
class FinalAuthorizationEnvelope:
    plan: PreAmendmentPlan
    amendment_sha256: str
    inputs: SevenInputInventory
    realized_templates: RealizedPreparationTemplates
    envelope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.plan) is not PreAmendmentPlan
            or type(self.inputs) is not SevenInputInventory
            or type(self.realized_templates) is not RealizedPreparationTemplates
            or self.inputs.rows[:6] != self.plan.existing_inputs.rows
            or self.inputs.rows[6].artifact_id
            != self.plan.amendment_template.artifact_id
            or self.inputs.rows[6].members[0].sha256 != self.amendment_sha256
        ):
            raise ProtocolError("OE-PPUR v4 final envelope topology drifted.")
        object.__setattr__(self, "envelope_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_workspace_sealed_final_envelope_v1",
            "pre_amendment_plan": self.plan.to_payload(),
            "pre_amendment_plan_sha256": self.plan.plan_hash,
            "amendment_input_template_sha256": (
                self.plan.amendment_template.template_hash
            ),
            "authorization_amendment_sha256": self.amendment_sha256,
            "seven_input_inventory": self.inputs.to_payload(),
            "seven_input_inventory_sha256": self.inputs.inventory_hash,
            "workspace_snapshot_sha256": self.plan.workspace.snapshot_hash,
            "execution_topology_sha256": self.plan.topology.contract_hash,
            "scientific_seals_sha256": self.plan.scientific.descriptor_hash,
            "preparation_templates_sha256": self.plan.templates.templates_hash,
            "realized_preparation_templates": self.realized_templates.to_payload(),
            "realized_preparation_templates_sha256": self.realized_templates.receipt_hash,
            "authorization_consumed": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }


def build_final_authorization_envelope(
    plan: PreAmendmentPlan,
    terms: AuthorizationTerms,
    amendment_raw: bytes,
) -> FinalAuthorizationEnvelope:
    validate_authorization_amendment_bytes(amendment_raw, plan=plan, terms=terms)
    inputs = build_seven_input_inventory(
        plan.existing_inputs,
        plan.amendment_template,
        amendment_raw,
    )
    amendment_sha256 = bytes_sha256(amendment_raw)
    realized = realize_preparation_templates(
        plan.templates,
        plan_sha256=plan.plan_hash,
        amendment_sha256=amendment_sha256,
    )
    return FinalAuthorizationEnvelope(
        plan=plan,
        amendment_sha256=amendment_sha256,
        inputs=inputs,
        realized_templates=realized,
    )


def final_envelope_bytes(envelope: FinalAuthorizationEnvelope) -> bytes:
    if type(envelope) is not FinalAuthorizationEnvelope:
        raise ProtocolError("OE-PPUR v4 final envelope is untyped.")
    return pretty_json_bytes(envelope.to_payload())


def commit_marker_bytes(envelope: FinalAuthorizationEnvelope) -> bytes:
    """Canonical marker written last by a separate NFS-safe publisher."""

    if type(envelope) is not FinalAuthorizationEnvelope:
        raise ProtocolError("OE-PPUR v4 commit marker envelope is untyped.")
    return pretty_json_bytes(
        {
            "schema_version": "oe_ppur_v4_preparation_commit_marker_v1",
            "status": "COMMITTED",
            "final_envelope_sha256": bytes_sha256(final_envelope_bytes(envelope)),
            "pre_amendment_plan_sha256": envelope.plan.plan_hash,
            "authorization_amendment_sha256": envelope.amendment_sha256,
            "member_writes_used_o_excl": True,
            "commit_marker_written_last": True,
            "authorization_consumed": True,
            "authorization_exhausted": True,
            "preparation_commit_is_scientific_complete": False,
            "target_labels_opened": False,
            "experiment_launched": False,
        }
    )


def validate_final_envelope_bytes(
    raw: bytes,
    *,
    expected: FinalAuthorizationEnvelope,
) -> Mapping[str, object]:
    if type(raw) is not bytes or type(expected) is not FinalAuthorizationEnvelope:
        raise ProtocolError("OE-PPUR v4 final envelope validation is untyped.")
    canonical = final_envelope_bytes(expected)
    if raw != canonical:
        raise ProtocolError("OE-PPUR v4 final envelope bytes drifted.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ProtocolError("OE-PPUR v4 final envelope is unreadable.") from exc
    if not isinstance(payload, dict) or payload != expected.to_payload():
        raise ProtocolError("OE-PPUR v4 final envelope payload drifted.")
    return payload


__all__ = (
    "FinalAuthorizationEnvelope",
    "build_final_authorization_envelope",
    "commit_marker_bytes",
    "final_envelope_bytes",
    "validate_final_envelope_bytes",
)
