"""Deterministic amendment content that binds an OE-PPUR v4 plan.

The functions here return bytes.  They never create, replace, or inspect the
canonical amendment file and cannot consume an execution lease.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping

from ...protocol import ProtocolError
from .hashing import bytes_sha256, pretty_json_bytes, require_nonempty_text
from .plan import PreAmendmentPlan


@dataclass(frozen=True, slots=True)
class AuthorizationTerms:
    authorization_basis: str
    authorized_by: str
    status: str = "AMENDMENT_ISSUED_SINGLE_USE_NOT_LAUNCH_AUTHORIZED"
    authorized_run_count: int = 1
    execution_authorized: bool = False
    consumed_test_reuse_authorized: bool = True
    separate_launch_authority_required: bool = True
    cross_run_recovery_allowed: bool = False

    def __post_init__(self) -> None:
        for role in ("authorization_basis", "authorized_by"):
            object.__setattr__(
                self, role, require_nonempty_text(getattr(self, role), role)
            )
        if (
            self.status != "AMENDMENT_ISSUED_SINGLE_USE_NOT_LAUNCH_AUTHORIZED"
            or type(self.authorized_run_count) is not int
            or self.authorized_run_count != 1
            or type(self.execution_authorized) is not bool
            or self.execution_authorized
            or type(self.consumed_test_reuse_authorized) is not bool
            or not self.consumed_test_reuse_authorized
            or type(self.separate_launch_authority_required) is not bool
            or not self.separate_launch_authority_required
            or type(self.cross_run_recovery_allowed) is not bool
            or self.cross_run_recovery_allowed
        ):
            raise ProtocolError("OE-PPUR v4 authorization terms drifted.")


def build_amendment_payload(
    plan: PreAmendmentPlan,
    terms: AuthorizationTerms,
) -> dict[str, object]:
    if type(plan) is not PreAmendmentPlan or type(terms) is not AuthorizationTerms:
        raise ProtocolError("OE-PPUR v4 amendment construction is untyped.")
    scientific = plan.scientific
    ordered_ids = [row.artifact_id for row in plan.existing_inputs.rows]
    ordered_ids.append(plan.amendment_template.artifact_id)
    return {
        "schema_version": "oe_ppur_v4_workspace_sealed_authorization_amendment_v1",
        "status": terms.status,
        "authorization_basis": terms.authorization_basis,
        "authorized_by": terms.authorized_by,
        "consumer_experiment_id": scientific.experiment_id,
        "consumer_output_artifact_id": scientific.output_artifact_id,
        "amendment_artifact_id": scientific.amendment_artifact_id,
        "authorized_run_count": 1,
        "execution_authorized": False,
        "separate_launch_authority_required": True,
        "consumed_test_reuse_authorized": True,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "cross_run_recovery_allowed": False,
        "pre_amendment_plan_sha256": plan.plan_hash,
        "amendment_input_template_sha256": plan.amendment_template.template_hash,
        "workspace_snapshot_sha256": plan.workspace.snapshot_hash,
        "existing_input_inventory_sha256": plan.existing_inputs.inventory_hash,
        "execution_topology_sha256": plan.topology.contract_hash,
        "scientific_seals_sha256": scientific.descriptor_hash,
        "predecessor_preservation_witness_sha256": plan.predecessor.witness_hash,
        "preparation_templates_sha256": plan.templates.templates_hash,
        "resolved_config_template_sha256": plan.templates.resolved_config.template_sha256,
        "input_manifest_template_sha256": plan.templates.input_manifest.template_sha256,
        "source_content_reuse_exception_sha256": plan.source_reuse_exception.exception_hash,
        "source_content_reuse_exception_scope": plan.source_reuse_exception.exception_scope,
        "workstation_topology_sha256": plan.workstation.receipt_hash,
        "preserved_v3_amendment_sha256": plan.predecessor.amendment_sha256,
        "v3_amendment_is_preservation_witness_only": True,
        "source_seal_sha256": scientific.source_seal_sha256,
        "protocol_seal_sha256": scientific.protocol_seal_sha256,
        "scientific_seal_sha256": scientific.scientific_seal_sha256,
        "lifecycle_seal_sha256": scientific.lifecycle_seal_sha256,
        "direct_input_artifact_ids": ordered_ids,
        "direct_input_count": 7,
        "target_labels_open_only_after_durable_preterminal_attestation": True,
        "predecessor_source_content_alias_used": True,
        "previous_stage90_operational_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_run_state_or_scratch_used": False,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
        "publication_status": scientific.publication_status,
        "terminal_decision": scientific.terminal_decision,
    }


def authorization_amendment_bytes(
    plan: PreAmendmentPlan,
    terms: AuthorizationTerms,
) -> bytes:
    return pretty_json_bytes(build_amendment_payload(plan, terms))


def authorization_amendment_sha256(
    plan: PreAmendmentPlan,
    terms: AuthorizationTerms,
) -> str:
    return bytes_sha256(authorization_amendment_bytes(plan, terms))


def validate_authorization_amendment_bytes(
    raw: bytes,
    *,
    plan: PreAmendmentPlan,
    terms: AuthorizationTerms,
) -> Mapping[str, object]:
    if type(raw) is not bytes:
        raise ProtocolError("OE-PPUR v4 amendment bytes are untyped.")
    expected = authorization_amendment_bytes(plan, terms)
    if raw != expected:
        raise ProtocolError("OE-PPUR v4 authorization amendment bytes drifted.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ProtocolError("OE-PPUR v4 authorization amendment is unreadable.") from exc
    if not isinstance(payload, dict) or payload != build_amendment_payload(plan, terms):
        raise ProtocolError("OE-PPUR v4 authorization amendment payload drifted.")
    return payload


__all__ = (
    "AuthorizationTerms",
    "authorization_amendment_bytes",
    "authorization_amendment_sha256",
    "build_amendment_payload",
    "validate_authorization_amendment_bytes",
)
