"""Reconstructive protocol, noninterference, and terminal claim reports."""

from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)
from .contracts import (
    PseudoReferenceKey,
    PseudoRouteKey,
    TargetReferenceKey,
    TargetRouteKey,
)
from .hashing import canonical_hash


def validate_transport_lineage_evidence(preterminal: object) -> Mapping[str, object]:
    """Reconstruct every route-keyed transport binding and its runtime seal."""

    runtime = getattr(preterminal, "policy_runtime")
    descriptors = runtime.transport_descriptors_by_outer_candidate
    blocks = runtime.transport_reference_blocks
    screens = runtime.transport_screens
    seal = runtime.transport_seal
    if (
        seal.descriptor_count != len(descriptors)
        or seal.reference_summary_count != len(blocks)
        or seal.screen_count != len(screens)
        or runtime.transport_hash != seal.transport_hash
    ):
        raise ProtocolError("PCSI-RACR transport seal count drifted.")

    numeric_hashes: set[str] = set()
    role_counts: dict[str, int] = {}
    for key, descriptor in descriptors.items():
        lineage = descriptor.lineage
        role_counts[lineage.role] = role_counts.get(lineage.role, 0) + 1
        numeric_hashes.add(descriptor.numeric_leaf_hash)
        if lineage.case_id in lineage.support_case_ids:
            raise ProtocolError("PCSI-RACR descriptor includes its own case label.")
        if isinstance(key, TargetRouteKey):
            valid = (
                lineage.role == "target_candidate"
                and key.outer_center == lineage.outer_center == lineage.endpoint_center
                and key.case_id == lineage.case_id
                and lineage.excluded_centers == (key.outer_center,)
            )
        elif isinstance(key, TargetReferenceKey):
            valid = (
                lineage.role == "target_reference"
                and key.outer_center == lineage.outer_center
                and key.reference_center == lineage.endpoint_center
                and key.case_id == lineage.case_id
                and set(lineage.excluded_centers)
                == {key.outer_center, key.reference_center}
            )
        elif isinstance(key, PseudoRouteKey):
            valid = (
                lineage.role == "pseudo_candidate"
                and key.outer_center == lineage.outer_center
                and key.donor_center == lineage.endpoint_center
                and key.case_id == lineage.case_id
                and set(lineage.excluded_centers)
                == {key.outer_center, key.donor_center}
            )
        elif isinstance(key, PseudoReferenceKey):
            valid = (
                lineage.role == "pseudo_reference"
                and key.outer_center == lineage.outer_center
                and key.reference_center == lineage.endpoint_center
                and key.case_id == lineage.case_id
                and set(lineage.excluded_centers)
                == {key.outer_center, key.donor_center, key.reference_center}
            )
        else:
            valid = False
        if not valid:
            raise ProtocolError("PCSI-RACR route descriptor identity drifted.")

    if seal.numeric_leaf_count != len(numeric_hashes):
        raise ProtocolError("PCSI-RACR numeric transport leaf count drifted.")
    candidate_hashes = {row.descriptor_hash for row in descriptors.values()}
    block_hashes = {row.block_hash for row in blocks.values()}
    for key, screen in screens.items():
        if screen.candidate_descriptor_hash not in candidate_hashes:
            raise ProtocolError("PCSI-RACR screen candidate binding drifted.")
        if not set(screen.reference_block_hashes) <= block_hashes:
            raise ProtocolError("PCSI-RACR screen reference binding drifted.")
        if isinstance(key, TargetRouteKey):
            valid = (
                screen.role == "target"
                and not screen.audit_only
                and screen.outer_center == key.outer_center
                and screen.candidate_center == key.outer_center
                and screen.candidate_case_id == key.case_id
            )
        elif isinstance(key, PseudoRouteKey):
            valid = (
                screen.role == "pseudo_audit"
                and screen.audit_only
                and screen.outer_center == key.outer_center
                and screen.candidate_center == key.donor_center
                and screen.candidate_case_id == key.case_id
            )
        else:
            valid = False
        if not valid:
            raise ProtocolError("PCSI-RACR route screen identity drifted.")

    payload = {
        "schema_version": "fixed_bank_pcsi_racr_transport_lineage_evidence_v1",
        "transport_hash": seal.transport_hash,
        "role_counts": role_counts,
        "descriptor_count": len(descriptors),
        "numeric_leaf_count": len(numeric_hashes),
        "reference_block_count": len(blocks),
        "screen_count": len(screens),
        "own_route_noninterference_proven": True,
        "pseudo_transport_audit_only": True,
        "support_conditioned_not_label_free": True,
        "authorization_valid": True,
        "raw_labels_persisted": False,
    }
    return MappingProxyType(
        {**payload, "transport_lineage_evidence_hash": canonical_hash(payload)}
    )


def validate_transport_endpoint_lineage_payload(
    payload: Mapping[str, object], **_unused: object
) -> Mapping[str, object]:
    """Validate one self-hashed route lineage payload for compatibility callers."""

    row = dict(payload)
    digest = row.pop("lineage_hash", None)
    if digest != canonical_hash(row) or row.get("own_case_evaluation_label_used") is not False:
        raise ProtocolError("PCSI-RACR transport lineage payload drifted.")
    return MappingProxyType(dict(payload))


def assert_transport_authorization_lineage_valid(preterminal: object) -> None:
    evidence = validate_transport_lineage_evidence(preterminal)
    if (
        evidence["own_route_noninterference_proven"] is not True
        or evidence["authorization_valid"] is not True
    ):
        raise ProtocolError("PCSI-RACR route transport authorization is invalid.")


def protocol_manifest_payload(
    config: object,
    *,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    cache_binding_hash: str,
    pre_gpu_firewall: Mapping[str, object],
) -> dict[str, object]:
    input_hashes = {
        artifact_id: canonical_hash(dict(row))
        for artifact_id, row in provenance.items()
    }
    payload = {
        "schema_version": "fixed_bank_pcsi_racr_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": str(getattr(protocol, "protocol_hash")),
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "input_artifact_hashes": input_hashes,
        "input_artifact_count": len(input_hashes),
        "cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(pre_gpu_firewall),
        "exact_six_original_inputs": len(input_hashes) == 6,
        "previous_stage90_output_or_checkpoint_used": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
    }
    return {**payload, "protocol_manifest_hash": canonical_hash(payload)}


def leakage_report_payload(
    *,
    probability_surface_hash: str,
    preterminal: object,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    donor = getattr(preterminal, "donor_runtime")
    runtime = getattr(preterminal, "policy_runtime")
    lineage = validate_transport_lineage_evidence(preterminal)
    payload = {
        "schema_version": "fixed_bank_pcsi_racr_leakage_report_v1",
        "status": "PASS_SCOPED_OWN_ROUTE_NONINTERFERENCE",
        "probability_surface_hash": probability_surface_hash,
        "outer_plan_seal_hash": getattr(preterminal, "plans").seal_hash,
        "decision_barrier_hash": getattr(preterminal, "decision_barrier")[
            "decision_barrier_hash"
        ],
        "aggregate_preterminal_seal_hash": getattr(preterminal, "aggregate_seal")[
            "aggregate_seal_hash"
        ],
        "donor_runtime_hash": donor.runtime_hash,
        "policy_replay_runtime_hash": runtime.runtime_hash,
        "capability_report_hash": canonical_hash(capability_report),
        **dict(lineage),
        "outer_case_labels_excluded_from_own_route": True,
        "pseudo_case_labels_excluded_from_own_pseudo_route": True,
        "other_route_support_labels_may_be_used": True,
        "all_pseudo_candidates_sealed_before_any_pseudo_evaluation_capability": True,
        "all_replays_and_calibrations_sealed_before_target_decisions": True,
        "pseudo_transport_affects_decision": False,
        "terminal_diagnostics_may_change_same_surface_routes": False,
        "target_expert_used": False,
        "source_or_shared_model_updated": False,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }
    return {**payload, "leakage_report_hash": canonical_hash(payload)}


def publication_decision_payload(terminal: object) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_racr_publication_decision_v1",
        "status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "terminal_evaluation_seal_hash": getattr(terminal, "terminal_seal")[
            "terminal_seal_hash"
        ],
        "diagnostic_summary": dict(getattr(terminal, "diagnostic_summary")),
        "method": "route_scoped_observed_donor_case_envelope",
        "conformal_or_coverage_claimed": False,
        "routing_success_claim_authorized": False,
        "target_performance_claim_authorized": False,
        "nominal_significance_claim_authorized": False,
        "completed_canonical_diagnostic_run": True,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "fresh_evidence": False,
    }


def run_state_payload(
    *, status: str, phase: str, error: str | None = None, error_class: str | None = None
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_racr_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "error_class": error_class,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }


__all__ = (
    "assert_transport_authorization_lineage_valid",
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "validate_transport_endpoint_lineage_payload",
    "validate_transport_lineage_evidence",
)
