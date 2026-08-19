"""Reconstructive protocol, leakage, and terminal claim reports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .constants import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash


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
        "schema_version": "fixed_bank_pdcb_protocol_manifest_v1",
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
        "exact_six_original_inputs": True,
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
    payload = {
        "schema_version": "fixed_bank_pdcb_leakage_report_v1",
        "status": "PASS",
        "probability_surface_hash": probability_surface_hash,
        "outer_plan_seal_hash": str(getattr(preterminal, "plans").seal_hash),
        "decision_barrier_hash": str(
            getattr(preterminal, "decision_barrier")["decision_barrier_hash"]
        ),
        "aggregate_preterminal_seal_hash": str(
            getattr(preterminal, "aggregate_seal")["aggregate_seal_hash"]
        ),
        "capability_report_hash": canonical_hash(capability_report),
        "all_physical_probabilities_sealed_before_any_label_access": True,
        "all_218_routes_globally_sealed_before_terminal_labels": True,
        "outer_case_labels_excluded_from_own_route": True,
        "double_exclusion_or_nested_voter_fit_used": False,
        "outer_target_center_labels_excluded_from_all_donor_features": True,
        "donor_feature_prior_scope": "q_not_in_outer_H_or_donor_J_or_e",
        "donor_response_scope": "J_not_equal_outer_H",
        "crossing_features_are_label_free": True,
        "complete_delete_one_donor_bagging_used": True,
        "target_expert_used": False,
        "source_or_shared_model_updated": False,
        "target_evaluation_labels_used_before_route_seal": False,
        "raw_labels_persisted": False,
        "sample_or_image_paths_persisted": False,
        "fresh_evidence": False,
    }
    return {**payload, "leakage_report_hash": canonical_hash(payload)}


def publication_decision_payload(terminal: object) -> dict[str, object]:
    summary = dict(getattr(terminal, "diagnostic_summary"))
    return {
        "schema_version": "fixed_bank_pdcb_publication_decision_v1",
        "status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "terminal_evaluation_seal_hash": str(
            getattr(terminal, "terminal_seal")["terminal_seal_hash"]
        ),
        "diagnostic_summary": summary,
        "unconfirmed_thesis_specific_mechanism_hypothesis": (
            "P-anchored_directional_crossing_helpfulness_with_outer-center-"
            "excluded_donor_training_and_complete_donor-deletion_aggregation"
        ),
        "generic_ensemble_or_calibration_method_novelty_claimed": False,
        "information_gate_is_formal_risk_control": False,
        "routing_success_claim_authorized": False,
        "routing_quality_claim_authorized": False,
        "target_performance_claim_authorized": False,
        "nominal_significance_claim_authorized": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "fresh_evidence": False,
    }


def run_state_payload(
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pdcb_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "error_class": error_class,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
