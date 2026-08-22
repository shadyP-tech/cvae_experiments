"""Protocol, leakage, runtime-state, and publication report payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from .constants import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash


def protocol_manifest_payload(
    config: object,
    *,
    protocol_hash: str,
    provenance: Mapping[str, Mapping[str, object]],
    cache_binding_hash: str,
    pre_gpu_firewall: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "fixed_bank_cbpupr_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": str(protocol_hash),
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "input_artifact_hashes": {
            key: canonical_hash(dict(value)) for key, value in provenance.items()
        },
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
    plan_seal_hash: str,
    aggregate_seal_hash: str,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema_version": "fixed_bank_cbpupr_leakage_report_v1",
        "status": "PASS_SCOPED_OWN_ROUTE_NONINTERFERENCE",
        "probability_surface_hash": probability_surface_hash,
        "outer_plan_seal_hash": plan_seal_hash,
        "aggregate_preterminal_seal_hash": aggregate_seal_hash,
        "capability_report_hash": canonical_hash(capability_report),
        "outer_case_labels_excluded_from_own_route": True,
        "posterior_fit_once_per_target_case_control": True,
        "posterior_model_fit_count": EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
        "pseudo_support_reopen_or_refit_used": False,
        "pseudo_posterior_reuses_sealed_J_minus_d_prediction": True,
        "pseudo_outer_H_support_rows_or_labels_enter_posterior_fit_or_normalization": False,
        "pseudo_outer_H_frozen_label_free_expert_fingerprint_covariates_present": True,
        "pseudo_posterior_is_outer_H_covariate_invariant": False,
        "outer_H_and_pseudo_J_excluded_from_actionable_endpoint_source_selection": True,
        "outer_H_and_pseudo_J_excluded_from_donor_calibration": True,
        "donor_calibration_unit": "center",
        "policy_replay_bias_used": False,
        "target_evaluation_labels_used_before_route_seal": False,
        "raw_labels_persisted": False,
        "sample_or_image_paths_persisted": False,
        "full_fitted_endpoint_state_DTOs_persisted": True,
        "full_fitted_posterior_model_DTOs_persisted": True,
        "all_fitted_DTO_outputs_replayed_during_validation": True,
        "optimizer_refits_during_bundle_validation": False,
        "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
        "fresh_evidence": False,
    }
    return {**payload, "leakage_report_hash": canonical_hash(payload)}


def publication_decision_payload(diagnostic_summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cbpupr_publication_decision_v1",
        "status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "diagnostic_summary": dict(diagnostic_summary),
        "bounded_interpretation": (
            "center_balanced_posterior_expected_utility_prefix_sensitivity_"
            "on_consumed_MIDOGpp_test_only"
        ),
        "routing_success_claim_authorized": False,
        "routing_quality_claim_authorized": False,
        "target_performance_claim_authorized": False,
        "nominal_significance_claim_authorized": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "fresh_evidence": False,
    }


def run_state_payload(
    *, status: str, phase: str, error: str | None = None, error_class: str | None = None
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_cbpupr_run_state_v1",
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
