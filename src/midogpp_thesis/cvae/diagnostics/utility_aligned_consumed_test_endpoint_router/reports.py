"""Claim-bound reports for the consumed-test endpoint-router diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence

from ...routing.residual_topup.hashing import canonical_sha256
from .experiment_contracts import (
    CLAIM_ROLE,
    DATASET_FAMILY,
    DEVELOPMENT_RESPONSE_COUNT,
    EXPERIMENT_ID,
    PRIMARY_CONTRASTS,
    PUBLICATION_STATUS,
    ROUTING_STATUS,
    STAGE_ID,
)
from .protocol import ConsumedTestEndpointRouterProtocol


def protocol_manifest_payload(
    config: object,
    *,
    protocol: ConsumedTestEndpointRouterProtocol,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    manifest_admission_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    unhashed = {
        "schema_version": (
            "midogpp_utility_aligned_consumed_test_endpoint_router_protocol_manifest_v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "publication_status": PUBLICATION_STATUS,
        "claim_role": CLAIM_ROLE,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "signed_protocol": protocol.to_payload(),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "manifest_admission_hash": manifest_admission_hash,
        "pre_gpu_firewall": dict(firewall),
        "input_artifact_count": len(getattr(config, "input_artifact_ids")),
        "global_source_control_provenance": "experiment_manifest_only",
        "computed_descriptive_cvae_diagnostics": [
            "posterior_mean_common_space_reconstruction_mse_surrogate",
            "latent_dimension_normalized_analytic_ps_kl",
            "linear_kernel_squared_mean_embedding_discrepancy",
        ],
        "exact_nelbo_computed": False,
        "reconstruction_kl_or_mmd_enter_learned_router": False,
        "learned_router_predictors": [
            "experiment_manifest_metadata_global_source_control",
            "unsigned_ensemble_first_support_action_probability_shift",
        ],
        "support_probability_shift_is_generative_compatibility": False,
        "Hxe_utility_action_semantics": "equal_union_B_plus_single_source_tail",
        "domain_mapping_member_shares_test_manifest_artifact_id": True,
        "support_labels_used": False,
        "development_prediction_seal_precedes_cross_center_development_labels": True,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal": True,
        "target_actions_are_static_per_center": True,
        "case_level_routing_used": False,
        "previous_stage90_output_or_amendment_consumed": False,
        "terminal_consumed_test_diagnostic_only": True,
    }
    return {**unhashed, "protocol_manifest_hash": canonical_sha256(unhashed)}


def development_label_access_report_payload(
    *, development_prediction_seal_hash: str,
    outer_target_ids: Sequence[str],
) -> dict[str, object]:
    unhashed = {
        "schema_version": (
            "midogpp_consumed_test_endpoint_router_development_label_access_v1"
        ),
        "status": "OPEN_AFTER_DEVELOPMENT_PREDICTION_SEAL",
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "outer_target_ids": list(outer_target_ids),
        "cross_center_evaluation_labels_used_as_development_q_labels": True,
        "same_outer_H_labels_excluded_from_each_H_model": True,
        "support_labels_opened": False,
        "development_response_count": DEVELOPMENT_RESPONSE_COUNT,
        "strict_H_q_e_exclusion": True,
        "raw_labels_persisted": False,
    }
    return {**unhashed, "report_hash": canonical_sha256(unhashed)}


def leakage_report_payload(**bindings: object) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_consumed_test_endpoint_router_leakage_report_v1"
        ),
        "status": "PASS",
        **bindings,
        "input_artifact_count": 6,
        "whole_case_support_evaluation_disjoint": True,
        "all_218_cases_partitioned_once": True,
        "support_labels_used": False,
        "strict_H_q_e_exclusion": True,
        "development_predictions_sealed_before_cross_center_labels": True,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal": True,
        "target_actions_are_static_per_center": True,
        "case_level_routing_used": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "target_labels_update_shared_model": False,
        "exact_nelbo_computed_or_claimed": False,
        "reconstruction_kl_or_mmd_enter_learned_router": False,
        "support_probability_shift_is_unsigned_classifier_sensitivity": True,
        "Hxe_is_hybrid_B_plus_single_source_tail": True,
        "previous_stage90_outputs_or_amendments_used": False,
        "stage50_stage60_or_stage70_results_used": False,
        "policy_or_action_update_after_same_outer_H_labels": False,
        "raw_labels_persisted": False,
    }


def runtime_summary_payload(
    preflight: Mapping[str, object],
    *,
    counts: Mapping[str, int],
    source_cache_staging: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_consumed_test_endpoint_router_runtime_summary_v1"
        ),
        "workstation_preflight": dict(preflight),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "source_cache_staging": dict(source_cache_staging),
        "generation_devices": ["cuda:0", "cuda:1"],
        "generation_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "array_storage_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "hash_validated_resume": True,
    }


def publication_decision_payload(
    terminal_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_consumed_test_endpoint_router_publication_decision_v1"
        ),
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": PUBLICATION_STATUS,
        "routing_status": ROUTING_STATUS,
        "claim_role": CLAIM_ROLE,
        "terminal_summary_hash": canonical_sha256(dict(terminal_summary)),
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "actual_learned_predictors": [
            "experiment_manifest_metadata_global_source_control",
            "unsigned_ensemble_first_support_action_probability_shift",
        ],
        "descriptive_cvae_diagnostics_are_not_nelbo_or_utility": True,
        "support_probability_shift_is_not_generative_compatibility": True,
        "Hxe_and_R_are_B_plus_tail_actions_not_standalone_expert_utility": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "model_update_authorized": False,
        "expert_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "generic_consumer_authorized": False,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    if status not in {"RUNNING", "FAILED", "COMPLETE"}:
        raise ValueError("Endpoint-router run state is invalid.")
    return {
        "schema_version": "midogpp_consumed_test_endpoint_router_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
        "promotion_eligible": False,
    }


__all__ = (
    "development_label_access_report_payload",
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "runtime_summary_payload",
)
