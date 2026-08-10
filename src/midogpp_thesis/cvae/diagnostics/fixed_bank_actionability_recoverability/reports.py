"""Claim-bound, exact-payload reports for the terminal diagnostic."""

from __future__ import annotations

from typing import Mapping

from .constants import GEOMETRY_IDS
from .experiment_contracts import CLAIM_ROLE, PUBLICATION_STATUS
from .protocol import ActionabilityRecoverabilityProtocol


def protocol_manifest_payload(
    config: object,
    *,
    protocol: ActionabilityRecoverabilityProtocol,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_fixed_bank_actionability_recoverability_protocol_manifest_v1"
        ),
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol": protocol.to_payload(),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "action_library": dict(getattr(config, "action_library")),
        "recoverability": dict(getattr(config, "recoverability")),
        "controls": dict(getattr(config, "controls")),
        "runtime": dict(getattr(config, "runtime")),
        "evaluation": dict(getattr(config, "evaluation")),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "probability_surface_recomputed_from_original_six_inputs": True,
        "strict_outer_H_and_nested_query_q_exclusion": True,
        "A0_and_A1_are_parallel_nonselectable_arms": True,
        "A1_reuses_exact_A0_rows_and_changes_fit_weights_only": True,
        "support_selector_candidate_set": "U_plus_eight_actions_within_geometry",
        "terminal_oracles_available_before_evaluation_labels": False,
        "single_class_cases_retained": True,
        "per_case_bacc_defined_or_persisted": False,
        "fresh_evidence": False,
        "terminal_consumed_test_diagnostic_only": True,
        "prior_stage90_output_or_scratch_consumed": False,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    feature_seal_hash: str,
    action_library_hash: str,
    model_seal_count: int,
    decision_count: int,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_fixed_bank_actionability_recoverability_leakage_report_v1"
        ),
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "label_free_feature_seal_hash": feature_seal_hash,
        "action_library_hash": action_library_hash,
        "loco_model_seal_count": model_seal_count,
        "pre_evaluation_method_decision_count": decision_count,
        "action_probabilities_and_features_sealed_before_any_label": True,
        "outer_H_labels_used_in_shared_models": False,
        "held_query_q_used_in_its_nested_fit": False,
        "candidate_e_query_rows_used_in_its_model": False,
        "all_G_R_P_models_sealed_before_same_H_support": True,
        "S_y_support_evaluation_whole_case_overlap_count": 0,
        "evaluation_labels_opened_after_all_method_and_permutation_seals": (
            capability_report.get("evaluation_labels_opened") is True
        ),
        "evaluation_labels_used_for_decisions": False,
        "terminal_oracles_used_for_decisions": False,
        "geometry_selected": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "shared_model_updated_with_target_labels": False,
        "metadata_artifact_used": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
        "raw_labels_persisted": False,
        "per_case_bacc_used": False,
    }


def publication_decision_payload(
    sealed_evaluation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_fixed_bank_actionability_recoverability_publication_v1"
        ),
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": PUBLICATION_STATUS,
        "claim_role": CLAIM_ROLE,
        "sealed_result_hash": sealed_evaluation.get("sealed_result_hash"),
        "geometry_ids": list(GEOMETRY_IDS),
        "global_method_ids": ["B"],
        "per_geometry_method_ids": [
            "U",
            "G",
            "R",
            "P",
            "S_y",
            "O_static",
            "O_case",
        ],
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "action_geometry_update_authorized": False,
        "geometry_selection_authorized": False,
        "policy_update_authorized": False,
        "model_update_authorized": False,
        "expert_update_authorized": False,
        "deployment_authorized": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": (
            "midogpp_fixed_bank_actionability_recoverability_run_state_v1"
        ),
        "status": status,
        "phase": phase,
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
    }
    if error is not None:
        payload["error"] = error
    return payload


__all__ = (
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
