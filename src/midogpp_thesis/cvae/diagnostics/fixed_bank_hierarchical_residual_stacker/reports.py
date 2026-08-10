"""Protocol, leakage, publication, and runner-state reports."""

from __future__ import annotations

from typing import Mapping

from .experiment_contracts import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_hierarchical_residual_stacker_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "protocol": dict(getattr(config, "protocol")),
        "probability_surface": dict(getattr(config, "probability_surface")),
        "features": dict(getattr(config, "features")),
        "hierarchical_model": dict(getattr(config, "hierarchical_model")),
        "target_support": dict(getattr(config, "target_support")),
        "stacker": dict(getattr(config, "stacker")),
        "controls": dict(getattr(config, "controls")),
        "evaluation": dict(getattr(config, "evaluation")),
        "claim_boundary": dict(getattr(config, "claim_boundary")),
        "support_selection_objective": "fixed_class_balanced_log_loss_only",
        "terminal_metric": "pooled_exact_bacc",
        "uncertainty_unit": "paired_whole_case_cluster",
        "soft_class_gate": True,
        "soft_composition": (
            "delta=(1-p_B_cal)*sum(alpha0*r)+p_B_cal*sum(alpha1*r)"
        ),
        "single_class_cases_retained": True,
        "per_case_bacc_defined_or_persisted": False,
        "predictions_and_features_sealed_before_any_label": True,
        "fresh_evidence": False,
        "terminal_consumed_test_diagnostic_only": True,
        "prior_stage90_output_or_scratch_consumed": False,
    }


def leakage_report_payload(
    *,
    prediction_seal_hash: str,
    feature_seal_hash: str,
    model_count: int,
    decision_count: int,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_hierarchical_residual_stacker_leakage_report_v1",
        "status": "PASS",
        "global_prediction_seal_hash": prediction_seal_hash,
        "label_free_feature_seal_hash": feature_seal_hash,
        "loco_target_model_family_count": model_count,
        "fold_method_decision_count": decision_count,
        "probabilities_and_case_features_sealed_before_any_label": True,
        "outer_H_labels_used_in_shared_models": False,
        "held_query_or_candidate_source_leakage": False,
        "all_R_and_P_models_sealed_before_same_H_support": True,
        "whole_case_support_evaluation_overlap_count": 0,
        "evaluation_labels_opened_after_all_method_and_permutation_seals": (
            capability_report.get("evaluation_labels_opened") is True
        ),
        "support_objective": "fixed_class_balanced_log_loss_only",
        "exact_bacc_used_for_grid_selection": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "shared_model_updated_with_target_labels": False,
        "metadata_artifact_used": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
        "evaluation_labels_used_for_decisions": False,
        "raw_labels_persisted": False,
        "per_case_bacc_used": False,
    }


def publication_decision_payload(evaluation: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_hierarchical_residual_stacker_publication_v1",
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "claim_role": (
            "known_fixed_bank_label_aware_case_oof_stacking_mechanism_diagnostic"
        ),
        "scientific_result_hash": evaluation.get("scientific_result_hash"),
        "method_ids": ["B", "B_cal", "G", "R", "P"],
        "fresh_evidence": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_hierarchical_residual_stacker_run_state_v1",
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
