"""Terminal consumed-only reports for the fixed-bank audit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from ...protocol import ProtocolError
from .experiment_contracts import (
    EXPERIMENT_ID,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_FEATURE_ROW_COUNT,
    EXPECTED_STRICT_TRAINING_ROW_COUNT,
    EXPECTED_SUPPORT_CASE_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    STAGE_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
)
from .serialization import canonical_hash


def protocol_manifest_payload(
    config: object,
    *,
    input_artifact_hashes: Mapping[str, str],
    test_cache_binding_hash: str,
    firewall: Mapping[str, object],
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_protocol_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "split": "test",
        "stage": STAGE_ID,
        "claim_scope": "diagnostic_only",
        "publication_status": PUBLICATION_STATUS,
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "input_artifact_hashes": dict(input_artifact_hashes),
        "test_cache_binding_hash": test_cache_binding_hash,
        "pre_gpu_firewall": dict(firewall),
        "parent_ledger_alias_artifact_id": TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
        "ledger_amendment_artifact_id": LEDGER_AMENDMENT_ARTIFACT_ID,
        "ledger_amendment_hash_chained": True,
        "ledger_consumer_whitelist": [EXPERIMENT_ID],
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
        "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
        "primary_response": "exact_nine_probability_ensemble_bacc_delta",
        "smooth_response": "exact_nine_probability_ensemble_soft_bacc_delta",
        "smooth_response_role": "wholly_separate_postseal_descriptive_only",
        "response_count": EXPECTED_FEATURE_ROW_COUNT,
        "strict_all_role_H_q_exclusion": True,
        "strict_training_row_count": EXPECTED_STRICT_TRAINING_ROW_COUNT,
        "candidate_e_history_retained_when_H_q_absent": True,
        "candidate_pool_excludes_H_and_q": True,
        "known_fixed_bank_reuse": True,
        "unseen_expert_transfer": False,
        "support_labels_used": False,
        "prediction_and_features_sealed_before_test_labels": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "prior_stage90_output_consumed": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
    }
    return {**unhashed, "protocol_manifest_hash": canonical_hash(unhashed)}


def prelabel_phase_payload(
    *,
    config_contract_hash: str,
    source_cache_lock_hash: str,
    development_prediction_seal_hash: str,
    feature_lock_hash: str,
) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_prelabel_phase_v1",
        "status": "SEALED_BEFORE_TEST_LABEL_ACCESS",
        "config_contract_hash": config_contract_hash,
        "source_cache_lock_hash": source_cache_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "fixed_bank_feature_lock_hash": feature_lock_hash,
        "candidate_feature_row_count": EXPECTED_FEATURE_ROW_COUNT,
        "support_probabilities_only_for_features": True,
        "evaluation_probabilities_used_as_features": False,
        "test_labels_opened": False,
        "support_labels_opened": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
    }
    return {**unhashed, "phase_hash": canonical_hash(unhashed)}


def leakage_report_payload(
    *,
    support_partition_lock_hash: str,
    development_prediction_seal_hash: str,
    feature_lock_hash: str,
    response_lock_hash: str,
    exact_crossfit_lock_hash: str,
    smooth_crossfit_lock_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_fixed_bank_leakage_report_v1",
        "status": "PASS",
        "support_partition_lock_hash": support_partition_lock_hash,
        "development_prediction_seal_hash": development_prediction_seal_hash,
        "fixed_bank_feature_lock_hash": feature_lock_hash,
        "fixed_bank_response_lock_hash": response_lock_hash,
        "exact_crossfit_lock_hash": exact_crossfit_lock_hash,
        "smooth_descriptive_crossfit_lock_hash": smooth_crossfit_lock_hash,
        "exact_response_is_primary": True,
        "smooth_response_is_wholly_separate_descriptive_only": True,
        "smooth_influences_exact_coefficients_selection_gate_or_decision": False,
        "test_labels_construct_postseal_response_rows": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "whole_case_support_evaluation_disjoint": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "test_labels_opened_after_global_prediction_and_feature_seals": True,
        "strict_all_role_H_q_exclusion": True,
        "strict_training_row_count": EXPECTED_STRICT_TRAINING_ROW_COUNT,
        "candidate_e_history_retained_when_legal": True,
        "candidate_pool_excludes_H_and_q": True,
        "outer_centers_are_inference_units": True,
        "seed_patch_and_case_rows_are_not_inference_units": True,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "known_fixed_bank_reuse": True,
        "unseen_expert_transfer": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "previous_stage90_outputs_used": False,
        "stage60_or_stage70_outputs_used": False,
        "diagnostic_only": True,
    }


def publication_decision_payload(audit_result: Mapping[str, object]) -> dict[str, object]:
    if (
        "primary_exact_gate_passed" not in audit_result
        or "exact_decision_hash" not in audit_result
    ):
        raise ProtocolError("Fixed-bank audit result is incomplete for publication.")
    informative = audit_result["primary_exact_gate_passed"]
    if type(informative) is not bool:
        raise ProtocolError("Fixed-bank exact gate must be boolean.")
    return {
        "schema_version": "midogpp_stage90_fixed_bank_publication_decision_v1",
        "decision": (
            "EXPLORATORY_FIXED_BANK_SIGNAL_REQUIRES_FRESH_CONFIRMATION"
            if informative
            else "NO_RELIABLE_FIXED_BANK_SIGNAL_RETAIN_EXACT_B_DIAGNOSTIC_BASELINE"
        ),
        "claim_role": "posthoc_diagnostic_screen",
        # Deliberately excludes the overall result hash: that hash includes the
        # isolated smooth description and would violate byte invariance under
        # a smooth-only perturbation.
        "exact_decision_hash": str(audit_result["exact_decision_hash"]),
        "publication_status": PUBLICATION_STATUS,
        "test_split_previously_consumed": True,
        "known_fixed_bank_reuse": True,
        "unseen_expert_transfer_claim": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "fresh_confirmation_claimed": False,
        "promotion_eligible": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_deployable_selection": False,
    }


def run_state_payload(
    status: str, phase: str, *, error: str | None = None
) -> dict[str, object]:
    if status not in {"RUNNING", "FAILED", "COMPLETE"}:
        raise ValueError("Fixed-bank run state is invalid.")
    return {
        "schema_version": "midogpp_stage90_fixed_bank_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "test_split_previously_consumed": True,
        "promotion_eligible": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
    }


__all__ = (
    "leakage_report_payload",
    "prelabel_phase_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
)
