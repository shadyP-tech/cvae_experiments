"""Path-free input and claim contracts for planned SCEPTRE v1."""

from __future__ import annotations

from .identity import (
    EXPERIMENT_ID,
    EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256,
    EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
    EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256,
    EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
    EXPECTED_SOURCE_EVALUATION_ROWS_SHA256,
    EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256,
    EXPECTED_SOURCE_PREDICTION_INDEX_SHA256,
    EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256,
    EXPECTED_SOURCE_UTILITY_LOCK_SHA256,
    EXPECTED_SOURCE_UTILITY_ROWS,
    EXPECTED_SOURCE_UTILITY_TABLE_SHA256,
    INPUT_ARTIFACT_IDS,
    PUBLICATION_STATUS,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    TERMINAL_DECISION,
)


def direct_input_policy_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v1_direct_input_policy_v1",
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "direct_input_count": len(INPUT_ARTIFACT_IDS),
        "all_direct_input_artifact_ids_unique": True,
        "allowed_direct_source_roles": [
            "frozen_routing_authorized_expert_bank",
            "frozen_generation_lock",
            "dataset_contract_annotation_manifest_only",
            "consumer_fenced_source_inner_development_alias",
            "consumer_specific_adaptive_reuse_amendment",
        ],
        "source_inner_alias_id": SOURCE_INNER_ALIAS_ARTIFACT_ID,
        "source_inner_amendment_id": SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
        "source_inner_reuse_role": "DEVELOPMENT_ONLY_ADAPTIVE_DESCRIPTIVE",
        "source_inner_utility_lock_sha256": EXPECTED_SOURCE_UTILITY_LOCK_SHA256,
        "source_inner_candidate_utility_sha256": EXPECTED_SOURCE_UTILITY_TABLE_SHA256,
        "source_inner_candidate_utility_row_count": EXPECTED_SOURCE_UTILITY_ROWS,
        "source_inner_case_confusions_sha256": EXPECTED_SOURCE_CASE_CONFUSIONS_SHA256,
        "source_inner_case_confusions_row_count": EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
        "source_inner_candidate_predictions_npz_sha256": (
            EXPECTED_SOURCE_PREDICTION_ARRAY_FILE_SHA256
        ),
        "source_inner_prediction_index_json_sha256": (
            EXPECTED_SOURCE_PREDICTION_INDEX_SHA256
        ),
        "source_inner_classifier_fits_csv_sha256": (
            EXPECTED_SOURCE_CLASSIFIER_FITS_SHA256
        ),
        "source_inner_classifier_fit_row_count": EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
        "source_inner_evaluation_rows_csv_sha256": (
            EXPECTED_SOURCE_EVALUATION_ROWS_SHA256
        ),
        "source_inner_evaluation_row_count": EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
        "source_inner_evidence_members_label_free": True,
        "source_inner_amendment_sha256": EXPECTED_SOURCE_REUSE_AMENDMENT_SHA256,
        "source_inner_original_policy_lock_mutated": False,
        "source_inner_original_policy_lock_reinterpreted": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_reports_or_run_states_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
        "previous_stage90_leases_or_amendments_used": False,
        "test_cache_capability_registered": False,
        "test_manifest_label_capability_registered": False,
        "test_consumption_ledger_capability_registered": False,
        "test_cache_resolution_status": "PENDING_SEPARATE_FUTURE_AUTHORIZATION",
        "test_manifest_resolution_status": "PENDING_SEPARATE_FUTURE_AUTHORIZATION",
        "parent_consumption_ledger_resolution_status": (
            "PENDING_SEPARATE_FUTURE_AUTHORIZATION"
        ),
        "execution_authorization_status": "ABSENT_NOT_AUTHORIZED",
        "input_path_resolution_deferred": True,
        "cross_run_recovery_allowed": False,
    }


def claim_boundary_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v1_claim_boundary_v1",
        "experiment_id": EXPERIMENT_ID,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": "diagnostic_only",
        "bounded_question": (
            "can_the_architecture_recover_routing_structure_already_present_"
            "in_the_observed_midogpp_utility_surface"
        ),
        "fresh_evidence": False,
        "historical_source_inner_adaptive_reuse_authorized": True,
        "adaptive_architecture_comparison": True,
        "comparisons_are_descriptive_only": True,
        "new_center_generalization_claimed": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "confidence_bound_claimed": False,
        "finite_sample_coverage_claimed": False,
        "significance_claimed": False,
        "thesis_confirmatory_improvement_claimed": False,
        "promotion_allowed": False,
        "deployment_claimed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "execution_authorized": False,
        "one_shot_test_label_reader_implemented": False,
        "consumed_test_reuse_authorized": False,
    }


__all__ = ("claim_boundary_payload", "direct_input_policy_payload")
