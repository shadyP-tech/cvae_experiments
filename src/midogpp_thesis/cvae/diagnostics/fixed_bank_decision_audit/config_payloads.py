"""Canonical scientific payloads for the fixed-bank decision audit."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .constants import (
    CONTROL_FAMILY_IDS,
    EXACT_FAMILY_IDS,
    EXACT_FAMILY_PREDICTORS,
    EXPECTED_EXACT_FOLD_COUNT,
    EXPECTED_EXACT_PREDICTION_COUNT,
    EXPECTED_SMOOTH_FOLD_COUNT,
    EXPECTED_SMOOTH_PREDICTION_COUNT,
    GLOBAL_SOURCE_EXACT_CONTROL,
    PRIMARY_R_FAMILY_ID,
    SECONDARY_CHALLENGER_FAMILY_IDS,
    SMOOTH_DESCRIPTIVE_FAMILY_IDS,
)
from .experiment_contracts import (
    CENTERS,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_CANDIDATE_COUNT_PER_QUERY,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_DESCRIPTIVE_SEED_ROW_COUNT,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_FEATURE_ROW_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_RESPONSE_ROW_COUNT,
    EXPECTED_STRICT_TRAINING_ROW_COUNT,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GENERATION_SEEDS,
    PUBLICATION_STATUS,
    STAGE_ID,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    TRAINING_SEEDS,
)


CLASSIFIER = ClassifierSpec(
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)

# Compatibility alias for config-contract tests and callers.  The scientific
# source of truth remains ``constants.SMOOTH_DESCRIPTIVE_FAMILY_IDS``.
SMOOTH_FAMILY_IDS = SMOOTH_DESCRIPTIVE_FAMILY_IDS

def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++",
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(EXPECTED_CASE_COUNTS_BY_CENTER),
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
        "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
        "evaluation_case_counts_by_center": dict(
            EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER
        ),
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "cross_fit_mode": "strict_all_role_H_q_holdout_known_e_reuse",
        "strict_crossfit_training_row_count": EXPECTED_STRICT_TRAINING_ROW_COUNT,
        "one_shared_model_per_heldout_H_q": True,
        "candidate_history_retained_when_H_q_absent_from_all_roles": True,
        "heldout_H_q_excluded_from_outer_query_and_candidate_roles": True,
        "candidate_pool_excludes_H_and_q": True,
        "candidate_count_per_query": EXPECTED_CANDIDATE_COUNT_PER_QUERY,
        "primary_response_name": "exact_bacc_delta",
        "primary_response_count": EXPECTED_RESPONSE_ROW_COUNT,
        "primary_response": (
            "tail_exact_nine_probability_ensemble_BACC_minus_"
            "base_exact_nine_probability_ensemble_BACC"
        ),
        "smooth_response_name": "smooth_bacc_delta",
        "smooth_response_role": "isolated_postseal_descriptive_crossfit_only",
        "smooth_response_may_affect_exact_fit_selection_gate_or_decision": False,
        "descriptive_seed_row_count": EXPECTED_DESCRIPTIVE_SEED_ROW_COUNT,
        "seed_rows_may_feed_model": False,
        "probabilities_averaged_before_single_threshold": True,
        "ensemble_probability_threshold": 0.5,
        "whole_case_support_evaluation_disjoint": True,
        "support_case_aggregation": "equal_weight_per_whole_case",
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "prediction_and_feature_seals_before_test_labels": True,
        "test_labels_construct_postseal_response_rows": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "target_actions_built": False,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_outputs_used": False,
        "previous_stage90_outputs_used": False,
    }


def canonical_features_payload() -> dict[str, object]:
    return {
        "family": "fixed_bank_case_aware_label_free_features_v1",
        "feature_row_unit": "candidate_H_q_e",
        "feature_row_count": EXPECTED_FEATURE_ROW_COUNT,
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "support_probabilities_are_label_free": True,
        "support_probabilities_averaged_across_exact_nine_seed_cells": True,
        "support_rows_aggregated_with_equal_case_weight": True,
        "evaluation_probabilities_used": False,
        "primitive_names": [
            "metadata_similarity",
            "pooled_row_weighted_abs_shift",
            "equal_case_abs_shift",
            "case_abs_shift_sd",
            "equal_case_signed_margin",
            "case_balanced_flip_rate",
            "case_balanced_entropy_change",
            "case_balanced_reconstruction",
            "case_balanced_kl",
            "case_balanced_log_mmd",
        ],
        "metadata_similarity_role": (
            "persisted_descriptive_only_not_used_by_any_exact_or_smooth_family"
        ),
        "derived_within_query_predictors": [
            "case_balanced_reconstruction_z",
            "case_balanced_kl_z",
            "case_balanced_log_mmd_z",
        ],
        "within_query_standardization_is_label_free": True,
        "blocked_permutation_unit": "heldout_outer_target_query_candidate_list",
        "blocked_permutation_seed": 90_902_026,
        "blocked_permutation_shift": 1,
        "technical_seed_rows_are_features": False,
    }


def canonical_model_payload() -> dict[str, object]:
    return {
        "family": "fixed_bank_shared_query_ridge_v1",
        "ridge_alpha": 1.0,
        "hyperparameter_selection": "none_predeclared_before_labels",
        "fit_unit": "one_model_per_heldout_outer_target_query",
        "training_row_count_per_fit": EXPECTED_STRICT_TRAINING_ROW_COUNT,
        "source_effect_coding": "centered_one_hot_known_fixed_bank",
        "source_effect_block_by_family": {
            family_id: family_id != "null_tied_exact_control"
            for family_id in EXACT_FAMILY_IDS
        },
        "local_predictor_limit": 3,
        "scaling_fit_on_training_fold_only": True,
        "ridge_cluster_unit": "outer_target_query",
        "exact_response": "exact_bacc_delta",
        "exact_family_ids": list(EXACT_FAMILY_IDS),
        "exact_family_predictors": {
            family_id: list(EXACT_FAMILY_PREDICTORS[family_id])
            for family_id in EXACT_FAMILY_IDS
        },
        "exact_prediction_row_count": EXPECTED_EXACT_PREDICTION_COUNT,
        "exact_fold_audit_row_count": EXPECTED_EXACT_FOLD_COUNT,
        "primary_r_family_id": PRIMARY_R_FAMILY_ID,
        "secondary_challenger_family_ids": list(
            SECONDARY_CHALLENGER_FAMILY_IDS
        ),
        "control_family_ids": list(CONTROL_FAMILY_IDS),
        "smooth_response": "smooth_bacc_delta",
        "smooth_family_ids": list(SMOOTH_DESCRIPTIVE_FAMILY_IDS),
        "smooth_prediction_row_count": EXPECTED_SMOOTH_PREDICTION_COUNT,
        "smooth_fold_audit_row_count": EXPECTED_SMOOTH_FOLD_COUNT,
        "smooth_models_are_separate_from_exact_models": True,
        "smooth_may_feed_exact_coefficients_selection_or_gate": False,
        "seed_rows_may_feed_model": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "selected_exact_nine_ensemble_bacc_delta",
        "primary_family_id": PRIMARY_R_FAMILY_ID,
        "global_control_family_id": GLOBAL_SOURCE_EXACT_CONTROL,
        "baseline_action_id": "B",
        "diagnostic_method_ids": ["B", "G", "R"],
        "method_rows_are_posthoc_diagnostic_screen": True,
        "method_rows_are_actions_or_policy": False,
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": 9,
        "query_count": EXPECTED_QUERY_COUNT,
        "confidence_level": 0.95,
        "metrics": [
            "selected_exact_gain_vs_B",
            "paired_R_minus_G",
            "top1_accuracy",
            "tie_aware_top1_accuracy",
            "spearman_rank_correlation",
            "pairwise_order_accuracy",
            "normalized_regret",
            "paired_regret_minus_G",
            "source_max_selection_share",
            "source_selection_entropy",
        ],
        "exact_gate": {
            "selected_gain_vs_B_ci95_lower_strictly_above": 0.0,
            "paired_R_minus_G_ci95_lower_strictly_above": 0.0,
            "spearman_ci95_lower_strictly_above": 0.0,
            "pairwise_accuracy_ci95_lower_strictly_above": 0.5,
            "regret_minus_G_ci95_upper_strictly_below": 0.0,
            "all_conditions_required": True,
        },
        "exact_B_abstention_when_gate_fails": True,
        "screening_gate_may_authorize_policy": False,
        "smooth_tables_have_no_decision_or_gate_fields": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "generated_cache_format": "float32_npy_memmap",
        "scientific_reductions_dtype": "float64",
        "phase_order": "two_gpu_source_streams_then_four_by_three_cpu_then_audit",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "development_coarse_task_count": 648,
        "development_classifier_fit_count": 5_184,
        "target_task_count": 0,
        "target_action_count": 0,
        "target_classifier_fit_count": 0,
        "maximum_total_classifier_fit_count": 5_184,
        "scratch_preference": [
            "/data/local/fixed_bank_decision_audit_v1",
            "artifact_parent",
        ],
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "consumed_test_data": True,
        "ledger_amendment_required_and_hash_chained": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "claim_role": "posthoc_diagnostic_screen",
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "proxy_is_nelbo": False,
        "proxy_is_downstream_utility": False,
        "test_labels_opened_only_after_prediction_and_feature_seals": True,
        "support_labels_used": False,
        "target_actions_built": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "screening_gate_may_authorize_policy": False,
        "promotion_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "previous_stage90_outputs_used": False,
    }


__all__ = (
    "CLASSIFIER",
    "EXACT_FAMILY_PREDICTORS",
    "SMOOTH_FAMILY_IDS",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_features_payload",
    "canonical_model_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
)
