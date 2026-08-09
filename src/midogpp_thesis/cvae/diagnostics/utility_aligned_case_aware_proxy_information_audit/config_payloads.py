"""Canonical scientific section payloads for the case-aware audit config."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    CENTERS,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_PROXY_FEATURE_ROW_COUNT,
    EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT,
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


def canonical_protocol_payload() -> dict[str, object]:
    """Return the immutable consumed-test split and response boundary."""

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
        "fixed_support_case_count_per_center": (
            FIXED_SUPPORT_CASE_COUNT_PER_CENTER
        ),
        "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
        "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
        "evaluation_case_counts_by_center": dict(
            EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER
        ),
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "cross_fit_mode": "strict_all_role_H_q_e_domain_holdout",
        "strict_crossfit_training_row_count": (
            EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT
        ),
        "primary_response_name": "exact_bacc_delta",
        "primary_response_unit": (
            "candidate_H_q_e_exact_nine_probability_ensemble"
        ),
        "primary_response_count": EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
        "primary_response": (
            "tail_exact_nine_probability_ensemble_BACC_minus_"
            "base_exact_nine_probability_ensemble_BACC"
        ),
        "diagnostic_response_name": "smooth_bacc_delta",
        "diagnostic_response_use": "post_seal_descriptive_only",
        "diagnostic_response_may_feed_fit_selection_or_gate": False,
        "descriptive_per_seed_utility_row_count": (
            EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT
        ),
        "descriptive_per_seed_rows_may_feed_model": False,
        "probabilities_averaged_before_single_threshold": True,
        "ensemble_probability_threshold": 0.5,
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction": True,
        "outer_target_H_excluded_from_fit_scaling_and_prediction": True,
        "pseudoquery_q_excluded_from_fit_scaling_and_prediction": True,
        "candidate_source_e_excluded_from_fit_scaling_and_prediction": True,
        "whole_case_support_evaluation_disjoint": True,
        "support_case_aggregation": "equal_weight_per_whole_case",
        "fixed_eight_case_support_is_diagnostic_only": True,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
        "development_predictions_sealed_before_test_labels": True,
        "test_labels_opened_only_after_global_prediction_seal": True,
        "test_labels_construct_postseal_response_rows": True,
        "label_derived_responses_feed_strict_crossfit_diagnostic_models": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "source_expert_updated": False,
        "target_expert_used": False,
        "target_actions_built": False,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "previous_stage90_outputs_used": False,
        "historical_or_quarantined_inputs_used": False,
    }


def canonical_proxy_features_payload() -> dict[str, object]:
    """Return the label-free, case-aware feature grammar."""

    return {
        "family": "predeclared_case_aware_compact_proxy_information_v1",
        "feature_row_unit": "candidate_H_q_e",
        "feature_row_count": EXPECTED_PROXY_FEATURE_ROW_COUNT,
        "fixed_support_case_count_per_center": (
            FIXED_SUPPORT_CASE_COUNT_PER_CENTER
        ),
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
        "primitive_formulas": {
            "metadata_similarity": (
                "exact_match_count_tumor_type_lab_or_origin_scanner_model_div_3"
            ),
            "pooled_row_weighted_abs_shift": (
                "mean_support_rows_abs(mean9_p_tail_minus_mean9_p_base)"
            ),
            "equal_case_abs_shift": (
                "mean_support_cases(mean_case_rows_abs(mean9_p_tail_minus_mean9_p_base))"
            ),
            "case_abs_shift_sd": (
                "population_sd_support_cases(mean_case_rows_abs(mean9_p_tail_minus_mean9_p_base))"
            ),
            "equal_case_signed_margin": (
                "mean_support_cases(mean_case_rows((mean9_p_tail_minus_mean9_p_base)*"
                "where(mean9_p_base_gte_0.5,1,-1)))"
            ),
            "case_balanced_flip_rate": (
                "mean_support_cases(mean_case_rows(indicator((mean9_p_base_minus_0.5)*"
                "(mean9_p_tail_minus_0.5)<0)))"
            ),
            "case_balanced_entropy_change": (
                "mean_support_cases(mean_case_rows(binary_entropy(mean9_p_tail)-"
                "binary_entropy(mean9_p_base)))"
            ),
            "case_balanced_reconstruction": (
                "mean_support_cases(mean_case_rows(reconstruction))"
            ),
            "case_balanced_kl": "mean_support_cases(mean_case_rows(analytic_kl))",
            "case_balanced_log_mmd": (
                "mean_support_cases(mean_exact9(log1p(linear_kernel_mmd2("
                "case_embedding_mean, generated_stream_mean))))"
            ),
        },
        "derived_within_query_predictors": [
            "case_balanced_reconstruction_z",
            "case_balanced_kl_z",
            "case_balanced_log_mmd_z",
        ],
        "within_query_standardization_uses_only_current_label_free_candidate_list": True,
        "within_query_standardization_uses_utility_or_evaluation_labels": False,
        "zero_variance_standardized_value": 0.0,
        "cyclic_directional_permutation": (
            "canonical_allowed_source_order_nonzero_rotation_by_one"
        ),
        "cyclic_directional_permutation_seed": 90_902_026,
        "cyclic_directional_permutation_shift": 1,
        "technical_seed_rows_are_features": False,
    }


def canonical_model_payload() -> dict[str, object]:
    """Return the fixed seven-family ridge screen."""

    return {
        "family": "fixed_alpha_cluster_weighted_ridge_case_aware_proxy_information_v1",
        "ridge_alpha": 1.0,
        "hyperparameter_selection": "none_predeclared_before_labels",
        "maximum_predictors_per_family": 3,
        "scaling_fit_on_training_fold_only": True,
        "ridge_cluster_unit": "outer_target_query",
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction": True,
        "primary_response": "exact_bacc_delta",
        "response_row_count": EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
        "diagnostic_response": "smooth_bacc_delta",
        "diagnostic_response_crossfit_role": (
            "separately_fit_descriptive_models_only"
        ),
        "diagnostic_response_may_feed_primary_model_or_gate": False,
        "descriptive_seed_row_count": EXPECTED_DESCRIPTIVE_SEED_UTILITY_ROW_COUNT,
        "descriptive_seed_rows_may_feed_model": False,
        "family_ids": [
            "equal_union_null",
            "metadata_only_control",
            "pooled_row_weighted_shift_control",
            "case_balanced_shift_compact",
            "case_balanced_rich_compact",
            "case_aware_hybrid_compact",
            "cyclic_directional_permutation_control",
        ],
        "family_predictors": {
            "equal_union_null": [],
            "metadata_only_control": ["metadata_similarity"],
            "pooled_row_weighted_shift_control": [
                "pooled_row_weighted_abs_shift"
            ],
            "case_balanced_shift_compact": [
                "equal_case_abs_shift",
                "case_abs_shift_sd",
                "equal_case_signed_margin",
            ],
            "case_balanced_rich_compact": [
                "case_balanced_reconstruction_z",
                "case_balanced_kl_z",
                "case_balanced_log_mmd_z",
            ],
            "case_aware_hybrid_compact": [
                "metadata_similarity",
                "case_balanced_log_mmd_z",
                "equal_case_abs_shift",
            ],
            "cyclic_directional_permutation_control": [
                "cyclic_equal_case_signed_margin",
                "cyclic_case_balanced_flip_rate",
                "cyclic_case_balanced_entropy_change",
            ],
        },
        "outer_target_centers_are_independent_units": True,
        "query_domains_are_nested_descriptive_units": True,
        "case_rows_are_equal_weighted_within_support_features": True,
        "seed_or_patch_rows_are_independent_units": False,
        "target_or_query_identity_predictors_used": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "outer_target_center_exact_bacc_proxy_information_screen",
        "primary_response": "exact_bacc_delta",
        "diagnostic_response": "smooth_bacc_delta",
        "diagnostic_response_timing": "computed_only_after_primary_prediction_seal",
        "diagnostic_response_may_change_primary_decision": False,
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": 9,
        "query_metric_row_count": 72,
        "query_metrics_are_descriptive_nested_within_centers": True,
        "metrics": [
            "spearman_proxy_utility",
            "pairwise_order_accuracy",
            "normalized_regret",
        ],
        "confidence_level": 0.95,
        "screening_candidate_family_ids": [
            "case_balanced_shift_compact",
            "case_balanced_rich_compact",
            "case_aware_hybrid_compact",
        ],
        "control_family_ids": [
            "equal_union_null",
            "metadata_only_control",
            "pooled_row_weighted_shift_control",
            "cyclic_directional_permutation_control",
        ],
        "screening_gate": {
            "outer_center_mean_spearman_ci95_lower_strictly_above": 0.0,
            "outer_center_pairwise_accuracy_ci95_lower_strictly_above": 0.5,
            "outer_center_normalized_regret_ci95_upper_strictly_below": 0.5,
            "mean_regret_strictly_below_each_control_family": True,
            "all_conditions_required": True,
        },
        "screening_gate_may_authorize_policy": False,
        "no_target_action_or_deployable_performance_evaluation": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "generated_cache_format": "float32_npy_memmap",
        "phase_order": "two_gpu_source_streams_then_four_by_three_cpu_development",
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
        "scratch_preference": ["/data/local", "artifact_parent"],
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "consumed_test_data": True,
        "consumed_validation_data": False,
        "user_authorized_consumed_test_repurposing": True,
        "test_consumption_ledger_acknowledged": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "proxy_information_audit_only": True,
        "cross_fitted_fixed_support_diagnostic": True,
        "fixed_eight_case_support_is_insufficient_for_policy": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "target_specific_router_success_claimed": False,
        "proxy_is_nelbo": False,
        "proxy_is_downstream_utility": False,
        "test_labels_opened_only_after_global_prediction_seal": True,
        "test_labels_construct_postseal_response_rows": True,
        "label_derived_responses_feed_strict_crossfit_diagnostic_models": True,
        "test_labels_used_for_feature_construction": False,
        "test_labels_used_for_policy_or_action_fit": False,
        "target_actions_built": False,
        "screening_gate_may_authorize_policy": False,
        "policy_update_authorized": False,
        "may_update_policy": False,
        "action_selection_authorized": False,
        "promotion_eligible": False,
        "oracle_eligible": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "may_feed_another_stage90_experiment": False,
    }


__all__ = (
    "CLASSIFIER",
    "canonical_claim_boundary_payload",
    "canonical_evaluation_payload",
    "canonical_model_payload",
    "canonical_protocol_payload",
    "canonical_proxy_features_payload",
    "canonical_runtime_payload",
)
