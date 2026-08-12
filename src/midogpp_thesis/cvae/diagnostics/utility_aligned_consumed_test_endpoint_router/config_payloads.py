"""Canonical immutable sections for the endpoint-router YAML contract."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...routing.local_marginal_utility.ridge import DEFAULT_RIDGE_ALPHAS
from ...routing.utility_aligned.ensemble_endpoint_contracts import (
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
)
from .experiment_contracts import (
    ACTION_IDS,
    CENTERS,
    DESCRIPTIVE_DEVELOPMENT_SEED_ROW_COUNT,
    DEVELOPMENT_RESPONSE_COUNT,
    EVALUATION_SPLIT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNT,
    EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_SUPPORT_ROW_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    PRIMARY_CONTRASTS,
    PUBLICATION_STATUS,
    STAGE_ID,
    SUPPORT_BOOTSTRAP_REPLICATES,
    SUPPORT_BOOTSTRAP_SEED,
    SUPPORT_CASE_COUNT_PER_CENTER,
    SUPPORT_PARTITION_NAMESPACE,
)


CLASSIFIER = ClassifierSpec(
    C=0.01, penalty="l2", solver="lbfgs", max_iter=3000,
    class_weight=None, random_state=23, l1_ratio=None,
    threshold_policy="predict", scaler_fit="synthetic_train_only",
)


def canonical_protocol_payload() -> dict[str, object]:
    return {
        "dataset_family": "MIDOG++", "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT, "centers": list(CENTERS),
        "excluded_center": "4", "training_seeds": [17, 42, 101],
        "generation_seeds": [17, 42, 101],
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(EXPECTED_CASE_COUNTS_BY_CENTER),
        "support_partition_rule": "canonical_sort_then_first_eight_whole_cases_per_center",
        "support_partition_is_seed_independent": True,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "fixed_support_case_count_per_center": SUPPORT_CASE_COUNT_PER_CENTER,
        "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
        "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
        "evaluation_case_counts_by_center": dict(EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER),
        "support_row_count": EXPECTED_SUPPORT_ROW_COUNT,
        "evaluation_row_count": EXPECTED_EVALUATION_ROW_COUNT,
        "evaluation_row_counts_by_center": dict(EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER),
        "class_coverage_checked_only_after_partition_membership_frozen": True,
        "whole_case_support_evaluation_disjoint": True,
        "all_218_cases_participate_once_as_support_or_evaluation": True,
        "support_labels_used": False,
        "development_response_name": "exact_nine_probability_ensemble_bacc_delta",
        "development_response_unit": "candidate_H_q_e",
        "development_response_count": DEVELOPMENT_RESPONSE_COUNT,
        "descriptive_per_seed_utility_row_count": DESCRIPTIVE_DEVELOPMENT_SEED_ROW_COUNT,
        "descriptive_per_seed_rows_may_feed_model": False,
        "strict_H_q_e_exclusion_in_fit_scaling_and_prediction": True,
        "development_predictions_sealed_before_development_labels": True,
        "cross_center_evaluation_labels_used_as_development_q_labels_after_development_seal": True,
        "global_source_control_provenance": "experiment_manifest_only",
        "domain_mapping_parsed_after_prelabel_manifest_admission": True,
        "separate_metadata_profile_artifact_used": False,
        "metadata_identity_or_label_predictor_used": False,
        "target_actions_are_static_per_center": True,
        "case_level_routing_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal": True,
        "source_expert_updated": False, "target_expert_used": False,
        "target_labels_update_shared_model": False,
        "stage50_outputs_used": False, "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "previous_stage90_outputs_or_amendments_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    return {
        "family": "utility_aligned_exact_nine_additive_tail_target_static_v1",
        "method_ids": list(ACTION_IDS), "single_source_tail_prefix": "Hxe::",
        "B_role": "immutable_equal_union_base_and_fail_closed_fallback",
        "U_role": "terminal_matched_uniform_additive_tail_control",
        "G_role": "diagnostic_global_model_selection",
        "R_role": "utility_aligned_target_static_routed_selection",
        "P_role": "same_capacity_deterministic_derangement_control",
        "Hxe_role": "terminal_descriptive_single_source_oracle_candidates",
        "target_source_count": 8, "target_base_per_source_per_class": 128,
        "target_topup_total_per_class": 128, "target_base_total_per_class": 1024,
        "target_matched_total_per_class": 1152, "inner_source_count": 7,
        "inner_base_per_source_per_class": 144, "inner_topup_total_per_class": 126,
        "inner_base_total_per_class": 1008, "inner_matched_total_per_class": 1134,
        "source_prefix_rows_per_class": 270,
        "target_physical_action_count_per_center": 10,
        "target_reported_method_count_per_center": 13,
        "G_R_P_reuse_selected_Hxe_predictions": True,
        "no_action_budget_temperature_strength_or_seed_search": True,
    }


def canonical_model_payload() -> dict[str, object]:
    return {
        "family": "candidate_specific_low_capacity_ridge_ensemble_endpoint_v1",
        "ridge_alpha_grid": [float(value) for value in DEFAULT_RIDGE_ALPHAS],
        "inner_selection": "strict_nested_leave_query_and_source_domain_out",
        "M0_predictors": ["global_source_control"],
        "M1_predictors": ["global_source_control", f"target_local::{SUPPORT_ACTION_PROBABILITY_SHIFT_NAME}"],
        "global_source_control_provenance": "experiment_manifest_only",
        "target_local_scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
        "target_local_scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
        "target_local_scalar_is_ensemble_first": True,
        "response": "exact_nine_probability_ensemble_bacc_delta",
        "response_row_count": DEVELOPMENT_RESPONSE_COUNT,
        "permutation_seed": 90_902_026,
        "permutation_family": "deterministic_nonidentity_candidate_derangement",
        "permutation_refits_same_capacity_model": True,
        "exact_nine_seed_cells_collapsed_before_model_fit": True,
        "technical_seed_cells_are_not_independent_units": True,
        "query_domains_are_model_selection_units": True,
        "target_support_bootstrap_replicates": SUPPORT_BOOTSTRAP_REPLICATES,
        "target_support_bootstrap_seed": SUPPORT_BOOTSTRAP_SEED,
        "target_support_bootstrap_unit": "whole_case",
        "cardinality_transfer_api": "evaluate_ensemble_cardinality_transfer",
        "target_policy_api": "build_ensemble_utility_policy",
        "R_policy_rule": "source_inner_transfer_and_capacity_gates_plus_selected_gain_lcb_gt_0",
        "R_fallback_action": "B",
        "simultaneous_prelabel_lcb_vs_U_G_P_required": False,
        "G_and_P_are_diagnostic_selections": True,
        "target_or_query_identity_predictors_used": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
        "probabilities_averaged_before_single_threshold": True,
        "ensemble_probability_threshold": 0.5,
        "evaluation_case_count": EXPECTED_EVALUATION_CASE_COUNT,
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "inference_unit": "target_center", "inference_center_count": 9,
        "technical_seed_repeats_are_not_independent_units": True,
        "oracle_diagnostics": ["selected_Hxe_source", "R_candidate_rank", "R_Hxe_top1_agreement", "normalized_oracle_gap"],
        "Hxe_oracle_role": "terminal_descriptive_only_no_plan_or_policy_update",
        "target_scoring_capability_requires_global_prelabel_seal": True,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal": True,
        "cross_center_evaluation_labels_may_have_opened_for_other_outer_targets": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"], "cuda_visible_devices": "0,1",
        "generation_workers_per_device": 1, "classifier_workers": 4,
        "classifier_threads_per_worker": 3, "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True, "tf32_enabled": False,
        "amp_enabled": False, "launch_blas_threads": 1,
        "array_storage_dtype": "float32", "scientific_reduction_dtype": "float64",
        "phase_order": "two_A5000_generation_then_four_by_three_CPU",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_stream_count": 81, "development_prediction_cell_count": 5_184,
        "target_physical_action_identity_count": 90,
        "target_prediction_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 5_994,
        "scratch_preference": ["/data/local", "artifact_parent"],
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS, "consumed_test_data": True,
        "consumed_validation_data": False, "user_authorized_consumed_test_repurposing": True,
        "test_consumption_ledger_direct_amendment_required": True,
        "method_development_is_posthoc": True, "terminal_stage90_diagnostic": True,
        "target_static_endpoint_diagnostic": True, "fresh_evidence": False,
        "fresh_confirmation": False, "routing_success_claimed": False,
        "routing_quality_claimed": False, "target_performance_claimed": False,
        "target_specific_router_success_claimed": False, "proxy_is_nelbo": False,
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal": True,
        "action_selection_authorized": False, "policy_update_authorized": False,
        "model_update_authorized": False, "expert_update_authorized": False,
        "promotion_eligible": False, "may_feed_stage50": False,
        "may_feed_stage60": False, "may_feed_stage70": False,
        "may_feed_another_stage90_experiment": False,
        "may_feed_another_experiment": False, "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False, "generic_consumer_authorized": False,
    }


__all__ = (
    "CLASSIFIER", "canonical_action_library_payload",
    "canonical_claim_boundary_payload", "canonical_evaluation_payload",
    "canonical_model_payload", "canonical_protocol_payload",
    "canonical_runtime_payload",
)
