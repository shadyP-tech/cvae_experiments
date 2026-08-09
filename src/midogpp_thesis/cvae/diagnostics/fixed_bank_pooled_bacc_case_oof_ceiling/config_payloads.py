"""Canonical scientific payloads for the pooled-BACC case-OOF ceiling."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    CENTERS,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_LOCO_DONOR_COUNT_PER_CANDIDATE,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_NULL_ACTION_COUNT,
    EXPECTED_PAIRWISE_ALTERNATIVE_COUNT_WHEN_G_IS_BASELINE,
    EXPECTED_PAIRWISE_ALTERNATIVE_COUNT_WHEN_G_IS_SOURCE,
    EXPECTED_PAIRWISE_DONOR_COUNT_WHEN_G_IS_BASELINE,
    EXPECTED_PAIRWISE_DONOR_COUNT_WHEN_G_IS_SOURCE,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
    EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PERMUTATION_COUNT,
    PUBLICATION_STATUS,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    SEED_PAIR_COUNT,
    STAGE_ID,
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

VARIANCE_FLOOR = 1.0e-6
CONFIDENCE_MULTIPLIER = 1.96
MINIMUM_GAIN = 0.0
TIE_TOLERANCE = 1.0e-12


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
        "exact_seed_pair_count": SEED_PAIR_COUNT,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(EXPECTED_CASE_COUNTS_BY_CENTER),
        "mixed_class_case_count": EXPECTED_MIXED_CLASS_CASE_COUNT,
        "negative_only_case_count": EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
        "positive_only_case_count": EXPECTED_POSITIVE_ONLY_CASE_COUNT,
        "single_class_cases_retained": True,
        "case_sufficient_statistic_fields": [
            "n_positive",
            "true_positive",
            "n_negative",
            "true_negative",
        ],
        "per_case_bacc_stored_or_used": False,
        "pooled_scope_requires_both_binary_classes": True,
        "support_utility": "pooled_exact_bacc",
        "uncertainty_unit": "paired_whole_case_cluster",
        "target_geometry": "direct_H_with_B_and_eight_Hxe_actions",
        "baseline_action_id": "B",
        "candidate_action_family": "Hxe",
        "candidate_action_id_encoding": "source_center_alias_for_Hxe",
        "candidate_pool_excludes_target_H": True,
        "candidate_source_count_per_target": (
            EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
        ),
        "action_count_per_target": EXPECTED_ACTION_COUNT_PER_TARGET,
        "target_action_identity_count": EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "oof_fold_count": OOF_FOLD_COUNT,
        "partition_seed": OOF_FOLD_SEED,
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "partition_unit": "whole_case_within_target_center",
        "each_case_evaluated_exactly_once": True,
        "heldout_fold_absent_from_its_support_and_decision_fit": True,
        "cross_role_case_reuse_only_in_other_folds": True,
        "center_fold_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "global_target_probability_seal_before_any_label_access": True,
        "support_labels_used": True,
        "support_label_scope": "same_H_nonheldout_folds_only",
        "support_labels_may_update_shared_model": False,
        "loco_and_pairwise_prior_seals_before_H_support_access": True,
        "evaluation_role_labels_inaccessible_until_all_observed_and_null_actions_sealed": True,
        "probabilities_averaged_before_single_threshold": True,
        "probability_threshold": 0.5,
        "support_case_aggregation": "pooled_row_weighted_confusion_sums",
        "source_expert_updated": False,
        "target_expert_used": False,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "label_free_cache_lineage": "stage70_derived_feature_cache_alias_only",
        "previous_stage90_outputs_used": False,
        "quarantined_v1_experiment_id": QUARANTINED_V1_EXPERIMENT_ID,
        "quarantined_v1_output_artifact_id": QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
        "quarantined_v1_output_used": False,
        "quarantined_v1_scratch_or_checkpoint_used": False,
        "v2_recomputed_from_original_six_inputs": True,
    }


def canonical_global_prior_payload() -> dict[str, object]:
    return {
        "family": "pooled_exact_bacc_LOCO_global_and_pairwise_priors_v2",
        "fit_unit": "one_G_H_and_candidate_pairwise_prior_set_per_target_H",
        "utility": "pooled_exact_bacc_from_aggregated_confusion_sums",
        "candidate_source_pool": "known_fixed_bank_sources_e_not_equal_H",
        "legal_donor_center_rule": "H_prime_not_in_H_or_e",
        "loco_donor_count_per_candidate": EXPECTED_LOCO_DONOR_COUNT_PER_CANDIDATE,
        "other_center_contribution_unit": "equal_weight_per_legal_donor_center",
        "G_H_effect": "U_H_prime_e_minus_U_H_prime_B",
        "G_H_candidate_selection": (
            "maximum_prior_mean_lexicographic_ties_only_if_95pct_LCB_vs_B_gt_0_"
            "otherwise_B"
        ),
        "G_H_uses_other_consumed_test_centers": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
        "G_H_sealed_before_H_support_access": True,
        "pairwise_prior_effect": "U_H_prime_e_minus_U_H_prime_selected_G_H",
        "pairwise_prior_uses_shared_legal_donors": True,
        "pairwise_alternatives_exclude_selected_source_G_H": True,
        "pairwise_alternative_count_when_G_H_is_B": (
            EXPECTED_PAIRWISE_ALTERNATIVE_COUNT_WHEN_G_IS_BASELINE
        ),
        "pairwise_alternative_count_when_G_H_is_source": (
            EXPECTED_PAIRWISE_ALTERNATIVE_COUNT_WHEN_G_IS_SOURCE
        ),
        "pairwise_donor_count_when_G_H_is_B": (
            EXPECTED_PAIRWISE_DONOR_COUNT_WHEN_G_IS_BASELINE
        ),
        "pairwise_donor_count_when_G_H_is_source": (
            EXPECTED_PAIRWISE_DONOR_COUNT_WHEN_G_IS_SOURCE
        ),
        "pairwise_priors_sealed_before_H_support_access": True,
        "prior_mean_estimator": "equal_weight_mean_of_legal_donor_effects",
        "prior_variance_estimator": (
            "max_sample_variance_of_legal_donor_effects_divided_by_J_and_vmin"
        ),
        "variance_floor": VARIANCE_FLOOR,
        "confidence_multiplier": CONFIDENCE_MULTIPLIER,
        "minimum_gain": MINIMUM_GAIN,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": "lexicographic_action_id",
        "evaluation_role_capability_used": False,
        "other_center_labels_accessed_only_by_loco_prior_capability": True,
        "hyperparameter_selection": "none_predeclared_before_labels",
    }


def canonical_posterior_payload() -> dict[str, object]:
    return {
        "family": "normal_normal_pooled_bacc_whole_case_cluster_v2",
        "fit_unit": "one_local_posterior_per_H_fold_and_candidate_e",
        "prior_source": "sealed_candidate_vs_selected_G_H_pairwise_prior",
        "support_observation": "pooled_exact_bacc_e_minus_selected_G_H",
        "support_labels_scope": "same_H_nonheldout_folds_only",
        "case_sufficient_statistics_only": True,
        "case_level_bacc_forbidden": True,
        "support_contrast_symbol": "D",
        "support_contrast_definition": "U_support_e_minus_U_support_selected_G_H",
        "uncertainty_unit": "paired_whole_case_cluster",
        "cluster_influence_definition": (
            "0.5_times_[n_c_pos_over_N_pos_times_(case_pos_accuracy_diff_minus_"
            "pooled_pos_accuracy_diff)_plus_n_c_neg_over_N_neg_times_(case_neg_"
            "accuracy_diff_minus_pooled_neg_accuracy_diff)]"
        ),
        "absent_case_class_term_policy": "omit_only_the_absent_class_term",
        "cluster_variance_estimator": "max_m_over_m_minus_1_sum_psi_squared_and_vmin",
        "posterior_variance": "inverse_of_inverse_V0_plus_inverse_Vf",
        "posterior_mean": "Vpost_times_mu0_over_V0_plus_D_over_Vf",
        "posterior_lower_bound": "mupost_minus_1.96_times_sqrt_Vpost",
        "variance_floor": VARIANCE_FLOOR,
        "confidence_multiplier": CONFIDENCE_MULTIPLIER,
        "minimum_gain": MINIMUM_GAIN,
        "hyperparameter_selection": "none_predeclared_before_labels",
        "no_cross_H_target_label_pooling_beyond_sealed_priors": True,
        "no_shared_target_label_fit": True,
    }


def canonical_decision_payload() -> dict[str, object]:
    return {
        "family": "abstaining_pooled_bacc_cluster_posterior_router_v2",
        "diagnostic_method_ids": ["B", "G_H", "R"],
        "hard_candidate_selection": True,
        "mixtures_allowed": False,
        "confidence_multiplier": CONFIDENCE_MULTIPLIER,
        "minimum_gain": MINIMUM_GAIN,
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": "lexicographic_action_id",
        "R_rule": "maximum_candidate_posterior_LCB_if_strictly_positive_else_G_H",
        "G_H_rule": "maximum_loco_prior_mean_with_positive_95pct_LCB_else_B",
        "B_abstention_when_no_positive_G_H_lower_bound": True,
        "all_center_fold_decisions_sealed_before_evaluation_capability": True,
        "all_permutation_null_actions_sealed_before_evaluation_capability": True,
        "expected_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "expected_permutation_null_action_count": EXPECTED_NULL_ACTION_COUNT,
        "evaluation_probabilities_may_affect_decision": False,
        "evaluation_labels_may_affect_decision": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "center_pooled_exact_bacc_over_whole_case_oof_predictions",
        "baseline_method_id": "B",
        "global_method_id": "G_H",
        "router_method_id": "R",
        "primary_contrasts": ["R-G_H", "R-B", "G_H-B"],
        "center_utility": "pooled_exact_bacc_from_aggregated_confusion_sums",
        "primary_aggregation": "equal_weight_per_target_center",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_repeats_are_not_independent_units": True,
        "single_class_cases_retained": True,
        "metrics": [
            "pooled_exact_bacc",
            "paired_R_minus_G_H",
            "paired_R_minus_B",
            "normalized_regret",
            "top1_accuracy",
            "tie_aware_top1_accuracy",
            "coverage",
            "source_selection_share",
        ],
        "zero_headroom_normalized_regret": 0.0,
        "zero_headroom_tolerance": TIE_TOLERANCE,
        "zero_headroom_interpretation": "no_routing_opportunity",
        "permutation_unit": (
            "complete_candidate_sufficient_statistic_block_derangement_within_"
            "H_fold_and_support_case"
        ),
        "permutation_primary_statistic": "equal_center_R_minus_G_H",
        "permutation_upper_tail_output_field": "one_sided_p_value",
        "permutation_lower_tail_output_field": "lower_tail_p_value",
        "permutation_two_sided_output_field": "two_sided_p_value",
        "permutation_upper_tail_p_value_formula": (
            "(1+count(null>=observed))/(K+1)"
        ),
        "permutation_lower_tail_p_value_formula": (
            "(1+count(null<=observed))/(K+1)"
        ),
        "permutation_two_sided_p_value_formula": "min(1,2*min(upper,lower))",
        "permutation_derangement_family": (
            "case_sha256_candidate_order_counter_splitmix64_nonzero_cyclic_"
            "shift_1_to_7_v1"
        ),
        "permutation_candidate_order": (
            "case_specific_sha256_of_seed_fold_id_case_id_action_then_action"
        ),
        "permutation_shift_generator": (
            "independent_counter_splitmix64_per_fold_case_permutation_index"
        ),
        "permutation_shift_range_inclusive": [1, 7],
        "permutation_zero_shift_allowed": False,
        "uniform_over_all_derangements": False,
        "permutation_seed": 90_912_026,
        "permutation_count": PERMUTATION_COUNT,
        "permutation_baseline_B_fixed": True,
        "permutation_eight_Hxe_multiset_preserved": True,
        "permutation_recomputes_same_pooled_bacc_cluster_posterior": True,
        "permutation_evaluation_donors_used": False,
        "permutation_actions_sealed_before_evaluation_labels": True,
        "permutation_decision_tie_break": (
            "lexicographic_action_id_no_evaluation_utility_access"
        ),
        "confidence_level": 0.95,
        "pooled_exact_metric_only_may_enter_gates": True,
        "results_are_terminal_consumed_test_diagnostics": True,
        "result_may_authorize_policy_or_action": False,
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
        "phase_order": (
            "two_gpu_source_streams_then_four_by_three_cpu_target_fits_then_"
            "capability_sealed_pooled_bacc_case_oof_audit"
        ),
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "target_task_count": EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
        "target_action_identity_count": EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "maximum_total_classifier_fit_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "scratch_preference": [
            "/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2",
            "artifact_parent",
        ],
        "v1_scratch_reuse_forbidden": True,
        "resume_policy": "hash_validated_atomic_phase_and_task_checkpoints",
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": "DO_NOT_PROMOTE",
        "consumed_test_data": True,
        "ledger_amendment_required_and_hash_chained": True,
        "method_development_is_posthoc": True,
        "terminal_stage90_diagnostic": True,
        "claim_role": "pooled_bacc_known_bank_case_oof_information_ceiling",
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "support_labels_used": True,
        "support_labels_local_only": True,
        "label_derived_LOCO_global_prior": True,
        "evaluation_labels_opened_only_after_all_observed_and_null_action_seals": True,
        "diagnostic_candidate_action_probabilities_built": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "screening_gate_may_authorize_policy": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "previous_stage90_outputs_used": False,
        "quarantined_v1_output_used": False,
        "quarantined_v1_scratch_or_checkpoint_used": False,
    }


__all__ = (
    "CLASSIFIER",
    "CONFIDENCE_MULTIPLIER",
    "MINIMUM_GAIN",
    "TIE_TOLERANCE",
    "VARIANCE_FLOOR",
    "canonical_claim_boundary_payload",
    "canonical_decision_payload",
    "canonical_evaluation_payload",
    "canonical_global_prior_payload",
    "canonical_posterior_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
)
