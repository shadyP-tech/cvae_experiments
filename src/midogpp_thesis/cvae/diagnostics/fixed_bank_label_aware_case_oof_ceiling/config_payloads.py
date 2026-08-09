"""Canonical scientific payloads for the label-aware case-OOF ceiling."""

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
    EXPECTED_TARGET_ACTION_IDENTITY_COUNT,
    EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PUBLICATION_STATUS,
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
        "evaluation_role_labels_inaccessible_until_all_decisions_sealed": True,
        "permutation_null_actions_sealed_before_evaluation_role_labels": True,
        "probabilities_averaged_before_single_threshold": True,
        "probability_threshold": 0.5,
        "support_case_aggregation": "equal_weight_per_whole_case",
        "source_expert_updated": False,
        "target_expert_used": False,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_outputs_used": False,
        "previous_stage90_outputs_used": False,
    }


def canonical_global_prior_payload() -> dict[str, object]:
    return {
        "family": "label_derived_LOCO_global_prior",
        "fit_unit": "one_G_H_per_heldout_target_center",
        "utility": "exact_nine_probability_ensemble_bacc_gain_vs_B",
        "candidate_source_pool": "known_fixed_bank_sources_e_not_equal_H",
        "other_center_contribution_unit": "equal_weight_per_target_center",
        "G_H_uses_other_consumed_test_centers": True,
        "H_labels_used_in_G_H": False,
        "G_H_shared_across_H": False,
        "G_H_hyperparameters_fixed_prelabel": True,
        "G_H_sealed_before_H_support_access": True,
        "H_prime_equal_e_absent_for_candidate_e": True,
        "evaluation_role_capability_used": False,
        "other_center_labels_accessed_only_by_loco_prior_capability": True,
        "prior_strength": 8.0,
        "variance_floor": 1.0e-6,
        "confidence_multiplier": 1.96,
        "minimum_gain": 0.0,
        "hyperparameter_selection": "none_predeclared_before_labels",
    }


def canonical_posterior_payload() -> dict[str, object]:
    return {
        "family": "fixed_hyperparameter_shrunk_action_utility_v1",
        "fit_unit": "one_local_posterior_per_H_fold_and_candidate_e",
        "prior_source": "label_derived_LOCO_global_prior",
        "support_observation": "exact_support_bacc_gain_vs_G_H",
        "support_labels_scope": "same_H_nonheldout_folds_only",
        "prior_strength": 8.0,
        "variance_floor": 1.0e-6,
        "confidence_multiplier": 1.96,
        "minimum_gain": 0.0,
        "hyperparameter_selection": "none_predeclared_before_labels",
        "no_cross_H_target_label_pooling_beyond_sealed_G_H": True,
        "no_shared_target_label_fit": True,
        "smooth_metric_role": "postseal_descriptive_only",
        "smooth_metric_may_affect_posterior_or_decision": False,
    }


def canonical_decision_payload() -> dict[str, object]:
    return {
        "family": "abstaining_label_aware_posterior_router_v1",
        "diagnostic_method_ids": ["B", "G_H", "R"],
        "hard_candidate_selection": True,
        "mixtures_allowed": False,
        "confidence_multiplier": 1.96,
        "minimum_gain": 0.0,
        "tie_tolerance": 1.0e-12,
        "tie_break": "lexicographic_action_id",
        "R_rule": "max_local_posterior_lower_bound_then_G_H_then_B",
        "G_H_rule": "max_sealed_loco_prior_mean_then_B",
        "B_abstention_when_no_positive_lower_bound": True,
        "all_center_fold_decisions_sealed_before_evaluation_capability": True,
        "all_permutation_null_decisions_sealed_before_evaluation_capability": True,
        "expected_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "evaluation_probabilities_may_affect_decision": False,
        "evaluation_labels_may_affect_decision": False,
        "smooth_metric_may_affect_decision": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "whole_case_oof_exact_nine_probability_ensemble_bacc",
        "baseline_method_id": "B",
        "global_method_id": "G_H",
        "router_method_id": "R",
        "primary_contrasts": ["R-G_H", "R-B", "G_H-B"],
        "primary_aggregation": "equal_weight_per_target_center",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_repeats_are_not_independent_units": True,
        "metrics": [
            "exact_bacc",
            "paired_R_minus_G_H",
            "paired_R_minus_B",
            "normalized_regret",
            "top1_accuracy",
            "tie_aware_top1_accuracy",
            "coverage",
            "source_selection_share",
        ],
        "permutation_unit": (
            "candidate_source_label_derangement_within_H_fold_and_support_case"
        ),
        "permutation_seed": 90_912_026,
        "permutation_count": 10_000,
        "permutation_baseline_B_fixed": True,
        "permutation_eight_Hxe_multiset_preserved": True,
        "permutation_evaluation_donors_used": False,
        "permutation_actions_sealed_before_evaluation_labels": True,
        "permutation_decision_tie_break": (
            "lexicographic_action_id_no_evaluation_utility_access"
        ),
        "confidence_level": 0.95,
        "exact_metric_only_may_enter_gates": True,
        "smooth_metric_role": "descriptive_only_no_gate_fields",
        "smooth_metric_may_affect_fit_selection_gate_or_decision": False,
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
            "capability_sealed_oof_audit"
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
            "/data/local/fixed_bank_label_aware_case_oof_ceiling_v1",
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
        "claim_role": "label_aware_known_bank_case_oof_ceiling",
        "candidate_generalization": "known_fixed_bank_reuse",
        "unseen_expert_transfer_claim": False,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "support_labels_used": True,
        "support_labels_local_only": True,
        "label_derived_LOCO_global_prior": True,
        "evaluation_labels_opened_only_after_all_decision_seals": True,
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
    }


__all__ = (
    "CLASSIFIER",
    "canonical_claim_boundary_payload",
    "canonical_decision_payload",
    "canonical_evaluation_payload",
    "canonical_global_prior_payload",
    "canonical_posterior_payload",
    "canonical_protocol_payload",
    "canonical_runtime_payload",
)
