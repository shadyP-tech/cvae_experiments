"""Canonical config sections for the multi-challenger diagnostic."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    ACTION_COUNT_PER_TARGET,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CALIBRATION_ALPHA,
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    FEATURE_ALPHA,
    FEATURE_NAMES,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    INTERCEPT_ALPHA,
    MARGIN_Z,
    METHOD_IDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    PRE_EVALUATION_METHOD_IDS,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    QUERY_ALPHA,
    ROUTING_STATUS,
    SCRATCH_ROOT,
    SOURCE_ALPHA,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    STAGE_ID,
    SUPPORT_PRIOR_CASES,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_DECISION,
    TERMINAL_ORACLE_IDS,
    TOP_K,
    TRAINING_SEEDS,
    WORKSTATION_PROFILE,
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
        "schema_version": "midogpp_fixed_bank_multi_challenger_protocol_v1",
        "stage": STAGE_ID,
        "dataset_family": DATASET_FAMILY,
        "evaluation_split": "test",
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "exact_seed_pair_count": 9,
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "partition_unit": "whole_case_within_target_center",
        "partition_seed": OOF_FOLD_SEED,
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "fold_count": OOF_FOLD_COUNT,
        "role_rotation": "eval=f_calibration=(f+1)%5_selection=remaining_three",
        "selection_calibration_evaluation_case_disjoint": True,
        "each_case_evaluated_exactly_once": True,
        "strict_outer_H_exclusion": True,
        "strict_donor_query_q_exclusion": True,
        "strict_candidate_source_e_exclusion": True,
        "strict_H_q_e_distinct": True,
        "all_probabilities_and_features_sealed_before_any_label_access": True,
        "each_fold_decision_invariant_to_held_evaluation_labels": True,
        "terminal_scoring_occurs_only_after_all_45_decision_seals": True,
        "original_six_inputs_only": True,
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "previous_stage90_outputs_or_predictions_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_a1_action_library_v1",
        "action_ids": "B_U_and_eight_A1_source_actions",
        "physical_action_count_per_target": ACTION_COUNT_PER_TARGET,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "baseline_rows_per_source_class": 128,
        "uniform_rows_per_source_class": 144,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "A1_selected_rows_per_class": 256,
        "A1_other_rows_per_class": 128,
        "A1_selected_row_weight": 23.0 / 16.0,
        "A1_other_row_weight": 7.0 / 8.0,
        "A1_selected_row_weight_fraction": "23/16",
        "A1_other_row_weight_fraction": "7/8",
        "candidate_pool_excludes_target_H": True,
        "probabilities_averaged_exact_nine_before_any_routing": True,
        "previous_probability_surfaces_used": False,
    }


def canonical_flip_features_payload() -> dict[str, object]:
    return {
        "family": "baseline_anchored_threshold_flip_case_features_v1",
        "hard_threshold": 0.5,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "features_label_free": True,
        "no_B_vs_U_flip_feature": True,
        "probabilities_ensemble_averaged_before_threshold": True,
        "zero_flip_rows_retained": True,
        "feature_hyperparameters_selected_after_labels": False,
    }


def canonical_routing_payload() -> dict[str, object]:
    return {
        "family": "support_anchored_hierarchical_multi_challenger_flip_router_v1",
        "candidate_menu_top_k": TOP_K,
        "candidate_menu_always_includes": "B",
        "candidate_menu_pool": "eight_frozen_A1_source_actions_only",
        "support_ranking_reference": "B",
        "support_ranking_objective": "pooled_exact_bacc_gain_vs_B",
        "support_prior_cases": SUPPORT_PRIOR_CASES,
        "support_prior_selected_after_labels": False,
        "anchor": "S_static_if_positive_else_B",
        "anchor_fallback": "S_static_or_B_when_S_static_equals_B",
        "selection_labels_use": "deterministic_top3_menu_and_S_static_anchor_only",
        "donor_response": "pooled_binomial_correct_flip_counts_by_direction",
        "donor_directions": ["0to1", "1to0"],
        "donor_model_families": ["G", "R", "P"],
        "G_model": (
            "pooled_direction_intercept_plus_penalized_candidate_source_"
            "and_query_effects"
        ),
        "R_model": "G_plus_fixed_label_free_case_action_features",
        "P_model": "R_capacity_with_blocked_complete_case_feature_permutation",
        "feature_alpha": FEATURE_ALPHA,
        "source_alpha": SOURCE_ALPHA,
        "query_alpha": QUERY_ALPHA,
        "intercept_alpha": INTERCEPT_ALPHA,
        "direction_model_hyperparameters_selected_after_labels": False,
        "calibration_model": (
            "target_local_menu_bound_gaussian_prior_direction_offsets_"
            "with_laplace_posterior_variance"
        ),
        "calibration_prior": "zero_mean_gaussian_precision_alpha",
        "sparse_calibration_policy": (
            "retain_donor_probability_with_prior_or_laplace_uncertainty"
        ),
        "calibration_alpha": CALIBRATION_ALPHA,
        "calibration_labels_use": "menu_actions_one_disjoint_calibration_fold_only",
        "target_labels_update_shared_model": False,
        "uncertainty_components": ["epistemic_parameter", "target_calibration"],
        "residual_outcome_variance_in_action_standard_error": False,
        "action_margin": "winner_expected_gain_minus_runner_up_expected_gain",
        "action_margin_z": MARGIN_Z,
        "action_margin_lcb_threshold": 0.0,
        "switch_rule": "best_nonanchor_only_when_winner_runner_up_margin_lcb_gt_zero",
        "invalid_calibration_fallback": "anchor",
        "single_challenger_control": "legacy_F_S_recomputed_from_original_inputs",
        "primary_router": PRIMARY_METHOD_ID,
        "calibrated_case_confidence_or_safety_claimed": False,
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "method_ids": list(METHOD_IDS),
        "pre_evaluation_method_ids": list(PRE_EVALUATION_METHOD_IDS),
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "B_role": "fixed_equal_union_baseline_and_menu_member",
        "U_role": "uniform_action_control_not_a_menu_candidate",
        "S_static_role": "support_ranked_static_anchor_or_B",
        "F_single_role": "recomputed_legacy_single_challenger_control",
        "G_multi_role": "multi_challenger_global_direction_control",
        "R_multi_role": "primary_hierarchical_label_free_feature_router",
        "P_multi_role": "blocked_permuted_feature_same_capacity_control",
        "O_menu_role": "terminal_best_per_case_action_within_sealed_menu",
        "O_binary_role": "terminal_best_per_case_of_B_and_S_static",
        "O_static_role": "terminal_best_static_A1_or_B",
        "O_case_role": "terminal_best_case_action_over_B_and_all_A1",
    }


def canonical_evaluation_payload() -> dict[str, object]:
    contrasts = [
        "R_multi-B",
        "R_multi-U",
        "R_multi-F_single",
        "R_multi-G_multi",
        "R_multi-P_multi",
        "R_multi-S_static",
    ]
    return {
        "primary_endpoint": "center_pooled_exact_bacc_whole_case_oof",
        "primary_method": PRIMARY_METHOD_ID,
        "primary_contrasts": contrasts,
        "diagnostic_recoverability_gate": {
            "gate_id": "all_primary_contrast_outer_center_lcbs_positive_v1",
            "lcb_field": "one_sided_95_lcb",
            "threshold": 0.0,
            "comparison": "strictly_greater_than",
            "required_contrast_count": len(contrasts),
            "pass_status": "PASS",
            "fail_status": "FAIL",
            "diagnostic_only": True,
        },
        "routing_metrics": [
            "top1_oracle_agreement",
            "top3_menu_oracle_coverage",
            "spearman",
            "normalized_oracle_gap",
            "fold_stability",
        ],
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": 9,
        "case_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "case_cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "scientific_reductions_dtype": "float64",
        "raw_labels_persisted": False,
        "per_case_bacc_persisted_or_used": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        # Shared frozen-source runtime contract.  Keep the experiment-local
        # name below as a redundant topology assertion, not as an alias that
        # the shared generator has to guess.
        "generation_workers_per_device": 1,
        "source_workers_per_device": 1,
        "source_generation_worker_count": 2,
        "persistent_source_workers": True,
        "gpu_source_phase_precedes_cpu_phase": True,
        "parent_cuda_context_forbidden": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "model_workers": 4,
        "model_threads_per_worker": 3,
        "bootstrap_workers": 4,
        "bootstrap_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "target_task_count": TARGET_TASK_COUNT,
        "physical_actions_per_target_task": ACTION_COUNT_PER_TARGET,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "maximum_total_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "direction_fit_task_count": 54,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": "hash_validated_phase_checkpoints_plus_deterministic_replay",
        "clean_scratch_only_after_closed_world_validation_pass": True,
        "two_fresh_process_validation_required": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "routing_status": ROUTING_STATUS,
        "claim_role": CLAIM_ROLE,
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "support_labels_used": True,
        "selection_calibration_evaluation_labels_role_separated_per_decision": True,
        "each_fold_plan_hash_invariant_to_its_held_evaluation_labels": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage50_stage60_stage70_or_another_experiment": False,
        "previous_stage90_outputs_or_scratch_used": False,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + (
    "CLASSIFIER",
)
