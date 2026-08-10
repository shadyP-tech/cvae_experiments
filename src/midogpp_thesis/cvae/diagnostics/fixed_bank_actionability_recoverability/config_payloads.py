"""Canonical payloads for the terminal actionability/recoverability study."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    A0_EFFECTIVE_ROWS_PER_CLASS,
    A0_OTHER_ROW_WEIGHT,
    A0_PHYSICAL_ROWS_PER_CLASS,
    A0_SELECTED_ROW_WEIGHT,
    A1_EFFECTIVE_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_PHYSICAL_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    BASE_ROWS_PER_CLASS,
    BASE_ROWS_PER_SOURCE_CLASS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CENTERS,
    CLAIM_ROLE,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_CENTER_FOLD_COUNT,
    EXPECTED_LOGICAL_ACTION_COUNT_PER_TARGET,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_PHYSICAL_ACTION_COUNT_PER_TARGET,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TARGET_LOGICAL_ACTION_IDENTITY_COUNT,
    EXPECTED_TARGET_PHYSICAL_ACTION_IDENTITY_COUNT,
    EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT,
    GENERATION_SEEDS,
    GEOMETRY_IDS,
    INPUT_ARTIFACT_IDS,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    OTHER_ROWS_PER_CLASS,
    PER_GEOMETRY_METHOD_IDS,
    PRE_EVALUATION_METHOD_IDS,
    PUBLICATION_STATUS,
    SELECTED_ROWS_PER_CLASS,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    STAGE_ID,
    TERMINAL_ORACLE_IDS,
    TRAINING_SEEDS,
    UNIFORM_ROWS_PER_CLASS,
    UNIFORM_ROWS_PER_SOURCE_CLASS,
)
from .protocol import canonical_consumed_test_protocol


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

ROUTER_FEATURE_NAMES = (
    "intercept",
    "baseline_probability_mean",
    "baseline_probability_sd",
    "uniform_delta_mean",
    "uniform_delta_abs_mean",
    "candidate_delta_mean",
    "candidate_delta_abs_mean",
    "candidate_delta_sd",
    "candidate_disagreement_vs_b",
    "candidate_disagreement_vs_u",
    "candidate_entropy_mean",
    "candidate_near_threshold_rate",
    "candidate_seed_sd_mean",
)
RIDGE_ALPHA = 1.0
PROBABILITY_EPSILON = 1.0e-4
NEAR_THRESHOLD_HALF_WIDTH = 0.1
STANDARDIZATION_SCALE_FLOOR = 1.0e-3
PERMUTATION_NAMESPACE = (
    "midogpp_actionability_recoverability_case_action_features_v1"
)


def canonical_protocol_payload() -> dict[str, object]:
    protocol = canonical_consumed_test_protocol().to_payload()
    return {
        **protocol,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "exact_seed_pair_count": len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(EXPECTED_CASE_COUNTS_BY_CENTER),
        "mixed_class_case_count": EXPECTED_MIXED_CLASS_CASE_COUNT,
        "negative_only_case_count": EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
        "positive_only_case_count": EXPECTED_POSITIVE_ONLY_CASE_COUNT,
        "single_class_cases_retained": True,
        "per_case_bacc_stored_or_used": False,
        "candidate_pool_excludes_target_H": True,
        "candidate_source_count_per_target": (
            EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
        ),
        "oof_fold_count": OOF_FOLD_COUNT,
        "partition_seed": OOF_FOLD_SEED,
        "partition_namespace": OOF_PARTITION_NAMESPACE,
        "partition_unit": "whole_case_within_target_center",
        "each_case_evaluated_exactly_once": True,
        "support_scope": "other_four_same_H_whole_case_folds",
        "evaluation_scope": "one_held_same_H_whole_case_fold",
        "heldout_fold_absent_from_its_support_and_decision_fit": True,
        "cross_role_case_reuse_only_in_other_folds": True,
        "center_fold_decision_count": EXPECTED_CENTER_FOLD_COUNT,
        "strict_outer_H_exclusion": True,
        "strict_nested_query_q_exclusion": True,
        "target_labels_used_for_shared_model": False,
        "terminal_endpoint": (
            "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        ),
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "original_six_inputs_only": True,
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "label_free_cache_lineage": "stage70_derived_feature_cache_alias_only",
        "previous_stage90_outputs_used": False,
        "signed_error_output_or_amendment_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_action_library_v1",
        "geometry_ids": list(GEOMETRY_IDS),
        "baseline_action_id": "B",
        "baseline_physical_fit_required": True,
        "uniform_action_id": "U",
        "uniform_physical_fit_required": True,
        "baseline_and_uniform_geometry_id": None,
        "source_action_id_format": "{geometry_id}::source={source_center}",
        "source_stream_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "baseline_rows_per_source_class": BASE_ROWS_PER_SOURCE_CLASS,
        "baseline_rows_per_class": BASE_ROWS_PER_CLASS,
        "uniform_rows_per_source_class": UNIFORM_ROWS_PER_SOURCE_CLASS,
        "uniform_rows_per_class": UNIFORM_ROWS_PER_CLASS,
        "uniform_action_is_one_physical_surface_shared_across_geometries": True,
        "A0": {
            "selected_rows_per_class": SELECTED_ROWS_PER_CLASS,
            "other_rows_per_class": OTHER_ROWS_PER_CLASS,
            "physical_rows_per_class": A0_PHYSICAL_ROWS_PER_CLASS,
            "selected_row_weight": A0_SELECTED_ROW_WEIGHT,
            "other_row_weight": A0_OTHER_ROW_WEIGHT,
            "effective_rows_per_class": A0_EFFECTIVE_ROWS_PER_CLASS,
        },
        "A1": {
            "selected_rows_per_class": SELECTED_ROWS_PER_CLASS,
            "other_rows_per_class": OTHER_ROWS_PER_CLASS,
            "physical_rows_per_class": A1_PHYSICAL_ROWS_PER_CLASS,
            "reuses_exact_A0_row_ids": True,
            "selected_row_weight": A1_SELECTED_ROW_WEIGHT,
            "other_row_weight": A1_OTHER_ROW_WEIGHT,
            "selected_row_weight_fraction": "23/16",
            "other_row_weight_fraction": "7/8",
            "effective_rows_per_class": A1_EFFECTIVE_ROWS_PER_CLASS,
        },
        "A0_and_A1_effective_training_mass_matched": True,
        "action_strength_sweep_used": False,
        "class_conditional_action_variant_used": False,
        "source_pair_action_used": False,
        "geometry_selection_used": False,
        "target_expert_used": False,
        "logical_action_count_per_target": (
            EXPECTED_LOGICAL_ACTION_COUNT_PER_TARGET
        ),
        "physical_action_count_per_target": (
            EXPECTED_PHYSICAL_ACTION_COUNT_PER_TARGET
        ),
        "target_logical_action_identity_count": (
            EXPECTED_TARGET_LOGICAL_ACTION_IDENTITY_COUNT
        ),
        "target_physical_action_identity_count": (
            EXPECTED_TARGET_PHYSICAL_ACTION_IDENTITY_COUNT
        ),
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "unique_classifier_fit_count": EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT,
        "probabilities_averaged_over_exact_nine_before_routing_or_scoring": True,
        "global_action_surface_sealed_before_any_label_access": True,
        "previous_stage90_probability_arrays_used": False,
    }


def canonical_recoverability_payload() -> dict[str, object]:
    return {
        "family": "strict_loco_case_action_utility_ridge_v1",
        "decision_unit": "whole_case_candidate_action_within_geometry",
        "feature_family": "baseline_anchored_label_free_case_action_features_v1",
        "feature_names": list(ROUTER_FEATURE_NAMES),
        "feature_count_including_intercept": len(ROUTER_FEATURE_NAMES),
        "probability_clip_epsilon": PROBABILITY_EPSILON,
        "near_threshold_half_width": NEAR_THRESHOLD_HALF_WIDTH,
        "baseline_predicted_class_branch_used": False,
        "candidate_identity_one_hot_or_learned_factor_used": False,
        "response": "class_balanced_proper_loss_gain_vs_u",
        "additive_diagnostic_response_allowed_but_not_primary": (
            "pooled_bacc_additive_gain"
        ),
        "response_uses_evaluation_fold": False,
        "outer_target_H_absent_from_fit_standardization_and_alpha_selection": True,
        "heldout_query_q_absent_from_nested_fit_and_standardization": True,
        "standardization_fit_scope": "legal_other_center_donor_rows_only",
        "standardization_scale_floor": STANDARDIZATION_SCALE_FLOOR,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_alpha_grid_or_selection_used": False,
        "G_definition": "candidate_specific_intercept_only_gain_ridge",
        "G_is_static_within_target_geometry": True,
        "R_definition": "aligned_label_free_case_action_gain_ridge",
        "P_definition": "same_capacity_candidate_block_permuted_gain_ridge",
        "G_R_P_selection": "maximum_predicted_gain_source_action",
        "G_R_P_frozen_minimum_gain": 0.0,
        "G_R_P_fallback_action": "U",
        "permutation_namespace": PERMUTATION_NAMESPACE,
        "permutation_unit": "candidate_block_within_query_case_geometry",
        "permutation_is_nonzero_cyclic_derangement": True,
        "permutation_applied_before_donor_fit": True,
        "permutation_applied_before_target_inference": True,
        "permutation_refits_same_capacity_model": True,
        "permutation_changes_labels_or_response_targets": False,
        "S_y_definition": (
            "same_H_support_pooled_exact_bacc_static_action_selector"
        ),
        "S_y_support_scope": "same_H_nonheldout_whole_cases_only",
        "S_y_evaluation_fold_absent": True,
        "S_y_candidate_set": "U_plus_eight_frozen_source_actions_per_geometry",
        "S_y_uniform_action_is_candidate": True,
        "S_y_tie_break": "U_then_numeric_source_center",
        "S_y_may_select_geometry_features_or_hyperparameters": False,
        "target_support_labels_update_shared_model": False,
        "prelabel_feature_surface_sealed_before_other_center_labels": True,
        "G_R_P_model_seals_written_before_target_support_labels": True,
        "B_U_G_R_P_S_y_decisions_sealed_before_evaluation_labels": True,
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "pre_evaluation_method_ids": list(PRE_EVALUATION_METHOD_IDS),
        "per_geometry_method_ids": list(PER_GEOMETRY_METHOD_IDS),
        "method_rows_carry_geometry_id": True,
        "global_B_has_no_geometry_id": True,
        "global_B_is_scored_but_never_selectable": True,
        "B_role": "fixed_equal_union_baseline",
        "U_role": "shared_144_per_source_uniform_action_control",
        "G_role": "target_independent_global_source_quality_control",
        "R_role": "label_free_case_action_recoverability_model",
        "P_role": "same_capacity_permuted_feature_alignment_control",
        "S_y_role": "evaluation_disjoint_labeled_target_support_ceiling",
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "O_static_role": "terminal_best_fixed_source_action_per_H_and_geometry",
        "O_case_role": "terminal_best_case_action_per_H_case_and_geometry",
        "terminal_oracles_are_pre_evaluation_methods": False,
        "terminal_oracles_can_train_select_or_seal_a_method": False,
        "geometry_winner_selection_authorized": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": (
            "center_pooled_exact_bacc_over_whole_case_oof_predictions"
        ),
        "actionability_contrasts": ["O_static-U", "O_case-O_static"],
        "recoverability_contrasts": ["R-U", "R-G", "R-P", "S_y-U"],
        "secondary_contrasts": ["U-B", "G-U", "S_y-R"],
        "contrasts_reported_separately_by_geometry": True,
        "cross_geometry_selection_or_promotion_contrast": False,
        "center_utility": "pooled_exact_bacc_from_aggregated_confusion_sums",
        "case_sufficient_statistic_fields": [
            "n_positive",
            "true_positive",
            "n_negative",
            "true_negative",
        ],
        "per_case_bacc_stored_or_used": False,
        "single_class_cases_retained": True,
        "primary_aggregation": "equal_weight_per_target_center",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_repeats_are_not_independent_units": True,
        "whole_case_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "whole_case_cluster_bootstrap_seed": BOOTSTRAP_SEED,
        "whole_case_cluster_bootstrap_scope": (
            "resample_cases_within_each_center_then_equal_center_aggregate"
        ),
        "center_level_confidence_interval": "paired_t_interval_over_nine_centers",
        "confidence_level": 0.95,
        "rank_stability_unit": "independent_same_H_support_and_evaluation_folds",
        "normalized_oracle_gap_reference": "U_to_O_static_with_degenerate_flag",
        "complementarity_reports_class_conditional_correctness": True,
        "O_static_and_O_case_computed_after_terminal_label_open_only": True,
        "results_are_terminal_consumed_test_diagnostics": True,
        "result_may_authorize_routing_action_geometry_or_later_experiment": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "source_generation_worker_count": 2,
        "persistent_source_workers": True,
        "gpu_source_phase_precedes_cpu_phase": True,
        "probability_materialization_device": "cpu",
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "model_workers": 4,
        "model_threads_per_worker": 3,
        "bootstrap_workers": 4,
        "bootstrap_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_forbidden": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "launch_blas_threads": 1,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 12_884_901_888,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "target_task_count": len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
        "physical_actions_per_target_task": EXPECTED_PHYSICAL_ACTION_COUNT_PER_TARGET,
        "logical_actions_per_target": EXPECTED_LOGICAL_ACTION_COUNT_PER_TARGET,
        "target_probability_cell_count": EXPECTED_TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT,
        "maximum_total_classifier_fit_count": EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT,
        "load_each_source_memmap_once_per_target_seed_task": True,
        "scratch_preference": [
            "/data/local/fixed_bank_actionability_recoverability_v1",
            "artifact_parent",
        ],
        "resume_policy": (
            "hash_validated_source_prediction_task_resume_plus_"
            "deterministic_phase_replay"
        ),
        "clean_scratch_only_after_closed_world_validation_pass": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": "DO_NOT_PROMOTE",
        "claim_role": CLAIM_ROLE,
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "terminal_stage90_diagnostic": True,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "support_labels_used": True,
        "support_labels_used_for_S_y_within_frozen_geometry_only": True,
        "other_center_labels_used_for_strict_outer_H_nested_q_model_fit": True,
        "evaluation_labels_opened_only_after_all_preterminal_seals": True,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "action_geometry_update_authorized": False,
        "geometry_selection_authorized": False,
        "policy_update_authorized": False,
        "model_update_authorized": False,
        "expert_update_authorized": False,
        "promotion_eligible": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "previous_stage90_outputs_used": False,
        "previous_prediction_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
        "signed_error_output_or_amendment_used": False,
    }


__all__ = (
    "CLASSIFIER",
    "PERMUTATION_NAMESPACE",
    "NEAR_THRESHOLD_HALF_WIDTH",
    "PROBABILITY_EPSILON",
    "RIDGE_ALPHA",
    "ROUTER_FEATURE_NAMES",
    "STANDARDIZATION_SCALE_FLOOR",
    "canonical_action_library_payload",
    "canonical_claim_boundary_payload",
    "canonical_controls_payload",
    "canonical_evaluation_payload",
    "canonical_protocol_payload",
    "canonical_recoverability_payload",
    "canonical_runtime_payload",
)
