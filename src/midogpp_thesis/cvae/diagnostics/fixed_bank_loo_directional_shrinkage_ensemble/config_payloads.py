"""Canonical config payloads for the directional-shrinkage diagnostic."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .experiment_contracts import (
    A1_EFFECTIVE_ROWS_PER_CLASS,
    A1_OTHER_ROWS_PER_CLASS,
    A1_OTHER_ROW_WEIGHT,
    A1_SELECTED_ROWS_PER_CLASS,
    A1_SELECTED_ROW_WEIGHT,
    ACTION_COUNT_PER_TARGET,
    ARM_IDS,
    BASE_ROWS_PER_SOURCE_CLASS,
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    DIRECTION_IDS,
    EVALUATION_SPLIT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_MIXED_CLASS_CASE_COUNT,
    EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
    EXPECTED_POSITIVE_ONLY_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    GENERATION_SEEDS,
    HARD_THRESHOLD,
    INPUT_ARTIFACT_IDS,
    K_GRID,
    METHOD_IDS,
    NULL_REPLICATES,
    NULL_SEED,
    PRE_TERMINAL_METHOD_IDS,
    PUBLICATION_STATUS,
    ROUTING_STATUS,
    SCRATCH_ROOT,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    STAGE_ID,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_DECISION,
    TERMINAL_ORACLE_IDS,
    TIE_TOLERANCE,
    TRAINING_SEEDS,
    UNIFORM_ROWS_PER_SOURCE_CLASS,
    WORKSTATION_PROFILE,
    W_GRID,
    W_RATIONAL_GRID,
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
    """Return the exact consumed-test, label-capability, and LOO protocol."""

    return {
        "schema_version": (
            "midogpp_fixed_bank_loo_directional_shrinkage_ensemble_protocol_v1"
        ),
        "stage": STAGE_ID,
        "dataset_family": DATASET_FAMILY,
        "evaluation_split": EVALUATION_SPLIT,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "consumed_test_data": True,
        "fresh_evidence": False,
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "exact_seed_pair_count": 9,
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
        "eligible_test_case_counts_by_center": dict(
            EXPECTED_CASE_COUNTS_BY_CENTER
        ),
        "mixed_class_case_count": EXPECTED_MIXED_CLASS_CASE_COUNT,
        "negative_only_case_count": EXPECTED_NEGATIVE_ONLY_CASE_COUNT,
        "positive_only_case_count": EXPECTED_POSITIVE_ONLY_CASE_COUNT,
        "single_class_cases_retained": True,
        "held_unit": "one_whole_case_or_group_c_within_target_center_H",
        "held_unit_count": EXPECTED_TOTAL_CASE_COUNT,
        "arbitrary_folds_used": False,
        "support_scope": "same_H_all_whole_cases_except_held_c",
        "terminal_evaluation_scope": "held_whole_case_c_only",
        "support_evaluation_whole_case_disjoint": True,
        "each_case_evaluated_exactly_once": True,
        "candidate_pool_excludes_target_H": True,
        "strict_outer_H_exclusion": True,
        "donor_query_scope": "q_not_in_H_or_e",
        "donor_prior_excludes_H_and_e": True,
        "support_labels_update_source_experts": False,
        "support_labels_update_shared_models": False,
        "all_physical_probabilities_globally_sealed_before_any_label_access": (
            True
        ),
        "role_scoped_label_capabilities_enforced": True,
        "route_scoped_support_grants_only": True,
        "all_held_case_endpoint_plans_sealed_before_terminal_label_access": (
            True
        ),
        "all_aggregate_method_seals_complete_before_terminal_label_access": (
            True
        ),
        "terminal_labels_never_train_tune_rank_or_select": True,
        "original_six_inputs_only": True,
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "stage50_outputs_used": False,
        "stage60_outputs_used": False,
        "stage70_prediction_scoring_or_policy_outputs_used": False,
        "label_free_cache_lineage": "stage70_derived_feature_cache_alias_only",
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_prediction_surfaces_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    """Return the immutable B/U/eight-A1 physical action surface."""

    return {
        "schema_version": (
            "fixed_bank_loo_directional_shrinkage_ensemble_action_library_v1"
        ),
        "action_ids": "B_U_and_eight_A1_source_actions",
        "baseline_action_id": "B",
        "uniform_control_action_id": "U",
        "source_action_id_format": "A1::source={source_center}",
        "candidate_source_count_per_target": 8,
        "physical_action_count_per_target": ACTION_COUNT_PER_TARGET,
        "target_task_count": TARGET_TASK_COUNT,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "baseline_rows_per_source_class": BASE_ROWS_PER_SOURCE_CLASS,
        "uniform_rows_per_source_class": UNIFORM_ROWS_PER_SOURCE_CLASS,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "A1_selected_rows_per_class": A1_SELECTED_ROWS_PER_CLASS,
        "A1_other_rows_per_class": A1_OTHER_ROWS_PER_CLASS,
        "A1_selected_row_weight": A1_SELECTED_ROW_WEIGHT,
        "A1_other_row_weight": A1_OTHER_ROW_WEIGHT,
        "A1_selected_row_weight_fraction": "23/16",
        "A1_other_row_weight_fraction": "7/8",
        "A1_effective_rows_per_class": A1_EFFECTIVE_ROWS_PER_CLASS,
        "action_strength_sweep_used": False,
        "class_conditional_action_variant_used": False,
        "source_pair_action_used": False,
        "geometry_selection_used": False,
        "target_expert_used": False,
        "probabilities_averaged_exact_nine_before_directional_scoring": True,
        "hard_probability_threshold": HARD_THRESHOLD,
        "hard_threshold_equal_maps_to_positive": True,
        "previous_probability_surfaces_used": False,
    }


def canonical_directional_ensemble_payload() -> dict[str, object]:
    """Return the exact directional score, arm grid, and composition rules."""

    arm_grid = [
        {
            "arm_id": arm_id,
            "K": k,
            "w": weight,
            "w_numerator": numerator,
            "w_denominator": denominator,
        }
        for k in K_GRID
        for (numerator, denominator), weight, arm_id in zip(
            W_RATIONAL_GRID, W_GRID, ARM_IDS[(K_GRID.index(k) * 3) :]
        )
    ]
    return {
        "schema_version": (
            "fixed_bank_loo_directional_shrinkage_ensemble_dcse_v1"
        ),
        "method_id": "DCSE_LOO",
        "direction_ids": list(DIRECTION_IDS),
        "branch_definition": "baseline_B_hard_prediction_before_candidate_flip",
        "zero_to_one_branch": "B_hard_equals_0_candidate_hard_equals_1",
        "one_to_zero_branch": "B_hard_equals_1_candidate_hard_equals_0",
        "support_score": (
            "pooled_additive_confusion_count_directional_exact_bacc_gain_vs_B"
        ),
        "per_case_bacc_used_for_scoring_or_selection": False,
        "G_definition": (
            "equal_center_mean_directional_gain_over_query_q_not_in_H_or_e"
        ),
        "G_query_scope": "q_not_in_H_or_e",
        "G_equal_center_aggregation": True,
        "K_grid": list(K_GRID),
        "w_grid": list(W_GRID),
        "w_rational_grid": [
            f"{numerator}/{denominator}"
            for numerator, denominator in W_RATIONAL_GRID
        ],
        "arm_grid": arm_grid,
        "arm_count": len(ARM_IDS),
        "arm_id_order": list(ARM_IDS),
        "all_arm_identities_retained_when_selected_endpoints_duplicate": True,
        "source_rank_rule": "descending_G_then_numeric_source",
        "top_K_scope": "eight_legal_non_target_sources_ranked_by_G",
        "endpoint_score": "w_times_S_plus_one_minus_w_times_G",
        "endpoint_score_arithmetic": "exact_rational_until_final_tie_check",
        "off_action_id": "OFF",
        "off_score": 0,
        "off_probability_source": "B",
        "selection_candidate_order": "OFF_then_numeric_source",
        "final_tie_tolerance": TIE_TOLERANCE,
        "arm_composition": (
            "mean_of_nine_selected_endpoint_probabilities_per_B_hard_branch"
        ),
        "off_endpoint_contributes_B_probability": True,
        "probabilities_averaged_before_final_threshold": True,
        "sole_final_threshold": HARD_THRESHOLD,
        "final_threshold_equal_maps_to_positive": True,
        "matched_G_method_id": "G_directional_matched",
        "matched_G_pipeline": "identical_nine_arm_pipeline_with_S_set_equal_to_G",
        "matched_G_uses_target_support_labels": False,
        "hidden_arm_selection_used": False,
        "hyperparameter_search_used": False,
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "method_ids": list(METHOD_IDS),
        "pre_terminal_method_ids": list(PRE_TERMINAL_METHOD_IDS),
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "B_role": "fixed_equal_union_baseline_and_OFF_probability",
        "U_role": "fixed_uniform_A1_control",
        "DCSE_LOO_role": "primary_nine_arm_whole_case_loo_directional_ensemble",
        "G_directional_matched_role": "matched_pipeline_with_S_equal_to_G",
        "DLOO_raw_role": "raw_directional_whole_case_loo_control",
        "LOO_frequency_committee_role": (
            "nested_delete_one_support_frequency_committee_control"
        ),
        "O_directional_static_role": "terminal_directional_static_oracle",
        "O_case_directional_role": "terminal_case_directional_oracle",
        "hard_vote_control": True,
        "unique_action_mean_control": True,
        "uniform_A1_control": True,
        "direction_decomposition_control": True,
        "nested_delete_one_support_frequency_control": True,
        "leave_one_arm_ablation": True,
        "whole_pipeline_delete_one_center_recomputation": True,
        "terminal_oracles_are_pre_terminal_methods": False,
        "terminal_oracles_can_train_tune_rank_select_or_seal": False,
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "center_pooled_exact_bacc_from_int64_confusion_sums",
        "outer_inference_unit": "target_center",
        "outer_inference_unit_count": len(CENTERS),
        "technical_seed_cells_are_not_independent_units": True,
        "primary_descriptive_contrasts": [
            "DCSE_LOO-B",
            "DCSE_LOO-U",
            "DCSE_LOO-G_directional_matched",
        ],
        "predeclared_descriptive_success_checks": [
            "full_DCSE_LOO_minus_B_strictly_positive",
            "full_DCSE_LOO_minus_U_strictly_positive",
            "both_primary_B_and_U_contrasts_positive_in_all_nine_"
            "whole_pipeline_center_deletions",
            "at_least_eight_of_nine_center_DCSE_LOO_minus_B_deltas_nonnegative",
            "every_leave_one_arm_DCSE_LOO_minus_B_contrast_strictly_positive",
        ],
        "matched_G_contrast_is_a_success_gate": False,
        "nominal_t_interval_is_a_success_gate": False,
        "jackknife_interval_is_a_success_gate": False,
        "null_summary_is_a_success_gate": False,
        "descriptive_t_interval": "two_sided_t8_over_nine_center_contrasts",
        "descriptive_jackknife": "whole_pipeline_delete_one_center_recomputation",
        "confusion_count_dtype": "int64",
        "scientific_reduction_dtype": "float64",
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "per_case_bacc_persisted_or_used": False,
        "results_are_terminal_consumed_test_diagnostics": True,
    }


def canonical_nulls_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_dcse_candidate_identity_null_v1",
        "family": "candidate_identity_null",
        "algorithm": "splitmix64_route_candidate_block_permutation_v1",
        "replicates": NULL_REPLICATES,
        "seed": NULL_SEED,
        "matrix_shape": [NULL_REPLICATES, EXPECTED_TOTAL_CASE_COUNT, 8],
        "matrix_dtype": "uint8",
        "permutation_scope": "one_route_local_eight_candidate_block",
        "same_permutation_for_paired_directions": True,
        "scrambled_surface": "support_S_candidate_identities_only",
        "baseline_B_fixed": True,
        "donor_prior_G_fixed": True,
        "physical_probability_surface_fixed": True,
        "canonical_endpoint_and_method_decisions_fixed": True,
        "null_replicate_endpoint_selections_recomputed": True,
        "plan_sealed_before_terminal_labels": True,
        "null_can_change_endpoint_or_method_decisions": False,
        "null_can_change_thresholds_or_success_checks": False,
        "output_role": "descriptive_only",
        "exchangeability_claimed": False,
        "p_value_computed": False,
        "confirmatory_gate_defined": False,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": (
            "fixed_bank_loo_directional_shrinkage_ensemble_runtime_v1"
        ),
        "workstation_profile": WORKSTATION_PROFILE,
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
        "persistent_source_workers": True,
        "persistent_generation_worker_count": 2,
        "multiprocessing_start_method": "spawn",
        "gpu_generation_phase_precedes_cpu_phase": True,
        "cuda_visible_devices_cleared_before_cpu_phase": True,
        "parent_cuda_context_forbidden": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "tf32_enabled": False,
        "amp_enabled": False,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
        "scientific_reductions_dtype": "float64",
        "generated_cache_format": "float32_npy_memmap",
        "probability_surface_format": "sealed_compressed_float32_npz",
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "launch_blas_threads": 1,
        "maximum_total_cpu_threads": 12,
        "cpu_phase_blas_thread_environment": {
            "OMP_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "3",
            "NUMEXPR_NUM_THREADS": "3",
        },
        "minimum_logical_cpu_count": 12,
        "minimum_physical_ram_bytes": 107_374_182_400,
        "minimum_artifact_disk_free_bytes": 8_589_934_592,
        "minimum_gpu_free_mib_per_device": 18_000,
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": SOURCE_PREFIX_ROWS_PER_CLASS,
        "target_task_count": TARGET_TASK_COUNT,
        "target_action_identity_count": len(CENTERS) * ACTION_COUNT_PER_TARGET,
        "physical_actions_per_target_task": ACTION_COUNT_PER_TARGET,
        "target_probability_cell_count": TARGET_PROBABILITY_CELL_COUNT,
        "target_unique_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "maximum_total_classifier_fit_count": TARGET_PROBABILITY_CELL_COUNT,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "owned_task_checkpoint_replay_allowed": False,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "resume_policy": (
            "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
        ),
        "successful_phase_checkpoint_cleanup_after_validated_global_seal": True,
        "two_fresh_process_validation_required": True,
        "previous_stage90_scratch_reuse_forbidden": True,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "routing_status": ROUTING_STATUS,
        "claim_role": CLAIM_ROLE,
        "bounded_interpretation": (
            "no_stable_incremental_target_support_advantage_vs_G"
        ),
        "consumed_test_data": True,
        "method_development_is_posthoc": True,
        "fresh_evidence": False,
        "fresh_confirmation": False,
        "terminal_stage90_diagnostic": True,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "downstream_utility_claimed": False,
        "source_expert_updated": False,
        "target_expert_used": False,
        "shared_model_updated_with_target_labels": False,
        "action_selection_authorized": False,
        "action_geometry_update_authorized": False,
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
        "previous_stage90_amendments_used": False,
        "previous_prediction_surface_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
        "confirmatory_p_value_or_gate_used": False,
    }


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + (
    "CLASSIFIER",
)
