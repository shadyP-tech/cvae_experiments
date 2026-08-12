"""Canonical config sections for the fixed flip-router diagnostic."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from .constants import (
    ACTION_COUNT_PER_TARGET,
    A1_OTHER_SAMPLE_WEIGHT,
    A1_SELECTED_SAMPLE_WEIGHT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CENTERS,
    FEATURE_NAMES,
    GENERATION_SEEDS,
    HARD_THRESHOLD,
    METHOD_IDS,
    MIN_GAIN,
    OOF_FOLD_COUNT,
    OOF_FOLD_SEED,
    OOF_PARTITION_NAMESPACE,
    OTHER_COUNT_PER_CLASS,
    PRE_EVALUATION_METHOD_IDS,
    RIDGE_ALPHA,
    RUNNER_UP_MARGIN,
    SAFE_Z,
    SCRATCH_ROOT,
    SELECTED_COUNT_PER_CLASS,
    SOURCE_PREFIX_ROWS_PER_CLASS,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_ORACLE_IDS,
    TRAINING_SEEDS,
    VARIANCE_FLOOR,
    WORKSTATION_PROFILE,
)
from .experiment_contracts import (
    CLAIM_ROLE,
    INPUT_ARTIFACT_IDS,
    PUBLICATION_STATUS,
    STAGE_ID,
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
        "schema_version": "midogpp_fixed_bank_labeled_support_flip_protocol_v1",
        "stage": STAGE_ID,
        "dataset_family": "MIDOG++",
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
        "all_probabilities_sealed_before_any_label_access": True,
        "each_fold_decision_invariant_to_and_sealed_without_its_held_evaluation_labels": True,
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
        "A1_selected_rows_per_class": SELECTED_COUNT_PER_CLASS,
        "A1_other_rows_per_class": OTHER_COUNT_PER_CLASS,
        "A1_selected_row_weight": A1_SELECTED_SAMPLE_WEIGHT,
        "A1_other_row_weight": A1_OTHER_SAMPLE_WEIGHT,
        "A1_selected_row_weight_fraction": "23/16",
        "A1_other_row_weight_fraction": "7/8",
        "candidate_pool_excludes_target_H": True,
        "probabilities_averaged_exact_nine_before_any_routing": True,
        "previous_probability_surfaces_used": False,
    }


def canonical_flip_features_payload() -> dict[str, object]:
    return {
        "family": "baseline_anchored_threshold_flip_case_features_v1",
        "hard_threshold": HARD_THRESHOLD,
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
        "family": "support_calibrated_case_conditional_threshold_flip_v1",
        "donor_model": "two_head_ridge_tp_tn_contribution",
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_alpha_selection_used": False,
        "variance_floor": VARIANCE_FLOOR,
        "static_challenger_candidates": "eight_A1_source_actions_only",
        "static_challenger_fallback": "B_when_no_positive_authorized_A1",
        "G_static_selection_objective": "unweighted_least_squares_exact_per_q_e_pooled_bacc_gain",
        "G_static_model": "gain_qe=grand_mean+query_effect_q+source_effect_e",
        "G_static_identifiability_constraints": [
            "sum_query_effects=0", "sum_source_effects=0"
        ],
        "G_static_selection_score": "grand_mean_plus_source_effect",
        "selection_labels_use": "select_one_static_A1_challenger_per_H_fold",
        "calibration_labels_use": "direction_shared_case_flip_calibration_only",
        "target_labels_update_shared_model": False,
        "heuristic_score_multiplier": SAFE_Z,
        "minimum_gain": MIN_GAIN,
        "runner_up_margin": RUNNER_UP_MARGIN,
        "nonadmitted_case_fallback": "B",
        "raw_router_descriptive_only": True,
        "heuristic_prediction_bound_descriptive_only": True,
        "calibrated_case_confidence_or_safety_claimed": False,
        "primary_router": "F_S",
    }


def canonical_controls_payload() -> dict[str, object]:
    return {
        "method_ids": list(METHOD_IDS),
        "pre_evaluation_method_ids": list(PRE_EVALUATION_METHOD_IDS),
        "terminal_oracle_ids": list(TERMINAL_ORACLE_IDS),
        "B_role": "fixed_equal_union_baseline_and_exact_fallback",
        "U_role": "uniform_action_control_not_a_flip_challenger",
        "G_static_role": "query_fixed_effect_adjusted_other_center_static_A1_source_or_B",
        "S_static_role": "selection_label_static_A1_source_or_B",
        "F_G_role": "case_flip_from_G_static",
        "F_S_role": "primary_heuristic_uncertainty_gated_case_flip_from_S_static",
        "F_P_role": "blocked_permuted_feature_same_capacity_control",
        "O_static_role": "terminal_best_static_A1_or_B",
        "O_case_role": "terminal_best_case_action_after_eval_open",
    }


def canonical_evaluation_payload() -> dict[str, object]:
    return {
        "primary_endpoint": "center_pooled_exact_bacc_whole_case_oof",
        "primary_method": "F_S",
        "primary_contrasts": ["F_S-B", "F_S-U", "F_S-F_G", "F_S-F_P", "F_S-S_static"],
        "diagnostic_recoverability_gate": {
            "gate_id": "all_primary_contrast_outer_center_lcbs_positive_v1",
            "lcb_field": "one_sided_95_lcb",
            "threshold": 0.0,
            "comparison": "strictly_greater_than",
            "required_contrast_count": 5,
            "pass_status": "PASS",
            "fail_status": "FAIL",
            "diagnostic_only": True,
        },
        "routing_metrics": ["top1_oracle_agreement", "spearman", "normalized_oracle_gap", "fold_stability"],
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
        "source_workers_per_device": 1,
        "generation_workers_per_device": 1,
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
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": "hash_validated_source_prediction_task_resume_plus_deterministic_phase_replay",
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


__all__ = tuple(name for name in globals() if name.startswith("canonical_")) + ("CLASSIFIER",)
