"""Canonical config payloads for prediction-only disagreement regret."""

from __future__ import annotations

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...routing.disagreement_regret_core.contracts import FEATURE_NAMES
from ...routing.disagreement_regret_core.design import (
    ACTION_L2_PENALTY,
    SHARED_L2_PENALTY,
)
from ...routing.disagreement_regret_core.fitting import (
    GRADIENT_TOLERANCE,
    MAX_NEWTON_ITERATIONS,
)
from ...routing.disagreement_regret_core.selection import FAMILY_WISE_ALPHA
from .experiment_contracts import (
    CENTERS,
    CLAIM_ROLE,
    EXCLUDED_CENTER,
    EXPECTED_TEST_FEATURE_DIM,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TRAIN_FEATURE_DIM,
    EXPECTED_TRAIN_ROW_COUNT,
    GENERATION_SEEDS,
    GEOMETRY_IDS,
    INPUT_ARTIFACT_IDS,
    MODEL_FAMILY_IDS,
    PUBLICATION_STATUS,
    SOURCE_SPLIT,
    STAGE_ID,
    SURFACE_IDS,
    TARGET_SPLIT,
    TRAINING_SEEDS,
)
from .protocol import canonical_prediction_only_protocol


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
        **canonical_prediction_only_protocol().to_payload(),
        "stage": STAGE_ID,
        "source_split": SOURCE_SPLIT,
        "target_split": TARGET_SPLIT,
        "centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "source_row_count": EXPECTED_TRAIN_ROW_COUNT,
        "target_row_count": EXPECTED_TEST_ROW_COUNT,
        "source_feature_dim": EXPECTED_TRAIN_FEATURE_DIM,
        "target_feature_dim": EXPECTED_TEST_FEATURE_DIM,
        "input_artifact_count": len(INPUT_ARTIFACT_IDS),
        "single_consumer_aliases_required": True,
        "source_oof_unit": "whole_case",
        "source_label_origin": "train_cache_metadata_train_rows_only",
        "strict_outer_target_H_exclusion": True,
        "source_oof_query_q_excluded_from_all_action_compositions": True,
        "source_oof_excluded_pair_fit_reuse_is_exact": True,
        "source_oof_physical_fit_task_count": 324,
        "source_oof_physical_classifier_fit_count": 5_184,
        "source_oof_oriented_prediction_context_count": 648,
        "source_oof_oriented_prediction_cell_count": 10_368,
        "source_oof_action_menu_size": 16,
        "target_inference_fit_task_count": 81,
        "target_inference_classifier_fit_count": 1_458,
        "target_inference_action_menu_size": 18,
        "total_physical_classifier_fit_count_before_test_admission": 6_642,
        "test_phase_classifier_fit_count": 0,
        "nested_query_q_models_used": False,
        "heldout_query_q_exclusion": "not_applicable_no_nested_selection",
        "candidate_source_e_response_query_excluded": True,
        "source_predictions_are_oof_for_every_labeled_row": True,
        "source_labels_open_only_after_source_probability_seal": True,
        "target_cache_admitted_only_after_model_bank_seal": True,
        "target_labels_never_opened": True,
        "target_labels_never_read_or_persisted": True,
        "target_scores_never_computed": True,
        "all_9928_target_rows_retained": True,
        "prior_stage90_outputs_used": False,
        "prior_stage90_predictions_or_checkpoints_used": False,
        "numbered_stage_result_or_policy_used": False,
    }


def canonical_action_library_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_prediction_only_action_library_v1",
        "geometry_ids": list(GEOMETRY_IDS),
        "geometry_selection_used": False,
        "baseline_action_id": "B",
        "control_action_id": "U",
        "candidate_action_id_format": "{geometry_id}::source={source_center}",
        "candidate_pool_excludes_outer_target_H": True,
        "source_oof_candidate_pool_excludes_query_q": True,
        "source_oof_B_and_U_exclude_query_q": True,
        "source_oof_action_menu_size": 16,
        "target_inference_action_menu_size": 18,
        "source_oof_excluded_pair_fit_reuse": (
            "one_physical_fit_per_unordered_H_q_pair_two_oriented_contexts"
        ),
        "nested_query_candidate_pool_used": False,
        "donor_B_and_U_may_include_H_source_history": False,
        "target_expert_used": False,
        "baseline_rows_per_source_class": 128,
        "uniform_rows_per_source_class": 144,
        "source_stream_prefix_rows_per_class": 270,
        "A0": {
            "selected_rows_per_class": 256,
            "other_rows_per_class": 128,
            "selected_row_weight": 1.0,
            "other_row_weight": 1.0,
        },
        "A1": {
            "selected_rows_per_class": 256,
            "other_rows_per_class": 128,
            "reuses_exact_A0_row_ids": True,
            "selected_row_weight": 1.4375,
            "other_row_weight": 0.875,
        },
        "source_oof_mass_normalization": {
            "B_global_factor": "8/7",
            "U_global_factor": "8/7",
            "A0_global_factor": "9/8",
            "A1_global_factor": "72/65",
            "A1_selected_effective_weight": "207/130",
            "A1_other_effective_weight": "63/65",
            "effective_mass_per_class": {
                "B": 1_024,
                "U": 1_152,
                "A0": 1_152,
                "A1": 1_152,
            },
            "sample_weight_scope": "logistic_regression_fit_only",
            "scaler_fit_used_sample_weight": False,
            "label_tuned": False,
        },
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_pairing": "cartesian_product_exact_nine_no_seed_selection",
        "probability_summary": (
            "exact_nine_mean_sd_and_winning_hard_vote_fraction"
        ),
        "action_strength_sweep_used": False,
        "class_conditional_action_variant_used": False,
        "source_pair_action_used": False,
    }


def canonical_regret_model_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_disagreement_regret_model_v1",
        "family": "known_bank_hierarchical_pairwise_regret_v1",
        "model_family_ids": list(MODEL_FAMILY_IDS),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "feature_scope": "label_free_case_action_disagreement_vs_B_and_U",
        "candidate_identity_mode": "known_bank_partial_pooling",
        "response": "source_oof_exact_bacc_regret_from_case_best",
        "response_scope": "source_train_only_never_target_test",
        "pair_weighting": "equal_query_then_absolute_regret_spread",
        "shared_l2_penalty": SHARED_L2_PENALTY,
        "action_l2_penalty": ACTION_L2_PENALTY,
        "max_newton_iterations": MAX_NEWTON_ITERATIONS,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "hyperparameter_search_used": False,
        "outer_target_H_absent_from_fit_and_standardization": True,
        "source_prediction_query_q_absent_from_every_action_fit_and_scaler": True,
        "nested_query_q_models_used": False,
        "heldout_query_q_exclusion": "not_applicable_no_nested_selection",
        "candidate_source_e_response_query_excluded": True,
        "G_definition": "static_candidate_identity_control",
        "R_definition": "aligned_disagreement_feature_model",
        "P_definition": "same_capacity_deterministic_feature_permutation_control",
        "permutation_changes_labels_or_responses": False,
        "selection_surfaces": list(SURFACE_IDS),
        "R_raw_definition": "maximum_predicted_pairwise_margin",
        "R_safe_definition": (
            "strict_simultaneous_one_sided_lcb_positive_vs_B_and_U_else_B"
        ),
        "family_wise_alpha": FAMILY_WISE_ALPHA,
        "safe_fallback_action_id": "B",
        "margin_aware_gate": True,
        "test_route_suggestions_are_unscored": True,
    }


def canonical_runtime_payload() -> dict[str, object]:
    return {
        "schema_version": (
            "fixed_bank_disagreement_regret_prediction_only_runtime_v1"
        ),
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "source_generation_workers": 2,
        "source_workers_per_device": 1,
        "gpu_phase_precedes_cpu_phase": True,
        "cpu_workers": 4,
        "threads_per_worker": 3,
        "maximum_total_cpu_threads": 12,
        "maximum_dense_fit_bytes": 536870912,
        "multiprocessing_start_method": "spawn",
        "gpu_and_cpu_phases_disjoint": True,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "scientific_reduction_dtype": "float64",
        "surface_storage_dtype": "float32",
        "source_oof_physical_classifier_fit_count": 5_184,
        "source_oof_oriented_prediction_cell_count": 10_368,
        "target_classifier_fit_count": 1_458,
        "total_physical_classifier_fit_count": 6_642,
        "test_phase_classifier_fit_count": 0,
        "source_oof_prediction_storage": "query_scoped_float32_npz",
        "classifier_parameter_storage": "read_only_float64_npy",
        "local_scratch_preferred": True,
        "hash_validated_resume": True,
    }


def canonical_outputs_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_prediction_only_outputs_v1",
        "target_row_count": EXPECTED_TEST_ROW_COUNT,
        "target_probability_rows_are_label_free": True,
        "persist_model_hashes": True,
        "persist_source_prediction_seal": True,
        "persist_model_bank_seal_before_test_admission": True,
        "persist_test_prediction_seal": True,
        "persist_full_candidate_contrasts": True,
        "persist_R_raw_and_R_safe": True,
        "persist_uncertainty_and_fallback_reason": True,
        "persist_action_and_threshold_crossing_counts": True,
        "forbidden_target_or_test_columns": [
            "label",
            "y_true",
            "bacc",
            "accuracy",
            "regret",
            "utility",
            "oracle",
            "nelbo",
            "downstream_metric",
        ],
        "source_training_aggregate_columns_allowed": [
            "source_exact_bacc_gain_vs_control",
            "source_exact_regret_from_case_best",
        ],
        "raw_source_label_columns_forbidden": True,
        "terminal_metric_table_exists": False,
        "output_is_policy": False,
    }


def canonical_claim_boundary_payload() -> dict[str, object]:
    return {
        "status": PUBLICATION_STATUS,
        "claim_role": CLAIM_ROLE,
        "maximum_claim": (
            "posthoc_source_oof_trained_router_produced_unscored_label_free_"
            "diagnostic_suggestions_for_all_consumed_test_rows"
        ),
        "not_fresh_evidence": True,
        "not_routing_success_evidence": True,
        "not_model_comparison_evidence": True,
        "not_statistical_equivalence_evidence": True,
        "not_policy_or_action_authorization": True,
        "not_promotion_or_deployment_evidence": True,
        "cannot_feed_another_experiment": True,
        "fresh_predeclared_whole_case_disjoint_evidence_required_for_claim": True,
    }


__all__ = (
    "CLASSIFIER",
    "canonical_action_library_payload",
    "canonical_claim_boundary_payload",
    "canonical_outputs_payload",
    "canonical_protocol_payload",
    "canonical_regret_model_payload",
    "canonical_runtime_payload",
)
