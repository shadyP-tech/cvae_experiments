"""Frozen scientific configuration for the executable SCALE-BP v2 system.

The YAML config repeats these values for auditability, but this module is the
closed-world authority.  Loading rejects both missing fields and additions so
an execution cannot silently tune the consumed test diagnostic.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .identity import (
    ACTION_FAMILIES,
    DIRECT_ACTIONS,
    DIRECTIONS,
    EXPECTED_PHYSICAL_CELL_COUNT,
    METRICS,
    SUPPORT_FOLD_COUNT,
)


SCIENTIFIC_SECTION_NAMES = (
    "action_geometry",
    "support_folds",
    "influence",
    "donor_prior",
    "local_residual",
    "empirical_bayes",
    "uncertainty",
    "selection",
    "admission",
    "controls",
)

METHOD_MENU = (
    "P_PROTECTED",
    "SCALE_BP_V2_PRIMARY",
    "SCALE_BP_V2_DONOR_ONLY",
    "SCALE_BP_V2_LOCAL_ONLY",
    "SCALE_BP_V2_SUPPORT_LABEL_PERMUTATION",
    "SCALE_BP_V2_CYCLIC_ACTION_IDENTITY_POISON",
    "SCALE_BP_V2_FULL_ENDPOINT_SENSITIVITY",
)

EVIDENCE_FEATURE_NAMES = (
    "case_row_count_log1p",
    "direction_branch_rate",
    "crossing_count_log1p",
    "crossing_rate",
    "protected_mean_on_branch",
    "endpoint_mean_on_branch",
    "protected_abs_margin_on_branch",
    "endpoint_abs_margin_on_branch",
    "signed_shift_on_crossings",
    "absolute_shift_on_crossings",
    "protected_entropy_on_crossings",
    "endpoint_entropy_on_crossings",
    "protected_seed_sd_on_crossings",
    "endpoint_seed_sd_on_crossings",
    "protected_vote_disagreement_on_crossings",
    "endpoint_vote_disagreement_on_crossings",
    "crossing_seed_support_fraction",
    "structural_noop",
)


def canonical_scientific_contracts_payload() -> dict[str, dict[str, object]]:
    """Return the exact, non-tunable scientific contract."""

    return {
        "action_geometry": {
            "schema_version": "scale_bp_v2_action_geometry_v1",
            "anchor": "P",
            "families": list(ACTION_FAMILIES),
            "directions": list(DIRECTIONS),
            "direct_actions": list(DIRECT_ACTIONS),
            "crossing_threshold": 0.5,
            "protected_portfolio": "3/5_I_PROTECTED_PLUS_2/5_R_PROTECTED",
            "primary_projection": "NEAREST_FLOAT32_ON_REQUIRED_SIDE_OF_HALF",
            "boundary_projection_primary": True,
            "calibrated_convex_blend_available": False,
            "calibrated_convex_blend_primary": False,
            "calibrated_convex_blend_status": (
                "DEFERRED_NO_LEGAL_CALIBRATION_SURFACE"
            ),
            "full_endpoint_primary": False,
            "full_endpoint_role": "SENSITIVITY_ONLY",
            "off_crossing_probabilities": "BYTE_EXACT_P",
            "unsupported_or_nonfinite_result": "BYTE_EXACT_P",
            "endpoint_derivation_B": "EXACT_NINE_MEAN_B",
            "endpoint_derivation_I": "DIRECTIONAL_EXTREME_OF_ELIGIBLE_A1_MEANS",
            "endpoint_derivation_R": "ROW_MEDIAN_OF_U_AND_ELIGIBLE_A1_MEANS",
            "physical_probability_cell_count": EXPECTED_PHYSICAL_CELL_COUNT,
            "source_pool_sizes_final_pseudo_nested": [10, 9, 8],
            "stored_dtype": "float32",
            "scientific_reduction_dtype": "float64",
        },
        "support_folds": {
            "schema_version": "scale_bp_v2_support_folds_v1",
            "support_definition": "H_MINUS_C",
            "fold_count": SUPPORT_FOLD_COUNT,
            "assignment": "SORTED_WHOLE_CASE_ROUND_ROBIN",
            "whole_case_patient_slide_group_disjoint": True,
            "held_case_excluded": True,
            "own_fold_excluded_from_prediction": True,
            "outer_held_case_excluded_from_every_oof_fit": True,
            "folds_execute_sequentially_inside_outer_worker": True,
            "fold_failure_result": "BYTE_EXACT_P",
        },
        "influence": {
            "schema_version": "scale_bp_v2_case_evidence_v1",
            "unit": "COMPLETE_CASE_BY_SIX_ACTION_RECTANGLE",
            "metrics": list(METRICS),
            "descriptor_names": list(EVIDENCE_FEATURE_NAMES),
            "case_descriptor_label_free": True,
            "probability_geometry_and_ensemble_disagreement_exposed": True,
            "threshold_switch_count_persisted_per_action": True,
            "harmful_switch_count_preterminal_available": False,
            "harmful_switch_count_status": (
                "UNAVAILABLE_PRETERMINAL_TARGET_LABELS_CLOSED"
            ),
            "latent_embedding_distance_available": False,
            "effective_source_training_support_available": False,
            "source_calibration_status_available": False,
            "unavailable_evidence_reason": (
                "ABSENT_FROM_FROZEN_DIRECT_ORIGINAL_INPUTS"
            ),
            "structural_noops_retained": True,
            "proxy_scores_are_not_true_utility": True,
            "nonfinite_result": "BYTE_EXACT_P",
        },
        "donor_prior": {
            "schema_version": "scale_bp_v2_donor_prior_v1",
            "final_fit_scope": "ALL_CENTERS_EXCEPT_H",
            "pseudo_fit_scope": "ALL_CENTERS_EXCEPT_H_AND_J",
            "nested_delete_center_scope": "ALL_CENTERS_EXCEPT_H_J_AND_K",
            "outer_center_excluded": True,
            "prediction_center_excluded": True,
            "delete_center_query_and_source_excluded": True,
            "delete_center_folds_independently_reconstructed": True,
            "equal_center_weighting": True,
            "equal_case_weighting_within_center": True,
            "ridge_alpha": 1.0,
            "maximum_abs_standardized_feature": 4.0,
            "minimum_independent_centers": 6,
            "target_support_updates_donor_coefficients": False,
            "target_support_updates_shared_scaler": False,
        },
        "local_residual": {
            "schema_version": "scale_bp_v2_local_residual_v1",
            "fit_scope": "EPHEMERAL_ROUTE_LOCAL_H_C_ONLY",
            "route_local_only": True,
            "updates_global_state": False,
            "crossfit_fold_count": SUPPORT_FOLD_COUNT,
            "residual_target": "REALIZED_MINUS_DONOR_PREDICTED_ACTION_VALUE",
            "ridge_alpha": 1.0,
            "held_case_and_own_fold_excluded": True,
            "support_labels_tune_hyperparameters": False,
            "insufficient_support_result": "BYTE_EXACT_P",
        },
        "empirical_bayes": {
            "schema_version": "scale_bp_v2_empirical_bayes_v1",
            "formula": "DONOR_MEAN_PLUS_WEIGHT_TIMES_LOCAL_CORRECTION",
            "shrinkage_signal_variance": "DONOR_BETWEEN_CENTER_HETEROGENEITY_SQUARED",
            "local_noise_variance": "LOCAL_OOF_RMSE_SQUARED_PLUS_LOCAL_ESTIMATOR_SE_SQUARED",
            "transport_rmse_double_counted_as_variance": False,
            "donor_estimator_se_coefficient": 1.0,
            "combined_estimator_se": "SQRT_DONOR_SE2_PLUS_WEIGHT2_LOCAL_SE2",
            "weight_bounded_0_1": True,
            "support_labels_tune_shared_state": False,
            "degenerate_variance_result": "DONOR_ONLY_OR_BYTE_EXACT_P",
        },
        "uncertainty": {
            "schema_version": "scale_bp_v2_uncertainty_v1",
            "computed_before_argmax": True,
            "base_multiplier": 1.2815515655446004,
            "selection_multiplier_floor": "SQRT_2_LOG_SIX",
            "descriptive_scale": "MAX_TRANSPORT_HETEROGENEITY_LOCAL_OOF_PLUS_ESTIMATOR_SE",
            "descriptive_only": True,
            "confidence_bound_claimed": False,
            "conformal_claimed": False,
            "finite_sample_coverage_claimed": False,
        },
        "selection": {
            "schema_version": "scale_bp_v2_selection_v1",
            "method_menu": list(METHOD_MENU),
            "direct_case_action_selection": True,
            "learned_prefix_layer": False,
            "P_is_first_class_candidate": True,
            "P_candidate_representation": (
                "IMPLICIT_ZERO_UTILITY_EXACT_FALLBACK"
            ),
            "P_candidate_expected_utility_anchor": 0.0,
            "P_candidate_assessment_emitted": False,
            "P_wins_without_unique_robust_safe_positive_action": True,
            "minimum_bacc_lower": 0.0,
            "maximum_brier_upper": 0.0,
            "maximum_log_upper": 0.0,
            "tie_tolerance": 1.0e-12,
            "tie_winner": "P_PROTECTED",
            "at_most_one_action": True,
            "exact_p_fallback": True,
        },
        "admission": {
            "schema_version": "scale_bp_v2_pseudo_admission_v1",
            "unit": "COMPLETE_H_J_D_BY_SIX_ACTION_RECTANGLES",
            "case_then_equal_center_aggregation": True,
            "opportunity_cases_only_for_rank_metrics": True,
            "minimum_opportunity_cases": 24,
            "minimum_represented_centers": 6,
            "minimum_within_case_spearman": 0.0,
            "maximum_normalized_oracle_gap": 1.0,
            "maximum_harmful_selected_policy_count": 0,
            "selected_actions_require_brier_and_log_nonworsening": True,
            "abort_before_terminal_on_failure": True,
            "thresholds_caller_overridable": False,
        },
        "controls": {
            "schema_version": "scale_bp_v2_controls_v1",
            "required_method_ids": list(METHOD_MENU),
            "global_only_control": "SCALE_BP_V2_DONOR_ONLY",
            "local_only_control": "SCALE_BP_V2_LOCAL_ONLY",
            "support_permutation_control": "SCALE_BP_V2_SUPPORT_LABEL_PERMUTATION",
            "action_identity_poison_control": "SCALE_BP_V2_CYCLIC_ACTION_IDENTITY_POISON",
            "full_endpoint_control": "SCALE_BP_V2_FULL_ENDPOINT_SENSITIVITY",
            "controls_may_authorize_primary": False,
            "controls_are_terminal_attribution_comparators": True,
            "raw_control_labels_may_be_persisted": False,
        },
    }


def frozen_scientific_contracts() -> Mapping[str, Mapping[str, object]]:
    """Expose an immutable top-level view for programmatic callers."""

    payload = canonical_scientific_contracts_payload()
    return MappingProxyType(
        {name: MappingProxyType(values) for name, values in payload.items()}
    )


__all__ = (
    "EVIDENCE_FEATURE_NAMES",
    "METHOD_MENU",
    "SCIENTIFIC_SECTION_NAMES",
    "canonical_scientific_contracts_payload",
    "frozen_scientific_contracts",
)
