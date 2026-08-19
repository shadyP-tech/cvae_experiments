"""Frozen scientific and claim protocol for PUMR."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    CLAIM_ROLE,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
    FINGERPRINT_FEATURE_COUNT,
    FINGERPRINT_STATISTIC_IDS,
    P_FALLBACK_MARGIN,
    MARGIN_MIN,
    RELIABILITY_AUC_MIN,
    RELIABILITY_BRIER_SKILL_MIN,
    ROBUST_MAD_SCALE,
    ROBUST_MAD_FLOOR,
    SIGN_PRESERVING_SHRINKAGE,
    SUPPORT_CROSSFIT_FOLD_COUNT,
    UTILITY_CELL_IDS,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash


def frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pumr_protocol_v1",
        "dataset_family": DATASET_FAMILY,
        "claim_dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "held_unit": "whole_case_or_group",
        "outer_support_scope": "H_minus_c",
        "double_exclusion_support_used": False,
        "utility_donor_scope": "J_not_equal_H",
        "source_prior_scope": "q_not_in_H_or_e",
        "donor_feature_source_prior_scope": "q_not_in_outer_H_or_donor_J_or_e",
        "outer_target_labels_excluded_from_all_donor_margin_calibration": True,
        "prior_rebinding_reuses_fitted_IRLS_basis_only": True,
        "outer_case_labels_enter_own_route": False,
        "target_support_labels_used": True,
        "unlabeled_target_deployment_claimed": False,
        "source_experts_updated": False,
        "shared_models_updated_with_target_support": False,
        "endpoint_methods": ["B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST", "P_PROTECTED"],
        "endpoint_nomination_used": False,
        "actionable_unit": "case_x_alternative_x_direction_complete_rectangle",
        "target_local_fingerprint": "B_U_and_eight_A1_exact_nine_sample_statistics",
        "fingerprint_feature_statistics": list(FINGERPRINT_STATISTIC_IDS),
        "fingerprint_feature_count": FINGERPRINT_FEATURE_COUNT,
        "fingerprint_uses_physical_probabilities_only": True,
        "target_posterior_model": "five_fold_route_local_standardized_balanced_logistic_C1",
        "target_posterior_support_scope": "H_minus_c_minus_validation_fold",
        "target_posterior_fold_count": SUPPORT_CROSSFIT_FOLD_COUNT,
        "target_posterior_fold_assignment": "label_free_hash_order_whole_case_round_robin",
        "target_posterior_validation_unit": "whole_case",
        "target_posterior_shared_across_routes": False,
        "target_posterior_natural_prevalence_correction": (
            "eta=pi*q/(pi*q+(1-pi)*(1-q))"
        ),
        "target_posterior_is_final_classifier": False,
        "posterior_expected_bacc_estimand": (
            "half_direction_sign_times_sum_eta_over_expected_positive_"
            "minus_sum_one_minus_eta_over_expected_negative"
        ),
        "posterior_expected_brier_estimand": "mean_q2_minus_p2_minus_2eta_times_q_minus_p",
        "posterior_expected_log_loss_estimand": (
            "mean_eta_log_p_over_q_plus_one_minus_eta_log_one_minus_p_over_one_minus_q"
        ),
        "posterior_robust_summary": "median_plus_or_minus_1_4826_MAD_across_five_folds",
        "posterior_robust_mad_scale": ROBUST_MAD_SCALE,
        "posterior_robust_mad_floor": ROBUST_MAD_FLOOR,
        "route_reliability_gate": (
            "support_OOF_AUC_gt_0_5_and_support_OOF_Brier_skill_gt_0"
        ),
        "route_reliability_auc_min_strict": RELIABILITY_AUC_MIN,
        "route_reliability_brier_skill_min_strict": RELIABILITY_BRIER_SKILL_MIN,
        "utility_feature_names": list(UTILITY_FEATURE_NAMES),
        "utility_cell_ids": list(UTILITY_CELL_IDS),
        "utility_features_use_labels": False,
        "structural_no_crossing_rows_retained": True,
        "utility_responses": list(UTILITY_RESPONSE_IDS),
        "utility_response_matches_actual_branch_local_composition": True,
        "primary_terminal_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "proper_loss_safety_estimands": ["equal_center_Brier", "equal_center_log_loss"],
        "proper_loss_safety_rule": (
            "mean_center_Brier_delta_vs_P_le_0_and_"
            "mean_center_log_loss_delta_vs_P_le_0"
        ),
        "proper_loss_safety_is_preterminal_selection_gate": True,
        "donor_response_regression_used": False,
        "utility_rows_are_not_independent_units": True,
        "margin_candidate_grid": "zero_positive_robust_scores_and_fixed_P_fallback",
        "P_fallback_margin": P_FALLBACK_MARGIN,
        "strict_positive_margin_min": MARGIN_MIN,
        "margin_objective": "median_donor_BACC_delta_minus_1_4826_MAD",
        "margin_constraints": (
            "every_training_donor_BACC_nonnegative_and_Brier_and_log_loss_nonpositive"
        ),
        "inner_margin_validation": "leave_one_donor_center_replay",
        "margin_tie_break": "larger_margin_then_fewer_selected_actions",
        "outer_target_labels_enter_margin_calibration": False,
        "selection": (
            "reliable_posterior_and_robust_BACC_above_margin_and_"
            "robust_Brier_and_log_loss_nonpositive_then_max_robust_BACC_"
            "with_B_I_R_tie_order"
        ),
        "whole_center_P_fallback_when_inner_donor_replay_unsafe": True,
        "composition": "branch_disjoint_selected_endpoint_else_exact_P",
        "sign_preserving_shrinkage": SIGN_PRESERVING_SHRINKAGE,
        "sign_preserving_shrinkage_tuned_on_test_labels": False,
        "exact_P_fallback_when_no_action_is_admissible": True,
        "bacc_only_control_predeclared": True,
        "zero_margin_control_predeclared": True,
        "blocked_fingerprint_control_predeclared": True,
        "blocked_fingerprint_unit": "complete_feature_row_cyclic_shift_within_case",
        "information_gate_is_terminal_only": True,
        "terminal_information_may_change_same_surface_routes": False,
        "protected_fallback": "P_PROTECTED",
        "target_evaluation_labels_used_before_route_seal": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_role": CLAIM_ROLE,
        "may_authorize_routing": False,
        "may_authorize_policy_update": False,
        "may_authorize_promotion": False,
        "may_authorize_deployment": False,
        "may_feed_another_experiment": False,
    }


@dataclass(frozen=True)
class FrozenProtocol:
    payload: dict[str, object] = field(default_factory=frozen_protocol_payload)
    protocol_hash: str = ""

    def __post_init__(self) -> None:
        canonical = frozen_protocol_payload()
        if self.payload != canonical:
            raise ProtocolError("PUMR frozen protocol drifted.")
        expected = canonical_hash(canonical)
        if self.protocol_hash and self.protocol_hash != expected:
            raise ProtocolError("PUMR protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {**self.payload, "protocol_hash": self.protocol_hash}


def build_frozen_protocol() -> FrozenProtocol:
    return FrozenProtocol()


__all__ = ("FrozenProtocol", "build_frozen_protocol", "frozen_protocol_payload")
