"""Frozen scientific and claim protocol for PCSI-PARC."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    FINGERPRINT_FEATURE_COUNT,
    FINGERPRINT_STATISTIC_IDS,
    PUBLICATION_STATUS,
    RIDGE_ALPHA,
    STAGE_ID,
    TERMINAL_DECISION,
    TRANSPORT_FEATURE_NAMES,
    UTILITY_ZERO_TOLERANCE,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .hashing import canonical_hash


def frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_parc_protocol_v1",
        "dataset_family": DATASET_FAMILY,
        "claim_dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "eligible_centers": list(CENTERS),
        "excluded_centers": [EXCLUDED_CENTER],
        "held_unit": "whole_case_or_group",
        "held_unit_count": EXPECTED_TOTAL_CASE_COUNT,
        "outer_support_scope": "H_minus_c",
        "double_exclusion_support_used": True,
        "double_exclusion_pair_count": EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
        "utility_donor_scope": "J_not_equal_H",
        "policy_regret_fit_scope": (
            "H_and_J_excluded_from_all_fit_normalization_and_response_roles"
        ),
        "source_prior_scope": "q_not_in_H_or_e",
        "actual_donor_feature_source_prior_scope": (
            "q_not_in_outer_H_or_training_donor_K_or_source_e"
        ),
        "pseudo_donor_feature_source_prior_scope": (
            "q_not_in_outer_H_or_pseudo_target_J_or_training_donor_K_or_source_e"
        ),
        "outer_target_labels_excluded_from_all_donor_models_and_replays": True,
        "pseudo_target_labels_excluded_from_models_predicting_that_pseudo_target": True,
        "prior_rebinding_reuses_fitted_IRLS_basis_only": True,
        "outer_case_labels_enter_own_route": False,
        "target_support_labels_used": True,
        "unlabeled_target_deployment_claimed": False,
        "source_experts_updated": False,
        "shared_models_updated_with_target_support": False,
        "endpoint_methods": ["B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST", "P_PROTECTED"],
        "endpoint_nomination_used": False,
        "actionable_unit": (
            "projected_output_equivalence_class_x_direction_with_whole_policy_authorization"
        ),
        "projection_geometry": "nearest_binary32_threshold_on_crossing_mask",
        "projection_zero_to_one_value": "binary32_0_5",
        "projection_one_to_zero_value": "binary32_predecessor_of_0_5",
        "projection_off_crossing_behavior": "preserve_P_byte_for_byte",
        "projected_equivalence_identity": (
            "complete_little_endian_binary32_output_vector"
        ),
        "projected_equivalence_collapse_before_modeling": True,
        "projected_equivalence_provenance_tie_order": "B_before_I_before_R",
        "target_local_fingerprint": "B_U_and_eight_A1_exact_nine_sample_statistics",
        "fingerprint_feature_statistics": list(FINGERPRINT_STATISTIC_IDS),
        "fingerprint_feature_count": FINGERPRINT_FEATURE_COUNT,
        "fingerprint_uses_physical_probabilities_only": True,
        "target_posterior_model": "route_local_standardized_balanced_logistic_C1",
        "target_posterior_support_scope": "H_minus_c",
        "target_posterior_shared_across_routes": False,
        "target_posterior_natural_prevalence_correction": (
            "eta=pi*q/(pi*q+(1-pi)*(1-q))"
        ),
        "target_posterior_is_final_classifier": False,
        "target_influence_estimand": (
            "half_sum_delta_times_eta_over_support_positive_minus_"
            "one_minus_eta_over_support_negative"
        ),
        "target_influence_is_calibrated_utility": False,
        "target_influence_admission_tolerance": UTILITY_ZERO_TOLERANCE,
        "utility_feature_names": list(UTILITY_FEATURE_NAMES),
        "utility_features_use_labels": False,
        "raw_alternative_identity_used_as_model_feature": False,
        "structural_projected_P_rows_enter_model": True,
        "utility_responses": list(UTILITY_RESPONSE_IDS),
        "utility_response_matches_actual_composition_geometry": True,
        "utility_model": "joint_three_response_weighted_ridge",
        "utility_ridge_alpha": RIDGE_ALPHA,
        "utility_direction_intercepts": ["zero_to_one", "one_to_zero"],
        "utility_direction_intercepts_penalized": False,
        "utility_shared_standardized_slope_count": len(UTILITY_FEATURE_NAMES),
        "center_dummy_effects_used": False,
        "equal_total_weight_per_donor_center": True,
        "equal_total_weight_per_case_within_donor_center": True,
        "equal_total_weight_per_surviving_equivalence_class": True,
        "utility_rows_are_not_independent_units": True,
        "projected_selection": (
            "crossing_and_target_influence_gt_1e_minus_15_and_predicted_Brier_le_0_"
            "and_predicted_log_loss_le_0_then_max_target_influence_with_B_I_R_"
            "provenance_tie_order"
        ),
        "predicted_BACC_used_as_per_cell_veto": False,
        "whole_policy_gain_vector": (
            "sum_predicted_BACC_negative_Brier_negative_log_loss"
        ),
        "pseudo_target_replay_reselects_and_recomposes_complete_policy": True,
        "policy_regret_vector": (
            "predicted_whole_policy_gain_minus_realized_whole_policy_gain"
        ),
        "policy_regret_correction": (
            "componentwise_max_of_zero_and_all_eight_pseudo_target_regrets"
        ),
        "policy_regret_is_worst_observed_donor_not_conformal": True,
        "transport_feature_count": len(TRANSPORT_FEATURE_NAMES),
        "transport_semantics": "support_conditioned_endpoint_reconstructed_P_B_I_R",
        "transport_endpoint_support_scope": "endpoint_target_T_minus_held_case_c",
        "transport_actual_source_prior_scope": "q_not_in_endpoint_target_T_or_source_e",
        "transport_donor_source_prior_scope": (
            "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
        ),
        "transport_source_prior_labels_used_upstream": True,
        "transport_route_local_support_labels_used_upstream": True,
        "transport_held_case_evaluation_capability_used_directly": False,
        "transport_pseudo_evaluation_capability_used_directly": False,
        "transport_terminal_evaluation_capability_used_directly": False,
        "transport_label_free_claim": False,
        "transport_uses_pre_equivalence_endpoint_crossing_rates": True,
        "transport_screens_sealed_before_pseudo_evaluation_capability_open": True,
        "transport_screens_sealed_before_terminal_evaluation_capability_open": True,
        "transport_identity_level_route_noninterference_required": True,
        "transport_identity_level_route_noninterference_proven": False,
        "transport_authorization_valid": False,
        "transport_protocol_status": "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK",
        "transport_scale": (
            "leave_one_center_median_MAD_with_1_4826_scale_and_1e_minus_12_floor"
        ),
        "transport_authorization_scope": (
            "outer_H_and_all_eight_pseudo_target_donors"
        ),
        "primary_policy_authorization": (
            "all_transport_screens_pass_and_all_three_corrected_gain_coordinates_"
            "strictly_positive"
        ),
        "equality_at_authorization_boundary_abstains": True,
        "projected_no_policy_regret_control_shares_projected_cell_model_hashes": True,
        "raw_full_action_PARC_control_uses_full_B_I_R_crossing_probabilities": True,
        "fresh_legacy_dual_veto_uses_sign_preserving_shrinkage_0_25": True,
        "blocked_fingerprint_control_predeclared": True,
        "projected_unprojected_and_legacy_response_model_hashes_distinct": True,
        "blocked_target_posterior_hash_distinct": True,
        "composition": "one_selected_equivalence_class_per_direction_else_exact_P",
        "exact_P_fallback_when_no_action_or_policy_is_authorized": True,
        "primary_terminal_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "proper_loss_safety_estimands": ["equal_center_Brier", "equal_center_log_loss"],
        "information_gate_is_terminal_only": True,
        "terminal_information_may_change_same_surface_routes": False,
        "protected_fallback": "P_PROTECTED",
        "held_case_evaluation_capability_used_before_route_seal": False,
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
            raise ProtocolError("PCSI-PARC frozen protocol drifted.")
        expected = canonical_hash(canonical)
        if self.protocol_hash and self.protocol_hash != expected:
            raise ProtocolError("PCSI-PARC protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {**self.payload, "protocol_hash": self.protocol_hash}


def build_frozen_protocol() -> FrozenProtocol:
    return FrozenProtocol()


__all__ = ("FrozenProtocol", "build_frozen_protocol", "frozen_protocol_payload")
