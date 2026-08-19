"""Frozen scientific and claim protocol for PDCB."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    CLAIM_ROLE,
    CROSSING_FEATURE_NAMES,
    CROSSING_LOGISTIC_RIDGE_ALPHA,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash


def frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pdcb_protocol_v1",
        "dataset_family": DATASET_FAMILY,
        "claim_dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "held_unit": "whole_case_or_group",
        "outer_support_scope": "H_minus_c",
        "double_exclusion_support_used": False,
        "crossing_donor_scope": "J_not_equal_H",
        "source_prior_scope": "q_not_in_H_or_e",
        "donor_feature_source_prior_scope": "q_not_in_outer_H_or_donor_J_or_e",
        "outer_target_labels_excluded_from_all_donor_models": True,
        "prior_rebinding_reuses_fitted_IRLS_basis_only": True,
        "outer_case_labels_enter_own_route": False,
        "target_support_labels_used": True,
        "unlabeled_target_deployment_claimed": False,
        "source_experts_updated": False,
        "shared_models_updated_with_target_support": False,
        "endpoint_methods": ["B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST", "P_PROTECTED"],
        "endpoint_nomination_used": False,
        "actionable_unit": "P_vs_B_or_I_or_R_hard_prediction_crossing",
        "crossing_feature_names": list(CROSSING_FEATURE_NAMES),
        "crossing_features_use_labels": False,
        "structural_no_crossing_rows_enter_model": False,
        "crossing_response": "alternative_hard_decision_correct_on_legal_donor_label",
        "crossing_helpfulness_probability_is_composed_BACC_gain": False,
        "crossing_helpfulness_role": "compatibility_proxy_for_fractional_composition",
        "primary_terminal_estimand": "equal_center_BACC_of_actual_composed_output_vs_P",
        "proper_loss_safety_estimands": ["equal_center_Brier", "equal_center_log_loss"],
        "proper_loss_safety_rule": (
            "mean_center_Brier_delta_vs_P_le_0_and_"
            "mean_center_log_loss_delta_vs_P_le_0"
        ),
        "proper_loss_safety_is_terminal_gate_only": True,
        "crossing_model": "center_balanced_shared_slope_ridge_logistic",
        "crossing_logistic_ridge_alpha": CROSSING_LOGISTIC_RIDGE_ALPHA,
        "center_dummy_effects_used": False,
        "equal_total_weight_per_donor_center": True,
        "equal_total_weight_per_case_within_donor_center": True,
        "crossing_rows_are_not_independent_units": True,
        "complete_delete_one_donor_fit_set": True,
        "delete_fit_independence_claimed": False,
        "robust_probability": "median_delete_one_donor_probability",
        "raw_weight": "max_0_2median_minus_1_times_fraction_delete_prob_gt_half",
        "composition": "P_plus_all_crossing_alternatives_normalized_by_1_plus_sum_raw_weights",
        "portfolio_anchor_pseudoweight": 1.0,
        "exact_P_fallback_when_all_crossing_weights_zero": True,
        "full_only_control_predeclared": True,
        "blocked_feature_permutation_control_predeclared": True,
        "blocked_permutation_unit": "complete_equal_size_case_blocks_within_donor_alternative_direction",
        "nonconverged_or_degenerate_fit_behavior": "neutral_probability_0_5_exact_P_fallback",
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
            raise ProtocolError("PDCB frozen protocol drifted.")
        expected = canonical_hash(canonical)
        if self.protocol_hash and self.protocol_hash != expected:
            raise ProtocolError("PDCB protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {**self.payload, "protocol_hash": self.protocol_hash}


def build_frozen_protocol() -> FrozenProtocol:
    return FrozenProtocol()


__all__ = ("FrozenProtocol", "build_frozen_protocol", "frozen_protocol_payload")
