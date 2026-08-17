"""Frozen scientific and claim protocol for the consumed-test successor."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    BACC_REGRET_TOLERANCE,
    CENTER_EFFECT_ALPHA,
    CLAIM_ROLE,
    CPU_WORKERS,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    LTT_FAMILYWISE_ALPHA,
    LTT_MAX_CENTER_HARM_RATE,
    LOG_LOSS_CLIP_EPSILON,
    MIN_DELETE_DONOR_POSITIVE,
    PROPER_LOSS_AGGREGATION,
    PROPER_LOSS_CLASS_WEIGHTING,
    PROPER_LOSS_TOLERANCE,
    PUBLICATION_STATUS,
    RIDGE_ALPHA,
    STAGE_ID,
    SUPPORT_DISPERSION_MULTIPLIER,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash


def frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_nested_donor_endpoint_regret_protocol_v1",
        "dataset_family": DATASET_FAMILY,
        "claim_dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "held_unit": "whole_case_or_group",
        "outer_support_scope": "H_minus_c",
        "unordered_pair_support_scope": "H_minus_c_minus_s",
        "unordered_pair_state_reused_for_two_ordered_voters": True,
        "regret_donor_scope": "J_not_equal_H",
        "source_prior_scope": "q_not_in_H_or_e",
        "donor_feature_source_prior_scope": "q_not_in_outer_H_or_donor_J_or_e",
        "outer_target_labels_excluded_from_all_donor_features": True,
        "prior_rebinding_reuses_fitted_IRLS_basis_only": True,
        "outer_case_labels_enter_own_route": False,
        "nested_voter_case_labels_enter_own_endpoint_fit": False,
        "source_experts_updated": False,
        "shared_models_updated_with_target_support": False,
        "endpoint_methods": ["B", "I_OPPORTUNITY_GATED", "R_NINE_ARM_ROBUST", "P_PROTECTED"],
        "endpoint_nomination": "same_target_nested_support_pooled_additive_BACC",
        "regret_response": "paired_additive_center_BACC_contribution_vs_P",
        "donor_training_row_scope": "all_cases_in_each_donor_center",
        "no_candidate_row_semantics": "protected_P_with_exact_zero_paired_regret",
        "proper_loss_response": "paired_center_mean_log_loss_contribution_vs_P",
        "proper_loss_aggregation": PROPER_LOSS_AGGREGATION,
        "proper_loss_class_weighting": PROPER_LOSS_CLASS_WEIGHTING,
        "proper_loss_clip_epsilon": LOG_LOSS_CLIP_EPSILON,
        "ridge_alpha": RIDGE_ALPHA,
        "center_effect_alpha": CENTER_EFFECT_ALPHA,
        "equal_total_weight_per_donor_center": True,
        "support_dispersion_multiplier": SUPPORT_DISPERSION_MULTIPLIER,
        "support_dispersion_is_confidence_bound": False,
        "minimum_delete_donor_positive": MIN_DELETE_DONOR_POSITIVE,
        "bacc_regret_tolerance": BACC_REGRET_TOLERANCE,
        "proper_loss_tolerance": PROPER_LOSS_TOLERANCE,
        "proper_loss_gate_is_point_estimate_no_worse_not_noninferiority_test": True,
        "center_block_ltt_familywise_alpha": LTT_FAMILYWISE_ALPHA,
        "center_block_ltt_max_harm_rate": LTT_MAX_CENTER_HARM_RATE,
        "center_block_ltt_is_feasibility_diagnostic": True,
        "center_block_ltt_statistical_authorization_enabled": False,
        "center_block_ltt_binomial_independence_claimed": False,
        "protected_fallback": "P_PROTECTED",
        "route_decision_label_blind": False,
        "protected_fallback_label_blind": False,
        "cpu_workers": CPU_WORKERS,
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
            raise ProtocolError("Nested-regret frozen protocol drifted.")
        expected = canonical_hash(canonical)
        if self.protocol_hash and self.protocol_hash != expected:
            raise ProtocolError("Nested-regret protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {**self.payload, "protocol_hash": self.protocol_hash}


def build_frozen_protocol() -> FrozenProtocol:
    return FrozenProtocol()


__all__ = ("FrozenProtocol", "build_frozen_protocol", "frozen_protocol_payload")
