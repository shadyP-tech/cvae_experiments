"""Frozen route-scoped scientific and claim protocol for PCSI-RACR."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    EVALUATION_SPLIT,
    EXCLUDED_CENTER,
    EXPECTED_FINAL_CASE_PREDICTION_COUNT,
    EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT,
    EXPECTED_TRANSPORT_SCREEN_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash
from .transport import TRANSPORT_PROTOCOL_CONTRACT


def frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_racr_route_scoped_protocol_v1",
        "dataset_family": DATASET_FAMILY,
        "claim_dataset_family": DATASET_FAMILY,
        "stage": STAGE_ID,
        "evaluation_split": EVALUATION_SPLIT,
        "eligible_centers": list(CENTERS),
        "excluded_center": EXCLUDED_CENTER,
        "eligible_row_count": EXPECTED_TEST_ROW_COUNT,
        "whole_case_route_count": EXPECTED_TOTAL_CASE_COUNT,
        "every_target_case_evaluated_once": True,
        "whole_test_dataset_reused": True,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_role": CLAIM_ROLE,
        "claim_boundary": "NON_GUARANTEE_CONSUMED_TEST_ONLY",
        "method_scope": "cross_fitted_case_policy_mosaic_diagnostic",
        "deployable_center_policy_claimed": False,
        "source_experts_frozen": True,
        "generation_lock_required": True,
        "source_expert_updated": False,
        "generated_embedding_mutated": False,
        "projection_is_probability_action_postprocessing_only": True,
        "not_nelbo_or_generative_compatibility": True,
        "not_reconstruction_or_fidelity_evidence": True,
        "not_downstream_utility_or_expert_selection_evidence": True,
        "target_route_support_scope": "H_minus_c",
        "target_route_own_label_excluded": True,
        "pseudo_route_support_scope": "J_minus_d",
        "pseudo_route_own_label_excluded": True,
        "target_model_exclusion": "outer_H",
        "pseudo_model_exclusion": "outer_H_and_pseudo_J",
        "target_reference_state": "K_minus_s_with_H_K_prior_exclusion",
        "pseudo_reference_state": "K_minus_s_with_H_J_K_prior_exclusion",
        "own_route_noninterference_required": True,
        "global_label_invariance_claimed": False,
        "other_leave_one_case_support_routes_may_change": True,
        "other_route_or_report_feedback_into_own_route_forbidden": True,
        "label_role_ledger_required": True,
        "phase_chain": [
            "PhysicalSeal",
            "PreEvaluationSeal",
            "PseudoLabelGrant",
            "PseudoReplaySeal",
            "CalibrationSeal",
            "DecisionSeal",
            "TargetTerminalLabelGrant",
        ],
        "candidate_eligibility": (
            "crossing_gt_0_and_target_influence_gt_1e_minus_15_and_"
            "predicted_raw_Brier_le_0_and_predicted_raw_log_loss_le_0"
        ),
        "candidate_selection": (
            "exact_maximum_target_influence_then_B_I_R_then_action_hash"
        ),
        "projected_and_raw_candidates_reselected_separately": True,
        "projected_off_mask_P_bytes_preserved": True,
        "fallback": "ABSTAIN_TO_P_BYTE_EXACT",
        "primary_envelope": "OBSERVED_DONOR_CASE_ENVELOPE",
        "residual": "predicted_favorable_vector_minus_realized_favorable_vector",
        "envelope": "max_zero_then_max_all_d_within_J_then_max_all_J",
        "all_pseudo_cases_included": True,
        "pseudo_transport_audit_only": True,
        "incomplete_pseudo_scope_invalidates_outer_geometry": True,
        "decision_gate": (
            "target_transport_pass_and_all_three_predicted_minus_margin_gt_zero"
        ),
        "equality_abstains": True,
        "conformal": False,
        "finite_sample_coverage": False,
        "tail_probability_claimed": False,
        "confidence_bound_claimed": False,
        "calibrated_uncertainty_claimed": False,
        "dependent_leave_one_case_pseudo_replays": True,
        "opportunity_imbalance_disclosed": True,
        "final_surfaces": [
            "P_PROTECTED",
            "PCSI_RACR_PROJECTED_OBSERVED_MAX",
            "PCSI_RACR_RAW_OBSERVED_MAX",
            "PCSI_RACR_PROJECTED_NO_ENVELOPE",
        ],
        "upper_median_is_unscored_annotation": True,
        "descriptor_match_is_unscored_annotation": True,
        "endpoint_model_fit_count": 3_488,
        "target_posterior_model_fit_count": (
            EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        ),
        "utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
        "pseudo_replay_count": EXPECTED_POLICY_REPLAY_COUNT,
        "role_bound_transport_descriptor_count": (
            EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT
        ),
        "numeric_transport_leaf_count": EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT,
        "transport_reference_summary_count": (
            EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT
        ),
        "transport_screen_count": EXPECTED_TRANSPORT_SCREEN_COUNT,
        "final_case_prediction_count": EXPECTED_FINAL_CASE_PREDICTION_COUNT,
        **dict(TRANSPORT_PROTOCOL_CONTRACT),
        "may_authorize_routing": False,
        "may_authorize_policy_update": False,
        "may_authorize_promotion": False,
        "may_authorize_deployment": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


@dataclass(frozen=True)
class FrozenProtocol:
    payload: dict[str, object] = field(default_factory=frozen_protocol_payload)
    protocol_hash: str = ""

    def __post_init__(self) -> None:
        canonical = frozen_protocol_payload()
        if self.payload != canonical:
            raise ProtocolError("PCSI-RACR frozen protocol drifted.")
        expected = canonical_hash(canonical)
        if self.protocol_hash and self.protocol_hash != expected:
            raise ProtocolError("PCSI-RACR protocol hash drifted.")
        object.__setattr__(self, "payload", canonical)
        object.__setattr__(self, "protocol_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {**self.payload, "protocol_hash": self.protocol_hash}


def build_frozen_protocol() -> FrozenProtocol:
    return FrozenProtocol()


__all__ = ("FrozenProtocol", "build_frozen_protocol", "frozen_protocol_payload")
