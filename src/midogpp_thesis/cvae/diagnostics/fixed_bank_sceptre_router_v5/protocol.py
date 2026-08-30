"""Frozen planned scientific protocol and permanent claim boundary."""

from __future__ import annotations

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash
from .development import (
    CALIBRATION_MAXIMUM_BRIER_DELTA,
    CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
    CALIBRATION_MINIMUM_BACC_GAIN,
    SUPPORT_MINIMUM_SHRUNK_BACC_GAIN,
    SUPPORT_PRIOR_EFFECTIVE_CASES,
)
from .experiment_contracts import INPUT_ARTIFACT_IDS
from .identity import (
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    POLICY_TRANSITION,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


PROTOCOL_SCHEMA = "sceptre_v5_executable_candidate_set_terminal_protocol_v1"


def protocol_payload() -> dict[str, object]:
    body = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "candidate_unit": "independently_trained_source_center_CVAE_family",
        "candidate_inventory": "EXACT_C_MINUS_H_PLUS_EXACT_B_FALLBACK",
        "architecture_role": (
            "TARGET_LABEL_ASSISTED_POST_HOC_DOWNSTREAM_UTILITY_POLICY_SENSITIVITY"
        ),
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_cells_per_family": 9,
        "seed_cells_are_nuisance_replications": True,
        "seed_selection_allowed": False,
        "source_inner_estimand": (
            "EXPERT_VS_EXPERT_RANKING_PRIOR_NOT_ADVANTAGE_OVER_EXACT_B"
        ),
        "source_inner_exact_b_outcomes_available": False,
        "strict_outer_center_exclusion": (
            "delete_all_q_equal_H_or_e_equal_H_before_transform_fit_or_tuning"
        ),
        "nested_lodo_hyperparameter_selection": True,
        "physical_surface": {
            "full_source_rows_per_class": 1024,
            "exact_b_source_count": 8,
            "exact_b_rows_per_source_per_class": 128,
            "target_expert_excluded": True,
            "candidate_target_rows_scored": False,
            "candidate_target_row_storage": "SEALED_MINUS_ONE_SENTINEL",
            "generation_lock_frozen": True,
            "expert_bank_frozen": True,
            "prediction_store_materialized_once": True,
        },
        "proposal_stage": {
            "complete_eight_expert_scores_persisted": True,
            "complete_eight_expert_order_persisted": True,
            "all_eight_experts_retained": True,
            "top_k_selected_from_consumed_results": False,
            "exact_b_score_invented": False,
            "target_labels_opened": False,
        },
        "support_stage": {
            "may_select_any_sealed_member": True,
            "target_support_classifier_labels_consumed": True,
            "label_blind_router_stage": False,
            "nelbo_compatibility_stage": False,
            "zero_centered_empirical_bayes_shrinkage": True,
            "prior_effective_cases": SUPPORT_PRIOR_EFFECTIVE_CASES,
            "minimum_shrunk_bacc_gain": SUPPORT_MINIMUM_SHRUNK_BACC_GAIN,
            "proper_losses_deferred_to_confirmation_safety_gate": True,
            "global_model_updates_from_target_support": False,
        },
        "confirmation_stage": {
            "same_support_selected_member_or_exact_b_only": True,
            "disjoint_target_calibration_classifier_labels_consumed": True,
            "label_blind_router_stage": False,
            "nelbo_compatibility_stage": False,
            "whole_case_paired_dirichlet_bootstrap": True,
            "joint_acceptance_probability": 0.8,
            "minimum_bacc_gain": CALIBRATION_MINIMUM_BACC_GAIN,
            "maximum_brier_delta": CALIBRATION_MAXIMUM_BRIER_DELTA,
            "maximum_log_loss_delta": CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
        },
        "target_protocol": {
            "fold_count_per_center": 5,
            "exact_fold_count": 45,
            "selection_calibration_evaluation_whole_case_disjoint": True,
            "evaluation_cases_scored_exactly_once": True,
            "route_policy_durable_before_evaluation_labels": True,
            "raw_labels_may_be_persisted": False,
            "durable_preterminal_attestation_required": True,
            "fresh_preterminal_validation_process_count": 2,
        },
        "phase_order": [
            "ALL_PROPOSAL_SETS_SEALED",
            "ALL_SUPPORT_DECISIONS_SEALED",
            "ALL_CONFIRMATION_DECISIONS_SEALED",
            "EXACT_45_FOLD_ROUTE_POLICY_SERIALIZED",
            "DURABLE_PRETERMINAL_ATTESTATION",
            "TERMINAL_EVALUATION_LABEL_ACCESS",
            "TWO_FRESH_FINAL_VALIDATIONS",
            "POSTVALIDATION_INDEX_AUTHENTICATED",
        ],
        "policy_transition": POLICY_TRANSITION,
        "policy_sensitivity_estimand": (
            "HELD_OUT_EVALUATION_ROUTE_MINUS_EXACT_B_UTILITY_NOT_ROUTE_COVERAGE"
        ),
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "execution_authorized": True,
        "consumed_test_reuse_authorized": True,
        "execution_authorized_only_by_exact_amendment_bytes": True,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "single_use_execution_identity": True,
        "cross_run_recovery_allowed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "deployable_label_blind_router_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "significance_claimed": False,
        "promotion_allowed": False,
        "deployment_claimed": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "protocol_hash": canonical_hash(body)}


def claim_boundary_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v5_claim_boundary_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": "diagnostic_only",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "execution_authorized": True,
        "consumed_test_reuse_authorized": True,
        "implementation_authorizes_execution": False,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "fresh_evidence": False,
        "architecture_can_recover_observed_routing_structure_only": True,
        "target_label_assisted_policy_sensitivity": True,
        "deployable_label_blind_router_claimed": False,
        "unseen_center_generalization_claimed": False,
        "routing_success_claimed": False,
        "nelbo_compatibility_claimed": False,
        "may_feed_another_experiment": False,
    }


def validate_protocol_payload(payload: object) -> None:
    if payload != protocol_payload():
        raise ProtocolError("SCEPTRE v5 protocol drifted.")


__all__ = (
    "claim_boundary_payload",
    "protocol_payload",
    "validate_protocol_payload",
)
