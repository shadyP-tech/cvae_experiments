"""Frozen protocol and terminal claim firewall for executable SCALE-BP v2.

This protocol intentionally describes a label-aware, post-hoc experiment on a
previously consumed test surface.  It is not the project's deployable,
label-free routing protocol and its output is permanently terminal.
"""

from __future__ import annotations

from collections.abc import Mapping

from .hashing import canonical_hash
from .identity import (
    ACTION_FAMILIES,
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    CENTERS,
    CLAIM_SCOPE,
    DIRECT_ACTIONS,
    DIRECTIONS,
    EXECUTION_REVISION,
    EXPECTED_CASE_COUNT,
    EXPECTED_DIRECT_ACTION_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPERIMENT_ID,
    EXCLUDED_CENTERS,
    GovernanceError,
    METRICS,
    PUBLICATION_STATUS,
    SUPPORT_FOLD_COUNT,
    TERMINAL_DECISION,
)


PROTOCOL_SCHEMA = "scale_bp_v2_single_use_terminal_protocol_v1"
CLAIM_FIREWALL_SCHEMA = "scale_bp_v2_terminal_claim_firewall_v1"


def _protocol_body() -> dict[str, object]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "dataset_family": "MIDOG++",
        "claim_dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "held_case_route_count": EXPECTED_CASE_COUNT,
        "eligible_center_ids": list(CENTERS),
        "excluded_center_ids": list(EXCLUDED_CENTERS),
        "held_unit": "whole_case_patient_slide_group",
        "execution_authorized": True,
        "implementation_request_alone_authorizes_execution": False,
        "explicit_single_use_authorization_required": True,
        "single_use_execution_identity": True,
        "consumed_test_reuse_authorized": True,
        "durable_external_authorization_lease_required": True,
        "lease_claimed_after_all_read_only_preflights": True,
        "lease_claimed_before_gpu_or_label_work": True,
        "output_or_scratch_deletion_restores_authorization": False,
        "lease_repair_removal_or_reuse_allowed": False,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "predecessor_outputs_used": False,
        "predecessor_artifacts_used": False,
        "predecessor_amendments_used": False,
        "predecessor_manifests_used": False,
        "predecessor_run_state_used": False,
        "predecessor_scratch_or_checkpoints_used": False,
        "historical_capability_journals_used": False,
        "source_experts_modified": False,
        "target_expert_available": False,
        "outer_center_H_excluded_from_every_donor_fit": True,
        "pseudo_center_J_excluded_from_own_prediction": True,
        "pseudo_held_case_d_excluded_from_own_fit": True,
        "candidate_source_exclusions_recomputed_per_fit": True,
        "held_case_c_excluded_from_every_preterminal_fit": True,
        "route_local_support": "H_MINUS_C",
        "support_fold_count": SUPPORT_FOLD_COUNT,
        "support_folds_whole_case_patient_slide_group_disjoint": True,
        "support_fold_prediction_excludes_own_labels": True,
        "support_labels_update_route_local_model_only": True,
        "support_labels_must_not_update_global_models": True,
        "support_labels_must_not_update_source_experts": True,
        "support_labels_must_not_update_shared_scalers": True,
        "support_labels_must_not_tune_hyperparameters": True,
        "target_identity_is_structural_not_predictive": True,
        "target_support_label_use_is_terminal_diagnostic_only": True,
        "project_label_free_deployment_protocol_satisfied": False,
        "target_terminal_labels_open_only_after_durable_decision_seal": True,
        "decision_seal_binds_all_case_actions_before_terminal_labels": True,
        "raw_labels_may_be_persisted": False,
        "label_capability_events_persist_hashes_not_labels": True,
        "action_families": list(ACTION_FAMILIES),
        "directions": list(DIRECTIONS),
        "direct_actions": list(DIRECT_ACTIONS),
        "direct_action_count": EXPECTED_DIRECT_ACTION_COUNT,
        "metrics": list(METRICS),
        "boundary_projection_primary": True,
        "full_endpoint_primary": False,
        "exact_p_fallback_required": True,
        "direct_case_action_selection": True,
        "learned_prefix_selection_forbidden": True,
        "preterminal_case_action_learnability_abort_required": True,
        "proxy_scores_are_not_true_utility": True,
        "bacc_brier_log_are_terminal_downstream_diagnostics": True,
        "cvae_compatibility_or_nelbo_claimed": False,
        "confidence_bound_claimed": False,
        "conformal_claimed": False,
        "finite_sample_coverage_claimed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "deployment_claimed": False,
        "promotion_allowed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def frozen_protocol_payload() -> dict[str, object]:
    """Return the immutable, path-independent protocol and its digest."""

    body = _protocol_body()
    return {**body, "protocol_hash": canonical_hash(body)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    """Fail closed unless *payload* is byte-semantically the frozen protocol."""

    if dict(payload) != frozen_protocol_payload():
        raise GovernanceError("SCALE-BP v2 frozen protocol drifted.")


def terminal_claim_firewall_payload() -> dict[str, object]:
    """Return the claim fields that every config/report must preserve."""

    body = {
        "schema_version": CLAIM_FIREWALL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "bounded_interpretation": (
            "single_post_hoc_scale_bp_v2_sensitivity_on_the_previously_"
            "consumed_midogpp_uniform_b_test_surface_only"
        ),
        "fresh_evidence": False,
        "project_label_free_deployment_protocol_satisfied": False,
        "target_support_label_use_is_terminal_diagnostic_only": True,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "cvae_compatibility_or_nelbo_claimed": False,
        "confidence_bound_claimed": False,
        "finite_sample_coverage_claimed": False,
        "promotion_eligible": False,
        "deployment_claimed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }
    return {**body, "claim_firewall_hash": canonical_hash(body)}


def validate_terminal_claim_firewall(payload: Mapping[str, object]) -> None:
    if dict(payload) != terminal_claim_firewall_payload():
        raise GovernanceError("SCALE-BP v2 terminal claim firewall drifted.")


__all__ = (
    "CLAIM_FIREWALL_SCHEMA",
    "GovernanceError",
    "PROTOCOL_SCHEMA",
    "frozen_protocol_payload",
    "terminal_claim_firewall_payload",
    "validate_protocol_payload",
    "validate_terminal_claim_firewall",
)
