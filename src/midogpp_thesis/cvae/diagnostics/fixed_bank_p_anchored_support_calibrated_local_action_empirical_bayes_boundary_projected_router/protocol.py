"""Leakage firewall and terminal claim boundary for SCALE-BP v1."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .identity import (
    ACTION_FAMILIES,
    DIRECTIONS,
    EXPERIMENT_ID,
    METRICS,
    METHOD_MENU,
    PUBLICATION_STATUS,
    SUPPORT_FOLD_COUNT,
    TERMINAL_DECISION,
)
from .source_seal import source_seal_identity


def _protocol_body() -> dict[str, object]:
    source_seal = source_seal_identity()
    return {
        "schema_version": "scale_bp_v1_terminal_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "claim_dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "split": "test",
        "split_previously_consumed": True,
        "eligible_test_row_count": 9928,
        "held_case_route_count": 218,
        "eligible_center_ids": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
        "excluded_center_ids": ["4"],
        "held_unit": "whole_case_patient_slide_group",
        "held_case_c_excluded_from_all_preterminal_fits": True,
        "route_identity_inventory_derived_from_exact_manifest_keys": True,
        "route_scope_witness_bound_to_final_and_pseudo_scopes": True,
        "support_and_evaluation_sample_key_hashes_required": True,
        "route_local_support": "H_minus_c",
        "support_fold_count": SUPPORT_FOLD_COUNT,
        "support_folds_whole_case_disjoint": True,
        "support_fold_prediction_excludes_own_labels": True,
        "support_labels_update_route_local_H_c_model_only": True,
        "support_labels_must_not_update_global_models": True,
        "support_labels_must_not_update_source_experts": True,
        "support_labels_must_not_update_shared_scalers_or_hyperparameters": True,
        "donor_prior_centers": "J_not_equal_H",
        "outer_center_H_excluded_from_every_donor_fit": True,
        "pseudo_center_J_excluded_from_own_prediction": True,
        "pseudo_held_case_d_excluded_from_own_fit": True,
        "candidate_source_exclusions_recomputed_per_fit": True,
        "target_expert_available": False,
        "source_experts_modified": False,
        "target_terminal_labels_may_open": False,
        "target_labels_open_only_after_durable_preterminal_attestation": True,
        "raw_labels_may_be_persisted": False,
        "action_families": list(ACTION_FAMILIES),
        "directions": list(DIRECTIONS),
        "metrics": list(METRICS),
        "method_menu": list(METHOD_MENU),
        "boundary_projection_primary": True,
        "full_endpoint_primary": False,
        "endpoint_receipt_rederived_from_exact_physical_rectangle": True,
        "physical_surfaces_issued_from_validated_read_only_memmaps": True,
        "physical_bank_is_factory_sealed_exact_810_cell_mapping": True,
        "physical_bank_cell_identity_file_offset_slice_and_row_hash_required": True,
        "physical_bank_symlink_out_of_root_missing_duplicate_and_overlap_forbidden": True,
        "physical_bank_receipt_hash_propagated_to_endpoint_evidence": True,
        "exact_p_fallback_required": True,
        "direct_case_action_selection": True,
        "learned_prefix_selection_forbidden": True,
        "preterminal_case_level_learnability_abort_required": True,
        "pseudo_admission_replays_exact_primary_and_all_frozen_controls": True,
        "pseudo_admission_requires_all_nine_outer_centers": True,
        "pseudo_admission_H_J_d_context_count": 1744,
        "every_outer_center_must_pass_before_final_routing": True,
        "pseudo_evidence_is_factory_sealed": True,
        "all_outer_evidence_bundle_is_factory_sealed": True,
        "admission_input_action_policy_and_oracle_roots_required": True,
        "realized_action_oracle_is_mechanically_derived": True,
        "caller_supplied_admission_metrics_or_oracle_forbidden": True,
        "terminal_denominators_derived_from_exact_pseudo_center_labels": True,
        "caller_supplied_terminal_denominators_forbidden": True,
        "pseudo_center_label_population_consistent_across_all_H_and_d": True,
        "terminal_label_requests_are_transient_and_nonserializable": True,
        "proxy_scores_are_not_true_utility": True,
        "confidence_bound_claimed": False,
        "conformal_claimed": False,
        "finite_sample_coverage_claimed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "consumed_test_reuse_authorized": False,
        "separate_future_execution_identity_required": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "promotion_allowed": False,
        "deployment_claimed": False,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
        "final_route_inventory_receipt_required": True,
        "final_route_task_and_result_count": 218,
        "source_manifest_required": True,
        "source_manifest_member": source_seal.manifest_member,
        "source_manifest_sha256": source_seal.manifest_sha256,
        "source_tree_sha256": source_seal.tree_sha256,
        "source_member_count": source_seal.member_count,
        "source_manifest_checked_before_any_gpu_or_label_access": True,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def frozen_protocol_payload() -> dict[str, object]:
    """Return the immutable protocol plus its path-independent digest."""

    body = _protocol_body()
    return {**body, "protocol_hash": canonical_hash(body)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    """Fail closed unless ``payload`` is exactly the frozen protocol."""

    if dict(payload) != frozen_protocol_payload():
        raise ProtocolError("SCALE-BP protocol contract drifted.")


__all__ = (
    "ProtocolError",
    "frozen_protocol_payload",
    "validate_protocol_payload",
)
