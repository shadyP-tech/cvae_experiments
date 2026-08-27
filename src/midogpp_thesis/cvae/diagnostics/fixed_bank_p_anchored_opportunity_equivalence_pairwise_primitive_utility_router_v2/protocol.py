"""Frozen terminal-only protocol for executable OE-PPUR v2 mechanics."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .identity import (
    ACTION_IDS,
    CENTERS,
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXCLUDED_CENTERS,
    EXPERIMENT_ID,
    EXPECTED_BANK_CONTENT_INDEX_SHA256,
    EXPECTED_CASE_COUNT,
    EXPECTED_GENERATION_CONTENT_INDEX_SHA256,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TEST_ROW_COUNT,
    FRESH_EVIDENCE,
    P_ACTION_ID,
    PUBLICATION_STATUS,
    PROBABILITY_COLUMN_IDS,
    TERMINAL_DECISION,
)


def claim_boundary_payload(*, execution_authorized: bool) -> dict[str, object]:
    """Return the invariant scientific firewall for either config state."""

    return {
        "schema_version": "oe_ppur_v2_terminal_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "fresh_evidence": FRESH_EVIDENCE,
        "execution_authorized": execution_authorized,
        "consumed_test_reuse_authorized": execution_authorized,
        "single_use_execution_identity": execution_authorized,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "cvae_compatibility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "deployment_claimed": False,
        "promotion_allowed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def _protocol_body() -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v2_single_use_terminal_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
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
        "candidate_action_ids": list(ACTION_IDS),
        "protected_probability_baseline_id": P_ACTION_ID,
        "probability_matrix_column_ids": list(PROBABILITY_COLUMN_IDS),
        "probability_matrix_shape": list(EXPECTED_PROBABILITY_MATRIX_SHAPE),
        "probability_matrix_dtype": "<f4",
        "canonical_admitted_row_binding_required": True,
        "parsed_probability_matrix_science_receipt_required": True,
        "probability_matrix_actual_bytes_must_be_parsed_read_only": True,
        "probability_matrix_finite_and_unit_bounded_required": True,
        "probability_shards_within_admitted_scratch_root_required": True,
        "worker_declared_row_count_is_scientific_evidence": False,
        "source_only_row_posterior": True,
        "target_H_excluded_from_every_fit_normalizer_calibrator_and_candidate_pool": True,
        "target_support_labels_for_final_routing": False,
        "query_to_compatibility_to_decision_to_expert_to_true_utility_order": True,
        "route_policy_proxy_is_true_utility": False,
        "route_policy_proxy_is_cvae_compatibility": False,
        "route_policy_proxy_is_nelbo_compatibility": False,
        "target_terminal_labels_closed_preterminal": True,
        "target_terminal_labels_ephemeral_aggregate_only": True,
        "typed_preterminal_decision_ledger_required": True,
        "preterminal_decision_ledger_frozen_before_labels": True,
        "two_artifact_only_fresh_process_attestations_required": True,
        "terminal_aggregate_capability_one_shot": True,
        "terminal_evaluated_case_count": EXPECTED_CASE_COUNT,
        "terminal_receipt_exact_ledger_binding_required": True,
        "preterminal_service_manifest_path_exposed": False,
        "structural_scientific_service_injection_allowed": False,
        "exact_P_on_any_failed_gate": True,
        "direct_input_count": 6,
        "direct_input_roles": list(DIRECT_INPUT_ROLES),
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "direct_input_order_exact": True,
        "direct_input_duplicates_forbidden": True,
        "predecessor_output_report_amendment_scratch_or_lease_reuse": False,
        "single_consumer_v2_amendment_required": True,
        "amendment_must_chain_directly_to_original_parent": True,
        "source_config_protocol_exact_binding_required": True,
        "bank_complete_content_index_required": True,
        "generation_complete_content_index_required": True,
        "bank_content_index_file_sha256_pinned": True,
        "generation_content_index_file_sha256_pinned": True,
        "bank_content_index_file_sha256": EXPECTED_BANK_CONTENT_INDEX_SHA256,
        "generation_content_index_file_sha256": (
            EXPECTED_GENERATION_CONTENT_INDEX_SHA256
        ),
        "admitted_input_location_binding_exact_matched_by_service_factory": True,
        "shared_protocol_source_member_sealed": True,
        "production_workstation_observations_caller_injectable": False,
        "admission_is_read_only": True,
        "authorization_lease_claimed_after_read_only_admission_only": True,
        "cross_run_recovery_allowed": False,
        "failure_after_lease": "FAILED_EXHAUSTED",
        "publication_and_claim_boundary": claim_boundary_payload(
            execution_authorized=False
        ),
    }


def frozen_protocol_payload() -> dict[str, object]:
    body = _protocol_body()
    return {**body, "protocol_hash": canonical_hash(body)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or dict(payload) != frozen_protocol_payload():
        raise ProtocolError("OE-PPUR v2 protocol contract drifted.")


def validate_claim_boundary(
    payload: Mapping[str, object], *, execution_authorized: bool
) -> None:
    if not isinstance(payload, Mapping) or dict(payload) != claim_boundary_payload(
        execution_authorized=execution_authorized
    ):
        raise ProtocolError("OE-PPUR v2 terminal claim firewall drifted.")


__all__ = (
    "ProtocolError",
    "claim_boundary_payload",
    "frozen_protocol_payload",
    "validate_claim_boundary",
    "validate_protocol_payload",
)
