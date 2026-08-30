"""Frozen scientific and governance protocol for OE-PPUR v4."""

from __future__ import annotations

from .hashing import canonical_hash
from .identity import (
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    EXPECTED_CASE_COUNT,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TEST_ROW_COUNT,
    FRESH_EVIDENCE,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


def frozen_protocol_payload() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "oe_ppur_v4_terminal_scientific_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "claim_dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
        "held_case_route_count": EXPECTED_CASE_COUNT,
        "probability_matrix_shape": list(EXPECTED_PROBABILITY_MATRIX_SHAPE),
        "scientific_method": "OE_PPUR_PAIRWISE_PRIMITIVE_UTILITY_UNCHANGED",
        "source_only_supervision_reuse": (
            "EXACT_IMMUTABLE_SOURCE_ONLY_CONTENT_ALIAS_NO_AUTHORITY_INHERITED"
        ),
        "source_only_supervision_reuse_exception": (
            "USER_AUTHORIZED_V4_ONLY_HASH_EXACT_CONTENT_PROVENANCE"
        ),
        "predecessor_no_feed_fence_acknowledged": True,
        "source_supervision_target_test_rows_present": False,
        "source_supervision_target_test_labels_present": False,
        "target_H_excluded_from_every_fit_normalizer_calibrator_and_candidate_pool": True,
        "source_inner_cross_fitting_required": True,
        "query_to_compatibility_to_decision_to_expert_to_true_utility_order": True,
        "proxy_is_not_true_utility": True,
        "terminal_labels_closed_preterminal": True,
        "terminal_labels_ephemeral_aggregate_only": True,
        "exact_P_on_any_failed_gate": True,
        "direct_input_count": 7,
        "direct_input_roles": list(DIRECT_INPUT_ROLES),
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "workspace_plan_bound_before_amendment": True,
        "resolved_config_template_bound_before_amendment": True,
        "input_manifest_template_bound_before_amendment": True,
        "git_head_tree_index_and_dirty_content_bound": True,
        "nfs_safe_in_place_commit_required": True,
        "v3_operational_state_reuse": False,
        "separate_launch_authority_required": True,
        "publication_and_claim_restrictions": claim_boundary_payload(False),
    }
    return {**body, "protocol_hash": canonical_hash(body)}


def claim_boundary_payload(execution_authorized: bool) -> dict[str, object]:
    return {
        "schema_version": "oe_ppur_v4_terminal_claim_boundary_v1",
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "fresh_evidence": FRESH_EVIDENCE,
        "execution_authorized": bool(execution_authorized),
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


__all__ = ("claim_boundary_payload", "frozen_protocol_payload")
