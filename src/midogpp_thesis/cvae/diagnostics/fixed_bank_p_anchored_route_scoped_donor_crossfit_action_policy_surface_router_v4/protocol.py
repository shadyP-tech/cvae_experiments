"""Frozen terminal consumed-test protocol for executable P-DCAPS v4."""

from __future__ import annotations

from typing import Mapping

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.protocol import (
    frozen_protocol_payload as frozen_v3_protocol_payload,
)
from .experiment_contracts import (
    AUTHORIZATION_DATE,
    EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256,
    EXPECTED_V2_SOURCE_MANIFEST_SHA256,
    EXPECTED_V2_SOURCE_MEMBER_COUNT,
    EXPECTED_V2_SOURCE_TREE_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256,
    EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
    EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
    EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256,
    EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT,
    EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256,
    V2_SOURCE_SNAPSHOT_SCHEMA,
    V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
    V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    V2_EXPERIMENT_ID,
    V2_OUTPUT_ARTIFACT_ID,
    V3_EXPERIMENT_ID,
    V3_OUTPUT_ARTIFACT_ID,
    canonical_hash,
)


PROTOCOL_SCHEMA = "pdcaps_v4_terminal_protocol_v1"


def frozen_protocol_payload() -> dict[str, object]:
    """Bind v3 scientific mechanics to one authorized v4 execution identity."""

    payload = dict(frozen_v3_protocol_payload())
    payload.pop("protocol_hash", None)
    payload.update(
        {
            "schema_version": PROTOCOL_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "execution_authorized": True,
            "consumed_test_reuse_authorized": True,
            "implementation_authorizes_execution": False,
            "authorization_is_separate_from_implementation_request": True,
            "source_code_or_implementation_request_alone_authorizes_execution": False,
            "authorization_basis": AUTHORIZATION_BASIS,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "authorization_date": AUTHORIZATION_DATE,
            "single_use_execution_identity": True,
            "authorization_exhausted": False,
            "separate_future_run_authorization_required": False,
            "v2_experiment_id": V2_EXPERIMENT_ID,
            "v2_output_artifact_id": V2_OUTPUT_ARTIFACT_ID,
            "v2_execution_status": "FAILED_EXHAUSTED",
            "v2_authorization_exhausted": True,
            "v2_retry_forbidden": True,
            "v3_experiment_id": V3_EXPERIMENT_ID,
            "v3_output_artifact_id": V3_OUTPUT_ARTIFACT_ID,
            "v3_workspace_status": "planned_non_executable",
            "v3_authorization_reused": False,
            "v3_output_used": False,
            "v3_amendment_used": False,
            "v3_run_state_used": False,
            "v3_scratch_or_checkpoint_used": False,
            "v3_probability_or_capability_history_used": False,
            "scientific_protocol_unchanged_from_v3": True,
            "scientific_method_changed_from_v3": False,
            "mechanical_repair_source_is_v3": True,
            "physical_probability_surface_recomputed_from_original_inputs": True,
            "previous_stage90_outputs_used": False,
            "previous_stage90_amendments_used": False,
            "previous_stage90_scratch_or_checkpoints_used": False,
            "inherited_v2_base_source_snapshot_schema": V2_SOURCE_SNAPSHOT_SCHEMA,
            "inherited_v2_base_source_manifest_sha256": (
                EXPECTED_V2_SOURCE_MANIFEST_SHA256
            ),
            "inherited_v2_base_source_tree_sha256": EXPECTED_V2_SOURCE_TREE_SHA256,
            "inherited_v2_base_source_member_count": EXPECTED_V2_SOURCE_MEMBER_COUNT,
            "v3_repair_source_snapshot_schema": V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA,
            "v3_repair_source_manifest_sha256": (
                EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256
            ),
            "v3_repair_source_tree_sha256": EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256,
            "v3_repair_source_member_count": EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT,
            "v4_execution_source_snapshot_schema": (
                V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA
            ),
            "v4_execution_source_manifest_sha256": (
                EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256
            ),
            "v4_execution_source_tree_sha256": (
                EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256
            ),
            "v4_execution_source_member_count": (
                EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT
            ),
            "combined_three_scope_source_seal_sha256": (
                EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256
            ),
            "source_scopes_are_disjoint": True,
            "target_labels_open_only_after_durable_preterminal_attestation": True,
            "raw_labels_may_be_persisted": False,
            "fresh_evidence": False,
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "may_feed_stage50": False,
            "may_feed_stage60": False,
            "may_feed_stage70": False,
            "may_feed_another_stage90": False,
            "may_feed_another_experiment": False,
            "routing_success_claimed": False,
            "downstream_utility_claimed": False,
            "nelbo_compatibility_claimed": False,
            "deployment_claimed": False,
            "promotion_allowed": False,
        }
    )
    return {**payload, "protocol_hash": canonical_hash(payload)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if dict(payload) != frozen_protocol_payload():
        raise ProtocolError("P-DCAPS v4 frozen protocol drifted.")


__all__ = ("PROTOCOL_SCHEMA", "frozen_protocol_payload", "validate_protocol_payload")
