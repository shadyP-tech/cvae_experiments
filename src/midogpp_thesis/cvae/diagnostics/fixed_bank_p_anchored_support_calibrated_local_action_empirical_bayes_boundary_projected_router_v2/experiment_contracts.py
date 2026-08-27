"""Immutable input, ledger, and output contracts for executable SCALE-BP v2."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_DIRECT_INPUT_COUNT,
    EXPERIMENT_ID,
    GovernanceError,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
)


LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router_ledger_"
    "amendment_v2.json"
)
LEDGER_AMENDMENT_SCHEMA_VERSION = (
    "midogpp_test_consumption_ledger_single_use_execution_authorization_v2"
)
PARENT_LEDGER_ARTIFACT_ID = "midogpp_uniform_b_test_consumption_ledger_v1"
PARENT_LEDGER_MEMBER = "reports/test_consumption_ledger.json"

FORBIDDEN_INPUT_FRAGMENTS = (
    "support_calibrated_local_action_empirical_bayes_boundary_projected_router_v1",
    "action_policy_surface_router",
    "center_balanced_posterior_utility_prefix_router",
    "pcsi_policy_regret_router",
    "probability_surface",
    "capability_history",
    "checkpoint",
    "scratch",
)
FORBIDDEN_PREDECESSOR_FIELDS = (
    "v1_output_used",
    "v1_amendment_used",
    "v1_label_capability_history_used",
    "v1_scratch_or_checkpoint_used",
    "previous_stage90_output_used",
    "previous_stage90_amendment_used",
    "previous_stage90_run_state_used",
    "previous_stage90_scratch_used",
    "previous_probability_surface_used",
    "cross_run_recovery_used",
)
NON_PROMOTION_FIELDS = (
    "fresh_evidence",
    "routing_success_claimed",
    "downstream_utility_claimed",
    "nelbo_compatibility_claimed",
    "confidence_bound_claimed",
    "finite_sample_coverage_claimed",
    "promotion_allowed",
    "deployment_claimed",
    "may_feed_stage50",
    "may_feed_stage60",
    "may_feed_stage70",
    "may_feed_another_stage90",
    "may_feed_another_experiment",
)


def validate_exact_input_fence(
    artifact_ids: Sequence[object],
    *,
    resolved_paths: Sequence[str | Path] = (),
) -> tuple[str, ...]:
    """Require the exact six new-v2/original inputs and reject history paths."""

    observed = tuple(str(value) for value in artifact_ids)
    if (
        observed != DIRECT_INPUT_ARTIFACT_IDS
        or len(observed) != EXPECTED_DIRECT_INPUT_COUNT
        or len(set(observed)) != EXPECTED_DIRECT_INPUT_COUNT
    ):
        raise GovernanceError("SCALE-BP v2 requires its exact six ordered inputs.")
    for value in (*observed, *(str(path) for path in resolved_paths)):
        folded = value.casefold()
        if any(fragment.casefold() in folded for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise GovernanceError("SCALE-BP v2 rejected predecessor or recovery input.")
    return observed


def validate_authorization_amendment(
    payload: Mapping[str, object],
    *,
    expected_source_manifest_sha256: str,
    expected_source_tree_sha256: str,
    expected_source_member_count: int,
) -> None:
    """Validate the one-shot authorization without treating it as evidence."""

    if (
        payload.get("schema_version") != LEDGER_AMENDMENT_SCHEMA_VERSION
        or payload.get("amendment_id") != AUTHORIZATION_AMENDMENT_ARTIFACT_ID
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("parent_artifact_id") != PARENT_LEDGER_ARTIFACT_ID
        or payload.get("parent_member") != PARENT_LEDGER_MEMBER
        or payload.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or payload.get("authorization_basis") != AUTHORIZATION_BASIS
        or payload.get("authorization_scope") != AUTHORIZATION_SCOPE
        or payload.get("execution_authorized") is not True
        or payload.get("consumed_test_reuse_authorized") is not True
        or payload.get("single_use_execution_identity") is not True
        or payload.get("authorization_exhausted") is not False
        or payload.get("authorization_state_at_issuance") != "UNCONSUMED"
        or payload.get("durable_external_authorization_lease_required") is not True
        or payload.get("lease_claimed_atomically_before_gpu_or_label_work") is not True
        or payload.get("output_or_scratch_deletion_restores_authorization") is not False
        or payload.get("lease_repair_removal_or_reuse_allowed") is not False
        or payload.get("direct_input_artifact_ids") != list(DIRECT_INPUT_ARTIFACT_IDS)
        or payload.get("direct_input_roles") != list(DIRECT_INPUT_ROLES)
        or payload.get("source_snapshot_manifest_sha256")
        != expected_source_manifest_sha256
        or payload.get("source_snapshot_tree_sha256") != expected_source_tree_sha256
        or payload.get("source_snapshot_member_count")
        != expected_source_member_count
        or payload.get("publication_status") != PUBLICATION_STATUS
        or payload.get("terminal_decision") != TERMINAL_DECISION
        or payload.get("canonical_output_relative_root")
        != CANONICAL_OUTPUT_RELATIVE_ROOT
        or any(payload.get(field) is not False for field in FORBIDDEN_PREDECESSOR_FIELDS)
        or any(payload.get(field) is not False for field in NON_PROMOTION_FIELDS)
    ):
        raise GovernanceError("SCALE-BP v2 authorization amendment drifted.")


__all__ = (
    "FORBIDDEN_INPUT_FRAGMENTS",
    "FORBIDDEN_PREDECESSOR_FIELDS",
    "LEDGER_AMENDMENT_FILENAME",
    "LEDGER_AMENDMENT_SCHEMA_VERSION",
    "NON_PROMOTION_FIELDS",
    "PARENT_LEDGER_ARTIFACT_ID",
    "PARENT_LEDGER_MEMBER",
    "TEST_CONSUMPTION_LEDGER_ARTIFACT_ID",
    "validate_authorization_amendment",
    "validate_exact_input_fence",
)
