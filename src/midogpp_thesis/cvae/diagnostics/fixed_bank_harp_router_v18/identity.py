"""Immutable identity and claim boundary for the isolated HARP v18 router."""

from __future__ import annotations

from ...routing.harp_protocol import canonical_hash


EXPERIMENT_ID = "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v18"
EXPERIMENT_NAME = "uniform_b_v2_consumed_test_fixed_bank_harp_router_v18"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_harp_router_v18"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_SCOPE = "diagnostic_only"
EXECUTION_REVISION = "v18_known_center_case_conditional_composite_signed_utility_aligned_metrics_nested_stacking"
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_harp_v18_"
    "source_train_case_conditional_composite_router_diagnostic"
)


def authorization_input_binding_payload(
    *,
    expert_bank_lock_hash: str,
    generation_lock_hash: str,
    test_cache_content_sha256: str,
    development_manifest_sha256: str,
    evaluation_manifest_sha256: str,
    parent_ledger_sha256: str,
) -> dict[str, object]:
    """Bind separate authority to the exact immutable scientific inputs."""

    body: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_authorized_input_binding_v18",
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "expert_bank_lock_hash": expert_bank_lock_hash,
        "generation_lock_hash": generation_lock_hash,
        "test_cache_content_sha256": test_cache_content_sha256,
        "development_manifest_sha256": development_manifest_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "parent_ledger_sha256": parent_ledger_sha256,
    }
    return {**body, "input_binding_hash": canonical_hash(body)}


def claim_boundary_payload(*, execution_authorized: bool) -> dict[str, object]:
    """Return the non-promotable consumed-test claim firewall."""

    body: dict[str, object] = {
        "schema_version": "midogpp_harp_stage90_claim_boundary_v18",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "execution_authorized": bool(execution_authorized),
        "implementation_authorizes_execution": False,
        "consumed_test_reuse": True,
        "architecture_recovery_is_descriptive_only": True,
        "unseen_center_generalization_claimed": False,
        "confirmatory_improvement_claimed": False,
        "routing_success_claimed": False,
        "deployment_or_promotion_allowed": False,
        "may_feed_stage60_or_stage70": False,
        "may_feed_any_other_experiment": False,
    }
    return {**body, "claim_boundary_hash": canonical_hash(body)}


__all__ = (
    "AUTHORIZATION_SCOPE",
    "CLAIM_SCOPE",
    "EXECUTION_REVISION",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "OUTPUT_ARTIFACT_ID",
    "PUBLICATION_STATUS",
    "TERMINAL_DECISION",
    "authorization_input_binding_payload",
    "claim_boundary_payload",
)
