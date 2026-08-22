"""Frozen six-input and workspace identities for the CBPUPR v3 repair."""

from __future__ import annotations

from .constants import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    EVALUATION_SPLIT,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)


CANONICAL_OUTPUT_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router/v3"
)
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v3_global_and_center_surface_"
    "lineage_mechanical_repair"
)
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_for_cbpupr_v3_global_and_center_surface_"
    "lineage_mechanical_repair_run"
)
LEDGER_AMENDMENT_SCHEMA_VERSION = "midogpp_test_consumption_ledger_amendment_v4"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_center_balanced_"
    "posterior_utility_prefix_router_test_cache_v3"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_center_balanced_"
    "posterior_utility_prefix_router_test_manifest_v3"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_route_"
    "scoped_center_balanced_posterior_utility_prefix_router_parent_v3"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_route_"
    "scoped_center_balanced_posterior_utility_prefix_router_amendment_v3"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_ledger_amendment_v3.json"
)

# These four v3 aliases are valid only when the separately authorized config,
# ledger amendment, catalog entry, and registry entry are frozen together.
# Merely importing this package cannot launch it.
WORKSPACE_ALIAS_PLACEHOLDER_IDS = (
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)
AUTHORIZED_INPUT_ROLES = (
    "frozen_source_expert_bank",
    "frozen_generation_lock",
    "fresh_v3_label_free_test_cache_alias",
    "fresh_v3_label_capability_manifest_alias",
    "fresh_v3_parent_consumption_ledger_alias",
    "this_v3_single_consumer_ledger_amendment",
)

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"

# Inputs must be the original bank/generation/cache/manifest/ledger/amendment
# chain.  Every predecessor diagnostic and numbered-stage product is rejected.
FORBIDDEN_INPUT_FRAGMENTS = (
    "50_all_candidate_utility_matrix",
    "60_routing_and_composition",
    "70_frozen_policy_downstream",
    "frozen_policy_downstream",
    "consumed_validation",
    "fixed_bank_loo_",
    "fixed_bank_disagreement_regret_prediction_only",
    "fixed_bank_actionability_recoverability",
    "fixed_bank_hierarchical_residual_stacker",
    "fixed_bank_label_aware_case_oof_ceiling",
    "fixed_bank_labeled_support_case_conditional_flip_router",
    "fixed_bank_multi_challenger_hierarchical_flip_router",
    "fixed_bank_p_anchored_directional_",
    "fixed_bank_p_anchored_crossfit_",
    "fixed_bank_p_anchored_boundary_projected_",
    "fixed_bank_p_anchored_simultaneous_shift_",
    "fixed_bank_p_anchored_route_scoped_boundary_projected_",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_v1",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_test_cache_v1",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_test_manifest_v1",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_parent_v1",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_amendment_v1",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_v2",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_test_cache_v2",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_test_manifest_v2",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_parent_v2",
    "fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_"
    "prefix_router_amendment_v2",
    "fixed_bank_pooled_bacc_case_oof_ceiling",
    "fixed_bank_signed_error_gate",
    "fixed_bank_support_static_router",
    "utility_aligned_",
    "residual_topup",
    "historical",
    "quarantine",
    "/scratch/",
    "/checkpoints/",
)
FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS = (
    "stage50", "stage60", "stage70", "/50_", "/60_", "/70_"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
