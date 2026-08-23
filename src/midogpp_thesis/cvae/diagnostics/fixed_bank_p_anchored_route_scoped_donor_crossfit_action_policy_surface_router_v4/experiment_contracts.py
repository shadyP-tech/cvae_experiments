"""Exact-six inputs and immutable governance anchors for executable P-DCAPS v4."""

from __future__ import annotations

from .identity import AUTHORIZATION_BASIS, AUTHORIZATION_SCOPE, EXPERIMENT_ID


CANONICAL_OUTPUT_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router/v4"
)
CANONICAL_SCRATCH_ROOT = (
    "/data/local/fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_v4"
)
LEDGER_AMENDMENT_SCHEMA_VERSION = "midogpp_test_consumption_ledger_amendment_v5"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_test_cache_v4"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_test_manifest_v4"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_parent_v4"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_amendment_v4"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_ledger_amendment_v4.json"
)

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)
AUTHORIZED_INPUT_ROLES = (
    "frozen_source_expert_bank",
    "frozen_generation_lock",
    "fresh_v4_label_free_test_cache_alias",
    "fresh_v4_label_capability_manifest_alias",
    "fresh_v4_parent_consumption_ledger_alias",
    "this_v4_single_consumer_ledger_amendment",
)

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)

# The v4 source snapshot normalizes this assignment, so pinning the immutable
# amendment bytes after source integration cannot create a hash cycle.
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "8ce04d560c6f133ffb914a04fe9c67a3cd418ae9edf0db3568fc212814e2be1d"
)

V2_SOURCE_SNAPSHOT_SCHEMA = "pdcaps_v2_source_snapshot_v1"
EXPECTED_V2_SOURCE_MANIFEST_SHA256 = (
    "3dc6d096ad607fe550eac47b114332fd6ac9ebec5d9cfb59e80897e9a982addc"
)
EXPECTED_V2_SOURCE_TREE_SHA256 = (
    "f457d8678eb93fe51520c9fcc188c8d44f8331aec6f37c88124a677bfcc2d5cb"
)
EXPECTED_V2_SOURCE_MEMBER_COUNT = 105

V3_REPAIR_SOURCE_SNAPSHOT_SCHEMA = "pdcaps_v3_repair_source_snapshot_v1"
EXPECTED_V3_REPAIR_SOURCE_MANIFEST_SHA256 = (
    "37b8e51f8d0900212ec4bfc8bd68b14ddbde1ed783eaacf86e8301dd9295b4a7"
)
EXPECTED_V3_REPAIR_SOURCE_TREE_SHA256 = (
    "df35265e8d27aa602c3ae6c3fcebdc2a3c4838effa4d9f1200e9deb12e7e0a3e"
)
EXPECTED_V3_REPAIR_SOURCE_MEMBER_COUNT = 13

V4_EXECUTION_SOURCE_SNAPSHOT_SCHEMA = "pdcaps_v4_execution_source_snapshot_v1"
V4_EXECUTION_SOURCE_SNAPSHOT_SCOPE = "complete_pdcaps_v4_executable_python_only"
EXPECTED_V4_EXECUTION_SOURCE_MANIFEST_SHA256 = (
    "e195064c0b3a30167ded9d10bb2355bc6ebc8c91c7fd07deb44b115611f8bdbf"
)
EXPECTED_V4_EXECUTION_SOURCE_TREE_SHA256 = (
    "6f116b8b6766a2c7f1e3d163a6b5226ee61556977ae07ed0f5ce3c93cead17b3"
)
EXPECTED_V4_EXECUTION_SOURCE_MEMBER_COUNT = 41
EXPECTED_COMBINED_THREE_SCOPE_SOURCE_SEAL_SHA256 = (
    "e059e75faa8543316ba19555f6b01ca25a0505ce5e28aed4308c28a17cdb5c9a"
)

V1_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v1"
)
V1_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v1"
)
V2_EXPERIMENT_ID = EXPERIMENT_ID[:-1] + "2"
V2_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v2"
)
V3_EXPERIMENT_ID = EXPERIMENT_ID[:-1] + "3"
V3_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v3"
)

# Any direct path carrying one of these fragments violates the exact-six fresh
# alias contract.  Source-code dependencies are separately pinned by seals.
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
    "fixed_bank_p_anchored_route_scoped_center_balanced_",
    "action_policy_surface_router_v1",
    "action_policy_surface_router_v2",
    "action_policy_surface_router_v3",
    "action_policy_surface_router_test_cache_v1",
    "action_policy_surface_router_test_cache_v2",
    "action_policy_surface_router_test_cache_v3",
    "action_policy_surface_router_test_manifest_v1",
    "action_policy_surface_router_test_manifest_v2",
    "action_policy_surface_router_test_manifest_v3",
    "action_policy_surface_router_parent_v1",
    "action_policy_surface_router_parent_v2",
    "action_policy_surface_router_parent_v3",
    "action_policy_surface_router_amendment_v1",
    "action_policy_surface_router_amendment_v2",
    "action_policy_surface_router_amendment_v3",
    "run_state",
    "probability_history",
    "capability_history",
    "residual_topup",
    "historical",
    "quarantine",
    "/scratch/",
    "/checkpoints/",
)

AUTHORIZATION_DATE = "2026-08-23"

__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
