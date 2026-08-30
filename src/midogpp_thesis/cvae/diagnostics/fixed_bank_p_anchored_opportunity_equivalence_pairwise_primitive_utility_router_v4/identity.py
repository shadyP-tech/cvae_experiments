"""Closed-world identity for the OE-PPUR v4 workspace-sealed successor."""

from __future__ import annotations


PACKAGE_NAME = (
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_"
    "router_v4"
)
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router.v4"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_v4"
)
EXPERIMENT_NAME = (
    "P-anchored opportunity-equivalence pairwise primitive-utility router v4"
)
CLI_SURFACE = "oe-ppur-v4"
CANONICAL_OUTPUT_RELATIVE_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_"
    "equivalence_pairwise_primitive_utility_router/v4"
)

PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_SCOPE = "diagnostic_only"
FRESH_EVIDENCE = False

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
SOURCE_SUPERVISION_ARTIFACT_ID = (
    "midogpp_stage90_oe_ppur_source_training_action_supervision_v4"
)
SOURCE_CONTENT_LINEAGE_ARTIFACT_ID = (
    "midogpp_stage90_oe_ppur_source_training_action_supervision_v3"
)
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_cache_v4"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_manifest_v4"
)
ORIGINAL_PARENT_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_parent_v4"
)
AUTHORIZATION_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_amendment_v4"
)
AUTHORIZATION_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_"
    "pairwise_primitive_utility_router_ledger_amendment_v4.json"
)

DIRECT_INPUT_ROLES = (
    "frozen_source_expert_bank",
    "frozen_generation_lock",
    "oe_v4_immutable_source_only_supervision_alias",
    "fresh_oe_v4_label_free_test_cache_alias",
    "fresh_oe_v4_canonical_manifest_alias",
    "fresh_oe_v4_original_parent_consumption_ledger_alias",
    "oe_v4_workspace_sealed_single_consumer_authorization_amendment",
)
DIRECT_INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    SOURCE_SUPERVISION_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    ORIGINAL_PARENT_LEDGER_ARTIFACT_ID,
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
)
EXPECTED_INPUT_KINDS = (
    "directory",
    "directory",
    "directory",
    "directory",
    "file",
    "file",
    "file",
)
INPUT_RELATIVE_MEMBERS = (
    "",
    "",
    "",
    "",
    "manifest.csv",
    "reports/test_consumption_ledger.json",
    AUTHORIZATION_AMENDMENT_FILENAME,
)
SOURCE_SUPERVISION_REQUIRED_MEMBERS = (
    "manifests/source_training_surface.json",
    "manifests/source_pool_lineage.json",
    "tables/source_rows.csv",
    "arrays/source_action_probabilities.npy",
    "manifests/content_index.json",
    "reports/validation_report.json",
)

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_BANK_CONTENT_INDEX_SHA256 = (
    "6b74fe794bd30cf6c1e42190427e506d1ff50ecd9280b9dcfee2a7592ec6a318"
)
EXPECTED_GENERATION_CONTENT_INDEX_SHA256 = (
    "086eb106a11fd52df5fc1f692d17a33edccaf6f707b2b9dd0fba15895d891d86"
)
EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "e9b6a05a6f4a4c982ce51eff7606b4ca303fd4c9df7dc466a6a3cc88fd93fe66"
)
EXPECTED_SOURCE_SURFACE_SHA256 = (
    "51084af5dfdf9ab7a34ccac2524b664c3df0860bdf02589b55c6d94310f968dd"
)
EXPECTED_SOURCE_ROW_ORDER_SHA256 = (
    "73bd8ade9944cbcf2e2dd9c4a8f4f247190ca68e880819a1f810a78ac64ae9bb"
)
EXPECTED_SOURCE_PRODUCER_SEAL_SHA256 = (
    "74bf5b5c01d50190a6a0639533f298cea7ece8d381fdbbd578fa82537c48ab91"
)
EXPECTED_SOURCE_RECOMPUTATION_RECEIPT_SHA256 = (
    "e6b641a605ecd85774ef7a6ad06c1f47c0abe68ecf6448f1aa3f0b4eee353241"
)
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_TEST_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
PRESERVED_V3_AMENDMENT_SHA256 = (
    "56269322ead01ef683c985d8f295b0369fb35ddef04d12115704f1df18a0c425"
)

EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_CASE_COUNT = 218
EXPECTED_PROBABILITY_MATRIX_SHAPE = (9_928, 7)
EXPECTED_TERMINAL_CASE_INVENTORY_SHA256 = (
    "d22568075a287af71d0f4477ba5e6265e43278cba4865f7775741cdbcdf2bcc6"
)

AUTHORIZATION_SCOPE = "one_terminal_consumed_test_oe_ppur_v4_run"
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_for_oe_ppur_v4_workspace_sealed_successor"
)
LEASE_DIRECTORY_NAME = ".oe_ppur_v4_single_use_authorization_consumed"
DEFAULT_SCRATCH_ROOT = (
    "/data/local/fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_v4"
)

FORBIDDEN_OPERATIONAL_PATH_FRAGMENTS = (
    "opportunity_equivalence_pairwise_primitive_utility_router/v1",
    "opportunity_equivalence_pairwise_primitive_utility_router/v2",
    "opportunity_equivalence_pairwise_primitive_utility_router/v3",
    "contracts/oe_ppur_v3",
    ".oe_ppur_v3_single_use_authorization_consumed",
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3",
    ".quarantine",
    "/checkpoints/",
    "/run_state/",
    "cross_run_recovery",
)

__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
