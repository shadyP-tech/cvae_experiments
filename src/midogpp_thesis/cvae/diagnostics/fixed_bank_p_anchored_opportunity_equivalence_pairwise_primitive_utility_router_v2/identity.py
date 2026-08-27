"""Closed-world identity for the executable OE-PPUR v2 successor.

These constants describe an executable *identity*, not current authority.  No
authorization amendment is shipped by this package and importing this module
cannot grant consumed-test or label access.
"""

from __future__ import annotations


PACKAGE_NAME = (
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_"
    "router_v2"
)
STEM = PACKAGE_NAME
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router.v2"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_v2"
)
EXPERIMENT_NAME = (
    "P-anchored opportunity-equivalence pairwise primitive-utility router v2"
)
CLI_SURFACE = (
    "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
    "utility-router-v2"
)

V1_EXPERIMENT_ID = EXPERIMENT_ID[:-1] + "1"
V1_OUTPUT_ARTIFACT_ID = OUTPUT_ARTIFACT_ID[:-1] + "1"

PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_SCOPE = "diagnostic_only"
FRESH_EVIDENCE = False

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTERS = ("4",)
ACTION_FAMILIES = ("B", "I", "R")
DIRECTIONS = ("zero_to_one", "one_to_zero")
ACTION_IDS = tuple(
    f"{family}::{direction}"
    for family in ACTION_FAMILIES
    for direction in DIRECTIONS
)
P_ACTION_ID = "P_PROTECTED"
PROBABILITY_COLUMN_IDS = (P_ACTION_ID, *ACTION_IDS)
EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_CASE_COUNT = 218
EXPECTED_ACTION_COUNT = len(ACTION_IDS)
EXPECTED_PROBABILITY_COLUMN_COUNT = len(PROBABILITY_COLUMN_IDS)
EXPECTED_PROBABILITY_MATRIX_SHAPE = (
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_PROBABILITY_COLUMN_COUNT,
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_cache_v2"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_opportunity_equivalence_pairwise_"
    "primitive_utility_router_test_manifest_v2"
)
ORIGINAL_PARENT_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_parent_v2"
)
AUTHORIZATION_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_amendment_v2"
)
AUTHORIZATION_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_"
    "pairwise_primitive_utility_router_ledger_amendment_v2.json"
)

DIRECT_INPUT_ROLES = (
    "frozen_source_expert_bank",
    "frozen_generation_lock",
    "fresh_oe_v2_label_free_test_cache_alias",
    "fresh_oe_v2_canonical_manifest_alias",
    "fresh_oe_v2_original_parent_consumption_ledger_alias",
    "oe_v2_only_single_consumer_authorization_amendment",
)
DIRECT_INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    ORIGINAL_PARENT_LEDGER_ARTIFACT_ID,
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
)
INPUT_ARTIFACT_IDS = DIRECT_INPUT_ARTIFACT_IDS
INPUT_ROLES = DIRECT_INPUT_ROLES

EXPECTED_INPUT_KINDS = (
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
    "manifest.csv",
    "reports/test_consumption_ledger.json",
    AUTHORIZATION_AMENDMENT_FILENAME,
)

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_BANK_CONTENT_INDEX_SHA256 = (
    "7fae389151618950102905f85c5fce300ef9821f15cf304d8469b5a147110279"
)
EXPECTED_GENERATION_CONTENT_INDEX_SHA256 = (
    "6b74fe794bd30cf6c1e42190427e506d1ff50ecd9280b9dcfee2a7592ec6a318"
)
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_TEST_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TERMINAL_CASE_INVENTORY_SHA256 = (
    "d22568075a287af71d0f4477ba5e6265e43278cba4865f7775741cdbcdf2bcc6"
)
EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)

AUTHORIZATION_SCOPE = "one_terminal_consumed_test_oe_ppur_v2_run"
AUTHORIZATION_BASIS = (
    "separate_explicit_user_authorization_for_oe_ppur_v2_terminal_run"
)

# Direct inputs may never point into predecessor or mutable run-history state.
# Exact bank/GenerationLock IDs are intentionally not included here.
FORBIDDEN_INPUT_PATH_FRAGMENTS = (
    "opportunity_equivalence_pairwise_primitive_utility_router/v1",
    "opportunity_equivalence_pairwise_primitive_utility_router_v1",
    ".quarantine",
    "/quarantine/",
    "/scratch/",
    "/checkpoints/",
    "/run_state/",
    "probability_history",
    "capability_history",
    "cross_run_recovery",
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
