"""Closed-world identity for the one-shot executable SCALE-BP v2 diagnostic.

The identity is deliberately unrelated to the planned v1 package at import and
artifact level.  Only the original Uniform-B bank/GenerationLock and newly
registered v2 aliases of the immutable test inputs are direct inputs.
"""

from __future__ import annotations


class GovernanceError(ValueError):
    """Fail-closed v2 contract or capability violation."""


PACKAGE_NAME = (
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router_v2"
)
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router.v2"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router_v2"
)
EXPERIMENT_NAME = (
    "P-anchored support-calibrated local-action empirical-Bayes "
    "boundary-projected router v2 terminal diagnostic"
)
CLI_SURFACE = (
    "fixed-bank-p-anchored-support-calibrated-local-action-empirical-bayes-"
    "boundary-projected-router-v2"
)

AUTHORIZATION_BASIS = (
    "explicit_user_authorization_2026_08_26_for_single_scale_bp_v2_"
    "consumed_test_terminal_diagnostic"
)
AUTHORIZATION_SCOPE = (
    "one_complete_scale_bp_v2_execution_on_the_previously_consumed_"
    "midogpp_uniform_b_test_surface"
)
EXECUTION_REVISION = "v2_single_use_consumed_test_terminal_diagnostic"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_SCOPE = "diagnostic_only"

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTERS = ("4",)
EXPECTED_CENTER_COUNT = 9
EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_CASE_COUNT = 218
EXPECTED_CASE_COUNTS_BY_CENTER = (
    ("0", 23),
    ("1", 20),
    ("2", 24),
    ("3", 39),
    ("5", 23),
    ("6", 23),
    ("7", 21),
    ("8", 22),
    ("9", 23),
)
EXPECTED_PHYSICAL_CELL_COUNT = 810
FEATURE_DIM = 3_840
EXPECTED_TRAINING_SEEDS = (17, 42, 101)
EXPECTED_GENERATION_SEEDS = (17, 42, 101)
SUPPORT_FOLD_COUNT = 4

ACTION_FAMILIES = ("B", "I", "R")
DIRECTIONS = ("zero_to_one", "one_to_zero")
DIRECT_ACTIONS = tuple(
    f"{family}::{direction}"
    for family in ACTION_FAMILIES
    for direction in DIRECTIONS
)
EXPECTED_DIRECT_ACTION_COUNT = 6
METRICS = ("bacc", "brier", "log")
P_METHOD_ID = "P_PROTECTED"
PRIMARY_METHOD_ID = "SCALE_BP_V2_PRIMARY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_support_calibrated_local_action_"
    "empirical_bayes_boundary_projected_router_test_cache_v2"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_support_calibrated_local_action_"
    "empirical_bayes_boundary_projected_router_test_manifest_v2"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router_parent_v2"
)
AUTHORIZATION_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router_authorization_amendment_v2"
)
LEDGER_AMENDMENT_ARTIFACT_ID = AUTHORIZATION_AMENDMENT_ARTIFACT_ID
DIRECT_INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
)
DIRECT_INPUT_ROLES = (
    "original_frozen_source_expert_bank",
    "original_frozen_generation_lock",
    "v2_alias_of_immutable_test_cache",
    "v2_alias_of_immutable_test_manifest",
    "v2_alias_of_immutable_original_parent_consumption_ledger",
    "v2_single_use_authorization_amendment",
)
EXPECTED_DIRECT_INPUT_COUNT = 6

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
EXPECTED_TEST_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_PARENT_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)

CANONICAL_OUTPUT_RELATIVE_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router/v2"
)
CANONICAL_SCRATCH_ROOT = (
    "/data/local/fixed_bank_p_anchored_support_calibrated_local_action_"
    "empirical_bayes_boundary_projected_router_v2"
)


if (
    len(CENTERS) != EXPECTED_CENTER_COUNT
    or sum(count for _, count in EXPECTED_CASE_COUNTS_BY_CENTER)
    != EXPECTED_CASE_COUNT
    or len(DIRECT_ACTIONS) != EXPECTED_DIRECT_ACTION_COUNT
    or len(DIRECT_INPUT_ARTIFACT_IDS) != EXPECTED_DIRECT_INPUT_COUNT
    or len(set(DIRECT_INPUT_ARTIFACT_IDS)) != EXPECTED_DIRECT_INPUT_COUNT
):  # pragma: no cover - import-time invariant
    raise RuntimeError("SCALE-BP v2 frozen identity constants are inconsistent.")


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
) + ("GovernanceError",)
