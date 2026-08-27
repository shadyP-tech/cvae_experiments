"""Stable SCALE-BP v1 identities and frozen scientific constants.

SCALE-BP v1 is an implementation-complete planning identity.  It does not
authorize another opening of the already-consumed MIDOG++ test labels.
"""

from __future__ import annotations


PACKAGE_NAME = (
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router"
)
STEM = PACKAGE_NAME
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router.v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router_v1"
)
EXPERIMENT_NAME = (
    "P-anchored support-calibrated local-action empirical-Bayes "
    "boundary-projected router v1"
)
CLI_SURFACE = (
    "fixed-bank-p-anchored-support-calibrated-local-action-empirical-bayes-"
    "boundary-projected-router"
)

PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
WORKSPACE_STATUS = "planned"
CLAIM_SCOPE = "diagnostic_only"

ACTION_FAMILIES = ("B", "I", "R")
DIRECTIONS = ("zero_to_one", "one_to_zero")
ACTION_IDS = tuple(
    f"{family}::{direction}"
    for family in ACTION_FAMILIES
    for direction in DIRECTIONS
)
# Compatibility name retained for persisted/test-facing action rectangles.
CELL_IDS = ACTION_IDS
METRICS = ("bacc", "brier", "log")
CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
PHYSICAL_CELL_COUNT = 810
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
EXPECTED_TEST_ROW_COUNT = 9928
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
SUPPORT_FOLD_COUNT = 4
RIDGE_ALPHA = 1.0
TIE_TOLERANCE = 1.0e-12
MINIMUM_OPPORTUNITY_CASES = 24
MINIMUM_REPRESENTED_CENTERS = 6
MINIMUM_WITHIN_CASE_SPEARMAN = 0.0
MAXIMUM_NORMALIZED_ORACLE_GAP = 1.0
MAXIMUM_HARMFUL_SELECTED_POLICY_COUNT = 0

P_METHOD_ID = "P_PROTECTED"
PRIMARY_METHOD_ID = "SCALE_BP_PRIMARY"
DONOR_ONLY_METHOD_ID = "SCALE_BP_DONOR_ONLY"
LOCAL_ONLY_METHOD_ID = "SCALE_BP_LOCAL_ONLY"
LEGACY_METHOD_ID = "LEGACY_SAME_RUN"
SUPPORT_PERMUTATION_METHOD_ID = "SCALE_BP_SUPPORT_LABEL_PERMUTATION"
CYCLIC_METHOD_ID = "SCALE_BP_CYCLIC_ACTION_IDENTITY_POISON"
FULL_ENDPOINT_METHOD_ID = "SCALE_BP_FULL_ENDPOINT_SENSITIVITY"
METHOD_MENU = (
    P_METHOD_ID,
    PRIMARY_METHOD_ID,
    DONOR_ONLY_METHOD_ID,
    LOCAL_ONLY_METHOD_ID,
    LEGACY_METHOD_ID,
    SUPPORT_PERMUTATION_METHOD_ID,
    CYCLIC_METHOD_ID,
    FULL_ENDPOINT_METHOD_ID,
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
