"""Stable identity for the planned OE-PPUR v1 terminal diagnostic.

This module contains names and closed-world constants only.  In particular,
importing it cannot grant label access or authorize execution.
"""

from __future__ import annotations


PACKAGE_NAME = (
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_"
    "router_v1"
)
STEM = PACKAGE_NAME
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router.v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "opportunity_equivalence_pairwise_primitive_utility_router_v1"
)
EXPERIMENT_NAME = (
    "P-anchored opportunity-equivalence pairwise primitive-utility router v1"
)
CLI_SURFACE = (
    "fixed-bank-p-anchored-opportunity-equivalence-pairwise-primitive-"
    "utility-router-v1"
)

PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
WORKSPACE_STATUS = "planned"
CLAIM_SCOPE = "diagnostic_only"
FRESH_EVIDENCE = False
EXECUTION_AUTHORIZED = False

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
METRICS = ("bacc", "brier", "log")
PRIMITIVES = ("delta_tp", "delta_tn", "delta_brier_sum", "delta_log_sum")
EXPECTED_TEST_ROW_COUNT = 9928
EXPECTED_CASE_COUNT = 218

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
ANNOTATION_MANIFEST_ARTIFACT_ID = "midogpp_dataset_contract_annotation_patch_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    ANNOTATION_MANIFEST_ARTIFACT_ID,
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
