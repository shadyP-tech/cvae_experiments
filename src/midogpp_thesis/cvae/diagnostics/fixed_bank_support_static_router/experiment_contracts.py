"""Frozen identities and topology for the support-static S4 diagnostic."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "support_static_router_s4.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "support_static_router_s4_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_ROLE = "posthoc_known_fixed_bank_support_static_router_s4_diagnostic"
ROUTING_STATUS = TERMINAL_DECISION
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_support_static_router_s4_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_support_static_router_s4_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_support_static_router_s4_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "support_static_router_s4_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "support_static_router_s4_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4_"
    "ledger_amendment_v1.json"
)

# The complete dependency surface is exactly these six ordered artifacts.
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

FORBIDDEN_INPUT_FRAGMENTS = (
    "50_all_candidate_utility_matrix",
    "60_routing_and_composition",
    "70_frozen_policy_downstream",
    "consumed_validation",
    "midogpp_output_uniform_b_v2_consumed_test",
    "fixed_bank_actionability_recoverability",
    "fixed_bank_decision_audit",
    "fixed_bank_disagreement_regret_prediction_only",
    "fixed_bank_hierarchical_residual_stacker",
    "fixed_bank_label_aware_case_oof_ceiling",
    "fixed_bank_labeled_support_case_conditional_flip_router",
    "fixed_bank_multi_challenger_hierarchical_flip_router",
    "fixed_bank_pooled_bacc_case_oof_ceiling",
    "fixed_bank_signed_error_gate",
    "utility_aligned_",
    "residual_topup",
    "case_aware_proxy",
    "ensemble_endpoint_proxy",
    "exact_tail_router",
    "historical",
    "quarantine",
    "/scratch/",
    "/checkpoints/",
)
FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS = (
    "stage50",
    "stage60",
    "stage70",
    "/50_",
    "/60_",
    "/70_",
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
MIDOGPP_CENTERS = CENTERS
EXCLUDED_CENTER = "4"
EVALUATION_SPLIT = "test"
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
SEED_PAIR_COUNT = 9

EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_TOTAL_CASE_COUNT = 218
EXPECTED_MIXED_CLASS_CASE_COUNT = 213
EXPECTED_NEGATIVE_ONLY_CASE_COUNT = 4
EXPECTED_POSITIVE_ONLY_CASE_COUNT = 1
EXPECTED_CASE_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    {
        "0": 23,
        "1": 20,
        "2": 24,
        "3": 39,
        "5": 23,
        "6": 23,
        "7": 21,
        "8": 22,
        "9": 23,
    }
)

OOF_FOLD_COUNT = 5
OOF_FOLD_SEED = 90_902_026
PARTITION_SEED = OOF_FOLD_SEED
OOF_PARTITION_NAMESPACE = (
    "midogpp_fixed_bank_support_static_router_s4_test_folds_v1"
)
PARTITION_NAMESPACE = OOF_PARTITION_NAMESPACE
EXPECTED_CENTER_FOLD_COUNT = len(CENTERS) * OOF_FOLD_COUNT
EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET = len(CENTERS) - 1

B_ACTION_ID = "B"
U_ACTION_ID = "U"
ACTION_COUNT_PER_TARGET = 10
TARGET_TASK_COUNT = len(CENTERS) * SEED_PAIR_COUNT
TARGET_PROBABILITY_CELL_COUNT = TARGET_TASK_COUNT * ACTION_COUNT_PER_TARGET
METHOD_IDS = ("B", "U", "G_static", "S4", "O_static", "O_case")
PRE_EVALUATION_METHOD_IDS = METHOD_IDS[:4]
TERMINAL_ORACLE_IDS = METHOD_IDS[4:]
HARD_THRESHOLD = 0.5
TIE_TOLERANCE = 1.0e-12

BASE_ROWS_PER_SOURCE_CLASS = 128
UNIFORM_ROWS_PER_SOURCE_CLASS = 144
SOURCE_PREFIX_ROWS_PER_CLASS = 270
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0
A1_EFFECTIVE_ROWS_PER_CLASS = 1_152

PERMUTATION_COUNT = 10_000
PERMUTATION_SEED = 90_912_026
NULL_DERANGEMENT_ALGORITHM = (
    "case_sha256_candidate_order_counter_splitmix64_nonzero_"
    "cyclic_shift_1_to_7_v1"
)
T_INTERVAL_CONFIDENCE_LEVEL = 0.95
T_INTERVAL_DEGREES_OF_FREEDOM = 8

SCRATCH_ROOT = "/data/local/fixed_bank_support_static_router_s4_v1"
WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "b1a97bd2c64f48075c07e9ba29fc5fd9c1679c16bdf51b4d7c78dcd9509aa11f"
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


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
