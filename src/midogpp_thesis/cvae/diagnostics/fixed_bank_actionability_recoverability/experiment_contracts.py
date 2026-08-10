"""Frozen workspace identities for the actionability/recoverability diagnostic."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "actionability_recoverability.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "actionability_recoverability_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
CLAIM_ROLE = "posthoc_actionability_recoverability_mechanism_diagnostic"
ROUTING_STATUS = "TERMINAL_ACTIONABILITY_RECOVERABILITY_DIAGNOSTIC_ONLY"
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_actionability_recoverability_diagnostic_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_actionability_recoverability_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_actionability_recoverability_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "actionability_recoverability_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "actionability_recoverability_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability_"
    "ledger_amendment_v1.json"
)

# The complete scientific dependency surface is exactly these six aliases.
# The cache is a label-free alias only. No prior Stage-90 or numbered-stage
# result, score, prediction, policy, checkpoint, or scratch product is admitted.
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
    "frozen_policy_downstream",
    "consumed_validation",
    "midogpp_output_uniform_b_v2_consumed_test",
    "fixed_bank_decision_audit",
    "fixed_bank_label_aware_case_oof_ceiling",
    "fixed_bank_pooled_bacc_case_oof_ceiling",
    "fixed_bank_hierarchical_residual_stacker",
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
OOF_FOLD_SEED = 90_902_029
OOF_PARTITION_NAMESPACE = (
    "midogpp_fixed_bank_actionability_recoverability_test_folds_v1"
)
EXPECTED_CENTER_FOLD_COUNT = len(CENTERS) * OOF_FOLD_COUNT
EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET = len(CENTERS) - 1

GEOMETRY_IDS = ("A0", "A1")
PRE_EVALUATION_METHOD_IDS = ("B", "U", "G", "R", "P", "S_y")
PER_GEOMETRY_METHOD_IDS = ("U", "G", "R", "P", "S_y")
TERMINAL_ORACLE_IDS = ("O_static", "O_case")

BASE_ROWS_PER_SOURCE_CLASS = 128
SELECTED_ROWS_PER_CLASS = 256
OTHER_ROWS_PER_CLASS = 128
UNIFORM_ROWS_PER_SOURCE_CLASS = 144
SOURCE_PREFIX_ROWS_PER_CLASS = 270
A0_SELECTED_ROW_WEIGHT = 1.0
A0_OTHER_ROW_WEIGHT = 1.0
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0
A0_PHYSICAL_ROWS_PER_CLASS = (
    SELECTED_ROWS_PER_CLASS
    + (EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET - 1) * OTHER_ROWS_PER_CLASS
)
A1_PHYSICAL_ROWS_PER_CLASS = A0_PHYSICAL_ROWS_PER_CLASS
A0_EFFECTIVE_ROWS_PER_CLASS = 1_152
A1_EFFECTIVE_ROWS_PER_CLASS = 1_152
UNIFORM_ROWS_PER_CLASS = (
    EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET * UNIFORM_ROWS_PER_SOURCE_CLASS
)
BASE_ROWS_PER_CLASS = (
    EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET * BASE_ROWS_PER_SOURCE_CLASS
)

# B + one physically shared U + eight source actions in each of A0 and A1.
EXPECTED_PHYSICAL_ACTION_COUNT_PER_TARGET = (
    2 + len(GEOMETRY_IDS) * EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
)
# The shared U has one logical identity in each geometry.
EXPECTED_LOGICAL_ACTION_COUNT_PER_TARGET = (
    1
    + len(GEOMETRY_IDS)
    * (1 + EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET)
)
EXPECTED_TARGET_PHYSICAL_ACTION_IDENTITY_COUNT = (
    len(CENTERS) * EXPECTED_PHYSICAL_ACTION_COUNT_PER_TARGET
)
EXPECTED_TARGET_LOGICAL_ACTION_IDENTITY_COUNT = (
    len(CENTERS) * EXPECTED_LOGICAL_ACTION_COUNT_PER_TARGET
)
EXPECTED_TARGET_PROBABILITY_CELL_COUNT = (
    EXPECTED_TARGET_PHYSICAL_ACTION_IDENTITY_COUNT * SEED_PAIR_COUNT
)
EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT = EXPECTED_TARGET_PROBABILITY_CELL_COUNT

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 90_912_028

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "66a345f6ee31717fcc7eb9fdba76f76e52bb032de987eabbf55b7ffdf5a66f09"
)
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
