"""Frozen workspace identities for the terminal pooled-BACC case-OOF ceiling."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "pooled_bacc_case_oof_ceiling.v2"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling_v2"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "pooled_bacc_case_oof_ceiling_v2"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
ROUTING_STATUS = "TERMINAL_POOLED_BACC_CASE_OOF_CEILING_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_pooled_bacc_case_oof_ceiling_test_cache_v2"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_pooled_bacc_case_oof_ceiling_test_manifest_v2"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_pooled_bacc_"
    "case_oof_ceiling_parent_v2"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_pooled_bacc_"
    "case_oof_ceiling_amendment_v2"
)

# The input fence is intentionally closed: neither the failed v1 output nor any
# Stage-50/60 result, Stage-70 prediction/scoring/policy result, or other
# Stage-90 result is admissible. The declared label-free cache alias retains its
# Stage-70-derived cache lineage without importing a Stage-70 result.
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

QUARANTINED_V1_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "label_aware_case_oof_ceiling.v1"
)
QUARANTINED_V1_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "label_aware_case_oof_ceiling_v1"
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
OOF_FOLD_SEED = 90_902_026
OOF_PARTITION_NAMESPACE = (
    "midogpp_fixed_bank_pooled_bacc_case_oof_ceiling_test_folds_v2"
)
EXPECTED_CENTER_FOLD_COUNT = len(CENTERS) * OOF_FOLD_COUNT
EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET = len(CENTERS) - 1
EXPECTED_ACTION_COUNT_PER_TARGET = 1 + EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
EXPECTED_TARGET_ACTION_IDENTITY_COUNT = len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET
EXPECTED_TARGET_PROBABILITY_CELL_COUNT = (
    EXPECTED_TARGET_ACTION_IDENTITY_COUNT * SEED_PAIR_COUNT
)
EXPECTED_LOCO_DONOR_COUNT_PER_CANDIDATE = len(CENTERS) - 2
EXPECTED_PAIRWISE_DONOR_COUNT_WHEN_G_IS_BASELINE = len(CENTERS) - 2
EXPECTED_PAIRWISE_DONOR_COUNT_WHEN_G_IS_SOURCE = len(CENTERS) - 3
EXPECTED_PAIRWISE_ALTERNATIVE_COUNT_WHEN_G_IS_BASELINE = len(CENTERS) - 1
EXPECTED_PAIRWISE_ALTERNATIVE_COUNT_WHEN_G_IS_SOURCE = len(CENTERS) - 2
PERMUTATION_COUNT = 10_000
EXPECTED_NULL_ACTION_COUNT = EXPECTED_CENTER_FOLD_COUNT * PERMUTATION_COUNT

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "4fa46736c08641d5df6cde1cdd0acb10ef09a4b18adf7b44c872f5ec651288da"
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
