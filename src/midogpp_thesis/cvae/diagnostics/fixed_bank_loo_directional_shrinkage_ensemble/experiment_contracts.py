"""Frozen workspace identities for the directional-shrinkage diagnostic."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .constants import (
    ACTION_COUNT_PER_TARGET,
    ARM_IDS,
    CENTERS,
    CLAIM_ROLE,
    DATASET_FAMILY,
    DIRECTION_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    HARD_THRESHOLD,
    K_GRID,
    METHOD_IDS,
    NULL_REPLICATES,
    NULL_SEED,
    PRE_TERMINAL_METHOD_IDS,
    PUBLICATION_STATUS,
    TARGET_PROBABILITY_CELL_COUNT,
    TARGET_TASK_COUNT,
    TERMINAL_DECISION,
    TERMINAL_ORACLE_IDS,
    TIE_TOLERANCE,
    W_GRID,
    W_RATIONAL_GRID,
)


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_v1"
)
CANONICAL_OUTPUT_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble/v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
ROUTING_STATUS = TERMINAL_DECISION
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_loo_directional_shrinkage_"
    "ensemble_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_loo_directional_shrinkage_"
    "ensemble_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "loo_directional_shrinkage_ensemble_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "loo_directional_shrinkage_ensemble_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_directional_shrinkage_ensemble_ledger_amendment_v1.json"
)

# Complete and ordered dependency surface.  In particular, there is no prior
# Stage-90 output, amendment, prediction surface, checkpoint, or scratch root.
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
    "fixed_bank_actionability_recoverability",
    "fixed_bank_decision_audit",
    "fixed_bank_disagreement_regret_prediction_only",
    "fixed_bank_hierarchical_residual_stacker",
    "fixed_bank_label_aware_case_oof_ceiling",
    "fixed_bank_labeled_support_case_conditional_flip_router",
    "fixed_bank_multi_challenger_hierarchical_flip_router",
    "fixed_bank_pooled_bacc_case_oof_ceiling",
    "fixed_bank_signed_error_gate",
    "fixed_bank_support_static_router_s4",
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

EXCLUDED_CENTER = "4"
EVALUATION_SPLIT = "test"
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
SEED_PAIR_COUNT = 9
EXPECTED_MIXED_CLASS_CASE_COUNT = 213
EXPECTED_NEGATIVE_ONLY_CASE_COUNT = 4
EXPECTED_POSITIVE_ONLY_CASE_COUNT = 1
EXPECTED_CASE_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    dict(EXPECTED_CASE_COUNTS_BY_CENTER)
)

BASE_ROWS_PER_SOURCE_CLASS = 128
UNIFORM_ROWS_PER_SOURCE_CLASS = 144
SOURCE_PREFIX_ROWS_PER_CLASS = 270
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0
A1_EFFECTIVE_ROWS_PER_CLASS = 1_152

SCRATCH_ROOT = (
    "/data/local/fixed_bank_loo_directional_shrinkage_ensemble_v1"
)
WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "05f800f1bd053528477abd1e67163612c01d44f56418f98961bcdf64677bdc52"
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
