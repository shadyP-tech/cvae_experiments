"""Frozen identities and topology for the terminal multi-challenger diagnostic."""

from __future__ import annotations


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "multi_challenger_hierarchical_flip_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_multi_challenger_"
    "hierarchical_flip_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "multi_challenger_hierarchical_flip_router_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
TERMINAL_DECISION = "DO_NOT_PROMOTE"
CLAIM_ROLE = (
    "posthoc_known_fixed_bank_multi_challenger_hierarchical_"
    "flip_router_diagnostic"
)
ROUTING_STATUS = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_multi_challenger_"
    "hierarchical_flip_router_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_multi_challenger_hierarchical_"
    "flip_router_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_multi_challenger_hierarchical_"
    "flip_router_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_multi_challenger_"
    "hierarchical_flip_router_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_multi_challenger_"
    "hierarchical_flip_router_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_multi_challenger_hierarchical_"
    "flip_router_ledger_amendment_v1.json"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
FEATURE_NAMES = (
    "flip_0_to_1_count",
    "flip_0_to_1_rate",
    "flip_1_to_0_count",
    "flip_1_to_0_rate",
    "zero_flip",
    "baseline_abs_margin_on_flip",
    "candidate_abs_margin_on_flip",
    "signed_probability_delta_on_flip",
    "seed_flip_robustness",
    "candidate_seed_disagreement_on_flip",
    "case_size",
)
METHOD_IDS = (
    "B",
    "U",
    "S_static",
    "F_single",
    "G_multi",
    "R_multi",
    "P_multi",
    "O_menu",
    "O_binary",
    "O_static",
    "O_case",
)
PRE_EVALUATION_METHOD_IDS = METHOD_IDS[:7]
TERMINAL_ORACLE_IDS = METHOD_IDS[7:]
PRIMARY_METHOD_ID = "R_multi"

OOF_FOLD_COUNT = 5
OOF_FOLD_SEED = 90_902_026
OOF_PARTITION_NAMESPACE = (
    "midogpp_fixed_bank_multi_challenger_hierarchical_flip_router_"
    "test_folds_v1"
)
TOP_K = 3
SUPPORT_PRIOR_CASES = 8.0
FEATURE_ALPHA = 1.0
SOURCE_ALPHA = 4.0
QUERY_ALPHA = 4.0
INTERCEPT_ALPHA = 0.25
CALIBRATION_ALPHA = 4.0
MARGIN_Z = 1.96
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 90_912_030

ACTION_COUNT_PER_TARGET = 10
TARGET_TASK_COUNT = 81
TARGET_PROBABILITY_CELL_COUNT = 810
SOURCE_PREFIX_ROWS_PER_CLASS = 270
SCRATCH_ROOT = (
    "/data/local/fixed_bank_multi_challenger_hierarchical_flip_router_v1"
)
WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "ba87cbf15ec4cbf85b2d93d41304dd9ed876d6ad0af87f5ba3d45b506deacc04"
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


__all__ = tuple(name for name in globals() if name.isupper())
