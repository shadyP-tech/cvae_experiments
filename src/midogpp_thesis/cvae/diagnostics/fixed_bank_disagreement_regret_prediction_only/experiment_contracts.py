"""Frozen workspace identities for the prediction-only disagreement router."""

from __future__ import annotations


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "disagreement_regret_prediction_only.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_"
    "disagreement_regret_prediction_only_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "disagreement_regret_prediction_only_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
CLAIM_ROLE = (
    "posthoc_source_oof_trained_full_consumed_test_prediction_only_diagnostic"
)
ROUTING_STATUS = "UNSCORED_DIAGNOSTIC_ROUTE_SUGGESTIONS_ONLY"
AUTHORIZATION_SCOPE = (
    "one_posthoc_source_oof_trained_full_consumed_test_"
    "prediction_only_diagnostic_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TRAIN_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_disagreement_regret_prediction_only_"
    "train_cache_v1"
)
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_disagreement_regret_prediction_only_"
    "test_cache_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "disagreement_regret_prediction_only_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_"
    "disagreement_regret_prediction_only_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_disagreement_regret_"
    "prediction_only_ledger_amendment_v1.json"
)

# No numbered-stage result or previous Stage-90 output is admitted.  The two
# promoted frozen assets are identities/weights only; the remaining inputs are
# single-consumer aliases owned by this diagnostic.
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TRAIN_CACHE_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
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
    "fixed_bank_signed_error_gate",
    "fixed_bank_hierarchical_residual_stacker",
    "fixed_bank_decision_audit",
    "case_aware_proxy",
    "utility_aligned_",
    "residual_topup",
    "historical",
    "quarantine",
    "/scratch/",
    "/checkpoints/",
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTER = "4"
SOURCE_SPLIT = "train"
TARGET_SPLIT = "test"
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
GEOMETRY_IDS = ("A0", "A1")
MODEL_FAMILY_IDS = ("G", "R", "P")
SURFACE_IDS = ("R_raw", "R_safe")

EXPECTED_TRAIN_ROW_COUNT = 9_648
EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_TRAIN_FEATURE_DIM = 3_840
EXPECTED_TEST_FEATURE_DIM = 3_840

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_TRAIN_CACHE_SHA256 = (
    "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
)
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
# Filled from the committed, canonical direct amendment.  Updating the
# amendment requires an explicit version bump and recomputation of this value.
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "a1a254bcfd6e5a6643e9a5e25b6f81f7b4fe38270efc47443415f066191c4533"
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
