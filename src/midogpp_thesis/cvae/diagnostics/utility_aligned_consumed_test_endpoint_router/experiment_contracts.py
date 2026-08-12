"""Frozen workspace identities for the consumed-test endpoint router."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_"
    "target_static_endpoint_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_utility_aligned_"
    "target_static_endpoint_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_utility_aligned_"
    "target_static_endpoint_router_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
CLAIM_ROLE = "posthoc_utility_aligned_target_static_endpoint_router_diagnostic"
ROUTING_STATUS = "TERMINAL_TARGET_STATIC_ENDPOINT_ROUTER_DIAGNOSTIC_ONLY"
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_utility_aligned_target_static_endpoint_router_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_utility_aligned_target_static_endpoint_router_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_utility_aligned_target_static_endpoint_router_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_utility_aligned_"
    "target_static_endpoint_router_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_utility_aligned_"
    "target_static_endpoint_router_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router_"
    "ledger_amendment_v1.json"
)

# This is the complete scientific dependency surface. In particular, the
# metadata control is reconstructed from the experiment-fenced manifest rather
# than admitted as a seventh artifact.
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
    "fixed_bank_actionability_recoverability",
    "utility_aligned_case_aware_proxy_information_audit",
    "utility_aligned_ensemble_endpoint_router",
    "utility_aligned_ensemble_endpoint_proxy_information_audit",
    "utility_aligned_exact_tail_router",
    "residual_topup",
    "midogpp_routing_metadata_profiles_v1",
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
EXPECTED_CASE_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    {"0": 23, "1": 20, "2": 24, "3": 39, "5": 23,
     "6": 23, "7": 21, "8": 22, "9": 23}
)
SUPPORT_CASE_COUNT_PER_CENTER = 8
EXPECTED_SUPPORT_CASE_COUNT = 72
EXPECTED_EVALUATION_CASE_COUNT = 146
EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    {"0": 15, "1": 12, "2": 16, "3": 31, "5": 15,
     "6": 15, "7": 13, "8": 14, "9": 15}
)
EXPECTED_SUPPORT_ROW_COUNT = 2_902
EXPECTED_EVALUATION_ROW_COUNT = 7_026
EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    {"0": 874, "1": 512, "2": 1_938, "3": 1_188, "5": 458,
     "6": 633, "7": 210, "8": 633, "9": 580}
)
SUPPORT_PARTITION_NAMESPACE = (
    "midogpp_utility_aligned_consumed_test_endpoint_router_support_v1"
)

DEVELOPMENT_RESPONSE_COUNT = 504
DESCRIPTIVE_DEVELOPMENT_SEED_ROW_COUNT = 4_536
EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET = len(CENTERS) - 1
ACTION_IDS = ("B", "U", "G", "R", "P", "Hxe")
PRIMARY_CONTRASTS = ("R-B", "R-U", "R-G", "R-P")
SUPPORT_BOOTSTRAP_REPLICATES = 32
SUPPORT_BOOTSTRAP_SEED = 90_703

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "600f1ec3618e0d55e09ed30213bbf0ad4776c099ea23c5b44efcedca50c82505"
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
