"""Frozen workspace identities for the terminal residual-stacker diagnostic."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "hierarchical_residual_stacker.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "hierarchical_residual_stacker_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
ROUTING_STATUS = "TERMINAL_HIERARCHICAL_RESIDUAL_STACKER_ONLY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_hierarchical_residual_stacker_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_hierarchical_residual_stacker_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_hierarchical_"
    "residual_stacker_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_hierarchical_"
    "residual_stacker_amendment_v1"
)

# This exact six-input fence is the complete scientific dependency surface.
# In particular it contains no metadata artifact, no Stage-50/60 result, no
# Stage-70 prediction/scoring/policy result, and no earlier Stage-90 artifact.
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

# Explicitly name every Stage-90 artifact present when v1 was frozen.  Runtime
# fence code may also reject by catalog stage; this list makes the pre-existing
# denylist auditable without importing the workspace registry at runtime.
FORBIDDEN_PRIOR_STAGE90_ARTIFACT_IDS = (
    "midogpp_virchow2_cache_integrity_audit",
    "midogpp_repository_migration_20260712_xai_master",
    "midogpp_dense_late_all_sources_v1",
    "midogpp_source_inner_class_conditional_positive_union_v1",
    "midogpp_phase2_target_support_adaptation_seed42",
    "midogpp_physical_multiscale_center_pooling_v1_failed_geometry_audit",
    "midogpp_physical_multiscale_annotation_local_pooling_v2_failed_geometry_audit",
    "midogpp_physical_multiscale_clipped_bbox_annotation_local_pooling_v3_geometry_audit",
    "midogpp_stage90_uniform_b_paired_reparameterization_snapshot_v1",
    "midogpp_output_uniform_b_paired_reparameterization_audit_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_dense_residual_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_local_marginal_utility_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_mmd_kmm_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_conditional_contrast_mmd_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_residual_topup_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_router_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_proxy_information_audit_v1",
    "midogpp_output_uniform_b_v2_consumed_test_utility_aligned_case_aware_proxy_information_audit_v1",
    "midogpp_output_uniform_b_v2_consumed_validation_utility_aligned_exact_tail_router_v1",
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_decision_audit_v1",
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling_v1",
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling_v2",
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
OOF_FOLD_SEED = 90_902_027
OOF_PARTITION_NAMESPACE = (
    "midogpp_fixed_bank_hierarchical_residual_stacker_test_folds_v1"
)
EXPECTED_CENTER_FOLD_COUNT = len(CENTERS) * OOF_FOLD_COUNT
EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET = len(CENTERS) - 1
EXPECTED_ACTION_COUNT_PER_TARGET = 1 + EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
EXPECTED_TARGET_ACTION_IDENTITY_COUNT = len(CENTERS) * EXPECTED_ACTION_COUNT_PER_TARGET
EXPECTED_TARGET_PROBABILITY_CELL_COUNT = (
    EXPECTED_TARGET_ACTION_IDENTITY_COUNT * SEED_PAIR_COUNT
)
EXPECTED_TARGET_CASE_ACTION_FEATURE_COUNT = (
    EXPECTED_TOTAL_CASE_COUNT * EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
)
EXPECTED_OUTER_CANDIDATE_MODEL_COUNT = (
    len(CENTERS) * EXPECTED_CANDIDATE_SOURCE_COUNT_PER_TARGET
)
EXPECTED_DONOR_CASE_ACTION_COUNT = (
    EXPECTED_TOTAL_CASE_COUNT * (len(CENTERS) - 1) * (len(CENTERS) - 2)
)
EXPECTED_DONOR_CLASS_RESPONSE_COUNT = (
    2 * EXPECTED_DONOR_CASE_ACTION_COUNT
    - (EXPECTED_NEGATIVE_ONLY_CASE_COUNT + EXPECTED_POSITIVE_ONLY_CASE_COUNT)
    * (len(CENTERS) - 1)
    * (len(CENTERS) - 2)
)

EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "e915134fc15901f1d5c43fb5fb974f1693282ca4622a2ade169eaa7487566b1b"
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
