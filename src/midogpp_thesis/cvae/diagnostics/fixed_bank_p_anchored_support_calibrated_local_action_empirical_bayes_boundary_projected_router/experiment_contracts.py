"""Exact-six input and inactive execution contracts for SCALE-BP v1."""

from __future__ import annotations

from .identity import EXPERIMENT_ID


CANONICAL_OUTPUT_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router/v1"
)
CANONICAL_SCRATCH_ROOT = (
    "/data/local/fixed_bank_p_anchored_support_calibrated_local_action_"
    "empirical_bayes_boundary_projected_router_v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_support_calibrated_local_action_"
    "empirical_bayes_boundary_projected_router_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_support_calibrated_local_action_"
    "empirical_bayes_boundary_projected_router_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "support_calibrated_local_action_empirical_bayes_boundary_projected_"
    "router_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_support_calibrated_"
    "local_action_empirical_bayes_boundary_projected_router_ledger_"
    "amendment_v1.json"
)
LEDGER_AMENDMENT_SCHEMA_VERSION = (
    "midogpp_test_consumption_ledger_non_authorizing_amendment_v1"
)

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)
INPUT_ROLES = (
    "frozen_source_expert_bank",
    "frozen_generation_lock",
    "registered_label_free_test_cache_alias",
    "registered_label_capability_manifest_alias",
    "registered_parent_consumption_ledger_alias",
    "registered_non_authorizing_ledger_amendment",
)
REGISTERED_CONSUMER_EXPERIMENT_IDS = (EXPERIMENT_ID,)
AUTHORIZED_CONSUMER_EXPERIMENT_IDS: tuple[str, ...] = ()

EXPECTED_BANK_LOCK_HASH = "9972a41dcd4814cd"
EXPECTED_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_TEST_CACHE_SEMANTIC_ID = "uniform_b_v2_descriptive_test_cache_v1"
EXPECTED_TEST_CACHE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
EXPECTED_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
EXPECTED_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
EXPECTED_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)

EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "9ca3843f21b2ca58a20d0671987fa87d61e22efc3c0b920d244ab0443129fc14"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
