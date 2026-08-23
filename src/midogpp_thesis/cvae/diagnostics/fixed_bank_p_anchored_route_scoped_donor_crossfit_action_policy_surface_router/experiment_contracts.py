"""Frozen direct-input and workspace identities for P-DCAPS."""

from __future__ import annotations

from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID


CANONICAL_OUTPUT_ROOT = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router/v1"
)

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
TEST_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_test_cache_v1"
)
TEST_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_fixed_bank_p_anchored_route_scoped_donor_crossfit_"
    "action_policy_surface_router_test_manifest_v1"
)
TEST_CONSUMPTION_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_parent_v1"
)
LEDGER_AMENDMENT_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_amendment_v1"
)
LEDGER_AMENDMENT_FILENAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_ledger_amendment_v1.json"
)

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    LEDGER_AMENDMENT_ARTIFACT_ID,
)

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

# Filled by the immutable amendment file and checked by config loading.  This
# is an integrity identity only; the amendment itself keeps execution false.
EXPECTED_LEDGER_AMENDMENT_SHA256 = (
    "e3cd6533b875f2fd4740ce5904bdf11f64b72c38604466d1e244fdd79a49f38c"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
)
