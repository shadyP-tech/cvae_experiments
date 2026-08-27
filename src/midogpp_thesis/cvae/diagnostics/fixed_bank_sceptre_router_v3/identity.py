"""Frozen identity for the executable, single-use SCEPTRE v3 diagnostic."""

from __future__ import annotations

from ..fixed_bank_sceptre_router.hashing import (
    canonical_bytes,
    canonical_hash,
    file_sha256,
    require_sha256,
)


PACKAGE_NAME = "fixed_bank_sceptre_router_v3"
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v3"
)
EXPERIMENT_NAME = "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v3"
OUTPUT_ARTIFACT_ID = f"midogpp_output_{EXPERIMENT_NAME}"
CLI_SURFACE = "fixed-bank-sceptre-router-v3"
WORKSPACE_STATUS = "diagnostic"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
AUTHORIZATION_DATE = "2026-08-28"
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_2026_08_28_for_one_sceptre_v3_terminal_"
    "consumed_test_runtime_repair_diagnostic"
)
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_sceptre_router_v3_runtime_repair_"
    "diagnostic"
)
EXECUTION_REVISION = "v3_terminal_consumed_test_runtime_repair"

V1_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v1"
)
V1_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v1"
)
V2_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v2"
)
V2_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v2"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
) + ("canonical_bytes", "canonical_hash", "file_sha256", "require_sha256")
