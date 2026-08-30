"""Frozen claim and package identities for SCEPTRE v5."""

from __future__ import annotations

from ..fixed_bank_sceptre_router.hashing import (
    canonical_bytes,
    canonical_hash,
    file_sha256,
    require_sha256,
)


PACKAGE_NAME = "fixed_bank_sceptre_router_v5"
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v5"
)
EXPERIMENT_NAME = "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v5"
OUTPUT_ARTIFACT_ID = f"midogpp_output_{EXPERIMENT_NAME}"
CLI_SURFACE = "fixed-bank-sceptre-router-v5"
WORKSPACE_STATUS = "diagnostic"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"

# The user's 2026-08-28 request explicitly authorizes one new terminal
# consumed-test diagnostic.  The code remains insufficient authority by
# itself: exact source-inner and execution-amendment bytes are required at
# admission, and the first durable lease claim permanently consumes the
# attempt whether it completes or fails.
EXECUTION_AUTHORIZED = True
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_2026_08_28_for_sceptre_v5_terminal_"
    "consumed_test_diagnostic_runner"
)
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_sceptre_router_v5_diagnostic"
)
AUTHORIZATION_DATE = "2026-08-28"
EXECUTION_REVISION = "v5_fit_semantics_terminal_consumed_test_diagnostic"

V1_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v1"
)
V2_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v2"
)
V3_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v3"
)
V1_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v1"
)
V2_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v2"
)
V3_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v3"
)
V4_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_sceptre_router.v4"
)
V4_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v4"
)

EXACT_B_ACTION = "B::exact_equal_union"
PROPOSAL_SET_ROLE = "SOURCE_INNER_EXPERT_RANKING_PRIOR_NOT_B_ADVANTAGE"
POLICY_TRANSITION = (
    "G_RANKED_FULL_SET_TO_SUPPORT_SELECTED_MEMBER_TO_SAME_MEMBER_OR_EXACT_B"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
) + ("canonical_bytes", "canonical_hash", "file_sha256", "require_sha256")
