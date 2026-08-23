"""Fresh executable identity for the authorized P-DCAPS v2 diagnostic."""

from __future__ import annotations

from ..identity import (
    ACTION_FAMILIES,
    ACTION_ONLY_METHOD_ID,
    ACTION_STRATA,
    CYCLIC_METHOD_ID,
    DIRECTIONS,
    DIRECT_INPUT_ROLES,
    LEGACY_METHOD_ID,
    METHOD_MENU,
    METRICS,
    PACKAGE_NAME,
    POLICY_ONLY_METHOD_ID,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    P_METHOD_ID,
    RIDGE_ALPHA,
    TERMINAL_DECISION,
    TIE_TOLERANCE,
    canonical_hash,
    require_sha256,
)


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v2"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v2"
)
EXPERIMENT_NAME = (
    "P-anchored donor-cross-fitted action-and-policy-surface router v2"
)
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_for_pdcaps_v2_terminal_consumed_test_"
    "diagnostic_run"
)
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_v2_diagnostic"
)
EXECUTION_REVISION = "v2_terminal_consumed_test_diagnostic"


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
) + ("canonical_hash", "require_sha256")
