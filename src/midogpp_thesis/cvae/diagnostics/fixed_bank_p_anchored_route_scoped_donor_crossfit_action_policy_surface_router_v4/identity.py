"""Fresh one-shot executable identity for P-DCAPS v4.

V4 executes the source-sealed v3 nullable-admission repair without reusing any
v1, v2, or v3 diagnostic output or execution state.  The consumed MIDOG++ test
split remains terminal diagnostic evidence only.
"""

from __future__ import annotations

from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_FAMILIES,
    ACTION_ONLY_METHOD_ID,
    ACTION_STRATA,
    CYCLIC_METHOD_ID,
    DIRECTIONS,
    DIRECT_INPUT_ROLES,
    LEGACY_METHOD_ID,
    METHOD_MENU,
    METRICS,
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


PACKAGE_NAME = (
    "fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_v4"
)
EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v4"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v4"
)
EXPERIMENT_NAME = (
    "P-anchored donor-cross-fitted action-and-policy-surface router v4 "
    "executable nullable-admission repair"
)
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_2026_08_23_for_pdcaps_v4_terminal_"
    "consumed_test_diagnostic_run"
)
AUTHORIZATION_SCOPE = (
    "one_terminal_consumed_test_fixed_bank_p_anchored_route_scoped_"
    "donor_crossfit_action_policy_surface_router_v4_diagnostic"
)
EXECUTION_REVISION = "v4_terminal_consumed_test_diagnostic"

V2_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v2"
)
V2_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v2"
)
V3_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router.v3"
)
V3_OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_p_anchored_"
    "route_scoped_donor_crossfit_action_policy_surface_router_v3"
)


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
) + ("canonical_hash", "require_sha256")
