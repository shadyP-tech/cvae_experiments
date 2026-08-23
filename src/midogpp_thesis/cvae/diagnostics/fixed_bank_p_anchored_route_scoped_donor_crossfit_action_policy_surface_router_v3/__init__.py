"""P-DCAPS v3 mechanical nullable-admission repair (planned only)."""

from .admission import (
    NullableStatistic,
    OuterAdmission,
    PseudoPolicyEvidence,
    build_outer_admission,
)
from .config import (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV3Config,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config,
)
from .execution_admission import BLOCKED_MESSAGE, assert_execution_authorized
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .method_controls import (
    AdmissionControlledMethodDecision,
    build_action_only_method_decision,
    build_cyclic_poison_method_decision,
    build_fixed_method_menu,
    build_legacy_method_decision,
    build_policy_only_method_decision,
    build_primary_method_decision,
    build_protected_method_decision,
    compose_method_prediction,
)
from .runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3,
)


__all__ = (
    "AdmissionControlledMethodDecision",
    "BLOCKED_MESSAGE",
    "EXPERIMENT_ID",
    "NullableStatistic",
    "OUTPUT_ARTIFACT_ID",
    "OuterAdmission",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV3Config",
    "PseudoPolicyEvidence",
    "assert_execution_authorized",
    "build_action_only_method_decision",
    "build_cyclic_poison_method_decision",
    "build_fixed_method_menu",
    "build_legacy_method_decision",
    "build_outer_admission",
    "build_policy_only_method_decision",
    "build_primary_method_decision",
    "build_protected_method_decision",
    "compose_method_prediction",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_config",
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3",
)
