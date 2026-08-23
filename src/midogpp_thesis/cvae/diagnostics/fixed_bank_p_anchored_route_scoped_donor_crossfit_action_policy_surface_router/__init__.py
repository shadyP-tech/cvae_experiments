"""P-DCAPS terminal consumed-test diagnostic implementation."""

from .admission import OuterAdmission, PseudoPolicyEvidence, build_outer_admission
from .composition import ComposedCenterPrediction
from .config import (
    PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig,
    load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config,
)
from .engine import (
    OuterActionPolicyResult,
    RouteActionDecision,
    fit_outer_action_policy_surface,
)
from .inventory import (
    CANONICAL_CASE_COUNT,
    CANONICAL_ROW_COUNT,
    ExpectedRouteInventory,
    InventoryCase,
)
from .lifecycle import PDCAPSLabelLifecycle, PreterminalOutputHashes
from .legacy_control import (
    LegacyControlDecision,
    LegacyControlSeal,
    LegacyControlSurface,
    LegacyPseudoReference,
    LegacyTargetPolicyDecision,
    build_legacy_control_decision,
    build_legacy_control_surface,
    seal_legacy_control,
)
from .method_controls import (
    ComposedMethodPrediction,
    MethodControlDecision,
    build_action_only_method_decision,
    build_cyclic_poison_method_decision,
    build_legacy_method_decision,
    build_policy_only_method_decision,
    build_primary_method_decision,
    build_protected_method_decision,
    compose_method_prediction,
)
from .runner import (
    run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router,
)
from .surface_set import (
    CYCLIC_CONTROL_ID,
    IDENTITY_CONTROL_ID,
    SealedActionSurfaceSet,
    seal_action_surface_set,
)
from .routing import (
    AuthorizedOuterPolicy,
    authorize_primary_policy,
    build_admission_from_pseudo_policies,
)


__all__ = (
    "AuthorizedOuterPolicy",
    "ComposedCenterPrediction",
    "ComposedMethodPrediction",
    "CANONICAL_CASE_COUNT",
    "CANONICAL_ROW_COUNT",
    "CYCLIC_CONTROL_ID",
    "ExpectedRouteInventory",
    "InventoryCase",
    "IDENTITY_CONTROL_ID",
    "LegacyControlDecision",
    "LegacyControlSeal",
    "LegacyControlSurface",
    "LegacyPseudoReference",
    "LegacyTargetPolicyDecision",
    "MethodControlDecision",
    "OuterActionPolicyResult",
    "OuterAdmission",
    "PDCAPSLabelLifecycle",
    "PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterConfig",
    "PseudoPolicyEvidence",
    "PreterminalOutputHashes",
    "RouteActionDecision",
    "SealedActionSurfaceSet",
    "authorize_primary_policy",
    "build_admission_from_pseudo_policies",
    "build_action_only_method_decision",
    "build_cyclic_poison_method_decision",
    "build_legacy_control_decision",
    "build_legacy_control_surface",
    "build_legacy_method_decision",
    "build_outer_admission",
    "build_policy_only_method_decision",
    "build_primary_method_decision",
    "build_protected_method_decision",
    "compose_method_prediction",
    "fit_outer_action_policy_surface",
    "load_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_config",
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router",
    "seal_legacy_control",
    "seal_action_surface_set",
)
