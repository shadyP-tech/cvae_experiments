"""HARP v16: target-support adapted hierarchical action-risk routing.

This package is intentionally independent of the sealed v13/v14 scientific
router.  It learns only from labeled MIDOG++ train-support cases for the same
center and exposes a label-free full-test routing interface.
"""

from .certificates import (
    ActionRiskCertificate,
    MenuRiskCalibration,
    certify_case_prediction,
    fit_menu_risk_calibration,
)
from .contracts import (
    ActionFamily,
    CasePrediction,
    Direction,
    EndpointPrediction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    canonical_probability_hex,
    float32_probability_hex,
)
from .crossfit import (
    SupportCrossfitResult,
    SupportOOFCasePrediction,
    SupportOOFRecord,
    leave_one_case_out_crossfit,
    validate_support_inventory,
)
from .effective_menu import build_effective_menu, group_effective_menus
from .features import (
    FittedFeatureMap,
    MECHANISM_FEATURE_PRIORITY,
    case_balanced_weights,
    fit_feature_map,
)
from .hierarchical import (
    NullSupportEndpointModel,
    RidgeHead,
    SupportEndpointModel,
    fit_support_endpoint_model,
)
from .outcome_normalization import (
    SupportFoldNormalizer,
    fit_support_fold_normalizer,
    fold_class_support_counts,
    normalize_action_outcomes,
)
from .policy import (
    FittedSupportRouter,
    HierarchyTrace,
    PolicyAdmission,
    RouteDecision,
    evaluate_support_policy_admission,
    fit_support_router,
    select_hierarchical_certificate,
)

__all__ = (
    "ActionFamily",
    "ActionRiskCertificate",
    "CasePrediction",
    "Direction",
    "EndpointPrediction",
    "FittedFeatureMap",
    "FittedSupportRouter",
    "HierarchyTrace",
    "LabelFreeAction",
    "LabelFreeCaseMenu",
    "MECHANISM_FEATURE_PRIORITY",
    "MenuRiskCalibration",
    "NullSupportEndpointModel",
    "PolicyAdmission",
    "RidgeHead",
    "RouteDecision",
    "RouterFitConfig",
    "SupportActionOutcome",
    "SupportCaseClassProfile",
    "SupportCrossfitResult",
    "SupportEndpointModel",
    "SupportOOFRecord",
    "SupportOOFCasePrediction",
    "SupportFoldNormalizer",
    "SurfaceRole",
    "canonical_probability_hex",
    "build_effective_menu",
    "case_balanced_weights",
    "certify_case_prediction",
    "evaluate_support_policy_admission",
    "fit_feature_map",
    "fit_support_fold_normalizer",
    "fit_menu_risk_calibration",
    "fit_support_endpoint_model",
    "fit_support_router",
    "float32_probability_hex",
    "fold_class_support_counts",
    "group_effective_menus",
    "leave_one_case_out_crossfit",
    "normalize_action_outcomes",
    "select_hierarchical_certificate",
    "validate_support_inventory",
)
