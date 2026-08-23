"""P-DCAPS donor-cross-fitted action-surface subsystem."""

from .contracts import (
    ActionCalibrationModel,
    ActionKey,
    ActionPrediction,
    ActionResponse,
    ActionStratumReliability,
    CalibratedAction,
    CalibratedActionSelection,
)
from .calibration_plan import (
    ActionCalibrationFamilies,
    build_optimized_action_calibration_families,
)
from .descriptors import (
    action_feature_names,
    build_action_descriptor,
    build_action_descriptor_matrix,
    predicted_metric_value,
)
from .reliability import (
    build_action_reliability_by_stratum,
    evaluate_action_stratum_reliability,
    finite_spearman,
)
from .responses import (
    LOG_CLIP_EPSILON,
    build_action_response,
    canonical_probabilities,
    probability_sha256,
    realized_favorable_utility,
)
from .ridge import fit_weighted_ridge, predict_weighted_ridge
from .runtime import (
    build_nested_action_calibration_models,
    build_reliability_oof_action_models,
    calibrate_action,
    calibrate_and_select_actions,
    calibrated_utility_for_prediction,
    fit_action_calibration_models,
    select_calibrated_action,
)
from .surface import (
    ActionDraft,
    ResponseDenominators,
    RouteActionDraftSurface,
    SealedActionCell,
    SealedActionSurface,
    SealedRouteActionSurface,
    build_route_action_draft_surface,
    open_pseudo_route_action_responses,
    seal_action_surface,
)
from .weights import HierarchicalWeightAudit, build_hierarchical_weights


__all__ = (
    "ActionCalibrationModel",
    "ActionCalibrationFamilies",
    "ActionDraft",
    "ActionKey",
    "ActionPrediction",
    "ActionResponse",
    "ActionStratumReliability",
    "CalibratedAction",
    "CalibratedActionSelection",
    "HierarchicalWeightAudit",
    "LOG_CLIP_EPSILON",
    "ResponseDenominators",
    "RouteActionDraftSurface",
    "SealedActionCell",
    "SealedActionSurface",
    "SealedRouteActionSurface",
    "action_feature_names",
    "build_action_descriptor",
    "build_action_descriptor_matrix",
    "build_action_reliability_by_stratum",
    "build_action_response",
    "build_hierarchical_weights",
    "build_nested_action_calibration_models",
    "build_optimized_action_calibration_families",
    "build_route_action_draft_surface",
    "build_reliability_oof_action_models",
    "calibrate_action",
    "calibrate_and_select_actions",
    "calibrated_utility_for_prediction",
    "canonical_probabilities",
    "evaluate_action_stratum_reliability",
    "finite_spearman",
    "fit_action_calibration_models",
    "fit_weighted_ridge",
    "predict_weighted_ridge",
    "predicted_metric_value",
    "probability_sha256",
    "realized_favorable_utility",
    "open_pseudo_route_action_responses",
    "seal_action_surface",
    "select_calibrated_action",
)
