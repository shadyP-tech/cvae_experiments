"""Policy-calibrated pairwise residual HARP v10 routing science core."""

from .acceptor import (
    SELECTION_FEATURE_NAMES,
    SelectedActionAcceptor,
    SelectedActionObservation,
    fit_selected_action_acceptor,
    selected_action_features,
)
from .admission import AdmissionConfig, OuterAdmission, evaluate_outer_admission
from .calibration import (
    DISABLED_ACCEPTANCE_THRESHOLD,
    PolicyCalibration,
    PolicyReplay,
    PolicyRiskConfig,
    calibrate_selected_policy,
)
from .contracts import (
    ActionScore,
    CasePrediction,
    Direction,
    LabelFreeAction,
    SourceActionOutcome,
    action_group,
    float32_probability_hex,
    probability_bytes_to_hex,
    probability_hex_to_bytes,
)
from .effective_menu import EffectiveMenu, build_effective_menu, group_effective_menus
from .model import (
    NestedPolicyFold,
    PairwiseFitConfig,
    PairwiseResidualRouterModel,
    SourceLODOResult,
    assemble_source_lodo_result,
    fit_prelabel_pseudo_target_fold,
    fit_source_lodo,
    predict_case,
    predict_target_actions,
)
from .policy import RouteDecision, select_policy_action
from .ranker import PairwiseRanker, ScaleOnlyTransform, fit_pairwise_ranker
from .residuals import (
    ResidualActionFeatures,
    assert_residual_identity,
    residual_feature_names,
    residualize_menu,
)


__all__ = (
    "ActionScore",
    "AdmissionConfig",
    "CasePrediction",
    "DISABLED_ACCEPTANCE_THRESHOLD",
    "Direction",
    "EffectiveMenu",
    "LabelFreeAction",
    "NestedPolicyFold",
    "OuterAdmission",
    "PairwiseFitConfig",
    "PairwiseRanker",
    "PairwiseResidualRouterModel",
    "PolicyCalibration",
    "PolicyReplay",
    "PolicyRiskConfig",
    "ResidualActionFeatures",
    "RouteDecision",
    "SELECTION_FEATURE_NAMES",
    "ScaleOnlyTransform",
    "SelectedActionAcceptor",
    "SelectedActionObservation",
    "SourceActionOutcome",
    "SourceLODOResult",
    "action_group",
    "assemble_source_lodo_result",
    "assert_residual_identity",
    "build_effective_menu",
    "calibrate_selected_policy",
    "evaluate_outer_admission",
    "fit_pairwise_ranker",
    "fit_prelabel_pseudo_target_fold",
    "fit_selected_action_acceptor",
    "fit_source_lodo",
    "float32_probability_hex",
    "group_effective_menus",
    "predict_case",
    "predict_target_actions",
    "probability_bytes_to_hex",
    "probability_hex_to_bytes",
    "residual_feature_names",
    "residualize_menu",
    "select_policy_action",
    "selected_action_features",
)
