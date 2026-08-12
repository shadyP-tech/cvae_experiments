"""Leakage-safe scientific core for labeled-support threshold-flip routing."""

from .calibration import (
    build_calibration_row,
    calibrated_gain,
    directional_raw_gains,
    fit_direction_shared_calibration,
)
from .contracts import (
    CalibrationRow,
    CaseActionFeatures,
    CaseDecision,
    ContributionTarget,
    DirectionSharedCalibration,
    DonorRow,
    HeadModel,
    StaticSelection,
    TwoHeadPrediction,
    TwoHeadRidgeModel,
    canonical_hash,
    hash_decision_inputs,
)
from .metrics import (
    BootstrapContrast,
    CaseConfusion,
    MethodScore,
    TerminalOracles,
    case_confusion,
    method_score,
    paired_case_bootstrap_contrast,
    pooled_bacc,
    router_metrics,
    terminal_oracles,
)
from .model import fit_two_head_ridge, predict_two_head
from .permutation import blocked_case_derangement, refit_blocked_permutation_control
from .query_fixed_effects import (
    QueryFixedEffectStaticFit,
    select_query_fixed_effect_static_source,
)
from .selection import select_case_action, select_static_source
from .targets import (
    contribution_target,
    exact_nine_mean_probabilities,
    hard_predictions,
    pooled_gain,
)

__all__ = (
    "BootstrapContrast",
    "CalibrationRow",
    "CaseActionFeatures",
    "CaseConfusion",
    "CaseDecision",
    "ContributionTarget",
    "DirectionSharedCalibration",
    "DonorRow",
    "HeadModel",
    "MethodScore",
    "QueryFixedEffectStaticFit",
    "StaticSelection",
    "TerminalOracles",
    "TwoHeadPrediction",
    "TwoHeadRidgeModel",
    "blocked_case_derangement",
    "build_calibration_row",
    "calibrated_gain",
    "canonical_hash",
    "case_confusion",
    "contribution_target",
    "directional_raw_gains",
    "exact_nine_mean_probabilities",
    "fit_direction_shared_calibration",
    "fit_two_head_ridge",
    "hard_predictions",
    "hash_decision_inputs",
    "method_score",
    "paired_case_bootstrap_contrast",
    "pooled_bacc",
    "pooled_gain",
    "predict_two_head",
    "refit_blocked_permutation_control",
    "router_metrics",
    "select_case_action",
    "select_query_fixed_effect_static_source",
    "select_static_source",
    "terminal_oracles",
)
