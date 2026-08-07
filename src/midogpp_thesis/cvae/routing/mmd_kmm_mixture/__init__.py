"""Protocol-fenced class-prior-controlled MMD/KMM routing mathematics.

This package intentionally exposes no runner or workspace binding. Experiment
scope, evidence consumption, scoring, and publication claims remain the
responsibility of separately fenced diagnostic packages.
"""

from .config import KMMGateConfig, KMMOptimizationConfig, PriorControlConfig
from .conditional import (
    CONDITIONAL_PROXY_FAMILY,
    ConditionalContrastConfig,
    ConditionalContrastProblem,
    ConditionalRouteResult,
    build_conditional_contrast_problem,
    build_conditional_prior_sensitivity_problems,
    build_conditional_seed_axis_problems,
    build_conditional_support_case_problems,
    case_equal_soft_class_kernel_means,
    route_conditional_contrast_mmd,
)
from .contracts import (
    CROSSFIT_COHORT_SUPPORT_ROLE,
    DirectionIdentityAudit,
    EnergyDirectionReference,
    FrozenNystroemFeatureMap,
    KMMRouteDecision,
    KMMWeightSolution,
    KernelMeanProblem,
    MMDKMMProtocol,
    SourceKernelReplica,
    SourceOnlyPriorPrediction,
    StabilityAudit,
    TargetSupportKernelFeatures,
    TransformedKernelFeatures,
)
from .feature_map import transform_frozen_nystroem
from .gates import audit_direction_identity, audit_weight_stability, route_mmd_kmm
from .kmm import solve_kmm_weights
from .moments import (
    build_kernel_mean_problem,
    build_prior_sensitivity_problems,
    build_seed_axis_problems,
    build_support_case_problems,
    case_equal_class_balanced_kernel_mean,
)
from .prior import (
    prepare_source_only_responsibilities,
    shift_binary_prior,
    shift_source_only_prior_prediction,
)

__all__ = (
    "CROSSFIT_COHORT_SUPPORT_ROLE",
    "CONDITIONAL_PROXY_FAMILY",
    "ConditionalContrastConfig",
    "ConditionalContrastProblem",
    "ConditionalRouteResult",
    "DirectionIdentityAudit",
    "EnergyDirectionReference",
    "FrozenNystroemFeatureMap",
    "KMMGateConfig",
    "KMMOptimizationConfig",
    "KMMRouteDecision",
    "KMMWeightSolution",
    "KernelMeanProblem",
    "MMDKMMProtocol",
    "PriorControlConfig",
    "SourceKernelReplica",
    "SourceOnlyPriorPrediction",
    "StabilityAudit",
    "TargetSupportKernelFeatures",
    "TransformedKernelFeatures",
    "audit_direction_identity",
    "audit_weight_stability",
    "build_kernel_mean_problem",
    "build_conditional_contrast_problem",
    "build_conditional_prior_sensitivity_problems",
    "build_conditional_seed_axis_problems",
    "build_conditional_support_case_problems",
    "build_prior_sensitivity_problems",
    "build_seed_axis_problems",
    "build_support_case_problems",
    "case_equal_class_balanced_kernel_mean",
    "case_equal_soft_class_kernel_means",
    "prepare_source_only_responsibilities",
    "route_mmd_kmm",
    "route_conditional_contrast_mmd",
    "shift_binary_prior",
    "shift_source_only_prior_prediction",
    "solve_kmm_weights",
    "transform_frozen_nystroem",
)
