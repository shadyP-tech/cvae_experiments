"""Protocol-fenced class-prior-controlled MMD/KMM routing mathematics.

This package intentionally exposes no runner or workspace binding.  It is a
label-free proxy core awaiting a separately authorized, unconsumed experiment
surface.
"""

from .config import KMMGateConfig, KMMOptimizationConfig, PriorControlConfig
from .contracts import (
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
    "build_prior_sensitivity_problems",
    "build_seed_axis_problems",
    "build_support_case_problems",
    "case_equal_class_balanced_kernel_mean",
    "prepare_source_only_responsibilities",
    "route_mmd_kmm",
    "shift_binary_prior",
    "shift_source_only_prior_prediction",
    "solve_kmm_weights",
    "transform_frozen_nystroem",
)
