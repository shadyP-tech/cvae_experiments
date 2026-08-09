"""Public facade for the consumed Stage-90 ensemble-endpoint diagnostic."""

from .config import (
    EnsembleEndpointRouterConfig,
    load_utility_aligned_ensemble_endpoint_router_config,
)
from .actions import (
    FrozenEnsembleEndpointAction,
    FrozenEnsembleEndpointActionLibrary,
    build_ensemble_endpoint_action_library,
    build_inner_ensemble_endpoint_actions,
    build_target_ensemble_endpoint_actions,
    inner_action_library_for,
)
from .diagnostic_plan import (
    FrozenEnsembleEndpointDiagnosticPlan,
    Stage90EnsembleDiagnosticPlanSet,
    build_ensemble_endpoint_diagnostic_plan,
    build_stage90_ensemble_diagnostic_plan_set,
)
from .endpoint_scoring import (
    build_source_inner_ensemble_response,
    build_support_action_shift,
    validate_source_inner_ensemble_responses,
)
from .features import (
    HeldoutEnsembleFeatureSurfaces,
    Stage90EnsembleFeatureSurfaceSet,
    build_heldout_ensemble_feature_surfaces,
    build_stage90_ensemble_feature_surface_set,
)
from .modeling import (
    HeldoutEnsembleRouterModels,
    Stage90EnsembleRouterModelSet,
    fit_stage90_ensemble_models,
    fit_stage90_heldout_ensemble_models,
)
from .scoring import (
    TargetEnsembleEndpointScore,
    TargetEnsembleEndpointScoreSet,
    build_terminal_hxe_oracle_diagnostics,
    score_target_action_ensemble_endpoint,
    validate_target_ensemble_endpoint_scores,
)


def run_utility_aligned_ensemble_endpoint_router_diagnostic(*args: object, **kwargs: object):
    from .runner import run_utility_aligned_ensemble_endpoint_router_diagnostic as run

    return run(*args, **kwargs)


__all__ = (
    "EnsembleEndpointRouterConfig",
    "FrozenEnsembleEndpointAction",
    "FrozenEnsembleEndpointActionLibrary",
    "FrozenEnsembleEndpointDiagnosticPlan",
    "HeldoutEnsembleFeatureSurfaces",
    "HeldoutEnsembleRouterModels",
    "Stage90EnsembleDiagnosticPlanSet",
    "Stage90EnsembleFeatureSurfaceSet",
    "Stage90EnsembleRouterModelSet",
    "TargetEnsembleEndpointScore",
    "TargetEnsembleEndpointScoreSet",
    "build_ensemble_endpoint_action_library",
    "build_ensemble_endpoint_diagnostic_plan",
    "build_heldout_ensemble_feature_surfaces",
    "build_inner_ensemble_endpoint_actions",
    "build_source_inner_ensemble_response",
    "build_stage90_ensemble_diagnostic_plan_set",
    "build_stage90_ensemble_feature_surface_set",
    "build_support_action_shift",
    "build_target_ensemble_endpoint_actions",
    "build_terminal_hxe_oracle_diagnostics",
    "fit_stage90_ensemble_models",
    "fit_stage90_heldout_ensemble_models",
    "inner_action_library_for",
    "load_utility_aligned_ensemble_endpoint_router_config",
    "run_utility_aligned_ensemble_endpoint_router_diagnostic",
    "score_target_action_ensemble_endpoint",
    "validate_source_inner_ensemble_responses",
    "validate_target_ensemble_endpoint_scores",
)
