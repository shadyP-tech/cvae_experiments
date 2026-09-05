"""HARP v17 pooled pairwise selected-policy router.

This is a predecessor-free scientific package.  It fits one source-only policy
over known-center ``(q, case)`` development rows, evaluates the complete nested
selection algorithm out of fold, and exposes only label-free target routing.
"""

from .admission import (
    ApproximateSourceOOFBounds,
    SourceOnlyAdmission,
    approximate_source_oof_bounds,
    build_source_only_admission,
)
from .composition import (
    build_baseline_composite,
    build_exact_u_composite,
    build_soft_topk_composite,
    soft_arm_id,
)
from .contracts import (
    AdmissionStatus,
    BASELINE_THRESHOLD,
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    PROBABILITY_CLIP,
    RouterFitConfig,
    SoftTopKComposite,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    canonical_probability_hex,
    decode_probability_hex,
    float32_probability_hex,
)
from .crossfit import (
    ArmSpec,
    FoldChoice,
    NestedCrossfitResult,
    candidate_arm_specs,
    center_stratified_folds,
    nested_source_crossfit,
    validate_source_inventory,
)
from .hashing import canonical_bytes, canonical_hash, canonical_value, require_sha256
from .modeling import (
    CaseModelPrediction,
    DirectionOpportunityHead,
    FittedFeatureTransform,
    PairwiseComparison,
    PairwiseRanker,
    PooledScienceModel,
    build_pairwise_comparisons,
    component_validation_losses,
    fit_feature_transform,
    fit_pairwise_ranker,
    fit_pooled_science_model,
)
from .policy import (
    PooledRouterPolicy,
    fit_source_router,
    route_decision_report,
    route_target_cases,
)
from .records import RouteDecision, SealedOOFSelection, SelectedOOFRecord
from .truth import (
    SupportTruthCapability,
    combine_truth_capabilities,
    score_selected_composite,
)


__all__ = (
    "AdmissionStatus",
    "ApproximateSourceOOFBounds",
    "ArmSpec",
    "BASELINE_THRESHOLD",
    "CaseModelPrediction",
    "CompositeKind",
    "Direction",
    "DirectionOpportunityHead",
    "FittedFeatureTransform",
    "FoldChoice",
    "LabelFreeAction",
    "LabelFreeCaseMenu",
    "NestedCrossfitResult",
    "PROBABILITY_CLIP",
    "PairwiseComparison",
    "PairwiseRanker",
    "PooledRouterPolicy",
    "PooledScienceModel",
    "RouteDecision",
    "RouterFitConfig",
    "SealedOOFSelection",
    "SelectedOOFRecord",
    "SoftTopKComposite",
    "SourceOnlyAdmission",
    "SupportActionOutcome",
    "SupportCaseClassProfile",
    "SupportTruthCapability",
    "SurfaceRole",
    "approximate_source_oof_bounds",
    "build_baseline_composite",
    "build_exact_u_composite",
    "build_pairwise_comparisons",
    "build_soft_topk_composite",
    "build_source_only_admission",
    "candidate_arm_specs",
    "canonical_bytes",
    "canonical_hash",
    "canonical_probability_hex",
    "canonical_value",
    "center_stratified_folds",
    "combine_truth_capabilities",
    "component_validation_losses",
    "decode_probability_hex",
    "fit_feature_transform",
    "fit_pairwise_ranker",
    "fit_pooled_science_model",
    "fit_source_router",
    "float32_probability_hex",
    "nested_source_crossfit",
    "require_sha256",
    "route_decision_report",
    "route_target_cases",
    "score_selected_composite",
    "soft_arm_id",
    "validate_source_inventory",
)
