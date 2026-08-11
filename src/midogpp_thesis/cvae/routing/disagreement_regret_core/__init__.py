"""Non-runnable disagreement-conditioned regret mathematics.

This package is intentionally an in-memory train/inference library.  It has no
experiment config, registry entry, runner, artifact writer, cache reader, or
terminal evaluator.  Consumed target/test labels are rejected; a separately
sealed consumed target may only produce terminal label-free diagnostics.
"""

from .contracts import (
    DEVELOPMENT_CLAIM_ROLE,
    DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
    FEATURE_NAMES,
    LABEL_FREE_INFERENCE_CLAIM_ROLE,
    LABEL_FREE_INFERENCE_SURFACE_ROLE,
    RESPONSE_SEMANTICS,
    SCORE_SEMANTICS,
    CandidateContrastRow,
    CaseActionFeatureRow,
    CaseActionResponseRow,
    DevelopmentSelectionDiagnostic,
    InferenceSelectionDiagnostic,
    DisagreementFeatureSurface,
    DisagreementRow,
    ExactRegretSurface,
    InferenceActionSchema,
    LabelFreeInferenceContext,
    PairwiseRegretModel,
    ProbabilityRow,
    SourceOOFLabelRow,
    SOURCE_OOF_TRAINING_SURFACE_ROLE,
)
from .controls import feature_surface_for_family
from .features import (
    build_disagreement_feature_surface,
    build_label_free_inference_feature_surface,
    build_source_oof_training_feature_surface,
)
from .model import (
    fit_known_bank_pairwise_models,
    score_label_free_inference_candidate_contrasts,
    score_target_candidate_contrasts,
)
from .model_bank import (
    PairwiseRegretModelBank,
    deserialize_pairwise_model_bank,
    freeze_pairwise_model_bank,
    serialize_pairwise_model_bank,
)
from .inference_contracts import assert_label_free_inference_context
from .provenance import DevelopmentContext, DevelopmentScope, assert_development_context
from .rewards import build_exact_regret_surface
from .runtime import (
    WorkstationRuntime,
    assert_dense_fit_within_budget,
    canonical_workstation_runtime,
    estimate_dense_fit_bytes,
    validate_runtime,
)
from .selection import (
    build_label_free_inference_selection_diagnostics,
    build_safe_selection_diagnostics,
)


__all__ = (
    "DEVELOPMENT_CLAIM_ROLE",
    "DEVELOPMENT_COMPOSITE_SURFACE_ROLE",
    "FEATURE_NAMES",
    "LABEL_FREE_INFERENCE_CLAIM_ROLE",
    "LABEL_FREE_INFERENCE_SURFACE_ROLE",
    "RESPONSE_SEMANTICS",
    "SCORE_SEMANTICS",
    "CandidateContrastRow",
    "CaseActionFeatureRow",
    "CaseActionResponseRow",
    "DevelopmentContext",
    "DevelopmentScope",
    "DevelopmentSelectionDiagnostic",
    "InferenceSelectionDiagnostic",
    "DisagreementFeatureSurface",
    "DisagreementRow",
    "ExactRegretSurface",
    "InferenceActionSchema",
    "LabelFreeInferenceContext",
    "PairwiseRegretModelBank",
    "PairwiseRegretModel",
    "ProbabilityRow",
    "SourceOOFLabelRow",
    "SOURCE_OOF_TRAINING_SURFACE_ROLE",
    "WorkstationRuntime",
    "assert_development_context",
    "assert_label_free_inference_context",
    "assert_dense_fit_within_budget",
    "build_disagreement_feature_surface",
    "build_label_free_inference_feature_surface",
    "build_label_free_inference_selection_diagnostics",
    "build_source_oof_training_feature_surface",
    "build_exact_regret_surface",
    "build_safe_selection_diagnostics",
    "canonical_workstation_runtime",
    "estimate_dense_fit_bytes",
    "feature_surface_for_family",
    "fit_known_bank_pairwise_models",
    "freeze_pairwise_model_bank",
    "deserialize_pairwise_model_bank",
    "serialize_pairwise_model_bank",
    "score_label_free_inference_candidate_contrasts",
    "score_target_candidate_contrasts",
    "validate_runtime",
)
