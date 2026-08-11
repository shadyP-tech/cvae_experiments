"""Compatibility facade for disagreement-regret contract objects."""

from ._validation import _canonical_id, _finite_probability
from .model_contracts import (
    DEVELOPMENT_CLAIM_ROLE,
    SCORE_SEMANTICS,
    CandidateContrastRow,
    DevelopmentSelectionDiagnostic,
    InferenceSelectionDiagnostic,
    PairwiseRegretModel,
)
from .inference_contracts import (
    LABEL_FREE_INFERENCE_CLAIM_ROLE,
    InferenceActionSchema,
    LabelFreeInferenceContext,
)
from .probability_contracts import (
    DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
    FEATURE_NAMES,
    LABEL_FREE_INFERENCE_SURFACE_ROLE,
    SOURCE_OOF_TRAINING_SURFACE_ROLE,
    CaseActionFeatureRow,
    DisagreementFeatureSurface,
    DisagreementRow,
    ProbabilityRow,
    SourceOOFLabelRow,
)
from .response_contracts import (
    RESPONSE_SEMANTICS,
    CaseActionResponseRow,
    ExactRegretSurface,
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
    "DevelopmentSelectionDiagnostic",
    "InferenceSelectionDiagnostic",
    "DisagreementFeatureSurface",
    "DisagreementRow",
    "ExactRegretSurface",
    "InferenceActionSchema",
    "LabelFreeInferenceContext",
    "PairwiseRegretModel",
    "ProbabilityRow",
    "SourceOOFLabelRow",
    "SOURCE_OOF_TRAINING_SURFACE_ROLE",
)
