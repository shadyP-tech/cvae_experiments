"""Small public facade for the modular correctness-proxy implementation."""

from .candidate_feature_permutation import (
    candidate_feature_permutation,
    permute_route_candidate_feature_blocks,
)
from .correctness_model import (
    case_proxy,
    fit_directional_correctness_model,
    fit_route_correctness_models,
    predict_correctness,
    support_calibrated_case_proxy,
)
from .correctness_observations import (
    score_directional_correctness_observations,
    score_route_correctness_observations,
    support_class_denominators,
)
from .correctness_products import (
    DirectionalCorrectnessModel,
    DirectionalCorrectnessObservation,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)
from .held_case_features import (
    build_label_free_case_candidate_features,
    build_label_free_features,
    case_directional_features,
)


__all__ = (
    "DirectionalCorrectnessModel",
    "DirectionalCorrectnessObservation",
    "LabelFreeDirectionalFeatures",
    "SupportClassDenominators",
    "build_label_free_case_candidate_features",
    "build_label_free_features",
    "candidate_feature_permutation",
    "case_directional_features",
    "case_proxy",
    "fit_directional_correctness_model",
    "fit_route_correctness_models",
    "permute_route_candidate_feature_blocks",
    "predict_correctness",
    "score_directional_correctness_observations",
    "score_route_correctness_observations",
    "support_calibrated_case_proxy",
    "support_class_denominators",
)
