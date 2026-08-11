"""Compatibility facade for pairwise regret fitting and target scoring."""

from ._solver import (
    BACKTRACK_STEPS,
    GRADIENT_TOLERANCE,
    MAX_NEWTON_ITERATIONS,
    STEP_TOLERANCE,
)
from .fitting import fit_known_bank_pairwise_models
from .scoring import (
    score_label_free_inference_candidate_contrasts,
    score_target_candidate_contrasts,
)


__all__ = (
    "BACKTRACK_STEPS",
    "GRADIENT_TOLERANCE",
    "MAX_NEWTON_ITERATIONS",
    "STEP_TOLERANCE",
    "fit_known_bank_pairwise_models",
    "score_label_free_inference_candidate_contrasts",
    "score_target_candidate_contrasts",
)
