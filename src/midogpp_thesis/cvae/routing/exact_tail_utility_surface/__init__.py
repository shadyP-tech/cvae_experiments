"""Production facade for the fresh exact additive-tail utility surface."""

from .bundle import REQUIRED_FILES, ExactTailUtilitySurfaceLock, validate_surface_bundle
from .config import (
    ExactTailUtilitySurfaceConfig,
    load_exact_tail_utility_surface_config,
)
from .contracts import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .ensemble_scoring import (
    ExactTailEnsembleEndpointLock,
    ScoredExactTailEnsembleEndpointRow,
    score_exact_tail_ensemble_endpoints,
)
from .probability_surface import (
    SealedProbabilitySurface,
    SealedSupportProbabilitySurface,
)
from .runner import run_exact_tail_utility_surface
from .support_shift_surface import (
    ExactTailSupportActionShiftLock,
    ExactTailSupportActionShiftRow,
    build_support_action_shift_rows,
)


__all__ = (
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "REQUIRED_FILES",
    "ExactTailEnsembleEndpointLock",
    "ExactTailSupportActionShiftLock",
    "ExactTailSupportActionShiftRow",
    "ExactTailUtilitySurfaceConfig",
    "ExactTailUtilitySurfaceLock",
    "ScoredExactTailEnsembleEndpointRow",
    "SealedProbabilitySurface",
    "SealedSupportProbabilitySurface",
    "build_support_action_shift_rows",
    "load_exact_tail_utility_surface_config",
    "run_exact_tail_utility_surface",
    "score_exact_tail_ensemble_endpoints",
    "validate_surface_bundle",
)
