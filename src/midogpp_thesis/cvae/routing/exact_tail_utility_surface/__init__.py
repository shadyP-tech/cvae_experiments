"""Production facade for the fresh exact additive-tail utility surface."""

from .bundle import REQUIRED_FILES, ExactTailUtilitySurfaceLock, validate_surface_bundle
from .config import (
    ExactTailUtilitySurfaceConfig,
    load_exact_tail_utility_surface_config,
)
from .contracts import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .runner import run_exact_tail_utility_surface


__all__ = (
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "REQUIRED_FILES",
    "ExactTailUtilitySurfaceConfig",
    "ExactTailUtilitySurfaceLock",
    "load_exact_tail_utility_surface_config",
    "run_exact_tail_utility_surface",
    "validate_surface_bundle",
)
