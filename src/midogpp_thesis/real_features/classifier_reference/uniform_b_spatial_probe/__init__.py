"""Bounded spatial successor diagnostic for canonical Uniform-B."""

from .cache import (
    assemble_b_spatial_features,
    build_uniform_b_spatial_cache,
    validate_uniform_b_spatial_cache,
)
from .config import (
    SpatialCacheConfig,
    SpatialProbeConfig,
    load_spatial_cache_config,
    load_spatial_probe_config,
)
from .runner import run_spatial_probe
from .validation import validate_spatial_probe_bundle

__all__ = [
    "SpatialCacheConfig",
    "SpatialProbeConfig",
    "assemble_b_spatial_features",
    "build_uniform_b_spatial_cache",
    "load_spatial_cache_config",
    "load_spatial_probe_config",
    "run_spatial_probe",
    "validate_spatial_probe_bundle",
    "validate_uniform_b_spatial_cache",
]
