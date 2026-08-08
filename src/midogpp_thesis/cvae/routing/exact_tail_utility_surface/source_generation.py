"""Stable facade for exact-tail source generation and feature components."""

from .source_checkpoint_store import load_component_arrays
from .source_contracts import (
    FeatureComponentRecord,
    GeneratedDevelopmentCache,
    SourceBlockRecord,
)
from .source_orchestration import materialize_generated_development_cache
from .source_planning import load_validated_generation_lock


__all__ = (
    "FeatureComponentRecord",
    "GeneratedDevelopmentCache",
    "SourceBlockRecord",
    "load_component_arrays",
    "load_validated_generation_lock",
    "materialize_generated_development_cache",
)
