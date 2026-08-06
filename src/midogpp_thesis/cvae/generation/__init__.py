"""Frozen generation contracts for routing-authorized CVAE experts."""

from .config import UniformBV2GenerationLockConfig, load_generation_lock_config
from .contracts import ControlReplicate, GenerationLock, SourceGenerationKey
from .generation import (
    GeneratedBlock,
    derived_composition_seed,
    derived_generation_seed,
    equal_union_replicate_plan,
    generate_source_block,
    source_generation_plan,
)
from .runner import build_generation_lock, read_generation_lock, run_generation_lock
from .validation import validate_generation_bundle

__all__ = (
    "ControlReplicate",
    "GeneratedBlock",
    "GenerationLock",
    "SourceGenerationKey",
    "UniformBV2GenerationLockConfig",
    "build_generation_lock",
    "derived_composition_seed",
    "derived_generation_seed",
    "equal_union_replicate_plan",
    "generate_source_block",
    "load_generation_lock_config",
    "read_generation_lock",
    "run_generation_lock",
    "source_generation_plan",
    "validate_generation_bundle",
)
