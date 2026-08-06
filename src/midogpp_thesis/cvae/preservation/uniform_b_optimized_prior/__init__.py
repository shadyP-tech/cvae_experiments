"""Capacity-and-composition redesign for the Uniform-B CVAE prior."""

from .config import OptimizedPriorConfig, load_optimized_prior_config
from .runner import run_optimized_prior_source_inner_study

__all__ = (
    "OptimizedPriorConfig",
    "load_optimized_prior_config",
    "run_optimized_prior_source_inner_study",
)
