"""Independent-source aggregate-posterior mixture plus GECO study."""

from .config import (
    AggregatePriorStudyConfig,
    load_aggregate_prior_study_config,
)
from .runner import run_aggregate_prior_source_inner_study
from .validation import validate_aggregate_prior_study_bundle

__all__ = [
    "AggregatePriorStudyConfig",
    "load_aggregate_prior_study_config",
    "run_aggregate_prior_source_inner_study",
    "validate_aggregate_prior_study_bundle",
]
