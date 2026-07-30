"""Bounded nonlinear decision-boundary diagnostic for canonical Uniform-B."""

from .config import (
    NonlinearProbeConfig,
    load_nonlinear_probe_config,
)
from .runner import run_nonlinear_probe
from .validation import validate_nonlinear_probe_bundle

__all__ = [
    "NonlinearProbeConfig",
    "load_nonlinear_probe_config",
    "run_nonlinear_probe",
    "validate_nonlinear_probe_bundle",
]
