"""SCALE-BP v1 implementation shell (planned and non-executable)."""

from .config import (
    ScaleBPConfig,
    load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config,
)
from .execution_admission import BLOCKED_MESSAGE, assert_execution_authorized
from .identity import EXPERIMENT_ID, OUTPUT_ARTIFACT_ID
from .runner import (
    run_support_calibrated_local_action_empirical_bayes_boundary_projected_router,
)


__all__ = (
    "BLOCKED_MESSAGE",
    "EXPERIMENT_ID",
    "OUTPUT_ARTIFACT_ID",
    "ScaleBPConfig",
    "assert_execution_authorized",
    "load_support_calibrated_local_action_empirical_bayes_boundary_projected_router_config",
    "run_support_calibrated_local_action_empirical_bayes_boundary_projected_router",
)
