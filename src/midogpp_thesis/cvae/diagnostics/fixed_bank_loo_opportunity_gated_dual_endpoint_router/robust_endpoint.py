"""Compatibility facade for the modular robust endpoint science."""

from .robust import (
    build_endpoint_arms,
    rank_sources_by_prior,
    select_robust_arm_decisions,
    select_robust_direction_decision,
)
from .robust_products import (
    DirectionRobustDecision,
    EndpointArm,
    RobustArmDecision,
    RobustCandidateScore,
)


__all__ = (
    "DirectionRobustDecision",
    "EndpointArm",
    "RobustArmDecision",
    "RobustCandidateScore",
    "build_endpoint_arms",
    "rank_sources_by_prior",
    "select_robust_arm_decisions",
    "select_robust_direction_decision",
)
