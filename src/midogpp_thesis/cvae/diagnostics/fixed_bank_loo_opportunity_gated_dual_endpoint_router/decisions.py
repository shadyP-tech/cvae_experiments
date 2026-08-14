"""Public decision facade for identification and robust endpoints."""

from .identification import select_case_identification_decision
from .identification_products import (
    CaseIdentificationDecision,
    DirectionIdentificationDecision,
    IdentificationCandidateScore,
)
from .robust import select_robust_arm_decisions, select_robust_direction_decision
from .robust_products import DirectionRobustDecision, EndpointArm, RobustArmDecision, RobustCandidateScore


__all__ = (
    "CaseIdentificationDecision",
    "DirectionIdentificationDecision",
    "DirectionRobustDecision",
    "EndpointArm",
    "IdentificationCandidateScore",
    "RobustArmDecision",
    "RobustCandidateScore",
    "select_case_identification_decision",
    "select_robust_arm_decisions",
    "select_robust_direction_decision",
)
