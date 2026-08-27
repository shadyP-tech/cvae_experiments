"""Direct finite-action selection with robust safety gates and exact-P fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import (
    ACTION_FAMILIES,
    ACTION_IDS,
    DIRECTIONS,
    PRIMARY_METHOD_ID,
    P_METHOD_ID,
    array_sha256,
    probability_vector,
)
from ..posterior.empirical_bayes import ActionEstimate
from ..posterior.uncertainty import DescriptiveBounds
from ..utility.actions import ActionRectangle


P_CANDIDATE_REPRESENTATION = "IMPLICIT_ZERO_UTILITY_EXACT_FALLBACK"


@dataclass(frozen=True, slots=True)
class SafetyThresholds:
    minimum_bacc_lower: float = 0.0
    maximum_brier_upper: float = 0.0
    maximum_log_upper: float = 0.0
    tie_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        values = (
            float(self.minimum_bacc_lower),
            float(self.maximum_brier_upper),
            float(self.maximum_log_upper),
            float(self.tie_tolerance),
        )
        if not all(math.isfinite(value) for value in values) or values[3] < 0.0:
            raise GovernanceError("SCALE-BP v2 safety thresholds drifted.")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    action_id: str
    estimate: ActionEstimate
    bounds: DescriptiveBounds
    eligible: bool
    reasons: tuple[str, ...]
    assessment_hash: str = field(init=False)

    def __post_init__(self) -> None:
        reasons = tuple(str(reason) for reason in self.reasons)
        if (
            self.action_id not in ACTION_IDS
            or self.estimate.action_id != self.action_id
            or self.bounds.action_id != self.action_id
            or self.bounds.estimate_hash != self.estimate.estimate_hash
            or self.bounds.mean != self.estimate.mean
            or not reasons
            or self.eligible != (reasons == ("ELIGIBLE_ROBUST_SAFE",))
        ):
            raise GovernanceError("SCALE-BP v2 candidate assessment drifted.")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "assessment_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_candidate_assessment_v1",
                    "action_id": self.action_id,
                    "estimate_hash": self.estimate.estimate_hash,
                    "bounds_hash": self.bounds.bounds_hash,
                    "eligible": self.eligible,
                    "reasons": reasons,
                }
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class RouteDecision:
    target_center: str
    case_id: str
    method_id: str
    selected_action_id: str | None
    reason: str
    emitted_probabilities: np.ndarray
    full_endpoint_probabilities: np.ndarray
    assessments: tuple[CandidateAssessment, ...]
    rectangle_hash: str
    selected_action_hash: str | None
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        emitted = probability_vector(self.emitted_probabilities)
        full = probability_vector(
            self.full_endpoint_probabilities, expected_length=len(emitted)
        )
        assessments = tuple(self.assessments)
        if (
            self.method_id not in {P_METHOD_ID, PRIMARY_METHOD_ID}
            or tuple(row.action_id for row in assessments) != ACTION_IDS
            or (self.method_id == P_METHOD_ID) != (self.selected_action_id is None)
            or (
                self.selected_action_id is not None
                and self.selected_action_id not in ACTION_IDS
            )
            or (self.selected_action_id is None) != (self.selected_action_hash is None)
            or not self.reason
            or not self.rectangle_hash
        ):
            raise GovernanceError("SCALE-BP v2 route decision drifted.")
        object.__setattr__(self, "emitted_probabilities", emitted)
        object.__setattr__(self, "full_endpoint_probabilities", full)
        object.__setattr__(self, "assessments", assessments)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_route_decision_v1",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "method_id": self.method_id,
                    "selected_action_id": self.selected_action_id,
                    "reason": self.reason,
                    "emitted_probability_sha256": array_sha256(emitted),
                    "full_endpoint_probability_sha256": array_sha256(full),
                    "assessment_hashes": tuple(
                        row.assessment_hash for row in assessments
                    ),
                    "rectangle_hash": self.rectangle_hash,
                    "selected_action_hash": self.selected_action_hash,
                    "p_candidate_representation": P_CANDIDATE_REPRESENTATION,
                    "p_candidate_expected_utility_anchor": 0.0,
                    "p_candidate_assessment_emitted": False,
                    "p_wins_without_unique_robust_safe_positive_action": True,
                    "exact_p_on_fallback": self.selected_action_id is None,
                    "at_most_one_action": True,
                }
            ),
        )

    @property
    def is_exact_p(self) -> bool:
        return self.selected_action_id is None


def assess_candidates(
    estimates: Sequence[ActionEstimate],
    bounds: Sequence[DescriptiveBounds],
    *,
    thresholds: SafetyThresholds = SafetyThresholds(),
) -> tuple[CandidateAssessment, ...]:
    estimates_by_id = {row.action_id: row for row in estimates}
    bounds_by_id = {row.action_id: row for row in bounds}
    if (
        tuple(estimates_by_id) != ACTION_IDS
        or tuple(bounds_by_id) != ACTION_IDS
        or len(estimates_by_id) != len(tuple(estimates))
        or len(bounds_by_id) != len(tuple(bounds))
    ):
        raise GovernanceError("SCALE-BP v2 selection requires complete pre-argmax inputs.")
    output: list[CandidateAssessment] = []
    for action_id in ACTION_IDS:
        estimate = estimates_by_id[action_id]
        envelope = bounds_by_id[action_id]
        failed: list[str] = []
        if estimate.structural_noop:
            failed.append("STRUCTURAL_NOOP")
        if not estimate.within_support:
            failed.append("OUTSIDE_SUPPORT")
        if not estimate.bank_viable:
            failed.append("BANK_NOT_VIABLE")
        if envelope.lower.bacc <= thresholds.minimum_bacc_lower:
            failed.append("BACC_LOWER_NOT_STRICTLY_POSITIVE")
        if envelope.upper.brier > thresholds.maximum_brier_upper:
            failed.append("BRIER_UPPER_UNSAFE")
        if envelope.upper.log > thresholds.maximum_log_upper:
            failed.append("LOG_UPPER_UNSAFE")
        reasons = tuple(failed) if failed else ("ELIGIBLE_ROBUST_SAFE",)
        output.append(
            CandidateAssessment(action_id, estimate, envelope, not failed, reasons)
        )
    return tuple(output)


def select_action(
    rectangle: ActionRectangle,
    estimates: Sequence[ActionEstimate],
    bounds: Sequence[DescriptiveBounds],
    *,
    thresholds: SafetyThresholds = SafetyThresholds(),
) -> RouteDecision:
    assessments = assess_candidates(estimates, bounds, thresholds=thresholds)
    if any(
        row.estimate.target_center != rectangle.target_center
        or row.estimate.case_id != rectangle.case_id
        or row.estimate.structural_noop
        != rectangle.cell(row.action_id).structural_noop
        for row in assessments
    ):
        raise GovernanceError("SCALE-BP v2 selection route identity drifted.")
    eligible = tuple(row for row in assessments if row.eligible)
    selected: CandidateAssessment | None = None
    reason = "EXACT_P_NO_ROBUST_SAFE_ACTION"
    if eligible:
        maximum = max(row.estimate.mean.bacc for row in eligible)
        tied = tuple(
            row
            for row in eligible
            if maximum - row.estimate.mean.bacc <= thresholds.tie_tolerance
        )
        if len(tied) == 1 and maximum > thresholds.minimum_bacc_lower:
            selected = tied[0]
            reason = "UNIQUE_MAXIMUM_ROBUST_SAFE_EXPECTED_BACC"
        else:
            reason = "EXACT_P_ACTION_TIE_OR_NONPOSITIVE_MAXIMUM"
    protected_p = rectangle.cells[0].action.protected_p
    if selected is None:
        return RouteDecision(
            rectangle.target_center,
            rectangle.case_id,
            P_METHOD_ID,
            None,
            reason,
            protected_p,
            protected_p,
            assessments,
            rectangle.rectangle_hash,
            None,
        )
    cell = rectangle.cell(selected.action_id)
    return RouteDecision(
        rectangle.target_center,
        rectangle.case_id,
        PRIMARY_METHOD_ID,
        selected.action_id,
        reason,
        cell.action.projected,
        cell.action.full_endpoint_control,
        assessments,
        rectangle.rectangle_hash,
        cell.action.action_hash,
    )


__all__ = (
    "CandidateAssessment",
    "P_CANDIDATE_REPRESENTATION",
    "RouteDecision",
    "SafetyThresholds",
    "assess_candidates",
    "select_action",
)
