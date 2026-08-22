"""Case-action eligibility and deterministic tie handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .canonical_probabilities import CanonicalProbabilityVector, canonical_hash
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    DIRECTION_IDS,
    UTILITY_ZERO_TOLERANCE,
)
from .posterior_expected_utility import PosteriorUtilityEstimate


ALTERNATIVE_ORDER = ALTERNATIVE_METHOD_IDS
DIRECTION_ORDER = DIRECTION_IDS

ELIGIBLE = "ELIGIBLE"
NONPOSITIVE_BACC = "NONPOSITIVE_EXPECTED_BACC"
BRIER_UNSAFE = "NEGATIVE_EXPECTED_BRIER_GAIN"
LOG_UNSAFE = "NEGATIVE_EXPECTED_LOG_GAIN"


@dataclass(frozen=True)
class ActionCandidate:
    """One label-free directional action for one physical case."""

    center: str
    case_id: str
    alternative_id: str
    direction: str
    control_id: str
    probabilities: CanonicalProbabilityVector
    estimate: PosteriorUtilityEstimate
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not self.center
            or not self.case_id
            or self.alternative_id not in ALTERNATIVE_ORDER
            or self.direction not in DIRECTION_ORDER
            or not self.control_id
            or self.estimate.center != self.center
            or self.estimate.case_id != self.case_id
            or self.estimate.action_id != self.action_id
            or self.estimate.direction != self.direction
            or self.estimate.control_id != self.control_id
        ):
            raise ProtocolError("CBPUPR action candidate identity drifted.")
        payload = {
            "schema_version": "cbpupr_action_candidate_v1",
            "center": self.center,
            "case_id": self.case_id,
            "alternative_id": self.alternative_id,
            "direction": self.direction,
            "control_id": self.control_id,
            "probability_sha256": self.probabilities.sha256,
            "estimate_hash": self.estimate.estimate_hash,
        }
        object.__setattr__(self, "action_hash", canonical_hash(payload))

    @property
    def action_id(self) -> str:
        return f"{self.alternative_id}::{self.direction}"

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ActionCandidate":
        row = cls(
            center=str(payload["center"]),
            case_id=str(payload["case_id"]),
            alternative_id=str(payload["alternative_id"]),
            direction=str(payload["direction"]),
            control_id=str(payload["control_id"]),
            probabilities=CanonicalProbabilityVector.from_payload(
                payload["probabilities"]  # type: ignore[arg-type]
            ),
            estimate=PosteriorUtilityEstimate.from_payload(
                payload["estimate"]  # type: ignore[arg-type]
            ),
        )
        if "action_hash" in payload and str(payload["action_hash"]) != row.action_hash:
            raise ProtocolError("CBPUPR action candidate hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "case_id": self.case_id,
            "alternative_id": self.alternative_id,
            "direction": self.direction,
            "control_id": self.control_id,
            "probabilities": self.probabilities.to_payload(),
            "estimate": self.estimate.to_payload(),
            "action_hash": self.action_hash,
        }


@dataclass(frozen=True)
class EligibilityDecision:
    candidate_hash: str
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_hash or not self.reason_codes:
            raise ProtocolError("CBPUPR eligibility decision is incomplete.")
        if self.eligible != (self.reason_codes == (ELIGIBLE,)):
            raise ProtocolError("CBPUPR eligibility status and reasons disagree.")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EligibilityDecision":
        return cls(
            str(payload["candidate_hash"]),
            bool(payload["eligible"]),
            tuple(str(value) for value in payload["reason_codes"]),  # type: ignore[index]
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_hash": self.candidate_hash,
            "eligible": self.eligible,
            "reason_codes": list(self.reason_codes),
        }


def assess_action(
    candidate: ActionCandidate,
    *,
    tolerance: float = UTILITY_ZERO_TOLERANCE,
) -> EligibilityDecision:
    """Require positive expected BACC and nonnegative proper-score gains."""

    utility = candidate.estimate.utility
    reasons: list[str] = []
    if utility.bacc_gain <= float(tolerance):
        reasons.append(NONPOSITIVE_BACC)
    if utility.brier_gain < -float(tolerance):
        reasons.append(BRIER_UNSAFE)
    if utility.log_gain < -float(tolerance):
        reasons.append(LOG_UNSAFE)
    return EligibilityDecision(
        candidate.action_hash,
        not reasons,
        (ELIGIBLE,) if not reasons else tuple(reasons),
    )


def select_best_eligible_action(
    candidates: Sequence[ActionCandidate],
    *,
    tolerance: float = UTILITY_ZERO_TOLERANCE,
) -> ActionCandidate | None:
    """Maximise the same BACC estimand used by the authorization gate.

    Exact/tolerance ties resolve B before I before R, then the immutable action
    hash.  Direction is already hash-bound and is not a separate preference.
    """

    rows = tuple(candidates)
    if len({row.action_hash for row in rows}) != len(rows):
        raise ProtocolError("CBPUPR candidate rectangle contains duplicates.")
    eligible = tuple(row for row in rows if assess_action(row, tolerance=tolerance).eligible)
    if not eligible:
        return None
    maximum = max(row.estimate.utility.bacc_gain for row in eligible)
    tied = tuple(
        row
        for row in eligible
        if abs(row.estimate.utility.bacc_gain - maximum) <= float(tolerance)
    )
    alternative_rank = {value: index for index, value in enumerate(ALTERNATIVE_ORDER)}
    return min(
        tied,
        key=lambda row: (
            alternative_rank[row.alternative_id],
            row.action_hash,
        ),
    )


__all__ = (
    "ALTERNATIVE_ORDER",
    "ActionCandidate",
    "BRIER_UNSAFE",
    "DIRECTION_ORDER",
    "ELIGIBLE",
    "EligibilityDecision",
    "LOG_UNSAFE",
    "NONPOSITIVE_BACC",
    "UTILITY_ZERO_TOLERANCE",
    "assess_action",
    "select_best_eligible_action",
)
