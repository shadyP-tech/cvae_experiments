"""Exact-rational DTOs for the preserved nine-arm robust endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import ARM_IDS, DIRECTION_IDS, EXACT_TIE_TOLERANCE, K_GRID, W_FRACTION_GRID, arm_id, candidate_sources
from .hashing import canonical_hash


ROBUST_METHOD_IDS = ("R_NINE_ARM_ROBUST", "G_DIRECTIONAL_MATCHED")


@dataclass(frozen=True, order=True)
class EndpointArm:
    k: int
    weight_numerator: int
    weight_denominator: int
    arm_id: str = field(init=False, compare=True)
    arm_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        weight = Fraction(int(self.weight_numerator), int(self.weight_denominator))
        if int(self.k) not in K_GRID or weight not in W_FRACTION_GRID:
            raise ProtocolError("OGDE robust endpoint arm drifted.")
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "weight_numerator", weight.numerator)
        object.__setattr__(self, "weight_denominator", weight.denominator)
        object.__setattr__(self, "arm_id", arm_id(self.k, weight))
        object.__setattr__(self, "arm_hash", canonical_hash(self._unhashed()))

    @property
    def weight(self) -> Fraction:
        return Fraction(self.weight_numerator, self.weight_denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_robust_endpoint_arm_v1",
            "arm_id": self.arm_id,
            "k": self.k,
            "weight_numerator": self.weight_numerator,
            "weight_denominator": self.weight_denominator,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "arm_hash": self.arm_hash}


@dataclass(frozen=True, order=True)
class RobustCandidateScore:
    source: str | None
    support_numerator: int
    support_denominator: int
    donor_numerator: int
    donor_denominator: int
    score_numerator: int
    score_denominator: int
    score_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        support = Fraction(int(self.support_numerator), int(self.support_denominator))
        donor = Fraction(int(self.donor_numerator), int(self.donor_denominator))
        score = Fraction(int(self.score_numerator), int(self.score_denominator))
        if self.source is None and any(value != 0 for value in (support, donor, score)):
            raise ProtocolError("OGDE robust OFF candidate must be exact zero.")
        object.__setattr__(self, "support_numerator", support.numerator)
        object.__setattr__(self, "support_denominator", support.denominator)
        object.__setattr__(self, "donor_numerator", donor.numerator)
        object.__setattr__(self, "donor_denominator", donor.denominator)
        object.__setattr__(self, "score_numerator", score.numerator)
        object.__setattr__(self, "score_denominator", score.denominator)
        object.__setattr__(self, "score_hash", canonical_hash(self._unhashed()))

    @property
    def score(self) -> Fraction:
        return Fraction(self.score_numerator, self.score_denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_robust_candidate_score_v1",
            "source": self.source,
            "support_numerator": self.support_numerator,
            "support_denominator": self.support_denominator,
            "donor_numerator": self.donor_numerator,
            "donor_denominator": self.donor_denominator,
            "score_numerator": self.score_numerator,
            "score_denominator": self.score_denominator,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "score_hash": self.score_hash}

    @classmethod
    def from_fractions(cls, source: str | None, support: Fraction, donor: Fraction, score: Fraction) -> "RobustCandidateScore":
        return cls(
            source,
            support.numerator, support.denominator,
            donor.numerator, donor.denominator,
            score.numerator, score.denominator,
        )


@dataclass(frozen=True, order=True)
class DirectionRobustDecision:
    method_id: str
    target_center: str
    case_id: str
    arm_id: str
    direction: str
    retained_sources: tuple[str, ...]
    candidate_scores: tuple[RobustCandidateScore, ...]
    selected_source: str | None
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        retained = tuple(self.retained_sources)
        rows = tuple(self.candidate_scores)
        if (
            self.method_id not in ROBUST_METHOD_IDS
            or self.arm_id not in ARM_IDS
            or self.direction not in DIRECTION_IDS
            or not self.case_id
            or not retained
            or any(source not in candidate_sources(self.target_center) for source in retained)
            or len(retained) != len(set(retained))
            or tuple(row.source for row in rows) != (None, *retained)
        ):
            raise ProtocolError("OGDE robust direction decision topology drifted.")
        maximum = max(row.score for row in rows)
        eligible = tuple(row.source for row in rows if maximum - row.score <= EXACT_TIE_TOLERANCE)
        expected = min(eligible, key=lambda source: -1 if source is None else int(source))
        if self.selected_source != expected:
            raise ProtocolError("OGDE robust decision violates exact OFF-first selection.")
        object.__setattr__(self, "retained_sources", retained)
        object.__setattr__(self, "candidate_scores", rows)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_robust_direction_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "direction": self.direction,
            "retained_sources": list(self.retained_sources),
            "candidate_scores": [row.to_payload() for row in self.candidate_scores],
            "selected_source": self.selected_source,
            "tie_rule": "exact_fraction_OFF_then_numeric_with_1e-12_tolerance",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class RobustArmDecision:
    method_id: str
    target_center: str
    case_id: str
    arm_id: str
    zero_to_one: DirectionRobustDecision
    one_to_zero: DirectionRobustDecision
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        identity = (self.method_id, self.target_center, self.case_id, self.arm_id)
        if (
            (self.zero_to_one.method_id, self.zero_to_one.target_center, self.zero_to_one.case_id, self.zero_to_one.arm_id) != identity
            or (self.one_to_zero.method_id, self.one_to_zero.target_center, self.one_to_zero.case_id, self.one_to_zero.arm_id) != identity
            or self.zero_to_one.direction != "zero_to_one"
            or self.one_to_zero.direction != "one_to_zero"
        ):
            raise ProtocolError("OGDE robust paired arm decision drifted.")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def decision_for_baseline_class(self, baseline_class: int) -> DirectionRobustDecision:
        return self.zero_to_one if int(baseline_class) == 0 else self.one_to_zero

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_robust_arm_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "zero_to_one": self.zero_to_one.to_payload(),
            "one_to_zero": self.one_to_zero.to_payload(),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


__all__ = (
    "DirectionRobustDecision",
    "EndpointArm",
    "ROBUST_METHOD_IDS",
    "RobustArmDecision",
    "RobustCandidateScore",
)
