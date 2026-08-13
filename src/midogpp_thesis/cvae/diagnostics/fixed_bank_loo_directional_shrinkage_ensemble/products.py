"""Immutable scientific products for the DCSE diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Sequence

from ...protocol import ProtocolError
from .constants import (
    ARM_IDS,
    B_ACTION_ID,
    CENTERS,
    DIRECTION_IDS,
    HARD_THRESHOLD,
    OFF_ACTION_ID,
    PRE_TERMINAL_METHOD_IDS,
    U_ACTION_ID,
    a1_action_id,
    arm_id,
    candidate_sources,
)
from .hashing import canonical_hash, require_sha256


def _text(value: object, role: str) -> str:
    result = str(value)
    if not result:
        raise ProtocolError(f"DCSE {role} must be non-empty.")
    return result


def _integer(value: object, role: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"DCSE {role} must be an integer >= {minimum}.")
    return int(value)


def _finite(value: object, role: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"DCSE {role} must be finite.")
    return result


def _direction(value: object) -> str:
    result = str(value)
    if result not in DIRECTION_IDS:
        raise ProtocolError("DCSE direction must be zero_to_one or one_to_zero.")
    return result


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


@dataclass(frozen=True, order=True)
class CaseIdentity:
    target_center: str
    case_id: str
    sample_id: str

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS:
            raise ProtocolError("DCSE identity has an unknown center.")
        _text(self.case_id, "case_id")
        _text(self.sample_id, "sample_id")

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class BinaryLabel:
    target_center: str
    case_id: str
    sample_id: str
    value: int
    label_scope: str

    def __post_init__(self) -> None:
        CaseIdentity(self.target_center, self.case_id, self.sample_id)
        if isinstance(self.value, bool) or self.value not in (0, 1):
            raise ProtocolError("DCSE labels must be binary integers.")
        _text(self.label_scope, "label_scope")

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id

    @property
    def label(self) -> int:
        return self.value

    def to_payload(self) -> dict[str, object]:
        # Used only in ephemeral science tests. Runtime persistence must never
        # serialize this payload into the result bundle.
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class CaseActionConfusion:
    """Whole-case additive statistics, including both B-defined flip paths."""

    target_center: str
    case_id: str
    action_id: str
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    flip_0to1_positive: int
    flip_0to1_negative: int
    flip_1to0_positive: int
    flip_1to0_negative: int
    counts_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS or self.action_id not in {
            B_ACTION_ID,
            U_ACTION_ID,
            *(a1_action_id(source) for source in candidate_sources(self.target_center)),
        }:
            raise ProtocolError("DCSE case/action confusion identity drifted.")
        _text(self.case_id, "case_id")
        for role in (
            "n_positive",
            "true_positive",
            "n_negative",
            "true_negative",
            "flip_0to1_positive",
            "flip_0to1_negative",
            "flip_1to0_positive",
            "flip_1to0_negative",
        ):
            _integer(getattr(self, role), role)
        if self.n_positive + self.n_negative <= 0:
            raise ProtocolError("DCSE case/action confusion cannot be empty.")
        if self.true_positive > self.n_positive or self.true_negative > self.n_negative:
            raise ProtocolError("DCSE correct counts exceed class denominators.")
        if (
            self.flip_0to1_positive > self.n_positive
            or self.flip_1to0_positive > self.n_positive
            or self.flip_0to1_negative > self.n_negative
            or self.flip_1to0_negative > self.n_negative
        ):
            raise ProtocolError("DCSE directional flips exceed class counts.")
        if self.action_id == B_ACTION_ID and any(
            (
                self.flip_0to1_positive,
                self.flip_0to1_negative,
                self.flip_1to0_positive,
                self.flip_1to0_negative,
            )
        ):
            raise ProtocolError("Baseline B cannot flip relative to itself.")
        object.__setattr__(self, "counts_hash", canonical_hash(self._unhashed()))

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    @property
    def false_negative(self) -> int:
        return self.n_positive - self.true_positive

    @property
    def false_positive(self) -> int:
        return self.n_negative - self.true_negative

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_case_action_confusion_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "flip_0to1_positive": self.flip_0to1_positive,
            "flip_0to1_negative": self.flip_0to1_negative,
            "flip_1to0_positive": self.flip_1to0_positive,
            "flip_1to0_negative": self.flip_1to0_negative,
            "dtype": "int64",
            "additive_sufficient_statistics": True,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "counts_hash": self.counts_hash}


# Common spelling in existing fixed-bank diagnostics.
CaseActionCounts = CaseActionConfusion


@dataclass(frozen=True, order=True)
class DirectionalGain:
    """An exact pooled directional BACC contribution."""

    query_center: str
    excluded_case_id: str | None
    source: str
    direction: str
    n_positive: int
    n_negative: int
    favorable_count: int
    adverse_count: int
    contributing_case_ids: tuple[str, ...]
    label_scope: str
    numerator: int = field(init=False, compare=True)
    denominator: int = field(init=False, compare=True)
    value: float = field(init=False, compare=True)
    gain_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        query = str(self.query_center)
        source = str(self.source)
        direction = _direction(self.direction)
        if query not in CENTERS or source not in candidate_sources(query):
            raise ProtocolError("DCSE directional gain violates query/source exclusion.")
        n_positive = _integer(self.n_positive, "n_positive", minimum=1)
        n_negative = _integer(self.n_negative, "n_negative", minimum=1)
        favorable = _integer(self.favorable_count, "favorable_count")
        adverse = _integer(self.adverse_count, "adverse_count")
        if direction == "zero_to_one":
            if favorable > n_positive or adverse > n_negative:
                raise ProtocolError("DCSE zero-to-one counts exceed their label classes.")
            fraction = Fraction(favorable, 2 * n_positive) - Fraction(adverse, 2 * n_negative)
        else:
            if favorable > n_negative or adverse > n_positive:
                raise ProtocolError("DCSE one-to-zero counts exceed their label classes.")
            fraction = Fraction(favorable, 2 * n_negative) - Fraction(adverse, 2 * n_positive)
        cases = tuple(sorted(str(case_id) for case_id in self.contributing_case_ids))
        if not cases or len(cases) != len(set(cases)):
            raise ProtocolError("DCSE directional gain needs unique contributing whole cases.")
        excluded = None if self.excluded_case_id is None else _text(self.excluded_case_id, "excluded_case_id")
        if excluded is not None and excluded in cases:
            raise ProtocolError("Held case c entered its own H-minus-c directional gain.")
        _text(self.label_scope, "label_scope")
        object.__setattr__(self, "query_center", query)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "excluded_case_id", excluded)
        object.__setattr__(self, "contributing_case_ids", cases)
        object.__setattr__(self, "numerator", fraction.numerator)
        object.__setattr__(self, "denominator", fraction.denominator)
        object.__setattr__(self, "value", float(fraction))
        object.__setattr__(self, "gain_hash", canonical_hash(self._unhashed()))

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_directional_gain_v1",
            "query_center": self.query_center,
            "excluded_case_id": self.excluded_case_id,
            "source": self.source,
            "direction": self.direction,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "favorable_count": self.favorable_count,
            "adverse_count": self.adverse_count,
            "contributing_case_ids": list(self.contributing_case_ids),
            "label_scope": self.label_scope,
            "exact_fraction": [self.numerator, self.denominator],
            "value": self.value,
            "pooled_additive_counts": True,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "gain_hash": self.gain_hash}


@dataclass(frozen=True, order=True)
class DonorPrior:
    heldout_center: str
    source: str
    direction: str
    query_gains: tuple[DirectionalGain, ...]
    numerator: int = field(init=False, compare=True)
    denominator: int = field(init=False, compare=True)
    value: float = field(init=False, compare=True)
    prior_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        heldout = str(self.heldout_center)
        source = str(self.source)
        direction = _direction(self.direction)
        if heldout not in CENTERS or source not in candidate_sources(heldout):
            raise ProtocolError("DCSE donor prior includes the held-out target expert.")
        gains = tuple(self.query_gains)
        expected_queries = tuple(center for center in CENTERS if center not in {heldout, source})
        if tuple(row.query_center for row in gains) != expected_queries:
            raise ProtocolError("DCSE donor G must use all and only q outside {H,e} in center order.")
        if any(
            row.source != source
            or row.direction != direction
            or row.excluded_case_id is not None
            for row in gains
        ):
            raise ProtocolError("DCSE donor G query gains drifted from H/e/direction.")
        exact = sum((row.exact for row in gains), Fraction(0, 1)) / len(gains)
        object.__setattr__(self, "heldout_center", heldout)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "query_gains", gains)
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "value", float(exact))
        object.__setattr__(self, "prior_hash", canonical_hash(self._unhashed()))

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_equal_center_donor_prior_v1",
            "heldout_center": self.heldout_center,
            "source": self.source,
            "direction": self.direction,
            "query_gains": [row.to_payload() for row in self.query_gains],
            "query_centers": [row.query_center for row in self.query_gains],
            "exact_fraction": [self.numerator, self.denominator],
            "value": self.value,
            "query_center_weighting": "equal",
            "q_not_in_H_e": True,
            "H_labels_used": False,
            "target_expert_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prior_hash": self.prior_hash}


@dataclass(frozen=True, order=True)
class EndpointArm:
    k: int
    weight_numerator: int
    weight_denominator: int
    arm_id: str = field(init=False, compare=True)
    arm_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        fraction = Fraction(self.weight_numerator, self.weight_denominator)
        identity = arm_id(int(self.k), fraction)
        object.__setattr__(self, "arm_id", identity)
        object.__setattr__(self, "weight_numerator", fraction.numerator)
        object.__setattr__(self, "weight_denominator", fraction.denominator)
        object.__setattr__(self, "arm_hash", canonical_hash(self._unhashed()))

    @property
    def weight(self) -> Fraction:
        return Fraction(self.weight_numerator, self.weight_denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_endpoint_arm_v1",
            "arm_id": self.arm_id,
            "K": self.k,
            "weight": [self.weight_numerator, self.weight_denominator],
            "OFF_score": [0, 1],
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "arm_hash": self.arm_hash}


@dataclass(frozen=True, order=True)
class CandidateScore:
    source: str | None
    support_numerator: int
    support_denominator: int
    prior_numerator: int
    prior_denominator: int
    score_numerator: int
    score_denominator: int

    def __post_init__(self) -> None:
        support = Fraction(self.support_numerator, self.support_denominator)
        prior = Fraction(self.prior_numerator, self.prior_denominator)
        score = Fraction(self.score_numerator, self.score_denominator)
        source = None if self.source is None else str(self.source)
        if source is None and (support or prior or score):
            raise ProtocolError("DCSE OFF must have exact zero support/prior/score.")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "support_numerator", support.numerator)
        object.__setattr__(self, "support_denominator", support.denominator)
        object.__setattr__(self, "prior_numerator", prior.numerator)
        object.__setattr__(self, "prior_denominator", prior.denominator)
        object.__setattr__(self, "score_numerator", score.numerator)
        object.__setattr__(self, "score_denominator", score.denominator)

    @classmethod
    def from_fractions(
        cls,
        source: str | None,
        support: Fraction,
        prior: Fraction,
        score: Fraction,
    ) -> "CandidateScore":
        return cls(
            source,
            support.numerator,
            support.denominator,
            prior.numerator,
            prior.denominator,
            score.numerator,
            score.denominator,
        )

    @property
    def support(self) -> Fraction:
        return Fraction(self.support_numerator, self.support_denominator)

    @property
    def prior(self) -> Fraction:
        return Fraction(self.prior_numerator, self.prior_denominator)

    @property
    def score(self) -> Fraction:
        return Fraction(self.score_numerator, self.score_denominator)

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "action_id": OFF_ACTION_ID if self.source is None else a1_action_id(self.source),
            "support_fraction": _fraction_payload(self.support),
            "prior_fraction": _fraction_payload(self.prior),
            "score_fraction": _fraction_payload(self.score),
            "support_value": float(self.support),
            "prior_value": float(self.prior),
            "score_value": float(self.score),
        }


@dataclass(frozen=True, order=True)
class DirectionDecision:
    method_id: str
    target_center: str
    case_id: str
    arm_id: str
    direction: str
    retained_sources: tuple[str, ...]
    scores: tuple[CandidateScore, ...]
    selected_source: str | None
    decision_rule: str
    decision_hash: str = field(init=False, compare=True)
    terminal_labels_used: bool = False

    def __post_init__(self) -> None:
        if self.method_id not in PRE_TERMINAL_METHOD_IDS:
            raise ProtocolError("DCSE direction decision method is not pre-terminal.")
        if self.target_center not in CENTERS or self.arm_id not in ARM_IDS:
            raise ProtocolError("DCSE direction decision target/arm drifted.")
        _text(self.case_id, "case_id")
        direction = _direction(self.direction)
        retained = tuple(str(source) for source in self.retained_sources)
        if not retained or len(retained) != len(set(retained)) or any(
            source not in candidate_sources(self.target_center) for source in retained
        ):
            raise ProtocolError("DCSE retained source menu is invalid.")
        scores = tuple(self.scores)
        if tuple(row.source for row in scores) != (None, *retained):
            raise ProtocolError("DCSE score rows must be OFF followed by the retained source order.")
        selected = None if self.selected_source is None else str(self.selected_source)
        if selected not in (None, *retained) or self.terminal_labels_used is not False:
            raise ProtocolError("DCSE selected source escaped the pre-terminal menu.")
        _text(self.decision_rule, "decision_rule")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "retained_sources", retained)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    @property
    def selected_action_id(self) -> str:
        return OFF_ACTION_ID if self.selected_source is None else a1_action_id(self.selected_source)

    @property
    def selected_score(self) -> Fraction:
        return next(row.score for row in self.scores if row.source == self.selected_source)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_direction_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "direction": self.direction,
            "retained_sources": list(self.retained_sources),
            "scores": [row.to_payload() for row in self.scores],
            "selected_source": self.selected_source,
            "selected_action_id": self.selected_action_id,
            "decision_rule": self.decision_rule,
            "OFF_precedes_numeric_source_ties": True,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class CaseArmDecision:
    method_id: str
    target_center: str
    case_id: str
    arm_id: str
    zero_to_one: DirectionDecision
    one_to_zero: DirectionDecision
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        rows = (self.zero_to_one, self.one_to_zero)
        if tuple(row.direction for row in rows) != DIRECTION_IDS or any(
            (row.method_id, row.target_center, row.case_id, row.arm_id)
            != (self.method_id, self.target_center, self.case_id, self.arm_id)
            for row in rows
        ):
            raise ProtocolError("DCSE case/arm directional decision identities drifted.")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def decision_for_baseline_class(self, baseline_class: int) -> DirectionDecision:
        if isinstance(baseline_class, bool) or baseline_class not in (0, 1):
            raise ProtocolError("DCSE branch requires a binary B hard class.")
        return self.zero_to_one if baseline_class == 0 else self.one_to_zero

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_case_arm_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "arm_id": self.arm_id,
            "directions": [self.zero_to_one.to_payload(), self.one_to_zero.to_payload()],
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class DirectionalControlDecision:
    method_id: str
    target_center: str
    case_id: str
    direction: str
    selected_source: str | None
    candidate_values: tuple[tuple[str | None, int, int], ...]
    nested_support_case_count: int
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.method_id not in {"DLOO_raw", "LOO_frequency_committee"}:
            raise ProtocolError("DCSE directional control method drifted.")
        if self.target_center not in CENTERS:
            raise ProtocolError("DCSE directional control target drifted.")
        _text(self.case_id, "case_id")
        direction = _direction(self.direction)
        rows = tuple(self.candidate_values)
        expected_sources = (None, *candidate_sources(self.target_center))
        if tuple(source for source, _n, _d in rows) != expected_sources:
            raise ProtocolError("DCSE directional control lacks OFF plus eight sources.")
        fractions = tuple(Fraction(numerator, denominator) for _s, numerator, denominator in rows)
        selected = None if self.selected_source is None else str(self.selected_source)
        if selected not in expected_sources:
            raise ProtocolError("DCSE directional control selected an illegal source.")
        nested_count = _integer(self.nested_support_case_count, "nested_support_case_count")
        if self.method_id == "DLOO_raw":
            if nested_count != 0 or fractions[0] != 0:
                raise ProtocolError("Raw DLOO must carry OFF=0 and no nested deletions.")
        else:
            if (
                nested_count <= 0
                or any(value < 0 for value in fractions)
                or sum(fractions, Fraction(0)) != 1
            ):
                raise ProtocolError(
                    "Frequency committee needs exact nonnegative nested-vote frequencies summing to one."
                )
            maximum = max(fractions)
            modal = next(
                source
                for (source, _n, _d), value in zip(rows, fractions, strict=True)
                if value == maximum
            )
            if selected != modal:
                raise ProtocolError(
                    "Frequency committee modal stability metadata drifted from its vote frequencies."
                )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "candidate_values", rows)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    @property
    def selected_value(self) -> Fraction:
        return next(
            Fraction(numerator, denominator)
            for source, numerator, denominator in self.candidate_values
            if source == self.selected_source
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_directional_control_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "selected_source": self.selected_source,
            "selected_action_id": (
                OFF_ACTION_ID
                if self.selected_source is None
                else a1_action_id(self.selected_source)
            ),
            "candidate_values": [
                [source, numerator, denominator]
                for source, numerator, denominator in self.candidate_values
            ],
            "nested_support_case_count": self.nested_support_case_count,
            "OFF_then_numeric_source_tie": True,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class CaseControlDecision:
    method_id: str
    target_center: str
    case_id: str
    zero_to_one: DirectionalControlDecision
    one_to_zero: DirectionalControlDecision
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        rows = (self.zero_to_one, self.one_to_zero)
        if tuple(row.direction for row in rows) != DIRECTION_IDS or any(
            (row.method_id, row.target_center, row.case_id)
            != (self.method_id, self.target_center, self.case_id)
            for row in rows
        ):
            raise ProtocolError("DCSE case control directional identities drifted.")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def decision_for_baseline_class(self, baseline_class: int) -> DirectionalControlDecision:
        if isinstance(baseline_class, bool) or baseline_class not in (0, 1):
            raise ProtocolError("DCSE control branch requires a binary B class.")
        return self.zero_to_one if baseline_class == 0 else self.one_to_zero

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_case_control_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "directions": [self.zero_to_one.to_payload(), self.one_to_zero.to_payload()],
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class MethodPrediction:
    target_center: str
    case_id: str
    sample_id: str
    method_id: str
    probability: float
    baseline_hard_class: int
    selected_sources_by_arm: tuple[str | None, ...]
    probability_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        CaseIdentity(self.target_center, self.case_id, self.sample_id)
        _text(self.method_id, "method_id")
        probability = _finite(self.probability, "prediction probability")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("DCSE prediction probability lies outside [0,1].")
        if isinstance(self.baseline_hard_class, bool) or self.baseline_hard_class not in (0, 1):
            raise ProtocolError("DCSE prediction needs a binary B hard class.")
        selections = tuple(None if value is None else str(value) for value in self.selected_sources_by_arm)
        if self.method_id in {"DCSE_LOO", "G_directional_matched"} and len(selections) != len(ARM_IDS):
            raise ProtocolError("DCSE endpoint prediction must preserve all nine arm selections.")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "selected_sources_by_arm", selections)
        object.__setattr__(self, "probability_hash", canonical_hash(self._unhashed()))

    @property
    def hard_prediction(self) -> int:
        return int(self.probability >= HARD_THRESHOLD)

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_method_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "method_id": self.method_id,
            "probability": self.probability,
            "hard_prediction": self.hard_prediction,
            "baseline_hard_class": self.baseline_hard_class,
            "selected_sources_by_arm": list(self.selected_sources_by_arm),
            "duplicate_arm_selections_preserved": True,
            "sole_threshold": HARD_THRESHOLD,
            "threshold_tie_class": 1,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "probability_hash": self.probability_hash}


@dataclass(frozen=True, order=True)
class CaseMethodConfusion:
    target_center: str
    case_id: str
    method_id: str
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS:
            raise ProtocolError("DCSE terminal confusion has an unknown center.")
        _text(self.case_id, "case_id")
        _text(self.method_id, "method_id")
        for role in ("true_positive", "true_negative", "false_positive", "false_negative"):
            _integer(getattr(self, role), role)
        if self.n_positive + self.n_negative <= 0:
            raise ProtocolError("DCSE terminal confusion cannot be empty.")

    @property
    def n_positive(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def n_negative(self) -> int:
        return self.true_negative + self.false_positive

    @property
    def case_key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_terminal_case_confusion_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "method_id": self.method_id,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "dtype": "int64",
            "per_case_bacc_persisted": False,
        }


@dataclass(frozen=True, order=True)
class PooledBacc:
    scope_id: str
    method_id: str
    case_count: int
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    sensitivity: float
    specificity: float
    bacc: float

    def __post_init__(self) -> None:
        _text(self.scope_id, "scope_id")
        _text(self.method_id, "method_id")
        _integer(self.case_count, "case_count", minimum=1)
        n_positive = _integer(self.n_positive, "n_positive", minimum=1)
        n_negative = _integer(self.n_negative, "n_negative", minimum=1)
        if not 0 <= self.true_positive <= n_positive or not 0 <= self.true_negative <= n_negative:
            raise ProtocolError("DCSE pooled BACC counts are invalid.")
        sensitivity = self.true_positive / n_positive
        specificity = self.true_negative / n_negative
        bacc = Fraction(self.true_positive, 2 * n_positive) + Fraction(
            self.true_negative, 2 * n_negative
        )
        if (
            abs(_finite(self.sensitivity, "sensitivity") - sensitivity) > 1.0e-15
            or abs(_finite(self.specificity, "specificity") - specificity) > 1.0e-15
            or abs(_finite(self.bacc, "bacc") - float(bacc)) > 1.0e-15
        ):
            raise ProtocolError("DCSE pooled BACC differs from its exact counts.")

    @property
    def exact(self) -> Fraction:
        return Fraction(self.true_positive, 2 * self.n_positive) + Fraction(
            self.true_negative, 2 * self.n_negative
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_pooled_bacc_v1",
            **self.__dict__,
            "exact_fraction": _fraction_payload(self.exact),
            "confusion_dtype": "int64",
            "reduction_dtype": "float64",
            "per_case_bacc_used": False,
        }


@dataclass(frozen=True, order=True)
class EqualCenterContrast:
    contrast_id: str
    method_id: str
    reference_id: str
    center_differences: tuple[tuple[str, int, int], ...]
    numerator: int = field(init=False, compare=True)
    denominator: int = field(init=False, compare=True)
    estimate: float = field(init=False, compare=True)

    def __post_init__(self) -> None:
        rows = tuple(self.center_differences)
        if tuple(center for center, _n, _d in rows) != CENTERS:
            raise ProtocolError("DCSE equal-center contrast must cover all nine centers in order.")
        fractions = tuple(Fraction(numerator, denominator) for _center, numerator, denominator in rows)
        exact = sum(fractions, Fraction(0, 1)) / len(fractions)
        object.__setattr__(self, "center_differences", rows)
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "estimate", float(exact))

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_equal_center_contrast_v1",
            "contrast_id": self.contrast_id,
            "method_id": self.method_id,
            "reference_id": self.reference_id,
            "center_differences": [
                [center, numerator, denominator]
                for center, numerator, denominator in self.center_differences
            ],
            "exact_fraction": _fraction_payload(self.exact),
            "estimate": self.estimate,
            "center_weighting": "equal",
        }


def decision_bundle_hash(rows: Sequence[CaseArmDecision]) -> str:
    values = tuple(rows)
    if not values:
        raise ProtocolError("DCSE cannot seal an empty decision bundle.")
    return canonical_hash(
        {
            "schema_version": "fixed_bank_dcse_decision_bundle_v1",
            "decisions": [row.to_payload() for row in values],
            "terminal_labels_used": False,
        }
    )


__all__ = (
    "BinaryLabel",
    "CandidateScore",
    "CaseActionConfusion",
    "CaseActionCounts",
    "CaseArmDecision",
    "CaseControlDecision",
    "CaseIdentity",
    "CaseMethodConfusion",
    "DirectionalGain",
    "DirectionalControlDecision",
    "DirectionDecision",
    "DonorPrior",
    "EndpointArm",
    "EqualCenterContrast",
    "MethodPrediction",
    "PooledBacc",
    "decision_bundle_hash",
)
