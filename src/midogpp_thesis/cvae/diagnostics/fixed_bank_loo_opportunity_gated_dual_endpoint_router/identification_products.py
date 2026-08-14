"""Persistable identification scores and OFF-first directional decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .constants import DIRECTION_IDS, TIE_TOLERANCE, candidate_sources
from .hashing import canonical_hash, require_sha256


IDENTIFICATION_METHOD_IDS = ("I_OPPORTUNITY_GATED", "I_FEATURE_BLOCK_PERMUTED")


@dataclass(frozen=True, order=True)
class IdentificationCandidateScore:
    target_center: str
    case_id: str
    direction: str
    source: str
    predicted_correctness: float | None
    directional_flip_count: int
    model_valid: bool
    case_proxy: float
    donor_prior: float
    case_scale: float
    donor_scale: float
    normalized_case_proxy: float
    normalized_donor_prior: float
    final_score: float
    opportunity_eligible: bool
    eligible: bool
    eligibility_reason: str
    model_hash: str
    score_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        predicted = None if self.predicted_correctness is None else float(self.predicted_correctness)
        values = tuple(
            float(value)
            for value in (
                self.case_proxy,
                self.donor_prior,
                self.case_scale,
                self.donor_scale,
                self.normalized_case_proxy,
                self.normalized_donor_prior,
                self.final_score,
            )
        )
        if (
            self.source not in candidate_sources(self.target_center)
            or self.direction not in DIRECTION_IDS
            or not self.case_id
            or (predicted is not None and (not math.isfinite(predicted) or not 0 <= predicted <= 1))
            or not all(math.isfinite(value) for value in values)
            or int(self.directional_flip_count) < 0
            or values[2] < 0
            or values[3] < 0
            or self.eligible and not self.opportunity_eligible
            or self.eligible != (self.opportunity_eligible and values[0] > 0.0)
            or not self.eligibility_reason
        ):
            raise ProtocolError("OGDE identification candidate score drifted.")
        object.__setattr__(self, "predicted_correctness", predicted)
        object.__setattr__(self, "directional_flip_count", int(self.directional_flip_count))
        for name, value in zip(
            (
                "case_proxy", "donor_prior", "case_scale", "donor_scale",
                "normalized_case_proxy", "normalized_donor_prior", "final_score",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "model_hash", require_sha256(self.model_hash, "model_hash"))
        object.__setattr__(self, "score_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_identification_candidate_score_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "source": self.source,
            "predicted_correctness": self.predicted_correctness,
            "directional_flip_count": self.directional_flip_count,
            "model_valid": bool(self.model_valid),
            "case_proxy": self.case_proxy,
            "donor_prior": self.donor_prior,
            "case_scale": self.case_scale,
            "donor_scale": self.donor_scale,
            "normalized_case_proxy": self.normalized_case_proxy,
            "normalized_donor_prior": self.normalized_donor_prior,
            "final_score": self.final_score,
            "opportunity_eligible": bool(self.opportunity_eligible),
            "eligible": bool(self.eligible),
            "eligibility_reason": self.eligibility_reason,
            "model_hash": self.model_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "score_hash": self.score_hash}


@dataclass(frozen=True, order=True)
class DirectionIdentificationDecision:
    method_id: str
    target_center: str
    case_id: str
    direction: str
    candidate_scores: tuple[IdentificationCandidateScore, ...]
    selected_source: str | None
    source_only_selected_source: str | None
    eligible_sources: tuple[str, ...]
    case_scale: float
    donor_scale: float
    fail_closed: bool
    decision_reason: str
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        rows = tuple(self.candidate_scores)
        expected_sources = candidate_sources(self.target_center)
        if (
            self.method_id not in IDENTIFICATION_METHOD_IDS
            or self.direction not in DIRECTION_IDS
            or tuple(row.source for row in rows) != expected_sources
            or any((row.target_center, row.case_id, row.direction) != (self.target_center, self.case_id, self.direction) for row in rows)
            or tuple(self.eligible_sources) != tuple(row.source for row in rows if row.eligible)
            or not math.isfinite(float(self.case_scale))
            or not math.isfinite(float(self.donor_scale))
            or not self.decision_reason
        ):
            raise ProtocolError("OGDE identification decision topology drifted.")
        eligible = tuple(row for row in rows if row.eligible)
        if self.fail_closed or not eligible or max(row.final_score for row in eligible) <= TIE_TOLERANCE:
            expected_selected = None
        else:
            maximum = max(row.final_score for row in eligible)
            expected_selected = min(
                (row.source for row in eligible if maximum - row.final_score <= TIE_TOLERANCE),
                key=int,
            )
        opportunity = tuple(row for row in rows if row.opportunity_eligible)
        if not opportunity:
            expected_source_only = None
        else:
            maximum = max(row.final_score for row in opportunity)
            expected_source_only = min(
                (row.source for row in opportunity if maximum - row.final_score <= TIE_TOLERANCE),
                key=int,
            )
        if self.selected_source != expected_selected or self.source_only_selected_source != expected_source_only:
            raise ProtocolError("OGDE identification decision violates frozen OFF/source selection.")
        object.__setattr__(self, "candidate_scores", rows)
        object.__setattr__(self, "eligible_sources", tuple(self.eligible_sources))
        object.__setattr__(self, "case_scale", float(self.case_scale))
        object.__setattr__(self, "donor_scale", float(self.donor_scale))
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_direction_identification_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "candidate_scores": [row.to_payload() for row in self.candidate_scores],
            "selected_source": self.selected_source,
            "source_only_selected_source": self.source_only_selected_source,
            "eligible_sources": list(self.eligible_sources),
            "case_scale": self.case_scale,
            "donor_scale": self.donor_scale,
            "fail_closed": bool(self.fail_closed),
            "decision_reason": self.decision_reason,
            "tie_rule": "OFF_then_numeric_source_with_1e-12_tolerance",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


@dataclass(frozen=True, order=True)
class CaseIdentificationDecision:
    method_id: str
    target_center: str
    case_id: str
    zero_to_one: DirectionIdentificationDecision
    one_to_zero: DirectionIdentificationDecision
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        identity = (self.method_id, self.target_center, self.case_id)
        if (
            (self.zero_to_one.method_id, self.zero_to_one.target_center, self.zero_to_one.case_id) != identity
            or (self.one_to_zero.method_id, self.one_to_zero.target_center, self.one_to_zero.case_id) != identity
            or self.zero_to_one.direction != "zero_to_one"
            or self.one_to_zero.direction != "one_to_zero"
        ):
            raise ProtocolError("OGDE paired identification decision drifted.")
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.method_id, self.target_center, self.case_id

    def decision_for_baseline_class(self, baseline_class: int) -> DirectionIdentificationDecision:
        return self.zero_to_one if int(baseline_class) == 0 else self.one_to_zero

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_case_identification_decision_v1",
            "method_id": self.method_id,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "zero_to_one": self.zero_to_one.to_payload(),
            "one_to_zero": self.one_to_zero.to_payload(),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


__all__ = (
    "CaseIdentificationDecision",
    "DirectionIdentificationDecision",
    "IDENTIFICATION_METHOD_IDS",
    "IdentificationCandidateScore",
)
