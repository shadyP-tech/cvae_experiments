"""Selection-only candidate-set posterior tournament against exact B.

The historical expert ranking is used only as a deterministic tie/audit prior.
The local effect is the paired target-support outcome relative to exact B.  A
zero-centered empirical-Bayes shrinkage factor reduces small-support effects;
it does not manufacture an unavailable source-inner B advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
    FamilyOutcome,
)
from ..fixed_bank_sceptre_router.partitions import ThreeRoleFold
from .development import (
    FrozenRoutingContext,
    SUPPORT_MINIMUM_SHRUNK_BACC_GAIN,
    SUPPORT_PRIOR_EFFECTIVE_CASES,
)
from .proposal_set import FrozenCandidateSetProposal


TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class CandidateSupportAssessment:
    candidate_center: str
    proposal_rank: int
    candidate_outcome_hash: str
    exact_b_outcome_hash: str
    case_count: int
    shrinkage_weight: float
    raw_bacc_gain: float
    raw_brier_delta: float
    raw_log_loss_delta: float
    shrunk_bacc_gain: float
    shrunk_brier_delta: float
    shrunk_log_loss_delta: float
    eligible: bool
    assessment_hash: str = ""

    def __post_init__(self) -> None:
        candidate = str(self.candidate_center)
        values = (
            self.shrinkage_weight,
            self.raw_bacc_gain,
            self.raw_brier_delta,
            self.raw_log_loss_delta,
            self.shrunk_bacc_gain,
            self.shrunk_brier_delta,
            self.shrunk_log_loss_delta,
        )
        if (
            not candidate
            or isinstance(self.proposal_rank, bool)
            or self.proposal_rank <= 0
            or isinstance(self.case_count, bool)
            or self.case_count <= 0
            or not 0.0 < float(self.shrinkage_weight) < 1.0
            or any(not math.isfinite(float(value)) for value in values)
            or not isinstance(self.eligible, bool)
        ):
            raise ProtocolError("SCEPTRE v4 support assessment is invalid.")
        for value, role in (
            (self.candidate_outcome_hash, "candidate outcome"),
            (self.exact_b_outcome_hash, "exact-B outcome"),
        ):
            require_sha256(value, role)
        weight = float(self.shrinkage_weight)
        expected_shrunk = tuple(
            weight * float(value)
            for value in (
                self.raw_bacc_gain,
                self.raw_brier_delta,
                self.raw_log_loss_delta,
            )
        )
        observed_shrunk = (
            float(self.shrunk_bacc_gain),
            float(self.shrunk_brier_delta),
            float(self.shrunk_log_loss_delta),
        )
        if observed_shrunk != expected_shrunk:
            raise ProtocolError("SCEPTRE v4 support shrinkage does not replay.")
        expected_eligible = (
            observed_shrunk[0] > SUPPORT_MINIMUM_SHRUNK_BACC_GAIN
        )
        if self.eligible is not expected_eligible:
            raise ProtocolError("SCEPTRE v4 support eligibility drifted.")
        body = self._payload_without_hash(candidate, weight)
        expected_hash = canonical_hash(body)
        if self.assessment_hash and self.assessment_hash != expected_hash:
            raise ProtocolError("SCEPTRE v4 support assessment hash drifted.")
        object.__setattr__(self, "candidate_center", candidate)
        object.__setattr__(self, "shrinkage_weight", weight)
        object.__setattr__(self, "assessment_hash", expected_hash)

    def _payload_without_hash(
        self, candidate: str | None = None, weight: float | None = None
    ) -> dict[str, object]:
        return {
            "schema_version": "sceptre_v4_candidate_support_assessment_v1",
            "candidate_center": (
                self.candidate_center if candidate is None else candidate
            ),
            "proposal_rank": self.proposal_rank,
            "candidate_outcome_hash": self.candidate_outcome_hash,
            "exact_b_outcome_hash": self.exact_b_outcome_hash,
            "case_count": self.case_count,
            "zero_anchor_prior_effective_cases": SUPPORT_PRIOR_EFFECTIVE_CASES,
            "shrinkage_weight": (
                self.shrinkage_weight if weight is None else weight
            ),
            "raw_delta_vs_exact_b": {
                "bacc": self.raw_bacc_gain,
                "brier": self.raw_brier_delta,
                "log_loss": self.raw_log_loss_delta,
            },
            "shrunk_delta_vs_exact_b": {
                "bacc": self.shrunk_bacc_gain,
                "brier": self.shrunk_brier_delta,
                "log_loss": self.shrunk_log_loss_delta,
            },
            "eligible": self.eligible,
            "selection_objective": "MAXIMUM_POSITIVE_SHRUNK_BACC_GAIN",
            "proper_losses_are_CALIBRATION_safety_gates": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "assessment_hash": self.assessment_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidateSupportAssessment":
        try:
            raw = payload["raw_delta_vs_exact_b"]
            shrunk = payload["shrunk_delta_vs_exact_b"]
            if not isinstance(raw, Mapping) or not isinstance(shrunk, Mapping):
                raise TypeError("delta mappings")
            value = cls(
                candidate_center=str(payload["candidate_center"]),
                proposal_rank=int(payload["proposal_rank"]),
                candidate_outcome_hash=str(payload["candidate_outcome_hash"]),
                exact_b_outcome_hash=str(payload["exact_b_outcome_hash"]),
                case_count=int(payload["case_count"]),
                shrinkage_weight=float(payload["shrinkage_weight"]),
                raw_bacc_gain=float(raw["bacc"]),
                raw_brier_delta=float(raw["brier"]),
                raw_log_loss_delta=float(raw["log_loss"]),
                shrunk_bacc_gain=float(shrunk["bacc"]),
                shrunk_brier_delta=float(shrunk["brier"]),
                shrunk_log_loss_delta=float(shrunk["log_loss"]),
                eligible=payload["eligible"],
                assessment_hash=str(payload["assessment_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE v4 support assessment payload is malformed.") from exc
        if value.to_payload() != dict(payload):
            raise ProtocolError("SCEPTRE v4 support assessment payload drifted.")
        return value


@dataclass(frozen=True, slots=True)
class SupportPosteriorDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    partition_hash: str
    selection_case_set_hash: str
    calibration_case_set_hash: str
    evaluation_case_set_hash: str
    routing_context_hash: str
    proposal_set_hash: str
    candidate_menu_hash: str
    exact_b_control_receipt_hash: str
    assessments: tuple[CandidateSupportAssessment, ...]
    selected_candidate: str | None
    route: str
    fallback_required: bool
    reason: str
    decision_hash: str = ""

    def __post_init__(self) -> None:
        target = str(self.target_center)
        expected_sources = legal_routing_sources(target)
        rows = tuple(self.assessments)
        if (
            tuple(row.candidate_center for row in rows) != expected_sources
            or tuple(row.proposal_rank for row in sorted(rows, key=lambda row: row.proposal_rank))
            != tuple(range(1, len(expected_sources) + 1))
        ):
            raise ProtocolError("SCEPTRE v4 support assessment inventory drifted.")
        for value, role in (
            (self.fold_hash, "fold"),
            (self.partition_hash, "partition"),
            (self.selection_case_set_hash, "selection case set"),
            (self.calibration_case_set_hash, "calibration case set"),
            (self.evaluation_case_set_hash, "evaluation case set"),
            (self.routing_context_hash, "routing context"),
            (self.proposal_set_hash, "proposal set"),
        ):
            require_sha256(value, role)
        if len(
            {
                self.selection_case_set_hash,
                self.calibration_case_set_hash,
                self.evaluation_case_set_hash,
            }
        ) != 3:
            raise ProtocolError("SCEPTRE v4 support roles overlap.")
        eligible = tuple(row for row in rows if row.eligible)
        expected_selected: str | None = None
        expected_reason = "NO_POSITIVE_SHRUNK_SUPPORT_GAIN_FALLBACK_TO_B"
        if eligible:
            maximum = max(row.shrunk_bacc_gain for row in eligible)
            tied = tuple(
                row
                for row in eligible
                if maximum - row.shrunk_bacc_gain <= TIE_TOLERANCE
            )
            if len(tied) == 1:
                expected_selected = tied[0].candidate_center
                expected_reason = "UNIQUE_MAXIMUM_POSITIVE_SHRUNK_SUPPORT_GAIN"
            else:
                expected_reason = "SUPPORT_POSTERIOR_TIE_FALLBACK_TO_B"
        expected_route = (
            EXACT_B_CANDIDATE
            if expected_selected is None
            else expected_selected
        )
        if (
            self.selected_candidate != expected_selected
            or self.route != expected_route
            or not isinstance(self.fallback_required, bool)
            or self.fallback_required is not (expected_selected is None)
            or self.reason != expected_reason
        ):
            raise ProtocolError("SCEPTRE v4 support route semantics drifted.")
        body = self._payload_without_hash(target, rows)
        expected_hash = canonical_hash(body)
        if self.decision_hash and self.decision_hash != expected_hash:
            raise ProtocolError("SCEPTRE v4 support decision hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "assessments", rows)
        object.__setattr__(self, "decision_hash", expected_hash)

    def _payload_without_hash(
        self,
        target: str | None = None,
        rows: tuple[CandidateSupportAssessment, ...] | None = None,
    ) -> dict[str, object]:
        assessments = self.assessments if rows is None else rows
        return {
            "schema_version": "sceptre_v4_support_posterior_decision_v1",
            "target_center": self.target_center if target is None else target,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "partition_hash": self.partition_hash,
            "selection_case_set_hash": self.selection_case_set_hash,
            "calibration_case_set_hash": self.calibration_case_set_hash,
            "evaluation_case_set_hash": self.evaluation_case_set_hash,
            "routing_context_hash": self.routing_context_hash,
            "proposal_set_hash": self.proposal_set_hash,
            "candidate_menu_hash": self.candidate_menu_hash,
            "exact_b_control_receipt_hash": self.exact_b_control_receipt_hash,
            "assessment_hashes": [row.assessment_hash for row in assessments],
            "selected_candidate": self.selected_candidate,
            "route": self.route,
            "fallback_required": self.fallback_required,
            "reason": self.reason,
            "support_can_replace_G_top1": True,
            "selection_model_updated_from_target_support": False,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "assessments": [row.to_payload() for row in self.assessments],
            "decision_hash": self.decision_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SupportPosteriorDecision":
        try:
            raw_assessments = payload["assessments"]
            if not isinstance(raw_assessments, list):
                raise TypeError("assessment list")
            assessments = tuple(
                CandidateSupportAssessment.from_payload(row)
                for row in raw_assessments
                if isinstance(row, Mapping)
            )
            if len(assessments) != len(raw_assessments) or payload.get(
                "assessment_hashes"
            ) != [row.assessment_hash for row in assessments]:
                raise ProtocolError("SCEPTRE v4 support assessment binding drifted.")
            value = cls(
                target_center=str(payload["target_center"]),
                fold_ordinal=int(payload["fold_ordinal"]),
                fold_hash=str(payload["fold_hash"]),
                partition_hash=str(payload["partition_hash"]),
                selection_case_set_hash=str(payload["selection_case_set_hash"]),
                calibration_case_set_hash=str(payload["calibration_case_set_hash"]),
                evaluation_case_set_hash=str(payload["evaluation_case_set_hash"]),
                routing_context_hash=str(payload["routing_context_hash"]),
                proposal_set_hash=str(payload["proposal_set_hash"]),
                candidate_menu_hash=str(payload["candidate_menu_hash"]),
                exact_b_control_receipt_hash=str(
                    payload["exact_b_control_receipt_hash"]
                ),
                assessments=assessments,
                selected_candidate=(
                    None
                    if payload["selected_candidate"] is None
                    else str(payload["selected_candidate"])
                ),
                route=str(payload["route"]),
                fallback_required=payload["fallback_required"],
                reason=str(payload["reason"]),
                decision_hash=str(payload["decision_hash"]),
            )
        except ProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE v4 support decision payload is malformed.") from exc
        if value.to_payload() != dict(payload):
            raise ProtocolError("SCEPTRE v4 support decision payload drifted.")
        return value


def select_support_candidate(
    outcomes: Iterable[FamilyOutcome],
    *,
    exact_b: FamilyOutcome,
    fold: ThreeRoleFold,
    partition_hash: str,
    proposal_set: FrozenCandidateSetProposal,
    routing_context: FrozenRoutingContext,
) -> SupportPosteriorDecision:
    """Select any sealed expert member, never only the G top-1 member."""

    if not isinstance(proposal_set, FrozenCandidateSetProposal):
        raise ProtocolError("SCEPTRE v4 support lacks its proposal set.")
    if not isinstance(routing_context, FrozenRoutingContext):
        raise ProtocolError("SCEPTRE v4 support lacks its routing context.")
    if not isinstance(fold, ThreeRoleFold):
        raise ProtocolError("SCEPTRE v4 support lacks its fold.")
    target = proposal_set.target_center
    rows = tuple(outcomes)
    expected_sources = legal_routing_sources(target)
    if (
        tuple(row.candidate_center for row in rows) != expected_sources
        or exact_b.candidate_center != EXACT_B_CANDIDATE
        or fold.target_center != target
        or exact_b.role != "SELECTION"
        or exact_b.fold_ordinal != fold.fold_ordinal
        or routing_context.partition_hash != partition_hash
        or proposal_set.frozen_model_sha256
        != routing_context.model_for_target(target).model_sha256
    ):
        raise ProtocolError("SCEPTRE v4 support lineage or inventory drifted.")
    scope = exact_b.scope_key
    if any(row.scope_key != scope for row in rows):
        raise ProtocolError("SCEPTRE v4 support outcomes do not share one scope.")
    if (
        scope[3] != partition_hash
        or scope[4] != fold.case_set_hash("SELECTION")
        or scope[5] != proposal_set.candidate_menu_hash
        or exact_b.exact_b_control_receipt_hash
        != proposal_set.exact_b_control_receipt_hash
    ):
        raise ProtocolError("SCEPTRE v4 support control binding drifted.")
    rank = {
        source: ordinal
        for ordinal, source in enumerate(proposal_set.ranked_sources, start=1)
    }
    assessments: list[CandidateSupportAssessment] = []
    for row in rows:
        if row.case_count != exact_b.case_count:
            raise ProtocolError("SCEPTRE v4 support case counts differ by action.")
        weight = row.case_count / (
            row.case_count + SUPPORT_PRIOR_EFFECTIVE_CASES
        )
        deltas = (
            row.confusion.bacc - exact_b.confusion.bacc,
            row.brier - exact_b.brier,
            row.log_loss - exact_b.log_loss,
        )
        shrunk = tuple(weight * value for value in deltas)
        assessments.append(
            CandidateSupportAssessment(
                candidate_center=row.candidate_center,
                proposal_rank=rank[row.candidate_center],
                candidate_outcome_hash=row.outcome_hash,
                exact_b_outcome_hash=exact_b.outcome_hash,
                case_count=row.case_count,
                shrinkage_weight=weight,
                raw_bacc_gain=deltas[0],
                raw_brier_delta=deltas[1],
                raw_log_loss_delta=deltas[2],
                shrunk_bacc_gain=shrunk[0],
                shrunk_brier_delta=shrunk[1],
                shrunk_log_loss_delta=shrunk[2],
                eligible=shrunk[0] > SUPPORT_MINIMUM_SHRUNK_BACC_GAIN,
            )
        )
    frozen = tuple(assessments)
    eligible = tuple(row for row in frozen if row.eligible)
    selected: str | None = None
    reason = "NO_POSITIVE_SHRUNK_SUPPORT_GAIN_FALLBACK_TO_B"
    if eligible:
        maximum = max(row.shrunk_bacc_gain for row in eligible)
        tied = tuple(
            row
            for row in eligible
            if maximum - row.shrunk_bacc_gain <= TIE_TOLERANCE
        )
        if len(tied) == 1:
            selected = tied[0].candidate_center
            reason = "UNIQUE_MAXIMUM_POSITIVE_SHRUNK_SUPPORT_GAIN"
        else:
            reason = "SUPPORT_POSTERIOR_TIE_FALLBACK_TO_B"
    return SupportPosteriorDecision(
        target_center=target,
        fold_ordinal=fold.fold_ordinal,
        fold_hash=fold.fold_hash,
        partition_hash=partition_hash,
        selection_case_set_hash=fold.case_set_hash("SELECTION"),
        calibration_case_set_hash=fold.case_set_hash("CALIBRATION"),
        evaluation_case_set_hash=fold.case_set_hash("EVALUATION"),
        routing_context_hash=routing_context.context_hash,
        proposal_set_hash=proposal_set.proposal_set_hash,
        candidate_menu_hash=proposal_set.candidate_menu_hash,
        exact_b_control_receipt_hash=proposal_set.exact_b_control_receipt_hash,
        assessments=frozen,
        selected_candidate=selected,
        route=EXACT_B_CANDIDATE if selected is None else selected,
        fallback_required=selected is None,
        reason=reason,
    )


__all__ = (
    "CandidateSupportAssessment",
    "SupportPosteriorDecision",
    "TIE_TOLERANCE",
    "select_support_candidate",
)
