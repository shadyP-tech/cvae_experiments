"""Disjoint calibration confirmation for the support-selected member."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
    FamilyOutcome,
)
from .development import (
    CALIBRATION_MAXIMUM_BRIER_DELTA,
    CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
    CALIBRATION_MINIMUM_BACC_GAIN,
    FrozenRoutingContext,
)
from .posterior import PairedCandidatePosterior
from .support_posterior import SupportPosteriorDecision


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    partition_hash: str
    selection_case_set_hash: str
    calibration_case_set_hash: str
    evaluation_case_set_hash: str
    routing_context_hash: str
    proposal_set_hash: str
    support_decision_hash: str
    candidate_menu_hash: str
    exact_b_control_receipt_hash: str
    support_selected_candidate: str | None
    posterior_hash: str | None
    candidate_outcome_hash: str | None
    exact_b_outcome_hash: str | None
    calibration_bacc_gain: float
    calibration_brier_delta: float
    calibration_log_loss_delta: float
    joint_acceptance_probability: float
    route: str
    accepted: bool
    reason: str
    decision_hash: str = ""

    def __post_init__(self) -> None:
        target = str(self.target_center)
        selected = self.support_selected_candidate
        if selected is not None and selected not in legal_routing_sources(target):
            raise ProtocolError("SCEPTRE v5 confirmation candidate is illegal.")
        for value, role in (
            (self.fold_hash, "confirmation fold"),
            (self.partition_hash, "confirmation partition"),
            (self.selection_case_set_hash, "selection cases"),
            (self.calibration_case_set_hash, "calibration cases"),
            (self.evaluation_case_set_hash, "evaluation cases"),
            (self.routing_context_hash, "routing context"),
            (self.proposal_set_hash, "proposal set"),
            (self.support_decision_hash, "support decision"),
        ):
            require_sha256(value, role)
        for value, role in (
            (self.candidate_menu_hash, "candidate menu"),
            (self.exact_b_control_receipt_hash, "exact-B control receipt"),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ProtocolError(f"SCEPTRE v5 {role} is invalid.")
        values = (
            float(self.calibration_bacc_gain),
            float(self.calibration_brier_delta),
            float(self.calibration_log_loss_delta),
            float(self.joint_acceptance_probability),
        )
        if any(not math.isfinite(value) for value in values) or not 0.0 <= values[3] <= 1.0:
            raise ProtocolError("SCEPTRE v5 confirmation metrics are invalid.")
        if selected is None:
            if any(
                value is not None
                for value in (
                    self.posterior_hash,
                    self.candidate_outcome_hash,
                    self.exact_b_outcome_hash,
                )
            ) or values != (0.0, 0.0, 0.0, 0.0):
                raise ProtocolError("SCEPTRE v5 support fallback opened calibration.")
            expected_accepted = False
            expected_route = EXACT_B_CANDIDATE
            expected_reason = "SUPPORT_FALLBACK_CALIBRATION_NOT_OPENED"
        else:
            for value, role in (
                (self.posterior_hash, "confirmation posterior"),
                (self.candidate_outcome_hash, "calibration candidate outcome"),
                (self.exact_b_outcome_hash, "calibration exact-B outcome"),
            ):
                require_sha256(value, role)
            expected_accepted = (
                values[3] >= 0.8
                and values[0] > CALIBRATION_MINIMUM_BACC_GAIN
                and values[1] <= CALIBRATION_MAXIMUM_BRIER_DELTA
                and values[2] <= CALIBRATION_MAXIMUM_LOG_LOSS_DELTA
            )
            expected_route = selected if expected_accepted else EXACT_B_CANDIDATE
            expected_reason = (
                "SAME_SUPPORT_MEMBER_CONFIRMED"
                if expected_accepted
                else "CALIBRATION_REJECT_FALLBACK_TO_B"
            )
        if (
            self.accepted is not expected_accepted
            or self.route != expected_route
            or self.reason != expected_reason
        ):
            raise ProtocolError("SCEPTRE v5 confirmation semantics drifted.")
        body = self._payload_without_hash(target, values)
        expected = canonical_hash(body)
        if self.decision_hash and self.decision_hash != expected:
            raise ProtocolError("SCEPTRE v5 confirmation hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "decision_hash", expected)

    def _payload_without_hash(
        self,
        target: str | None = None,
        values: tuple[float, float, float, float] | None = None,
    ) -> dict[str, object]:
        metrics = (
            (
                self.calibration_bacc_gain,
                self.calibration_brier_delta,
                self.calibration_log_loss_delta,
                self.joint_acceptance_probability,
            )
            if values is None
            else values
        )
        return {
            "schema_version": "sceptre_v5_confirmation_decision_v1",
            "target_center": self.target_center if target is None else target,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "partition_hash": self.partition_hash,
            "selection_case_set_hash": self.selection_case_set_hash,
            "calibration_case_set_hash": self.calibration_case_set_hash,
            "evaluation_case_set_hash": self.evaluation_case_set_hash,
            "routing_context_hash": self.routing_context_hash,
            "proposal_set_hash": self.proposal_set_hash,
            "support_decision_hash": self.support_decision_hash,
            "candidate_menu_hash": self.candidate_menu_hash,
            "exact_b_control_receipt_hash": self.exact_b_control_receipt_hash,
            "support_selected_candidate": self.support_selected_candidate,
            "posterior_hash": self.posterior_hash,
            "candidate_outcome_hash": self.candidate_outcome_hash,
            "exact_b_outcome_hash": self.exact_b_outcome_hash,
            "calibration_delta_vs_exact_b": {
                "bacc": metrics[0],
                "brier": metrics[1],
                "log_loss": metrics[2],
            },
            "joint_acceptance_probability": metrics[3],
            "thresholds": {
                "joint_acceptance_probability": 0.8,
                "minimum_bacc_gain": CALIBRATION_MINIMUM_BACC_GAIN,
                "maximum_brier_delta": CALIBRATION_MAXIMUM_BRIER_DELTA,
                "maximum_log_loss_delta": CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
            },
            "accepted": self.accepted,
            "route": self.route,
            "reason": self.reason,
            "support_selected_member_may_not_change_in_calibration": True,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "decision_hash": self.decision_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ConfirmationDecision":
        try:
            deltas = payload["calibration_delta_vs_exact_b"]
            if not isinstance(deltas, Mapping):
                raise TypeError("calibration deltas")
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
                support_decision_hash=str(payload["support_decision_hash"]),
                candidate_menu_hash=str(payload["candidate_menu_hash"]),
                exact_b_control_receipt_hash=str(
                    payload["exact_b_control_receipt_hash"]
                ),
                support_selected_candidate=(
                    None
                    if payload["support_selected_candidate"] is None
                    else str(payload["support_selected_candidate"])
                ),
                posterior_hash=(
                    None
                    if payload["posterior_hash"] is None
                    else str(payload["posterior_hash"])
                ),
                candidate_outcome_hash=(
                    None
                    if payload["candidate_outcome_hash"] is None
                    else str(payload["candidate_outcome_hash"])
                ),
                exact_b_outcome_hash=(
                    None
                    if payload["exact_b_outcome_hash"] is None
                    else str(payload["exact_b_outcome_hash"])
                ),
                calibration_bacc_gain=float(deltas["bacc"]),
                calibration_brier_delta=float(deltas["brier"]),
                calibration_log_loss_delta=float(deltas["log_loss"]),
                joint_acceptance_probability=float(
                    payload["joint_acceptance_probability"]
                ),
                route=str(payload["route"]),
                accepted=payload["accepted"],
                reason=str(payload["reason"]),
                decision_hash=str(payload["decision_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE v5 confirmation payload is malformed.") from exc
        if value.to_payload() != dict(payload):
            raise ProtocolError("SCEPTRE v5 confirmation payload drifted.")
        return value


def apply_confirmation_gate(
    support: SupportPosteriorDecision,
    *,
    posterior: PairedCandidatePosterior | None,
    candidate: FamilyOutcome | None,
    exact_b: FamilyOutcome | None,
    routing_context: FrozenRoutingContext,
) -> ConfirmationDecision:
    """Confirm the same support-selected expert or deterministically use B."""

    if not isinstance(support, SupportPosteriorDecision):
        raise ProtocolError("SCEPTRE v5 confirmation requires support decision.")
    if not isinstance(routing_context, FrozenRoutingContext):
        raise ProtocolError("SCEPTRE v5 confirmation requires routing context.")
    if support.routing_context_hash != routing_context.context_hash:
        raise ProtocolError("SCEPTRE v5 confirmation context drifted.")
    selected = support.selected_candidate
    if selected is None:
        return ConfirmationDecision(
            target_center=support.target_center,
            fold_ordinal=support.fold_ordinal,
            fold_hash=support.fold_hash,
            partition_hash=support.partition_hash,
            selection_case_set_hash=support.selection_case_set_hash,
            calibration_case_set_hash=support.calibration_case_set_hash,
            evaluation_case_set_hash=support.evaluation_case_set_hash,
            routing_context_hash=support.routing_context_hash,
            proposal_set_hash=support.proposal_set_hash,
            support_decision_hash=support.decision_hash,
            candidate_menu_hash=support.candidate_menu_hash,
            exact_b_control_receipt_hash=support.exact_b_control_receipt_hash,
            support_selected_candidate=None,
            posterior_hash=None,
            candidate_outcome_hash=None,
            exact_b_outcome_hash=None,
            calibration_bacc_gain=0.0,
            calibration_brier_delta=0.0,
            calibration_log_loss_delta=0.0,
            joint_acceptance_probability=0.0,
            route=EXACT_B_CANDIDATE,
            accepted=False,
            reason="SUPPORT_FALLBACK_CALIBRATION_NOT_OPENED",
        )
    if (
        not isinstance(posterior, PairedCandidatePosterior)
        or not isinstance(candidate, FamilyOutcome)
        or not isinstance(exact_b, FamilyOutcome)
        or posterior.candidate_center != selected
        or posterior.support_decision_hash != support.decision_hash
        or posterior.proposal_set_hash != support.proposal_set_hash
        or candidate.candidate_center != selected
        or exact_b.candidate_center != EXACT_B_CANDIDATE
        or candidate.scope_key != exact_b.scope_key
        or candidate.role != "CALIBRATION"
        or candidate.case_set_hash != support.calibration_case_set_hash
        or candidate.candidate_menu_hash != support.candidate_menu_hash
        or exact_b.exact_b_control_receipt_hash
        != support.exact_b_control_receipt_hash
    ):
        raise ProtocolError("SCEPTRE v5 confirmation evidence lineage drifted.")
    deltas = (
        candidate.confusion.bacc - exact_b.confusion.bacc,
        candidate.brier - exact_b.brier,
        candidate.log_loss - exact_b.log_loss,
    )
    accepted = (
        posterior.joint_acceptance_probability >= 0.8
        and deltas[0] > CALIBRATION_MINIMUM_BACC_GAIN
        and deltas[1] <= CALIBRATION_MAXIMUM_BRIER_DELTA
        and deltas[2] <= CALIBRATION_MAXIMUM_LOG_LOSS_DELTA
    )
    return ConfirmationDecision(
        target_center=support.target_center,
        fold_ordinal=support.fold_ordinal,
        fold_hash=support.fold_hash,
        partition_hash=support.partition_hash,
        selection_case_set_hash=support.selection_case_set_hash,
        calibration_case_set_hash=support.calibration_case_set_hash,
        evaluation_case_set_hash=support.evaluation_case_set_hash,
        routing_context_hash=support.routing_context_hash,
        proposal_set_hash=support.proposal_set_hash,
        support_decision_hash=support.decision_hash,
        candidate_menu_hash=support.candidate_menu_hash,
        exact_b_control_receipt_hash=support.exact_b_control_receipt_hash,
        support_selected_candidate=selected,
        posterior_hash=posterior.posterior_hash,
        candidate_outcome_hash=candidate.outcome_hash,
        exact_b_outcome_hash=exact_b.outcome_hash,
        calibration_bacc_gain=deltas[0],
        calibration_brier_delta=deltas[1],
        calibration_log_loss_delta=deltas[2],
        joint_acceptance_probability=posterior.joint_acceptance_probability,
        route=selected if accepted else EXACT_B_CANDIDATE,
        accepted=accepted,
        reason=(
            "SAME_SUPPORT_MEMBER_CONFIRMED"
            if accepted
            else "CALIBRATION_REJECT_FALLBACK_TO_B"
        ),
    )


__all__ = ("ConfirmationDecision", "apply_confirmation_gate")
