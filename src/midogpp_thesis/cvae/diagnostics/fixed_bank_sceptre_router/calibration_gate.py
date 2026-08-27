"""Lineage-bound disjoint calibration safety gate with exact-B abstention."""

from __future__ import annotations

from dataclasses import dataclass
import math

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .hashing import canonical_hash, require_sha256
from .outcome_surface import EXACT_B_CANDIDATE, FamilyOutcome
from .partitions import FOLD_COUNT
from .support_tournament import SupportTournamentDecision
from .policy_contracts import (
    CALIBRATION_MAXIMUM_BRIER_DELTA,
    CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
    CALIBRATION_MINIMUM_BACC_GAIN,
)


@dataclass(frozen=True, slots=True)
class CalibrationGateDecision:
    target_center: str
    fold_ordinal: int
    fold_hash: str
    partition_hash: str
    selection_case_set_hash: str
    calibration_case_set_hash: str
    evaluation_case_set_hash: str
    candidate_menu_hash: str
    exact_b_control_receipt_hash: str
    candidate_menu_payload_sha256: str
    exact_b_control_payload_sha256: str
    router_bundle_hash: str
    decision_policy_sha256: str
    frozen_model_hash: str
    g_proposal_hash: str
    g_proposed_candidate: str | None
    support_decision_hash: str
    support_selected_candidate: str | None
    uncertainty_decision_hash: str | None
    uncertainty_phase_capability_identity_hash: str | None
    uncertainty_accepted: bool
    candidate_outcome_hash: str | None
    exact_b_outcome_hash: str | None
    route: str
    accepted: bool
    candidate_center: str | None
    calibration_bacc_gain: float
    calibration_brier_delta: float
    calibration_log_loss_delta: float
    minimum_bacc_gain: float
    maximum_brier_delta: float
    maximum_log_loss_delta: float
    reason: str
    decision_hash: str = ""

    def __post_init__(self) -> None:
        target = str(self.target_center)
        if (
            target not in CENTERS
            or isinstance(self.fold_ordinal, bool)
            or self.fold_ordinal not in range(FOLD_COUNT)
        ):
            raise ProtocolError("SCEPTRE calibration decision scope drifted.")
        fold_hash = require_sha256(self.fold_hash, "calibration fold")
        partition = require_sha256(self.partition_hash, "calibration partition")
        selection_cases = require_sha256(
            self.selection_case_set_hash, "selection case set"
        )
        calibration_cases = require_sha256(
            self.calibration_case_set_hash, "calibration case set"
        )
        if selection_cases == calibration_cases:
            raise ProtocolError("SCEPTRE selection and calibration cases overlap.")
        evaluation_cases = require_sha256(
            self.evaluation_case_set_hash, "evaluation case set"
        )
        if len({selection_cases, calibration_cases, evaluation_cases}) != 3:
            raise ProtocolError("SCEPTRE calibration role case sets are not disjoint.")
        menu_hash = _identifier(self.candidate_menu_hash, "candidate menu")
        control_hash = _identifier(
            self.exact_b_control_receipt_hash, "exact-B control receipt"
        )
        menu_payload = require_sha256(
            self.candidate_menu_payload_sha256, "candidate-menu payload"
        )
        control_payload = require_sha256(
            self.exact_b_control_payload_sha256, "exact-B control payload"
        )
        router_bundle = require_sha256(
            self.router_bundle_hash, "calibration frozen router bundle"
        )
        decision_policy = require_sha256(
            self.decision_policy_sha256, "calibration frozen decision policy"
        )
        frozen_model = require_sha256(
            self.frozen_model_hash, "calibration frozen target model"
        )
        g_proposal_hash = require_sha256(
            self.g_proposal_hash, "calibration G proposal"
        )
        support_hash = require_sha256(self.support_decision_hash, "support decision")
        minimum = _nonnegative_finite(self.minimum_bacc_gain, "minimum BACC gain")
        maximum_brier = _nonpositive_finite(
            self.maximum_brier_delta, "maximum Brier delta"
        )
        maximum_log = _nonpositive_finite(
            self.maximum_log_loss_delta, "maximum log-loss delta"
        )
        deltas = (
            float(self.calibration_bacc_gain),
            float(self.calibration_brier_delta),
            float(self.calibration_log_loss_delta),
        )
        if not isinstance(self.uncertainty_accepted, bool):
            raise ProtocolError("SCEPTRE calibration uncertainty flag is invalid.")
        if any(not math.isfinite(value) for value in deltas):
            raise ProtocolError("SCEPTRE calibration deltas are non-finite.")
        legal = set(legal_routing_sources(target))
        g_proposed = self.g_proposed_candidate
        if g_proposed is not None and g_proposed not in legal:
            raise ProtocolError("SCEPTRE G-proposed candidate is outside C minus H.")
        support_selected = self.support_selected_candidate
        if support_selected is not None and support_selected not in legal:
            raise ProtocolError("SCEPTRE support-selected candidate is outside C minus H.")
        if self.reason == "SUPPORT_DECISION_FALLBACK":
            expected_accepted = False
            expected_candidate = None
            expected_route = EXACT_B_CANDIDATE
            if (
                support_selected is not None
                or self.candidate_outcome_hash is not None
                or self.exact_b_outcome_hash is not None
                or self.uncertainty_decision_hash is not None
                or self.uncertainty_phase_capability_identity_hash is not None
                or self.uncertainty_accepted is not False
            ):
                raise ProtocolError("SCEPTRE support fallback carries a candidate outcome.")
            if deltas != (0.0, 0.0, 0.0):
                raise ProtocolError("SCEPTRE support fallback carries calibration deltas.")
            candidate_hash = None
            exact_b_hash = None
            uncertainty_hash = None
            uncertainty_capability_hash = None
        else:
            if support_selected is None or g_proposed != support_selected:
                raise ProtocolError("SCEPTRE calibration lacks a support-selected candidate.")
            candidate_hash = require_sha256(
                self.candidate_outcome_hash, "calibration candidate outcome"
            )
            exact_b_hash = require_sha256(
                self.exact_b_outcome_hash, "calibration exact-B outcome"
            )
            uncertainty_hash = require_sha256(
                self.uncertainty_decision_hash, "calibration uncertainty decision"
            )
            uncertainty_capability_hash = require_sha256(
                self.uncertainty_phase_capability_identity_hash,
                "calibration uncertainty capability",
            )
            safe = (
                self.uncertainty_accepted is True
                and deltas[0] > minimum
                and deltas[1] <= maximum_brier
                and deltas[2] <= maximum_log
            )
            expected_accepted = safe
            expected_candidate = support_selected if safe else None
            expected_route = support_selected if safe else EXACT_B_CANDIDATE
            expected_reason = (
                "CALIBRATION_ACCEPT"
                if safe
                else (
                    "CALIBRATION_UNCERTAINTY_REJECT_FALLBACK"
                    if self.uncertainty_accepted is False
                    else "CALIBRATION_POINT_GATE_REJECT_FALLBACK"
                )
            )
            if self.reason != expected_reason:
                raise ProtocolError("SCEPTRE calibration decision reason drifted.")
        if (
            self.accepted is not expected_accepted
            or self.candidate_center != expected_candidate
            or self.route != expected_route
        ):
            raise ProtocolError("SCEPTRE calibration route semantics drifted.")
        body = {
            "schema_version": "sceptre_calibration_gate_decision_v3",
            "target_center": target,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": fold_hash,
            "partition_hash": partition,
            "selection_case_set_hash": selection_cases,
            "calibration_case_set_hash": calibration_cases,
            "evaluation_case_set_hash": evaluation_cases,
            "candidate_menu_hash": menu_hash,
            "exact_b_control_receipt_hash": control_hash,
            "candidate_menu_payload_sha256": menu_payload,
            "exact_b_control_payload_sha256": control_payload,
            "router_bundle_hash": router_bundle,
            "decision_policy_sha256": decision_policy,
            "frozen_model_hash": frozen_model,
            "g_proposal_hash": g_proposal_hash,
            "g_proposed_candidate": g_proposed,
            "support_decision_hash": support_hash,
            "support_selected_candidate": support_selected,
            "uncertainty_decision_hash": uncertainty_hash,
            "uncertainty_phase_capability_identity_hash": (
                uncertainty_capability_hash
            ),
            "uncertainty_accepted": self.uncertainty_accepted,
            "candidate_outcome_hash": candidate_hash,
            "exact_b_outcome_hash": exact_b_hash,
            "candidate_center": expected_candidate,
            "calibration_bacc_gain": deltas[0],
            "calibration_brier_delta": deltas[1],
            "calibration_log_loss_delta": deltas[2],
            "thresholds": {
                "minimum_bacc_gain": minimum,
                "maximum_brier_delta": maximum_brier,
                "maximum_log_loss_delta": maximum_log,
            },
            "accepted": expected_accepted,
            "route": expected_route,
            "reason": self.reason,
        }
        expected_hash = canonical_hash(body)
        if self.decision_hash and self.decision_hash != expected_hash:
            raise ProtocolError("SCEPTRE calibration decision hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "fold_hash", fold_hash)
        object.__setattr__(self, "partition_hash", partition)
        object.__setattr__(self, "selection_case_set_hash", selection_cases)
        object.__setattr__(self, "calibration_case_set_hash", calibration_cases)
        object.__setattr__(self, "evaluation_case_set_hash", evaluation_cases)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "exact_b_control_receipt_hash", control_hash)
        object.__setattr__(self, "candidate_menu_payload_sha256", menu_payload)
        object.__setattr__(self, "exact_b_control_payload_sha256", control_payload)
        object.__setattr__(self, "router_bundle_hash", router_bundle)
        object.__setattr__(self, "decision_policy_sha256", decision_policy)
        object.__setattr__(self, "frozen_model_hash", frozen_model)
        object.__setattr__(self, "g_proposal_hash", g_proposal_hash)
        object.__setattr__(self, "g_proposed_candidate", g_proposed)
        object.__setattr__(self, "support_decision_hash", support_hash)
        object.__setattr__(self, "uncertainty_decision_hash", uncertainty_hash)
        object.__setattr__(
            self,
            "uncertainty_phase_capability_identity_hash",
            uncertainty_capability_hash,
        )
        object.__setattr__(self, "candidate_outcome_hash", candidate_hash)
        object.__setattr__(self, "exact_b_outcome_hash", exact_b_hash)
        object.__setattr__(self, "accepted", expected_accepted)
        object.__setattr__(self, "candidate_center", expected_candidate)
        object.__setattr__(self, "route", expected_route)
        object.__setattr__(self, "calibration_bacc_gain", deltas[0])
        object.__setattr__(self, "calibration_brier_delta", deltas[1])
        object.__setattr__(self, "calibration_log_loss_delta", deltas[2])
        object.__setattr__(self, "minimum_bacc_gain", minimum)
        object.__setattr__(self, "maximum_brier_delta", maximum_brier)
        object.__setattr__(self, "maximum_log_loss_delta", maximum_log)
        object.__setattr__(self, "decision_hash", expected_hash)


def apply_calibration_gate(
    support: SupportTournamentDecision,
    *,
    uncertainty: object | None,
    candidate: FamilyOutcome | None,
    exact_b: FamilyOutcome | None,
    frozen_router: object,
    minimum_bacc_gain: float = CALIBRATION_MINIMUM_BACC_GAIN,
    maximum_brier_delta: float = CALIBRATION_MAXIMUM_BRIER_DELTA,
    maximum_log_loss_delta: float = CALIBRATION_MAXIMUM_LOG_LOSS_DELTA,
) -> CalibrationGateDecision:
    from .router_bundle_freeze import FrozenPrelabelRouter
    from .uncertainty import UncertaintyRouteDecision

    if not isinstance(support, SupportTournamentDecision):
        raise ProtocolError("SCEPTRE calibration requires a typed support decision.")
    if not isinstance(frozen_router, FrozenPrelabelRouter):
        raise ProtocolError("SCEPTRE calibration requires its full frozen router.")
    model = frozen_router.model_for_target(support.target_center)
    minimum = _nonnegative_finite(minimum_bacc_gain, "minimum BACC gain")
    maximum_brier = _nonpositive_finite(
        maximum_brier_delta, "maximum Brier delta"
    )
    maximum_log = _nonpositive_finite(
        maximum_log_loss_delta, "maximum log-loss delta"
    )
    if (
        minimum != CALIBRATION_MINIMUM_BACC_GAIN
        or maximum_brier != CALIBRATION_MAXIMUM_BRIER_DELTA
        or maximum_log != CALIBRATION_MAXIMUM_LOG_LOSS_DELTA
    ):
        raise ProtocolError("SCEPTRE calibration thresholds drifted from frozen values.")
    if (
        support.router_bundle_hash != frozen_router.router_bundle_hash
        or support.partition_hash != frozen_router.partition_hash
        or support.decision_policy_sha256 != frozen_router.decision_policy_sha256
        or support.frozen_model_hash != model.model_sha256
        or support.candidate_menu_hash != model.candidate_menu_hash
        or support.candidate_menu_payload_sha256
        != model.candidate_menu_payload_sha256
        or support.exact_b_control_receipt_hash
        != model.exact_b_control_receipt_hash
        or support.exact_b_control_payload_sha256
        != model.exact_b_control_payload_sha256
        or minimum != frozen_router.calibration_minimum_bacc_gain
        or maximum_brier != frozen_router.calibration_maximum_brier_delta
        or maximum_log != frozen_router.calibration_maximum_log_loss_delta
    ):
        raise ProtocolError("SCEPTRE calibration frozen-router lineage differs.")
    expected_scope = (
        support.target_center,
        support.fold_ordinal,
        "CALIBRATION",
        support.partition_hash,
        support.calibration_case_set_hash,
        support.candidate_menu_hash,
    )
    if support.fallback_required:
        if candidate is not None or exact_b is not None or uncertainty is not None:
            raise ProtocolError("SCEPTRE support fallback opened calibration evidence.")
        bacc_gain = brier_delta = log_delta = 0.0
        accepted, selected, reason = False, None, "SUPPORT_DECISION_FALLBACK"
        candidate_hash = None
        exact_b_hash = None
        uncertainty_hash = None
        uncertainty_capability_hash = None
        uncertainty_accepted = False
    else:
        if (
            not isinstance(uncertainty, UncertaintyRouteDecision)
            or uncertainty.role != "CALIBRATION"
            or uncertainty.target_center != support.target_center
            or uncertainty.fold_ordinal != support.fold_ordinal
            or uncertainty.fold_hash != support.fold_hash
            or uncertainty.partition_hash != support.partition_hash
            or uncertainty.role_case_set_hash != support.calibration_case_set_hash
            or uncertainty.router_bundle_hash != support.router_bundle_hash
            or uncertainty.g_proposal_hash != support.g_proposal_hash
            or uncertainty.predecessor_decision_hash != support.decision_hash
            or uncertainty.candidate_menu_hash != support.candidate_menu_hash
            or uncertainty.exact_b_control_receipt_hash
            != support.exact_b_control_receipt_hash
            or uncertainty.bootstrap_config_hash
            != frozen_router.dirichlet_config.config_hash
            or uncertainty.g_proposed_candidate != support.selected_candidate
            or uncertainty.selected_candidate
            not in (None, support.selected_candidate)
            or candidate is None
            or exact_b is None
            or exact_b.candidate_center != EXACT_B_CANDIDATE
            or candidate.candidate_center != support.selected_candidate
            or candidate.scope_key != exact_b.scope_key
            or exact_b.scope_key != expected_scope
            or exact_b.case_set_hash != support.calibration_case_set_hash
            or exact_b.exact_b_control_receipt_hash
            != support.exact_b_control_receipt_hash
        ):
            raise ProtocolError(
                "SCEPTRE calibration uncertainty/candidate lineage differs from support."
            )
        bacc_gain = candidate.confusion.bacc - exact_b.confusion.bacc
        brier_delta = candidate.brier - exact_b.brier
        log_delta = candidate.log_loss - exact_b.log_loss
        accepted = (
            uncertainty.accepted
            and bacc_gain > minimum
            and brier_delta <= maximum_brier
            and log_delta <= maximum_log
        )
        selected = candidate.candidate_center if accepted else None
        reason = (
            "CALIBRATION_ACCEPT"
            if accepted
            else (
                "CALIBRATION_UNCERTAINTY_REJECT_FALLBACK"
                if not uncertainty.accepted
                else "CALIBRATION_POINT_GATE_REJECT_FALLBACK"
            )
        )
        candidate_hash = candidate.outcome_hash
        exact_b_hash = exact_b.outcome_hash
        uncertainty_hash = uncertainty.decision_hash
        uncertainty_capability_hash = uncertainty.phase_capability_identity_hash
        uncertainty_accepted = uncertainty.accepted
    return CalibrationGateDecision(
        target_center=support.target_center,
        fold_ordinal=support.fold_ordinal,
        fold_hash=support.fold_hash,
        partition_hash=support.partition_hash,
        selection_case_set_hash=support.selection_case_set_hash,
        calibration_case_set_hash=support.calibration_case_set_hash,
        evaluation_case_set_hash=support.evaluation_case_set_hash,
        candidate_menu_hash=support.candidate_menu_hash,
        exact_b_control_receipt_hash=support.exact_b_control_receipt_hash,
        candidate_menu_payload_sha256=support.candidate_menu_payload_sha256,
        exact_b_control_payload_sha256=support.exact_b_control_payload_sha256,
        router_bundle_hash=support.router_bundle_hash,
        decision_policy_sha256=support.decision_policy_sha256,
        frozen_model_hash=support.frozen_model_hash,
        g_proposal_hash=support.g_proposal_hash,
        g_proposed_candidate=support.g_proposed_candidate,
        support_decision_hash=support.decision_hash,
        support_selected_candidate=support.selected_candidate,
        uncertainty_decision_hash=uncertainty_hash,
        uncertainty_phase_capability_identity_hash=uncertainty_capability_hash,
        uncertainty_accepted=uncertainty_accepted,
        candidate_outcome_hash=candidate_hash,
        exact_b_outcome_hash=exact_b_hash,
        route=selected or EXACT_B_CANDIDATE,
        accepted=accepted,
        candidate_center=selected,
        calibration_bacc_gain=bacc_gain,
        calibration_brier_delta=brier_delta,
        calibration_log_loss_delta=log_delta,
        minimum_bacc_gain=minimum,
        maximum_brier_delta=maximum_brier,
        maximum_log_loss_delta=maximum_log,
        reason=reason,
    )


def _nonnegative_finite(value: object, role: str) -> float:
    parsed = _finite(value, role)
    if parsed < 0.0:
        raise ProtocolError(f"SCEPTRE {role} must be nonnegative.")
    return parsed


def _nonpositive_finite(value: object, role: str) -> float:
    parsed = _finite(value, role)
    if parsed > 0.0:
        raise ProtocolError(f"SCEPTRE {role} must be nonpositive.")
    return parsed


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"SCEPTRE {role} is invalid.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"SCEPTRE {role} is invalid.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"SCEPTRE {role} must be finite.")
    return parsed


def _identifier(value: object, role: str) -> str:
    text = "" if value is None else str(value)
    if not text or text.strip() != text:
        raise ProtocolError(f"SCEPTRE {role} is invalid.")
    return text


__all__ = (
    "CALIBRATION_MAXIMUM_BRIER_DELTA",
    "CALIBRATION_MAXIMUM_LOG_LOSS_DELTA",
    "CALIBRATION_MINIMUM_BACC_GAIN",
    "CalibrationGateDecision",
    "apply_calibration_gate",
)
