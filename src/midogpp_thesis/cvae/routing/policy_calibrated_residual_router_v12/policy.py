"""Exact physical top-1 or byte-identical baseline policy for HARP v12."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .admission import OuterAdmission
from .calibration import PolicyCalibration
from .contracts import CasePrediction
from .effective_menu import EffectiveMenu
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class RouteDecision:
    outer_target_id: str
    case_id: str
    selected_action_id: str
    probability_hex: tuple[str, ...]
    exact_b_fallback: bool
    reason: str
    ranked_action_ids: tuple[str, ...]
    selected_score_hash: str | None
    prediction_hash: str
    admission_hash: str
    calibration_hash: str
    acceptance_probability: float
    acceptance_threshold: float
    rank_margin: float
    rank_margin_threshold: float
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.exact_b_fallback != (self.selected_action_id == "B"):
            raise ProtocolError("HARP v12 route fallback semantics are malformed.")
        if self.exact_b_fallback != (self.selected_score_hash is None):
            raise ProtocolError("HARP v12 selected score/fallback semantics disagree.")
        object.__setattr__(
            self,
            "route_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_exact_top1_route_v12",
                    "outer_target_id": self.outer_target_id,
                    "case_id": self.case_id,
                    "selected_action_id": self.selected_action_id,
                    "probability_hex": self.probability_hex,
                    "exact_b_fallback": self.exact_b_fallback,
                    "reason": self.reason,
                    "ranked_action_ids": self.ranked_action_ids,
                    "selected_score_hash": self.selected_score_hash,
                    "prediction_hash": self.prediction_hash,
                    "admission_hash": self.admission_hash,
                    "calibration_hash": self.calibration_hash,
                    "acceptance_probability": self.acceptance_probability,
                    "acceptance_threshold": self.acceptance_threshold,
                    "rank_margin": self.rank_margin,
                    "rank_margin_threshold": self.rank_margin_threshold,
                    "target_evaluation_labels_used": False,
                    "probability_mixture_used": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "case_id": self.case_id,
            "selected_action_id": self.selected_action_id,
            "probability_hex": list(self.probability_hex),
            "exact_b_fallback": self.exact_b_fallback,
            "reason": self.reason,
            "ranked_action_ids": list(self.ranked_action_ids),
            "selected_score_hash": self.selected_score_hash,
            "prediction_hash": self.prediction_hash,
            "admission_hash": self.admission_hash,
            "calibration_hash": self.calibration_hash,
            "acceptance_probability": self.acceptance_probability,
            "acceptance_threshold": self.acceptance_threshold,
            "rank_margin": self.rank_margin,
            "rank_margin_threshold": self.rank_margin_threshold,
            "route_hash": self.route_hash,
        }


def select_policy_action(
    menu: EffectiveMenu,
    prediction: CasePrediction,
    admission: OuterAdmission,
    calibration: PolicyCalibration,
) -> RouteDecision:
    h = menu.outer_target_id
    if (
        menu.query_center_id != h
        or prediction.outer_target_id != h
        or prediction.query_center_id != h
        or prediction.case_id != menu.case_id
        or prediction.menu_hash != menu.menu_hash
        or admission.outer_target_id != h
        or calibration.outer_target_id != h
    ):
        raise ProtocolError("HARP v12 route inputs crossed outer/case/menu roles.")
    selected_id = "B"
    probability = menu.baseline_probability_hex
    score_hash = None
    top = prediction.raw_top_action_id
    if not menu.actions:
        reason = "EXACT_B_NO_ACTIVE_ACTION"
    elif top == "B":
        reason = "EXACT_B_VIRTUAL_BASELINE_RANKED_TOP1"
    elif not admission.admitted:
        reason = "EXACT_B_OUTER_RANK_ADMISSION_FAILED"
    elif not calibration.calibrated:
        reason = "EXACT_B_POLICY_RISK_CALIBRATION_FAILED"
    elif prediction.acceptance_probability < calibration.acceptance_threshold:
        reason = "EXACT_B_SELECTED_ACTION_ACCEPTANCE_BELOW_THRESHOLD"
    elif prediction.rank_margin < calibration.rank_margin_threshold:
        reason = "EXACT_B_SELECTED_ACTION_MARGIN_BELOW_THRESHOLD"
    else:
        action = next((row for row in menu.actions if row.action_id == top), None)
        score = prediction.score_for(top)
        if action is None or score is None or action.action_hash != score.action_hash:
            raise ProtocolError("HARP v12 selected action drifted from its sealed menu.")
        selected_id = top
        probability = action.action_probability_hex
        score_hash = score.score_hash
        reason = "ROUTED_POLICY_CALIBRATED_EXACT_TOP1"
    decision = RouteDecision(
        outer_target_id=h,
        case_id=menu.case_id,
        selected_action_id=selected_id,
        probability_hex=probability,
        exact_b_fallback=selected_id == "B",
        reason=reason,
        ranked_action_ids=prediction.ranked_action_ids,
        selected_score_hash=score_hash,
        prediction_hash=prediction.prediction_hash,
        admission_hash=admission.admission_hash,
        calibration_hash=calibration.calibration_hash,
        acceptance_probability=prediction.acceptance_probability,
        acceptance_threshold=calibration.acceptance_threshold,
        rank_margin=prediction.rank_margin,
        rank_margin_threshold=calibration.rank_margin_threshold,
    )
    if decision.exact_b_fallback and decision.probability_hex != menu.baseline_probability_hex:
        raise ProtocolError("HARP v12 exact-B fallback is not byte-identical.")
    return decision


__all__ = ("RouteDecision", "select_policy_action")
