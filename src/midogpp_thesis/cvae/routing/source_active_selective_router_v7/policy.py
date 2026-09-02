"""Exact top-1 target selection with a byte-identical exact-B fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .admission import OuterAdmission
from .calibration import SelectiveCalibration
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
    prediction_hash: str
    admission_hash: str
    calibration_hash: str
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.exact_b_fallback != (self.selected_action_id == "B"):
            raise ProtocolError("Route decision fallback semantics are malformed.")
        object.__setattr__(
            self,
            "route_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_exact_top1_route_v7",
                    "outer_target_id": self.outer_target_id,
                    "case_id": self.case_id,
                    "selected_action_id": self.selected_action_id,
                    "probability_hex": self.probability_hex,
                    "exact_b_fallback": self.exact_b_fallback,
                    "reason": self.reason,
                    "prediction_hash": self.prediction_hash,
                    "admission_hash": self.admission_hash,
                    "calibration_hash": self.calibration_hash,
                    "target_evaluation_labels_used": False,
                }
            ),
        )


def select_exact_top1(
    menu: EffectiveMenu,
    prediction: CasePrediction,
    admission: OuterAdmission,
    calibration: SelectiveCalibration,
) -> RouteDecision:
    """Apply the frozen source-only policy to one label-free target menu."""

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
        raise ProtocolError("Target route inputs crossed outer target, case, or menu roles.")

    reason: str
    selected_action_id = "B"
    probability = menu.baseline_probability_hex
    if not menu.actions or prediction.top_action_id is None:
        reason = "EXACT_B_NO_ACTIVE_ACTION"
    elif not admission.admitted:
        reason = "EXACT_B_OUTER_ADMISSION_FAILED"
    elif not calibration.calibrated:
        reason = "EXACT_B_POLICY_CALIBRATION_FAILED"
    elif prediction.opportunity_probability < calibration.opportunity_threshold:
        reason = "EXACT_B_OPPORTUNITY_BELOW_THRESHOLD"
    elif not prediction.passes_rank_margin(calibration.rank_margin_threshold):
        reason = "EXACT_B_RANK_MARGIN_BELOW_THRESHOLD"
    else:
        action = next(
            (row for row in menu.actions if row.action_id == prediction.top_action_id), None
        )
        if action is None:
            raise ProtocolError("Predicted top action is absent from the sealed effective menu.")
        score = next(row for row in prediction.action_scores if row.action_id == action.action_id)
        if score.action_hash != action.action_hash:
            raise ProtocolError("Predicted top action hash drifted from the sealed menu.")
        selected_action_id = action.action_id
        probability = action.action_probability_hex
        reason = "ROUTED_EXACT_TOP1"
    decision = RouteDecision(
        outer_target_id=h,
        case_id=menu.case_id,
        selected_action_id=selected_action_id,
        probability_hex=probability,
        exact_b_fallback=selected_action_id == "B",
        reason=reason,
        prediction_hash=prediction.prediction_hash,
        admission_hash=admission.admission_hash,
        calibration_hash=calibration.calibration_hash,
    )
    if decision.exact_b_fallback and decision.probability_hex != menu.baseline_probability_hex:
        raise ProtocolError("Exact-B fallback is not byte-identical to baseline B.")
    return decision


__all__ = ("RouteDecision", "select_exact_top1")
