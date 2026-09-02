"""Exact top-1 selection inside a certified-safe set, else exact baseline B."""

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
    safe_action_ids: tuple[str, ...]
    selected_certificate_hash: str | None
    prediction_hash: str
    admission_hash: str
    calibration_hash: str
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.exact_b_fallback != (self.selected_action_id == "B"):
            raise ProtocolError("HARP v8 route fallback semantics are malformed.")
        if self.exact_b_fallback != (self.selected_certificate_hash is None):
            raise ProtocolError("HARP v8 selected certificate/fallback semantics disagree.")
        object.__setattr__(
            self,
            "route_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_exact_safe_top1_route_v8",
                    "outer_target_id": self.outer_target_id,
                    "case_id": self.case_id,
                    "selected_action_id": self.selected_action_id,
                    "probability_hex": self.probability_hex,
                    "exact_b_fallback": self.exact_b_fallback,
                    "reason": self.reason,
                    "safe_action_ids": self.safe_action_ids,
                    "selected_certificate_hash": self.selected_certificate_hash,
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
        raise ProtocolError("HARP v8 route inputs crossed outer/case/menu roles.")
    selected_id = "B"
    probability = menu.baseline_probability_hex
    certificate_hash = None
    top = prediction.top_action_id
    if not menu.actions:
        reason = "EXACT_B_NO_ACTIVE_ACTION"
    elif top is None:
        reason = "EXACT_B_NO_CERTIFIED_SAFE_ACTION"
    elif not admission.admitted:
        reason = "EXACT_B_OUTER_ADMISSION_FAILED"
    elif not calibration.calibrated:
        reason = "EXACT_B_POLICY_CALIBRATION_FAILED"
    else:
        certificate = next(row for row in prediction.action_certificates if row.action_id == top)
        if (
            1.0 - certificate.harm_probability_ucb
            < calibration.certificate_confidence_threshold
        ):
            reason = "EXACT_B_CERTIFICATE_CONFIDENCE_BELOW_THRESHOLD"
        elif not prediction.passes_rank_margin(calibration.rank_margin_threshold):
            reason = "EXACT_B_SAFE_SET_MARGIN_BELOW_THRESHOLD"
        else:
            action = next((row for row in menu.actions if row.action_id == top), None)
            if action is None or action.action_hash != certificate.action_hash:
                raise ProtocolError("HARP v8 selected certificate drifted from sealed menu.")
            selected_id = top
            probability = action.action_probability_hex
            certificate_hash = certificate.certificate_hash
            reason = "ROUTED_CERTIFIED_SAFE_EXACT_TOP1"
    decision = RouteDecision(
        outer_target_id=h,
        case_id=menu.case_id,
        selected_action_id=selected_id,
        probability_hex=probability,
        exact_b_fallback=selected_id == "B",
        reason=reason,
        safe_action_ids=prediction.safe_action_ids,
        selected_certificate_hash=certificate_hash,
        prediction_hash=prediction.prediction_hash,
        admission_hash=admission.admission_hash,
        calibration_hash=calibration.calibration_hash,
    )
    if decision.exact_b_fallback and decision.probability_hex != menu.baseline_probability_hex:
        raise ProtocolError("HARP v8 exact-B fallback is not byte-identical.")
    return decision


__all__ = ("RouteDecision", "select_exact_top1")
