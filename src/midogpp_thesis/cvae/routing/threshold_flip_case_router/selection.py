"""Static A1 selection and heuristic per-case B-versus-challenger decisions."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .calibration import calibrated_gain
from .contracts import (
    CaseActionFeatures,
    CaseDecision,
    ContributionTarget,
    DirectionSharedCalibration,
    StaticSelection,
    TwoHeadPrediction,
)
from .targets import pooled_gain


SAFE_Z = 1.96


def select_static_source(
    action_targets: Mapping[str, Sequence[ContributionTarget]],
    *,
    minimum_gain: float = 0.0,
) -> StaticSelection:
    """Select one A1 action on same-H selection cases, otherwise exact B."""

    if float(minimum_gain) != 0.0:
        raise ProtocolError("The diagnostic freezes the static minimum gain at zero.")
    candidates: list[tuple[float, str]] = []
    for action_id, targets in action_targets.items():
        action = str(action_id)
        if action == "B":
            continue
        if not action.startswith("A1"):
            raise ProtocolError("Static challenger selection is restricted to A1 actions.")
        rows = tuple(targets)
        if not rows or any(row.action_id != action for row in rows):
            raise ProtocolError("Static action target identity drifted.")
        candidates.append((pooled_gain(rows), action))
    if not candidates:
        return StaticSelection("B", 0.0, 0.0, True)
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    best_gain, best_action = ranked[0]
    runner_up = max(0.0, ranked[1][0] if len(ranked) > 1 else 0.0)
    if best_gain <= 0.0:
        return StaticSelection("B", 0.0, max(best_gain, 0.0), True)
    return StaticSelection(best_action, float(best_gain), float(runner_up), False)


def select_case_action(
    *,
    method_id: str,
    challenger: StaticSelection,
    features: CaseActionFeatures,
    prediction: TwoHeadPrediction,
    calibration: DirectionSharedCalibration,
    z: float = SAFE_Z,
    minimum_gain: float = 0.0,
    runner_up_margin: float = 0.0,
) -> CaseDecision:
    """Apply the frozen descriptive score bound; this is not calibrated confidence."""

    if float(z) != SAFE_Z or float(minimum_gain) != 0.0 or float(runner_up_margin) != 0.0:
        raise ProtocolError("Safe router thresholds are frozen at z=1.96 and zero margins.")
    if challenger.action_id == "B":
        return _fallback(method_id, features.case_id, "B", "static_fallback_B")
    if features.action_id != challenger.action_id or prediction.action_id != challenger.action_id:
        raise ProtocolError("Case inputs do not match the sealed static challenger.")
    if not calibration.valid:
        return _fallback(method_id, features.case_id, challenger.action_id, "single_class_calibration")
    if not features.has_flips:
        return _fallback(method_id, features.case_id, challenger.action_id, "zero_flip")
    mean, standard_error = calibrated_gain(calibration, prediction, features)
    lower = mean - SAFE_Z * standard_error
    if lower <= 0.0:
        selected, reason = "B", "nonpositive_lcb"
    elif mean <= 0.0:
        selected, reason = "B", "nonpositive_gain"
    else:
        selected, reason = challenger.action_id, "heuristic_positive_bound"
    return CaseDecision(
        method_id=str(method_id),
        case_id=features.case_id,
        selected_action_id=selected,
        challenger_action_id=challenger.action_id,
        predicted_gain=float(mean),
        standard_error=float(standard_error),
        lower_confidence_bound=float(lower),
        reason=reason,
    )


def _fallback(method_id: str, case_id: str, challenger: str, reason: str) -> CaseDecision:
    return CaseDecision(
        method_id=str(method_id),
        case_id=str(case_id),
        selected_action_id="B",
        challenger_action_id=str(challenger),
        predicted_gain=0.0,
        standard_error=0.0,
        lower_confidence_bound=0.0,
        reason=reason,
    )


__all__ = ("SAFE_Z", "select_case_action", "select_static_source")
