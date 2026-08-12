"""Winner-versus-runner-up margin routing with joint covariance."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ActionScore,
    CandidateMenu,
    DIRECTIONS,
    DirectionalCalibration,
    DirectionalLogitModel,
    MultiChallengerDecision,
)


MARGIN_Z = 1.96


def select_action_with_margin(
    *,
    case_id: str,
    method_id: str,
    menu: CandidateMenu,
    scores: Sequence[ActionScore],
    models: Mapping[str, DirectionalLogitModel],
    calibrations: Mapping[str, DirectionalCalibration],
    z: float = MARGIN_Z,
) -> MultiChallengerDecision:
    """Switch away from the support-static anchor only on a positive margin LCB."""

    if float(z) != MARGIN_Z or set(models) != set(DIRECTIONS) or set(calibrations) != set(DIRECTIONS):
        raise ProtocolError("Multi-challenger decision topology drifted.")
    by_action = {row.action_id: row for row in scores}
    if len(by_action) != len(tuple(scores)) or set(by_action) != set(menu.action_ids):
        raise ProtocolError("Multi-challenger score menu drifted.")
    if not all(calibrations[direction].valid for direction in DIRECTIONS):
        return _fallback_invalid(case_id, method_id, menu, by_action)
    ranked = tuple(
        sorted(by_action.values(), key=lambda row: (-row.expected_gain, row.action_id))
    )
    best, runner_up = ranked[:2]
    epistemic_variance, calibration_variance = _paired_margin_variance(
        best, runner_up, models=models, calibrations=calibrations
    )
    margin = best.expected_gain - runner_up.expected_gain
    margin_standard_error = math.sqrt(epistemic_variance + calibration_variance)
    margin_lcb = margin - MARGIN_Z * margin_standard_error
    if best.action_id == menu.anchor_action_id:
        selected = menu.anchor_action_id
        reason = "anchor_top_ranked"
    elif margin_lcb > 0.0:
        selected = best.action_id
        reason = "positive_winner_runner_up_margin_lcb"
    else:
        selected = menu.anchor_action_id
        reason = "nonpositive_winner_runner_up_margin_lcb"
    selected_score = by_action[selected]
    return MultiChallengerDecision(
        case_id=str(case_id),
        method_id=str(method_id),
        anchor_action_id=menu.anchor_action_id,
        selected_action_id=selected,
        best_action_id=best.action_id,
        runner_up_action_id=runner_up.action_id,
        predicted_gain=selected_score.expected_gain,
        action_margin=margin,
        epistemic_standard_error=math.sqrt(epistemic_variance),
        calibration_standard_error=math.sqrt(calibration_variance),
        margin_standard_error=margin_standard_error,
        margin_lcb=margin_lcb,
        reason=reason,
        menu_hash=menu.menu_hash,
    )


def _paired_margin_variance(
    first: ActionScore,
    second: ActionScore,
    *,
    models: Mapping[str, DirectionalLogitModel],
    calibrations: Mapping[str, DirectionalCalibration],
) -> tuple[float, float]:
    epistemic = 0.0
    calibration = 0.0
    for direction in DIRECTIONS:
        difference = np.asarray(first.model_gradients[direction], dtype=np.float64) - np.asarray(
            second.model_gradients[direction], dtype=np.float64
        )
        covariance = np.asarray(models[direction].covariance, dtype=np.float64)
        if difference.shape != (models[direction].dimension,):
            raise ProtocolError("Paired action gradient dimension drifted.")
        epistemic += max(float(difference @ covariance @ difference), 0.0)
        calibration_difference = (
            first.calibration_gradients[direction]
            - second.calibration_gradients[direction]
        )
        calibration += (
            calibration_difference
            * calibration_difference
            * calibrations[direction].offset_variance
        )
    return float(epistemic), float(calibration)


def _fallback_invalid(
    case_id: str,
    method_id: str,
    menu: CandidateMenu,
    scores: Mapping[str, ActionScore],
) -> MultiChallengerDecision:
    anchor = scores[menu.anchor_action_id]
    ranked = tuple(
        sorted(scores.values(), key=lambda row: (-row.expected_gain, row.action_id))
    )
    return MultiChallengerDecision(
        case_id=str(case_id),
        method_id=str(method_id),
        anchor_action_id=menu.anchor_action_id,
        selected_action_id=menu.anchor_action_id,
        best_action_id=ranked[0].action_id,
        runner_up_action_id=ranked[1].action_id,
        predicted_gain=anchor.expected_gain,
        action_margin=ranked[0].expected_gain - ranked[1].expected_gain,
        epistemic_standard_error=0.0,
        calibration_standard_error=0.0,
        margin_standard_error=0.0,
        margin_lcb=0.0,
        reason="invalid_calibration_anchor_fallback",
        menu_hash=menu.menu_hash,
    )


__all__ = ("MARGIN_Z", "select_action_with_margin")
