"""Frozen posterior-utility selection with donor-held abstention margins."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import (
    BACC_ONLY_METHOD_ID,
    DIRECTION_IDS,
    FULL_ONLY_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PORTFOLIO_METHOD_ID,
)
from .margin_calibration import select_prediction_for_direction
from .utility_contracts import (
    DirectionalDecision,
    MarginCalibration,
    PosteriorUtilityPrediction,
    UtilityDescriptor,
)


def select_directional_actions(
    descriptors: Sequence[UtilityDescriptor],
    utility_predictions: Sequence[PosteriorUtilityPrediction],
    calibration: MarginCalibration,
    *,
    policy_id: str,
) -> tuple[DirectionalDecision, ...]:
    """Select at most one action per direction, otherwise preserve exact P."""

    rows = tuple(descriptors)
    predictions = {row.descriptor_hash: row for row in utility_predictions}
    descriptor_hashes = {row.descriptor_hash for row in rows}
    if (
        policy_id
        not in {
            MODEL_BASED_METHOD_ID,
            BACC_ONLY_METHOD_ID,
            FULL_ONLY_METHOD_ID,
            PERMUTATION_METHOD_ID,
        }
        or len(rows) != 6
        or len(predictions) != 6
        or descriptor_hashes != set(predictions)
        or len({(row.alternative, row.direction) for row in rows}) != 6
    ):
        raise ProtocolError("PUMR directional selection rectangle drifted.")
    center = rows[0].target_center
    case = rows[0].case_id
    if (
        any(row.target_center != center or row.case_id != case for row in rows)
        or calibration.outer_target_center != center
        or any(predictions[row.descriptor_hash].key != row.key for row in rows)
    ):
        raise ProtocolError("PUMR directional selection scope drifted.")

    require_proper = policy_id != BACC_ONLY_METHOD_ID
    margin = 0.0 if policy_id == FULL_ONLY_METHOD_ID else calibration.selected_margin
    calibration_required = policy_id in {
        MODEL_BASED_METHOD_ID,
        PERMUTATION_METHOD_ID,
    }
    force_fallback = calibration_required and not calibration.authorized
    decisions: list[DirectionalDecision] = []
    for direction in DIRECTION_IDS:
        direction_rows = tuple(row for row in rows if row.direction == direction)
        candidate_predictions = tuple(
            predictions[row.descriptor_hash] for row in direction_rows
        )
        selected = (
            None
            if force_fallback
            else select_prediction_for_direction(
                candidate_predictions,
                margin=margin,
                require_proper=require_proper,
            )
        )
        decisions.append(
            DirectionalDecision(
                center,
                case,
                policy_id,
                direction,
                selected.alternative if selected is not None else PORTFOLIO_METHOD_ID,
                selected.robust_bacc_lower if selected is not None else 0.0,
                tuple(
                    [
                        row.utility_hash
                        for row in sorted(
                            candidate_predictions,
                            key=lambda value: value.alternative,
                        )
                    ]
                    + [calibration.calibration_hash]
                ),
            )
        )
    return tuple(decisions)


__all__ = ("select_directional_actions",)
