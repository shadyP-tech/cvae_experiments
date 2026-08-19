"""Deterministic one-action-per-direction selection with exact P fallback."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BACC_ONLY_METHOD_ID,
    DELETE_POSITIVE_FRACTION_MIN,
    DIRECTION_IDS,
    FULL_ONLY_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    RESIDUAL_MARGIN_MULTIPLIER,
    UTILITY_ZERO_TOLERANCE,
)
from .utility_contracts import (
    DirectionalDecision,
    UtilityDescriptor,
    UtilityPrediction,
)


def select_directional_actions(
    descriptors: Sequence[UtilityDescriptor],
    predictions: Sequence[UtilityPrediction],
    *,
    policy_id: str,
) -> tuple[DirectionalDecision, ...]:
    rows = tuple(descriptors)
    predicted = {row.descriptor_hash: row for row in predictions}
    if (
        policy_id
        not in {
            MODEL_BASED_METHOD_ID,
            BACC_ONLY_METHOD_ID,
            FULL_ONLY_METHOD_ID,
            PERMUTATION_METHOD_ID,
        }
        or len(rows) != 6
        or len(predicted) != 6
        or {row.descriptor_hash for row in rows} != set(predicted)
        or len({(row.alternative, row.direction) for row in rows}) != 6
    ):
        raise ProtocolError("PDSUR directional selection rectangle drifted.")
    center = rows[0].target_center
    case = rows[0].case_id
    if any(row.target_center != center or row.case_id != case for row in rows):
        raise ProtocolError("PDSUR directional selection spans cases.")

    decisions: list[DirectionalDecision] = []
    alternative_order = {
        alternative: index for index, alternative in enumerate(ALTERNATIVE_METHOD_IDS)
    }
    for direction in DIRECTION_IDS:
        candidates: list[tuple[float, int, UtilityDescriptor]] = []
        direction_rows = tuple(row for row in rows if row.direction == direction)
        for descriptor in direction_rows:
            utility = predicted[descriptor.descriptor_hash]
            if descriptor.crossing_count == 0:
                continue
            if policy_id == FULL_ONLY_METHOD_ID:
                score = utility.full("bacc_contribution_delta")
                admissible = score > UTILITY_ZERO_TOLERANCE
            else:
                score = utility.robust("bacc_contribution_delta") - (
                    RESIDUAL_MARGIN_MULTIPLIER
                    * utility.scale("bacc_contribution_delta")
                )
                admissible = (
                    score > UTILITY_ZERO_TOLERANCE
                    and utility.fraction("bacc_contribution_delta")
                    >= DELETE_POSITIVE_FRACTION_MIN
                )
                if policy_id in {MODEL_BASED_METHOD_ID, PERMUTATION_METHOD_ID}:
                    admissible = admissible and (
                        utility.robust("brier_contribution_delta") <= 0.0
                        and utility.robust("log_loss_contribution_delta") <= 0.0
                        and utility.fraction("brier_contribution_delta")
                        >= DELETE_POSITIVE_FRACTION_MIN
                        and utility.fraction("log_loss_contribution_delta")
                        >= DELETE_POSITIVE_FRACTION_MIN
                    )
            if admissible:
                candidates.append(
                    (score, -alternative_order[descriptor.alternative], descriptor)
                )
        selected = max(candidates, default=None, key=lambda value: (value[0], value[1]))
        decisions.append(
            DirectionalDecision(
                center,
                case,
                policy_id,
                direction,
                selected[2].alternative if selected is not None else PORTFOLIO_METHOD_ID,
                float(selected[0]) if selected is not None else 0.0,
                tuple(
                    predicted[row.descriptor_hash].prediction_hash
                    for row in sorted(
                        direction_rows,
                        key=lambda row: alternative_order[row.alternative],
                    )
                ),
            )
        )
    return tuple(decisions)


__all__ = ("select_directional_actions",)
