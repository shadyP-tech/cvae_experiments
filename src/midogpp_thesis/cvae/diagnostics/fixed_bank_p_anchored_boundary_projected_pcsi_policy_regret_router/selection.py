"""Frozen target-influence selection with an outer-donor Pareto veto."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BACC_ONLY_METHOD_ID,
    DIRECTION_IDS,
    FULL_ONLY_METHOD_ID,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    UTILITY_ZERO_TOLERANCE,
)
from .sample_influence_contracts import InfluencePrediction
from .utility_contracts import (
    DirectionalDecision,
    UtilityDescriptor,
    UtilityPrediction,
)


def select_directional_actions(
    descriptors: Sequence[UtilityDescriptor],
    influence_predictions: Sequence[InfluencePrediction],
    donor_predictions: Sequence[UtilityPrediction],
    *,
    policy_id: str,
) -> tuple[DirectionalDecision, ...]:
    """Select at most one action per direction, otherwise preserve exact P."""

    rows = tuple(descriptors)
    influences = {row.descriptor_hash: row for row in influence_predictions}
    donors = {row.descriptor_hash: row for row in donor_predictions}
    descriptor_hashes = {row.descriptor_hash for row in rows}
    if (
        policy_id not in {
            MODEL_BASED_METHOD_ID,
            BACC_ONLY_METHOD_ID,
            FULL_ONLY_METHOD_ID,
            PERMUTATION_METHOD_ID,
        }
        or len(rows) != 6
        or len(influences) != 6
        or len(donors) != 6
        or descriptor_hashes != set(influences)
        or descriptor_hashes != set(donors)
        or len({(row.alternative, row.direction) for row in rows}) != 6
    ):
        raise ProtocolError("PCSI-PARC directional selection rectangle drifted.")
    center = rows[0].target_center
    case = rows[0].case_id
    if any(row.target_center != center or row.case_id != case for row in rows):
        raise ProtocolError("PCSI-PARC directional selection spans cases.")

    decisions: list[DirectionalDecision] = []
    alternative_order = {
        alternative: index for index, alternative in enumerate(ALTERNATIVE_METHOD_IDS)
    }
    for direction in DIRECTION_IDS:
        direction_rows = tuple(row for row in rows if row.direction == direction)
        candidates: list[tuple[float, int, UtilityDescriptor]] = []
        for descriptor in direction_rows:
            target = influences[descriptor.descriptor_hash]
            donor = donors[descriptor.descriptor_hash]
            if (
                target.key != descriptor.key
                or target.crossing_count != descriptor.crossing_count
            ):
                raise ProtocolError("PCSI-PARC influence/descriptor binding drifted.")
            score = target.target_score
            positive_target = (
                descriptor.crossing_count > 0 and score > UTILITY_ZERO_TOLERANCE
            )
            donor_bacc_safe = donor.robust("bacc_contribution_delta") > 0.0
            donor_proper_safe = (
                donor.robust("brier_contribution_delta") <= 0.0
                and donor.robust("log_loss_contribution_delta") <= 0.0
            )
            if policy_id == BACC_ONLY_METHOD_ID:
                admissible = positive_target
            elif policy_id == FULL_ONLY_METHOD_ID:
                admissible = positive_target and donor_proper_safe
            else:
                admissible = positive_target and donor_bacc_safe and donor_proper_safe
            if admissible:
                candidates.append(
                    (score, -alternative_order[descriptor.alternative], descriptor)
                )
        selected = max(candidates, default=None, key=lambda row: (row[0], row[1]))
        ordered_rows = tuple(
            sorted(direction_rows, key=lambda row: alternative_order[row.alternative])
        )
        decisions.append(
            DirectionalDecision(
                center,
                case,
                policy_id,
                direction,
                selected[2].alternative if selected is not None else PORTFOLIO_METHOD_ID,
                float(selected[0]) if selected is not None else 0.0,
                tuple(
                    digest
                    for row in ordered_rows
                    for digest in (
                        influences[row.descriptor_hash].influence_hash,
                        donors[row.descriptor_hash].prediction_hash,
                    )
                ),
            )
        )
    return tuple(decisions)


__all__ = ("select_directional_actions",)
