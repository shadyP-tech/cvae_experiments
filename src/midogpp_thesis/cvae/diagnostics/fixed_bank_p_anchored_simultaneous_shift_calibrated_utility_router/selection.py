"""Frozen selection from simultaneous shift-calibrated utility bounds."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    BLOCKED_CONTROL_METHOD_ID,
    BOUND_STRICT_TOLERANCE,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
)
from .utility_contracts import (
    DirectionalDecision,
    EnvelopeCalibration,
    UtilityCertificate,
    UtilityDescriptor,
)


def select_certificate_for_direction(
    certificates: Sequence[UtilityCertificate],
) -> UtilityCertificate | None:
    """Select one admissible endpoint; P wins every numerical tie at zero."""

    rows = tuple(certificates)
    if (
        len(rows) != len(ALTERNATIVE_METHOD_IDS)
        or len({row.alternative for row in rows}) != len(ALTERNATIVE_METHOD_IDS)
        or len({row.direction for row in rows}) != 1
        or len({(row.target_center, row.case_id, row.policy_id) for row in rows}) != 1
    ):
        raise ProtocolError("PSSCUR directional certificate set drifted.")
    order = {
        alternative: index
        for index, alternative in enumerate(ALTERNATIVE_METHOD_IDS)
    }
    admissible = tuple(
        row
        for row in rows
        if row.admissible and row.lower_bacc_delta > BOUND_STRICT_TOLERANCE
    )
    return max(
        admissible,
        default=None,
        key=lambda row: (row.lower_bacc_delta, -order[row.alternative]),
    )


def select_directional_actions(
    descriptors: Sequence[UtilityDescriptor],
    certificates: Sequence[UtilityCertificate],
    calibration: EnvelopeCalibration,
    *,
    policy_id: str,
) -> tuple[DirectionalDecision, ...]:
    """Select at most one action per direction, otherwise preserve exact P."""

    descriptor_rows = tuple(descriptors)
    certificate_rows = tuple(certificates)
    descriptor_keys = {row.key for row in descriptor_rows}
    certificate_keys = {
        (row.target_center, row.case_id, row.alternative, row.direction)
        for row in certificate_rows
    }
    if (
        policy_id not in COMPOSED_POLICY_IDS
        or len(descriptor_rows) != 6
        or len(certificate_rows) != 6
        or descriptor_keys != certificate_keys
        or {row.policy_id for row in certificate_rows} != {policy_id}
    ):
        raise ProtocolError("PSSCUR directional selection rectangle drifted.")
    center = descriptor_rows[0].target_center
    case = descriptor_rows[0].case_id
    if (
        any(
            row.target_center != center or row.case_id != case
            for row in descriptor_rows
        )
        or calibration.outer_target_center != center
        or any(
            row.calibration_hash != calibration.calibration_hash
            for row in certificate_rows
        )
    ):
        raise ProtocolError("PSSCUR directional selection scope drifted.")

    decisions: list[DirectionalDecision] = []
    authorization_required = policy_id in {
        PRIMARY_METHOD_ID,
        BLOCKED_CONTROL_METHOD_ID,
    }
    for direction in DIRECTION_IDS:
        candidates = tuple(
            row for row in certificate_rows if row.direction == direction
        )
        selected = (
            select_certificate_for_direction(candidates)
            if calibration.authorized or not authorization_required
            else None
        )
        decisions.append(
            DirectionalDecision(
                center,
                case,
                policy_id,
                direction,
                (
                    selected.alternative
                    if selected is not None
                    else PORTFOLIO_METHOD_ID
                ),
                selected.lower_bacc_delta if selected is not None else 0.0,
                tuple(
                    [
                        row.certificate_hash
                        for row in sorted(
                            candidates, key=lambda value: value.alternative
                        )
                    ]
                    + [calibration.calibration_hash]
                ),
            )
        )
    return tuple(decisions)


__all__ = ("select_certificate_for_direction", "select_directional_actions")
