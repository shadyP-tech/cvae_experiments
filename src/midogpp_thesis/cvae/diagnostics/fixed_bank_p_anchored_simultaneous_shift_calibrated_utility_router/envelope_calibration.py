"""Nested donor-held calibration for the simultaneous utility envelope."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DIRECTION_IDS,
    MODEL_BASED_METHOD_ID,
    UTILITY_ZERO_TOLERANCE,
)
from .selection import select_certificate_for_direction
from .shift_certificate import certify_utility
from .simultaneous_envelope import FittedEnvelopeModel, fit_simultaneous_envelope
from .tail_risk import lower_tail_mean, upper_tail_mean
from .utility_contracts import (
    DonorUtilityRow,
    EnvelopeCalibration,
    InnerDonorReplay,
    PosteriorUtilityPrediction,
)


def calibrate_envelope(
    *,
    outer_target_center: str,
    control_id: str,
    predictions: Sequence[PosteriorUtilityPrediction],
    donor_rows: Sequence[DonorUtilityRow],
) -> EnvelopeCalibration:
    """Fit the final model and audit it with whole-donor nested replay."""

    utilities = tuple(predictions)
    responses = tuple(donor_rows)
    donors = tuple(center for center in CENTERS if center != outer_target_center)
    if (
        outer_target_center not in CENTERS
        or not utilities
        or not responses
        or {row.target_center for row in utilities} != set(donors)
        or {row.outer_target_center for row in responses} != {outer_target_center}
        or {row.donor_center for row in responses} != set(donors)
        or {row.control_id for row in utilities} != {control_id}
        or {row.descriptor_hash for row in utilities}
        != {row.descriptor_hash for row in responses}
    ):
        raise ProtocolError("PSSCUR envelope calibration scope drifted.")

    final_model = fit_simultaneous_envelope(
        utilities, responses, allowed_donors=donors
    )
    inner: list[InnerDonorReplay] = []
    for held_donor in donors:
        training_donors = tuple(
            center for center in donors if center != held_donor
        )
        model = fit_simultaneous_envelope(
            utilities, responses, allowed_donors=training_donors
        )
        selected_count, bacc, brier, log_loss = _evaluate_donors(
            model,
            utilities,
            responses,
            allowed_donors=(held_donor,),
        )
        inner.append(
            InnerDonorReplay(
                outer_target_center,
                control_id,
                held_donor,
                len(training_donors),
                selected_count,
                bacc[held_donor],
                brier[held_donor],
                log_loss[held_donor],
                model.model_hash,
            )
        )

    selected_count, _bacc, _brier, _log_loss = _evaluate_donors(
        final_model,
        utilities,
        responses,
        allowed_donors=donors,
    )
    lower_bacc = lower_tail_mean([row.bacc_delta for row in inner])
    upper_brier = upper_tail_mean([row.brier_delta for row in inner])
    upper_log = upper_tail_mean([row.log_loss_delta for row in inner])
    nonvacuous = selected_count > 0 and sum(
        row.selected_action_count for row in inner
    ) > 0
    authorized = (
        nonvacuous
        and lower_bacc >= -UTILITY_ZERO_TOLERANCE
        and upper_brier <= UTILITY_ZERO_TOLERANCE
        and upper_log <= UTILITY_ZERO_TOLERANCE
    )
    return EnvelopeCalibration(
        outer_target_center,
        control_id,
        authorized,
        final_model.residual_scales,
        final_model.feature_references,
        final_model.direction_envelopes,
        selected_count,
        lower_bacc,
        upper_brier,
        upper_log,
        tuple(inner),
        final_model.source_utility_hash,
        final_model.source_response_hash,
    )


def _evaluate_donors(
    model: FittedEnvelopeModel,
    predictions: Sequence[PosteriorUtilityPrediction],
    responses: Sequence[DonorUtilityRow],
    *,
    allowed_donors: Sequence[str],
) -> tuple[
    int,
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    allowed = tuple(allowed_donors)
    response_by_hash = {row.descriptor_hash: row for row in responses}
    groups: dict[tuple[str, str, str], list[PosteriorUtilityPrediction]] = {}
    for prediction in predictions:
        if prediction.target_center in set(allowed):
            groups.setdefault(
                (
                    prediction.target_center,
                    prediction.case_id,
                    prediction.direction,
                ),
                [],
            ).append(prediction)
    bacc = {center: 0.0 for center in allowed}
    brier = {center: 0.0 for center in allowed}
    log_loss = {center: 0.0 for center in allowed}
    selected_count = 0
    for (donor, _case, direction), utilities in sorted(groups.items()):
        if direction not in DIRECTION_IDS:
            raise ProtocolError("PSSCUR donor replay direction drifted.")
        certificates = tuple(
            certify_utility(
                response_by_hash[row.descriptor_hash],
                row,
                model,
                policy_id=MODEL_BASED_METHOD_ID,
                calibration_hash=model.model_hash,
            )
            for row in utilities
        )
        selected = select_certificate_for_direction(certificates)
        if selected is None:
            continue
        response = response_by_hash[selected.descriptor_hash]
        if response.donor_center != donor:
            raise ProtocolError("PSSCUR donor replay binding drifted.")
        bacc[donor] += response.bacc_contribution_delta
        brier[donor] += response.brier_contribution_delta
        log_loss[donor] += response.log_loss_contribution_delta
        selected_count += 1
    return selected_count, bacc, brier, log_loss


__all__ = ("calibrate_envelope",)
