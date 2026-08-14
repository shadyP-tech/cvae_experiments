"""Terminal-only identification reselection after delete-center G recomputation."""

from __future__ import annotations

from collections.abc import Sequence

from ...protocol import ProtocolError
from .constants import DIRECTION_IDS, IDENTIFICATION_CASE_WEIGHT, IDENTIFICATION_DONOR_WEIGHT, TIE_TOLERANCE, candidate_sources
from .identification_products import (
    CaseIdentificationDecision,
    DirectionIdentificationDecision,
    IdentificationCandidateScore,
)


def _reselect_direction(
    base: DirectionIdentificationDecision,
    priors: dict[tuple[str, str], object],
) -> DirectionIdentificationDecision:
    donor_values = {
        source: float(getattr(priors[(source, base.direction)], "value"))
        for source in candidate_sources(base.target_center)
    }
    donor_scale = sum(abs(value) for value in donor_values.values()) / len(donor_values)
    rows: list[IdentificationCandidateScore] = []
    for previous in base.candidate_scores:
        donor = donor_values[previous.source]
        normalized_donor = 0.0 if donor_scale == 0.0 else donor / donor_scale
        final = (
            float(IDENTIFICATION_CASE_WEIGHT) * previous.normalized_case_proxy
            + float(IDENTIFICATION_DONOR_WEIGHT) * normalized_donor
        )
        rows.append(
            IdentificationCandidateScore(
                previous.target_center,
                previous.case_id,
                previous.direction,
                previous.source,
                previous.predicted_correctness,
                previous.directional_flip_count,
                previous.model_valid,
                previous.case_proxy,
                donor,
                previous.case_scale,
                donor_scale,
                previous.normalized_case_proxy,
                normalized_donor,
                final,
                previous.opportunity_eligible,
                previous.eligible,
                previous.eligibility_reason,
                previous.model_hash,
            )
        )
    fail_closed = bool(base.case_scale == 0.0)
    eligible = tuple(row for row in rows if row.eligible)
    if fail_closed or not eligible or max(row.final_score for row in eligible) <= TIE_TOLERANCE:
        selected = None
    else:
        maximum = max(row.final_score for row in eligible)
        selected = min((row.source for row in eligible if maximum - row.final_score <= TIE_TOLERANCE), key=int)
    opportunity = tuple(row for row in rows if row.opportunity_eligible)
    if not opportunity:
        source_only = None
    else:
        maximum = max(row.final_score for row in opportunity)
        source_only = min((row.source for row in opportunity if maximum - row.final_score <= TIE_TOLERANCE), key=int)
    return DirectionIdentificationDecision(
        base.method_id,
        base.target_center,
        base.case_id,
        base.direction,
        tuple(rows),
        selected,
        source_only,
        tuple(row.source for row in rows if row.eligible),
        base.case_scale,
        donor_scale,
        fail_closed,
        "terminal_delete_center_G_normalization_reselection",
    )


def reselect_identification_for_priors(
    base: CaseIdentificationDecision,
    donor_priors: Sequence[object],
) -> CaseIdentificationDecision:
    priors = {(str(getattr(row, "source")), str(getattr(row, "direction"))): row for row in donor_priors}
    expected = tuple((source, direction) for source in candidate_sources(base.target_center) for direction in DIRECTION_IDS)
    if tuple(priors) != expected:
        raise ProtocolError("OGDE delete-center reselection lacks exact donor-prior topology.")
    zero_to_one = _reselect_direction(base.zero_to_one, priors)
    one_to_zero = _reselect_direction(base.one_to_zero, priors)
    return CaseIdentificationDecision(
        base.method_id,
        base.target_center,
        base.case_id,
        zero_to_one,
        one_to_zero,
    )


__all__ = ("reselect_identification_for_priors",)
