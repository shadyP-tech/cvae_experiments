"""Exact-rational directional shrinkage decisions with OFF-first ties."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import (
    ARM_IDS,
    DIRECTION_IDS,
    TIE_TOLERANCE,
    candidate_sources,
)
from .endpoint_library import EndpointLibrary, build_endpoint_library
from .products import (
    CandidateScore,
    CaseArmDecision,
    CaseControlDecision,
    CaseActionConfusion,
    DirectionalGain,
    DirectionalControlDecision,
    DirectionDecision,
    DonorPrior,
    EndpointArm,
)
from .loo_plans import WholeCaseLooPlan
from .scoring import directional_hard_flip_gain


EXACT_TIE_TOLERANCE = Fraction(1, 10**12)


def _gain_map(
    rows: Sequence[DirectionalGain] | Sequence[DonorPrior],
    *,
    target_center: str,
    direction: str,
) -> dict[str, Fraction]:
    result: dict[str, Fraction] = {}
    for row in rows:
        if row.direction != direction:
            continue
        if isinstance(row, DirectionalGain):
            if row.query_center != target_center:
                continue
        else:
            if row.heldout_center != target_center:
                continue
        if row.source in result:
            raise ProtocolError("DCSE gain surface duplicates a source/direction cell.")
        result[row.source] = row.exact
    expected = candidate_sources(target_center)
    if tuple(result) != expected:
        raise ProtocolError("DCSE gain surface lacks exact eight-source canonical coverage.")
    return result


def rank_sources_by_prior(
    priors: Mapping[str, Fraction], *, target_center: str
) -> tuple[str, ...]:
    expected = candidate_sources(target_center)
    if tuple(priors) != expected:
        raise ProtocolError("DCSE G ranking must cover eight canonical non-target sources.")
    # Candidate prescreening uses exact G.  The frozen tolerance belongs only
    # to the final endpoint score selection, never to top-K membership.
    return tuple(sorted(expected, key=lambda source: (-priors[source], int(source))))


def _choose_exact(scores: Sequence[CandidateScore]) -> str | None:
    if not scores or scores[0].source is not None:
        raise ProtocolError("DCSE decision scores must begin with OFF.")
    best = max(row.score for row in scores)
    # Source order is already OFF then numeric.  Apply the frozen tolerance as
    # an exact rational gap, so values around 1e-12 cannot acquire a binary64
    # rounding-dependent decision.
    eligible = tuple(
        row.source for row in scores if best - row.score <= EXACT_TIE_TOLERANCE
    )
    # The retained menu is ordered by exact G, but the *final* tie rule is an
    # independent OFF-then-numeric-source order.
    return min(
        eligible,
        key=lambda source: (-1 if source is None else int(source)),
    )


def choose_with_frozen_tolerance(
    sources: Sequence[str | None],
    scores: Sequence[float],
    *,
    tie_tolerance: float = TIE_TOLERANCE,
) -> str | None:
    """Fallback for non-rational inputs; never used for canonical Fraction paths."""

    identities = tuple(sources)
    values = tuple(float(value) for value in scores)
    if (
        float(tie_tolerance) != TIE_TOLERANCE
        or len(identities) != len(values)
        or not identities
        or identities[0] is not None
    ):
        raise ProtocolError("DCSE frozen-tolerance tie inputs drifted.")
    maximum = max(values)
    return next(source for source, value in zip(identities, values, strict=True) if maximum - value <= tie_tolerance)


def select_direction_decision(
    *,
    method_id: str,
    target_center: str,
    case_id: str,
    arm: EndpointArm,
    direction: str,
    support_gains: Sequence[DirectionalGain] | Sequence[DonorPrior],
    donor_priors: Sequence[DonorPrior],
) -> DirectionDecision:
    """Rank by G, retain top K, then maximize w*S+(1-w)*G against OFF=0."""

    target = str(target_center)
    if direction not in DIRECTION_IDS or arm.arm_id not in ARM_IDS:
        raise ProtocolError("DCSE direction/arm drifted.")
    if method_id not in {"DCSE_LOO", "G_directional_matched"}:
        raise ProtocolError("DCSE endpoint decision method drifted.")
    priors = _gain_map(donor_priors, target_center=target, direction=direction)
    support = _gain_map(support_gains, target_center=target, direction=direction)
    if method_id == "G_directional_matched" and support != priors:
        raise ProtocolError("Matched G must execute the identical pipeline with S := G.")
    retained = rank_sources_by_prior(priors, target_center=target)[: arm.k]
    scores = [CandidateScore.from_fractions(None, Fraction(0), Fraction(0), Fraction(0))]
    for source in retained:
        score = arm.weight * support[source] + (1 - arm.weight) * priors[source]
        scores.append(CandidateScore.from_fractions(source, support[source], priors[source], score))
    selected = _choose_exact(scores)
    return DirectionDecision(
        method_id=method_id,
        target_center=target,
        case_id=str(case_id),
        arm_id=arm.arm_id,
        direction=direction,
        retained_sources=retained,
        scores=tuple(scores),
        selected_source=selected,
        decision_rule=(
            "exact_fraction_max_OFF_then_numeric_source_tie"
            if method_id == "DCSE_LOO"
            else "matched_G_exact_fraction_max_OFF_then_numeric_source_tie_S_equals_G"
        ),
    )


def select_arm_decision(
    *,
    method_id: str,
    target_center: str,
    case_id: str,
    arm: EndpointArm,
    support_gains: Sequence[DirectionalGain] | Sequence[DonorPrior],
    donor_priors: Sequence[DonorPrior],
) -> CaseArmDecision:
    rows = tuple(
        select_direction_decision(
            method_id=method_id,
            target_center=target_center,
            case_id=case_id,
            arm=arm,
            direction=direction,
            support_gains=support_gains,
            donor_priors=donor_priors,
        )
        for direction in DIRECTION_IDS
    )
    return CaseArmDecision(
        method_id,
        str(target_center),
        str(case_id),
        arm.arm_id,
        rows[0],
        rows[1],
    )


def select_arm_decisions(
    *,
    method_id: str,
    target_center: str,
    case_id: str,
    support_gains: Sequence[DirectionalGain] | Sequence[DonorPrior],
    donor_priors: Sequence[DonorPrior],
    library: EndpointLibrary | None = None,
) -> tuple[CaseArmDecision, ...]:
    endpoint = library or build_endpoint_library(method_id)
    if endpoint.method_id != method_id:
        raise ProtocolError("DCSE endpoint library/method mismatch.")
    return tuple(
        select_arm_decision(
            method_id=method_id,
            target_center=target_center,
            case_id=case_id,
            arm=arm,
            support_gains=support_gains,
            donor_priors=donor_priors,
        )
        for arm in endpoint.arms
    )


def select_matched_g_decisions(
    *,
    target_center: str,
    case_id: str,
    donor_priors: Sequence[DonorPrior],
) -> tuple[CaseArmDecision, ...]:
    return select_arm_decisions(
        method_id="G_directional_matched",
        target_center=target_center,
        case_id=case_id,
        support_gains=donor_priors,
        donor_priors=donor_priors,
    )


def _choose_fraction_values(
    values: Mapping[str | None, Fraction], *, tolerance: Fraction
) -> str | None:
    expected = (None, *sorted((source for source in values if source is not None), key=int))
    if tuple(values) != expected:
        raise ProtocolError("DCSE control values must be OFF then numeric sources.")
    maximum = max(values.values())
    return next(source for source in expected if maximum - values[source] <= tolerance)


def select_raw_directional_loo_control(
    *,
    target_center: str,
    case_id: str,
    support_gains: Sequence[DirectionalGain],
) -> CaseControlDecision:
    target = str(target_center)
    output: list[DirectionalControlDecision] = []
    for direction in DIRECTION_IDS:
        gains = _gain_map(support_gains, target_center=target, direction=direction)
        values: dict[str | None, Fraction] = {
            None: Fraction(0),
            **{source: gains[source] for source in sorted(gains, key=int)},
        }
        selected = _choose_fraction_values(values, tolerance=EXACT_TIE_TOLERANCE)
        output.append(
            DirectionalControlDecision(
                "DLOO_raw",
                target,
                str(case_id),
                direction,
                selected,
                tuple(
                    (source, value.numerator, value.denominator)
                    for source, value in values.items()
                ),
                0,
            )
        )
    return CaseControlDecision(
        "DLOO_raw", target, str(case_id), output[0], output[1]
    )


def select_nested_frequency_committee_control(
    *,
    plan: WholeCaseLooPlan,
    support_counts: Sequence[CaseActionConfusion],
) -> CaseControlDecision:
    """Delete each H-minus-c support case and vote using exact rational DLOO."""

    target = plan.target_center
    support_cases = tuple(plan.support_case_ids)
    if len(support_cases) < 2:
        raise ProtocolError("DCSE nested frequency committee requires two support cases.")
    output: list[DirectionalControlDecision] = []
    for direction in DIRECTION_IDS:
        votes = {source: 0 for source in (None, *candidate_sources(target))}
        for nested_deleted_case in support_cases:
            nested_scope = tuple(
                case for case in support_cases if case != nested_deleted_case
            )
            values: dict[str | None, Fraction] = {None: Fraction(0)}
            for source in candidate_sources(target):
                gain = directional_hard_flip_gain(
                    support_counts,
                    query_center=target,
                    source=source,
                    direction=direction,
                    excluded_case_id=nested_deleted_case,
                    contributing_case_ids=nested_scope,
                    label_scope=(
                        f"nested_route_support::H={target}::c={plan.case_id}::"
                        f"deleted_j={nested_deleted_case}"
                    ),
                )
                values[source] = gain.exact
            winner = _choose_fraction_values(
                values, tolerance=EXACT_TIE_TOLERANCE
            )
            votes[winner] += 1
        frequencies = {
            source: Fraction(votes[source], len(support_cases)) for source in votes
        }
        # Frequencies are exact; ties use OFF then numeric source without an
        # additional numerical tolerance.
        selected = _choose_fraction_values(frequencies, tolerance=Fraction(0))
        output.append(
            DirectionalControlDecision(
                "LOO_frequency_committee",
                target,
                plan.case_id,
                direction,
                selected,
                tuple(
                    (source, value.numerator, value.denominator)
                    for source, value in frequencies.items()
                ),
                len(support_cases),
            )
        )
    return CaseControlDecision(
        "LOO_frequency_committee",
        target,
        plan.case_id,
        output[0],
        output[1],
    )


# Public spelling requested by the runtime integration surface.
select_arm_decision_for_case = select_arm_decision


__all__ = (
    "EXACT_TIE_TOLERANCE",
    "choose_with_frozen_tolerance",
    "rank_sources_by_prior",
    "select_arm_decision",
    "select_arm_decision_for_case",
    "select_arm_decisions",
    "select_direction_decision",
    "select_matched_g_decisions",
    "select_nested_frequency_committee_control",
    "select_raw_directional_loo_control",
)
