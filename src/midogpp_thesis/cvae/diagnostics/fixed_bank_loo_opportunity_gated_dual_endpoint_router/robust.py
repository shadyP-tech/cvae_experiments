"""Exact top-K donor prescreen and preserved nine-arm robust endpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import DIRECTION_IDS, EXACT_TIE_TOLERANCE, K_GRID, W_RATIONAL_GRID, candidate_sources
from .donor_prior import DonorPrior
from .response_products import DirectionalGain
from .robust_products import (
    DirectionRobustDecision,
    EndpointArm,
    ROBUST_METHOD_IDS,
    RobustArmDecision,
    RobustCandidateScore,
)
from .split_plans import WholeCaseLooPlan


def build_endpoint_arms() -> tuple[EndpointArm, ...]:
    return tuple(EndpointArm(k, numerator, denominator) for k in K_GRID for numerator, denominator in W_RATIONAL_GRID)


def _gain_map(
    rows: Sequence[DirectionalGain] | Sequence[DonorPrior] | Sequence[object],
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
        elif getattr(row, "heldout_center", None) != target_center:
            continue
        if row.source in result:
            raise ProtocolError("OGDE robust gain surface duplicates a source/direction.")
        result[row.source] = row.exact
    if tuple(result) != candidate_sources(target_center):
        raise ProtocolError("OGDE robust gain surface lacks canonical eight-source coverage.")
    return result


def rank_sources_by_prior(priors: Mapping[str, Fraction], *, target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if tuple(priors) != candidate_sources(target):
        raise ProtocolError("OGDE donor ranking lacks canonical eight-source coverage.")
    return tuple(sorted(priors, key=lambda source: (-priors[source], int(source))))


def _choose_exact(rows: Sequence[RobustCandidateScore]) -> str | None:
    if not rows or rows[0].source is not None:
        raise ProtocolError("OGDE robust score menu must begin with OFF.")
    maximum = max(row.score for row in rows)
    eligible = tuple(row.source for row in rows if maximum - row.score <= EXACT_TIE_TOLERANCE)
    return min(eligible, key=lambda source: -1 if source is None else int(source))


def select_robust_direction_decision(
    *,
    method_id: str,
    target_center: object,
    case_id: object,
    arm: EndpointArm,
    direction: object,
    support_gains: Sequence[DirectionalGain] | Sequence[DonorPrior],
    donor_priors: Sequence[DonorPrior],
) -> DirectionRobustDecision:
    target, case, direction_id = str(target_center), str(case_id), str(direction)
    if method_id not in ROBUST_METHOD_IDS or direction_id not in DIRECTION_IDS:
        raise ProtocolError("OGDE robust method/direction drifted.")
    prior = _gain_map(donor_priors, target_center=target, direction=direction_id)
    support = _gain_map(support_gains, target_center=target, direction=direction_id)
    if method_id == "G_DIRECTIONAL_MATCHED" and support != prior:
        raise ProtocolError("OGDE matched-G endpoint requires S exactly equal to G.")
    retained = rank_sources_by_prior(prior, target_center=target)[: arm.k]
    scores = [RobustCandidateScore.from_fractions(None, Fraction(0), Fraction(0), Fraction(0))]
    for source in retained:
        score = arm.weight * support[source] + (1 - arm.weight) * prior[source]
        scores.append(RobustCandidateScore.from_fractions(source, support[source], prior[source], score))
    selected = _choose_exact(scores)
    return DirectionRobustDecision(
        method_id,
        target,
        case,
        arm.arm_id,
        direction_id,
        retained,
        tuple(scores),
        selected,
    )


def select_robust_arm_decisions(
    plan: WholeCaseLooPlan,
    support_gains: Sequence[DirectionalGain],
    donor_priors: Sequence[DonorPrior],
    *,
    method_id: str = "R_NINE_ARM_ROBUST",
    arms: Sequence[EndpointArm] | None = None,
) -> tuple[RobustArmDecision, ...]:
    library = tuple(build_endpoint_arms() if arms is None else arms)
    if tuple(row.arm_id for row in library) != tuple(row.arm_id for row in build_endpoint_arms()):
        raise ProtocolError("OGDE robust endpoint must preserve all nine frozen arms in order.")
    if method_id not in ROBUST_METHOD_IDS:
        raise ProtocolError("OGDE robust endpoint method drifted.")
    support: Sequence[DirectionalGain] | Sequence[DonorPrior]
    support = donor_priors if method_id == "G_DIRECTIONAL_MATCHED" else support_gains
    output: list[RobustArmDecision] = []
    for arm in library:
        directional = tuple(
            select_robust_direction_decision(
                method_id=method_id,
                target_center=plan.target_center,
                case_id=plan.case_id,
                arm=arm,
                direction=direction,
                support_gains=support,
                donor_priors=donor_priors,
            )
            for direction in DIRECTION_IDS
        )
        output.append(
            RobustArmDecision(
                method_id,
                plan.target_center,
                plan.case_id,
                arm.arm_id,
                directional[0],
                directional[1],
            )
        )
    return tuple(output)


__all__ = (
    "build_endpoint_arms",
    "rank_sources_by_prior",
    "select_robust_arm_decisions",
    "select_robust_direction_decision",
)
