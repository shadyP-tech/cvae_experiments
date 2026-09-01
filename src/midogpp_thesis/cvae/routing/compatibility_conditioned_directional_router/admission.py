"""Non-vacuous source-only learnability admission."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    AdmissionThresholds,
    LearnabilityAdmission,
    SourceAdmissionCase,
)
from .hashing import canonical_hash


DEFAULT_THRESHOLDS = AdmissionThresholds()


def _sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def _tau(cases: Sequence[SourceAdmissionCase]) -> float:
    concordant = discordant = predicted_ties = realized_ties = 0
    for case in cases:
        members = [("B", 0.0, 0.0)] + [
            (row.action_id, row.predicted_score, row.observed.bacc_gain)
            for row in case.candidates
        ]
        for left, right in combinations(members, 2):
            predicted = _sign(left[1] - right[1])
            realized = _sign(left[2] - right[2])
            if predicted == 0 and realized == 0:
                continue
            if predicted == 0:
                predicted_ties += 1
            elif realized == 0:
                realized_ties += 1
            elif predicted == realized:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + predicted_ties)
        * (concordant + discordant + realized_ties)
    )
    return 0.0 if denominator == 0.0 else (concordant - discordant) / denominator


def evaluate_source_only_admission(
    cases: Sequence[SourceAdmissionCase],
    *,
    thresholds: AdmissionThresholds = DEFAULT_THRESHOLDS,
) -> LearnabilityAdmission:
    """Gate target routing using only strict source-center-OOF evidence."""

    rows = tuple(sorted(tuple(cases), key=lambda row: (row.query_center_id, row.case_id)))
    keys = tuple((row.query_center_id, row.case_id) for row in rows)
    if (
        not isinstance(thresholds, AdmissionThresholds)
        or not rows
        or any(not isinstance(row, SourceAdmissionCase) for row in rows)
        or len(set(keys)) != len(keys)
    ):
        raise ProtocolError("Source-only admission cases are empty, duplicated, or untyped.")
    centers = tuple(sorted({row.query_center_id for row in rows}))
    sign_correct = sign_total = top1_correct = selected_count = 0
    harmful = proper = selected_cases = 0
    for case in rows:
        for candidate in case.candidates:
            predicted_sign = _sign(candidate.predicted_score)
            realized_sign = _sign(candidate.observed.bacc_gain)
            if predicted_sign != 0 or realized_sign != 0:
                sign_total += 1
                sign_correct += int(predicted_sign == realized_sign)
        predicted_members = [("B", 0.0)] + [
            (candidate.action_id, candidate.predicted_score)
            for candidate in case.candidates
        ]
        realized_members = [("B", 0.0)] + [
            (candidate.action_id, candidate.observed.bacc_gain)
            for candidate in case.candidates
        ]
        predicted_winner = min(predicted_members, key=lambda row: (-row[1], row[0]))[0]
        realized_max = max(value for _, value in realized_members)
        realized_winners = {action for action, value in realized_members if value == realized_max}
        top1_correct += int(predicted_winner in realized_winners)
        selected = tuple(candidate for candidate in case.candidates if candidate.safe_selected)
        if len(selected) > 1:
            raise ProtocolError("Source admission case contains multiple sealed selections.")
        if selected:
            selected_cases += 1
            selected_count += 1
            chosen = selected[0]
            harmful += int(chosen.observed.bacc_gain < 0.0)
            proper += int(
                chosen.observed.brier_delta > 0.0 or chosen.observed.log_delta > 0.0
            )
    sign_accuracy = sign_correct / sign_total if sign_total else 0.0
    top1_accuracy = top1_correct / len(rows)
    safe_coverage = selected_cases / len(rows)
    delete_center_taus = tuple(
        _tau(tuple(row for row in rows if row.query_center_id != held))
        for held in centers
    )
    minimum_tau = min(delete_center_taus) if delete_center_taus else 0.0
    reasons: list[str] = []
    if len(centers) < thresholds.minimum_center_count:
        reasons.append("INSUFFICIENT_SOURCE_CENTERS")
    if len(rows) < thresholds.minimum_case_count:
        reasons.append("INSUFFICIENT_SOURCE_CASES")
    if sign_accuracy < thresholds.minimum_sign_accuracy:
        reasons.append("SIGN_ACCURACY_BELOW_FLOOR")
    if top1_accuracy < thresholds.minimum_top1_accuracy:
        reasons.append("TOP1_ACCURACY_BELOW_FLOOR")
    if minimum_tau <= thresholds.minimum_delete_center_tau:
        reasons.append("DELETE_CENTER_RANK_LOWER_BOUND_NONPOSITIVE")
    if selected_count == 0:
        reasons.append("ZERO_SAFE_SOURCE_SELECTIONS")
    if safe_coverage < thresholds.minimum_safe_coverage:
        reasons.append("SAFE_COVERAGE_BELOW_FLOOR")
    if harmful > thresholds.maximum_harmful_selected:
        reasons.append("HARMFUL_SOURCE_SELECTION")
    if proper > thresholds.maximum_proper_loss_violations:
        reasons.append("PROPER_LOSS_SOURCE_VIOLATION")
    source_hash = canonical_hash(
        tuple(
            (
                row.query_center_id,
                row.case_id,
                tuple(
                    (
                        candidate.action_id,
                        candidate.predicted_score,
                        candidate.opportunity_probability,
                        candidate.safe_selected,
                        candidate.observed.as_tuple(),
                    )
                    for candidate in row.candidates
                ),
            )
            for row in rows
        )
    )
    return LearnabilityAdmission(
        passed=not reasons,
        center_ids=centers,
        case_count=len(rows),
        sign_accuracy=sign_accuracy,
        top1_accuracy=top1_accuracy,
        minimum_delete_center_tau=minimum_tau,
        safe_coverage=safe_coverage,
        selected_count=selected_count,
        harmful_selected_count=harmful,
        proper_loss_violation_count=proper,
        reasons=tuple(reasons),
        source_oof_hash=source_hash,
    )


__all__ = ("DEFAULT_THRESHOLDS", "evaluate_source_only_admission")
