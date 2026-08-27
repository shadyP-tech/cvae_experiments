"""Non-vacuous source-OOF admission for pairwise action routing."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import (
    AdmissionCase,
    AdmissionDecisionReceipt,
    AdmissionReport,
    AdmissionThresholds,
    ActionSelectionEvidence,
    BaccRankingPolicy,
    CandidatePoolReceipt,
    CenterAdmission,
    DEFAULT_ADMISSION_THRESHOLDS,
    P_ACTION_ID,
    PairwiseRankerModel,
    SelectionDecision,
    OpportunityCaseReceipt,
    UncertaintyCalibration,
    canonical_sha256,
)


def seal_admission_decision(
    *,
    center_id: object,
    case_id: object,
    decision: SelectionDecision,
    candidate_evidence: Sequence[ActionSelectionEvidence],
    candidate_pool: CandidatePoolReceipt,
    pairwise_model: PairwiseRankerModel,
    uncertainty_calibration: UncertaintyCalibration,
    opportunity_receipt: OpportunityCaseReceipt,
    ranking_policy: BaccRankingPolicy,
) -> AdmissionDecisionReceipt:
    """Recompute and seal the exact pre-label decision for terminal joining."""

    if not isinstance(decision, SelectionDecision):
        raise ProtocolError("Admission sealing requires a typed selection decision.")
    return AdmissionDecisionReceipt(
        center_id=str(center_id),
        case_id=str(case_id),
        selection_decision=decision,
        candidate_evidence=tuple(candidate_evidence),
        candidate_pool=candidate_pool,
        pairwise_model=pairwise_model,
        uncertainty_calibration=uncertainty_calibration,
        opportunity_receipt=opportunity_receipt,
        ranking_policy=ranking_policy,
    )


def _tau_b(cases: Sequence[AdmissionCase]) -> tuple[float, int]:
    concordant = discordant = ties_prediction = ties_realized = comparisons = 0
    for case in cases:
        rows = tuple(
            [(P_ACTION_ID, 0.0, 0.0)]
            + [
                (row.action_id, row.predicted_score, row.realized_bacc_gain)
                for row in case.candidates
            ]
        )
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                predicted_delta = rows[left_index][1] - rows[right_index][1]
                realized_delta = rows[left_index][2] - rows[right_index][2]
                predicted_sign = (predicted_delta > 0.0) - (predicted_delta < 0.0)
                realized_sign = (realized_delta > 0.0) - (realized_delta < 0.0)
                if predicted_sign == 0 and realized_sign == 0:
                    continue
                comparisons += 1
                if predicted_sign == 0:
                    ties_prediction += 1
                elif realized_sign == 0:
                    ties_realized += 1
                elif predicted_sign == realized_sign:
                    concordant += 1
                else:
                    discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_prediction)
        * (concordant + discordant + ties_realized)
    )
    return (
        0.0 if denominator == 0.0 else (concordant - discordant) / denominator,
        comparisons,
    )


def _metrics(cases: Sequence[AdmissionCase]) -> dict[str, float | int]:
    rows = tuple(cases)
    active_rows = tuple(case for case in rows if case.candidates)
    sign_correct = sign_total = top1_correct = selected = harmful = proper = 0
    for case in rows:
        for candidate in case.candidates:
            predicted_sign = (candidate.predicted_score > 0.0) - (
                candidate.predicted_score < 0.0
            )
            realized_sign = (candidate.realized_bacc_gain > 0.0) - (
                candidate.realized_bacc_gain < 0.0
            )
            if predicted_sign != 0 or realized_sign != 0:
                sign_total += 1
                sign_correct += int(predicted_sign == realized_sign)
            if candidate.action_id == case.selected_action_id:
                selected += 1
                harmful += int(candidate.realized_bacc_gain < 0.0)
                proper += int(
                    candidate.realized_brier_delta > 0.0
                    or candidate.realized_log_delta > 0.0
                )
        if not case.candidates:
            continue
        predicted = [(P_ACTION_ID, 0.0)] + [
            (candidate.action_id, candidate.predicted_score) for candidate in case.candidates
        ]
        realized = [(P_ACTION_ID, 0.0)] + [
            (candidate.action_id, candidate.realized_bacc_gain) for candidate in case.candidates
        ]
        predicted_winner = min(predicted, key=lambda row: (-row[1], row[0]))[0]
        realized_max = max(value for _, value in realized)
        realized_top = {action for action, value in realized if value == realized_max}
        top1_correct += int(predicted_winner in realized_top)
    tau, comparisons = _tau_b(rows)
    case_count = len(rows)
    active_case_count = len(active_rows)
    return {
        "case_count": case_count,
        "active_case_count": active_case_count,
        "sign_accuracy": sign_correct / sign_total if sign_total else 0.0,
        "pairwise_tau_b": tau,
        "pairwise_comparison_count": comparisons,
        "top1_accuracy": top1_correct / active_case_count if active_case_count else 0.0,
        "selected_count": selected,
        "safe_coverage": selected / case_count if case_count else 0.0,
        "harmful_selected_count": harmful,
        "proper_loss_violation_count": proper,
    }


def _center_result(
    center_id: str,
    cases: Sequence[AdmissionCase],
    thresholds: AdmissionThresholds,
) -> CenterAdmission:
    values = _metrics(cases)
    reasons: list[str] = []
    if int(values["selected_count"]) <= 0:
        reasons.append("center_zero_safe_selections")
    if float(values["sign_accuracy"]) < thresholds.min_worst_center_sign_accuracy:
        reasons.append("center_sign_accuracy_below_floor")
    if float(values["pairwise_tau_b"]) <= thresholds.min_rank_lower_bound:
        reasons.append("center_pairwise_ordering_nonpositive")
    if int(values["harmful_selected_count"]) > thresholds.max_harmful_selected:
        reasons.append("center_harmful_selection_detected")
    if int(values["proper_loss_violation_count"]) > thresholds.max_proper_loss_violations:
        reasons.append("center_proper_loss_violation_detected")
    return CenterAdmission(
        center_id=center_id,
        passed=not reasons,
        case_count=int(values["case_count"]),
        selected_count=int(values["selected_count"]),
        sign_accuracy=float(values["sign_accuracy"]),
        pairwise_tau_b=float(values["pairwise_tau_b"]),
        unique_surface_top1_accuracy=float(values["top1_accuracy"]),
        harmful_selected_count=int(values["harmful_selected_count"]),
        proper_loss_violation_count=int(values["proper_loss_violation_count"]),
        reasons=tuple(reasons),
    )


def _equal_center_metrics(
    grouped: dict[str, list[AdmissionCase]],
) -> dict[str, float | int]:
    center_values = tuple(_metrics(group) for _, group in sorted(grouped.items()))
    totals = _metrics(tuple(case for group in grouped.values() for case in group))
    for name in ("sign_accuracy", "pairwise_tau_b", "top1_accuracy", "safe_coverage"):
        totals[name] = sum(float(values[name]) for values in center_values) / len(center_values)
    return totals


def evaluate_source_only_admission(
    cases: Sequence[AdmissionCase],
    *,
    thresholds: AdmissionThresholds = DEFAULT_ADMISSION_THRESHOLDS,
) -> AdmissionReport:
    """Evaluate predeclared source-only ordering, safety, and coverage gates.

    Admission cannot pass with zero selections.  The rank lower bound is the
    minimum delete-center tau-b, not a descriptive full-surface point estimate.
    Once the global surface passes, centers that fail their local stability
    gate remain independently sealed to protected P.
    """

    if thresholds != DEFAULT_ADMISSION_THRESHOLDS:
        raise ProtocolError("Admission thresholds are frozen and cannot be tuned post hoc.")
    rows = tuple(sorted(tuple(cases), key=lambda row: (row.center_id, row.case_id)))
    keys = tuple((row.center_id, row.case_id) for row in rows)
    if not rows or len(set(keys)) != len(keys):
        raise ProtocolError("Source-only admission cases are empty or duplicated.")
    grouped: dict[str, list[AdmissionCase]] = defaultdict(list)
    for row in rows:
        grouped[row.center_id].append(row)
    center_results = tuple(
        _center_result(center, group, thresholds) for center, group in sorted(grouped.items())
    )
    values = _equal_center_metrics(grouped)
    delete_center_taus = tuple(
        sum(
            float(_metrics(group)["pairwise_tau_b"])
            for center, group in grouped.items()
            if center != held_center
        )
        / (len(grouped) - 1)
        for held_center in sorted(grouped)
    ) if len(grouped) > 1 else (0.0,)
    rank_lower = min(delete_center_taus) if delete_center_taus else 0.0
    reasons: list[str] = []
    if len(grouped) < thresholds.min_center_count:
        reasons.append("insufficient_source_centers")
    if int(values["active_case_count"]) < thresholds.min_unique_active_cases:
        reasons.append("insufficient_unique_active_cases")
    if int(values["pairwise_comparison_count"]) < thresholds.min_pairwise_comparisons:
        reasons.append("insufficient_pairwise_comparisons")
    if float(values["sign_accuracy"]) < thresholds.min_sign_accuracy:
        reasons.append("sign_accuracy_below_floor")
    if float(values["top1_accuracy"]) < thresholds.min_unique_surface_top1_accuracy:
        reasons.append("unique_surface_top1_below_floor")
    if rank_lower <= thresholds.min_rank_lower_bound:
        reasons.append("minimum_delete_center_tau_b_nonpositive")
    if int(values["selected_count"]) <= 0:
        reasons.append("zero_safe_selections")
    if float(values["safe_coverage"]) < thresholds.min_safe_coverage:
        reasons.append("safe_coverage_below_floor")
    if int(values["harmful_selected_count"]) > thresholds.max_harmful_selected:
        reasons.append("harmful_selection_detected")
    if int(values["proper_loss_violation_count"]) > thresholds.max_proper_loss_violations:
        reasons.append("proper_loss_violation_detected")

    globally_admissible = not reasons
    admitted = tuple(
        row.center_id for row in center_results if globally_admissible and row.passed
    )
    sealed = tuple(row.center_id for row in center_results if row.center_id not in admitted)
    return AdmissionReport(
        passed=bool(admitted),
        center_results=center_results,
        admitted_center_ids=admitted,
        sealed_to_p_center_ids=sealed,
        case_count=len(rows),
        unique_active_case_count=int(values["active_case_count"]),
        pairwise_comparison_count=int(values["pairwise_comparison_count"]),
        selected_count=int(values["selected_count"]),
        sign_accuracy=float(values["sign_accuracy"]),
        pairwise_tau_b=float(values["pairwise_tau_b"]),
        minimum_delete_center_tau_b=float(rank_lower),
        unique_surface_top1_accuracy=float(values["top1_accuracy"]),
        safe_coverage=float(values["safe_coverage"]),
        harmful_selected_count=int(values["harmful_selected_count"]),
        proper_loss_violation_count=int(values["proper_loss_violation_count"]),
        reasons=tuple(reasons),
    )


__all__ = ("evaluate_source_only_admission", "seal_admission_decision")
