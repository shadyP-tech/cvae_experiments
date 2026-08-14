"""Terminal-only oracle, identification, probability, and utility diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, EXACT_TIE_TOLERANCE, candidate_sources
from .identification_products import CaseIdentificationDecision
from .prediction_products import MethodPrediction
from .response_products import BinaryLabel, CaseActionConfusion
from .terminal_products import (
    DirectionalOracleDecision,
    IdentificationMetrics,
    OracleCandidateUtility,
    ProbabilityMetrics,
)


def _utility(
    row: CaseActionConfusion,
    direction: str,
    *,
    center_n_positive: int,
    center_n_negative: int,
) -> Fraction:
    """Return one case's additive contribution to center-pooled BACC.

    A whole case may legitimately contain only one class.  Case-oracle choices
    therefore use the target center's pooled class denominators, never a
    per-case BACC denominator.
    """

    if center_n_positive <= 0 or center_n_negative <= 0:
        raise ProtocolError(
            "OGDE terminal directional utility requires both classes per center."
        )
    if direction == "zero_to_one":
        return Fraction(row.flip_0to1_positive, 2 * center_n_positive) - Fraction(
            row.flip_0to1_negative, 2 * center_n_negative
        )
    if direction == "one_to_zero":
        return Fraction(row.flip_1to0_negative, 2 * center_n_negative) - Fraction(
            row.flip_1to0_positive, 2 * center_n_positive
        )
    raise ProtocolError("OGDE terminal oracle direction drifted.")


def _oracle(
    method_id: str,
    target: str,
    case: str,
    direction: str,
    utilities: dict[str, Fraction],
) -> DirectionalOracleDecision:
    rows = (
        OracleCandidateUtility(None, 0, 1),
        *(
            OracleCandidateUtility(source, utilities[source].numerator, utilities[source].denominator)
            for source in candidate_sources(target)
        ),
    )
    maximum = max(row.exact for row in rows)
    eligible = tuple(row.source for row in rows if maximum - row.exact <= EXACT_TIE_TOLERANCE)
    selected = min(eligible, key=lambda source: -1 if source is None else int(source))
    return DirectionalOracleDecision(method_id, target, case, direction, rows, selected)


def build_case_directional_oracles(
    confusions: Sequence[CaseActionConfusion],
) -> tuple[DirectionalOracleDecision, ...]:
    indexed = {(row.target_center, row.case_id, row.action_id): row for row in confusions}
    cases = tuple(sorted({(row.target_center, row.case_id) for row in confusions}))
    center_denominators: dict[str, tuple[int, int]] = {}
    for target in CENTERS:
        baseline_rows = tuple(
            row
            for row in confusions
            if row.target_center == target and row.action_id == "B"
        )
        if not baseline_rows:
            continue
        center_denominators[target] = (
            sum(row.n_positive for row in baseline_rows),
            sum(row.n_negative for row in baseline_rows),
        )
    output: list[DirectionalOracleDecision] = []
    for target, case in cases:
        try:
            center_n_positive, center_n_negative = center_denominators[target]
        except KeyError as exc:
            raise ProtocolError(
                "OGDE terminal case oracle lacks baseline center denominators."
            ) from exc
        for direction in DIRECTION_IDS:
            try:
                utilities = {
                    source: _utility(
                        indexed[(target, case, f"A1::source={source}")],
                        direction,
                        center_n_positive=center_n_positive,
                        center_n_negative=center_n_negative,
                    )
                    for source in candidate_sources(target)
                }
            except KeyError as exc:
                raise ProtocolError(
                    "OGDE terminal case oracle lacks a candidate confusion row."
                ) from exc
            output.append(_oracle("O_CASE_DIRECTIONAL", target, case, direction, utilities))
    return tuple(output)


def build_static_directional_oracles(
    confusions: Sequence[CaseActionConfusion],
) -> tuple[DirectionalOracleDecision, ...]:
    output: list[DirectionalOracleDecision] = []
    for target in CENTERS:
        target_rows = tuple(row for row in confusions if row.target_center == target)
        for direction in DIRECTION_IDS:
            utilities: dict[str, Fraction] = {}
            for source in candidate_sources(target):
                rows = tuple(row for row in target_rows if row.action_id == f"A1::source={source}")
                positive = sum(row.n_positive for row in rows)
                negative = sum(row.n_negative for row in rows)
                if direction == "zero_to_one":
                    favorable = sum(row.flip_0to1_positive for row in rows)
                    adverse = sum(row.flip_0to1_negative for row in rows)
                else:
                    favorable = sum(row.flip_1to0_negative for row in rows)
                    adverse = sum(row.flip_1to0_positive for row in rows)
                if positive <= 0 or negative <= 0:
                    raise ProtocolError(
                        "OGDE terminal directional utility requires both classes per center."
                    )
                utilities[source] = (
                    Fraction(favorable, 2 * positive)
                    - Fraction(adverse, 2 * negative)
                    if direction == "zero_to_one"
                    else Fraction(favorable, 2 * negative)
                    - Fraction(adverse, 2 * positive)
                )
            output.append(_oracle("O_DIRECTIONAL_STATIC", target, "__STATIC__", direction, utilities))
    return tuple(output)


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = _ranks(left), _ranks(right)
    x -= np.mean(x, dtype=np.float64)
    y -= np.mean(y, dtype=np.float64)
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return 0.0 if denominator == 0.0 else float(np.sum(x * y) / denominator)


def score_identification_metrics(
    decisions: Sequence[CaseIdentificationDecision],
    case_oracles: Sequence[DirectionalOracleDecision],
) -> IdentificationMetrics:
    route_decisions = {
        (row.target_center, row.case_id, direction): row.decision_for_baseline_class(0 if direction == "zero_to_one" else 1)
        for row in decisions
        for direction in DIRECTION_IDS
    }
    oracles = {row.key: row for row in case_oracles if row.method_id == "O_CASE_DIRECTIONAL"}
    if not route_decisions or set(route_decisions) != set(oracles):
        raise ProtocolError("OGDE identification metrics lack aligned case-direction oracles.")
    oracle_off = predicted_off = true_off = true_active = correct_off = correct_active = 0
    exact_action = active_top1 = active_count = 0
    correlations: list[float] = []
    gaps: list[float] = []
    normalized: list[float] = []
    for key in sorted(route_decisions):
        decision, oracle = route_decisions[key], oracles[key]
        predicted = decision.selected_source
        truth = oracle.selected_source
        oracle_off += int(truth is None)
        predicted_off += int(predicted is None)
        if truth is None:
            true_off += 1
            correct_off += int(predicted is None)
        else:
            true_active += 1
            correct_active += int(predicted is not None)
            active_count += 1
            active_top1 += int(predicted == truth)
        exact_action += int(predicted == truth)
        sources = (None, *candidate_sources(decision.target_center))
        predicted_scores = [0.0, *(row.final_score for row in decision.candidate_scores)]
        oracle_scores = [float(oracle.utility_for(source)) for source in sources]
        correlations.append(_spearman(predicted_scores, oracle_scores))
        best = max(oracle_scores)
        selected_utility = float(oracle.utility_for(predicted))
        gap = best - selected_utility
        gaps.append(gap)
        span = best - min(oracle_scores)
        normalized.append(0.0 if span == 0.0 else gap / span)
    count = len(route_decisions)
    precision = correct_off / predicted_off if predicted_off else 0.0
    recall = correct_off / true_off if true_off else 0.0
    active_recall = correct_active / true_active if true_active else 0.0
    method_ids = {row.method_id for row in decisions}
    if len(method_ids) != 1:
        raise ProtocolError("OGDE identification metrics mix methods.")
    return IdentificationMetrics(
        next(iter(method_ids)), count, oracle_off, predicted_off,
        precision, recall, 0.5 * (recall + active_recall),
        exact_action / count, active_top1 / active_count if active_count else 0.0,
        active_count, float(np.mean(correlations, dtype=np.float64)),
        float(np.mean(gaps, dtype=np.float64)),
        float(np.mean(normalized, dtype=np.float64)),
    )


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(y), dtype=np.float64), logit))
    beta = np.asarray([0.0, 1.0], dtype=np.float64)
    for _ in range(50):
        eta = np.clip(design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(fitted * (1.0 - fitted), 1.0e-12, None)
        information = design.T @ (weights[:, None] * design)
        gradient = design.T @ (y - fitted)
        try:
            update = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            break
        beta += update
        if float(np.max(np.abs(update))) <= 1.0e-12:
            break
    return float(beta[0]), float(beta[1])


def calibration_parameters(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0 or not np.all(np.isin(y, (0, 1))):
        raise ProtocolError("OGDE calibration inputs are empty or unaligned.")
    return _calibration(y, p)


def score_probability_metrics(
    predictions: Sequence[MethodPrediction], labels: Sequence[BinaryLabel]
) -> ProbabilityMetrics:
    rows = tuple(predictions)
    truth = {row.key: row.value for row in labels}
    indexed = {row.key: row for row in rows}
    if not rows or len(indexed) != len(rows) or set(indexed) != set(truth) or len({row.method_id for row in rows}) != 1:
        raise ProtocolError("OGDE terminal probability metrics lack aligned unique predictions.")
    keys = sorted(indexed)
    y = np.asarray([truth[key] for key in keys], dtype=np.float64)
    p = np.asarray([indexed[key].probability for key in keys], dtype=np.float64)
    brier = float(np.mean((p - y) ** 2, dtype=np.float64))
    clipped = np.clip(p, 1.0e-12, 1.0 - 1.0e-12)
    log_loss = float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped), dtype=np.float64))
    center_rows: list[tuple[str, float]] = []
    hard = p >= 0.5
    for center in CENTERS:
        mask = np.asarray([key[0] == center for key in keys], dtype=bool)
        positive, negative = y[mask] == 1, y[mask] == 0
        if not bool(np.any(positive)) or not bool(np.any(negative)):
            raise ProtocolError("OGDE center BACC requires both classes.")
        sensitivity = float(np.mean(hard[mask][positive], dtype=np.float64))
        specificity = float(np.mean(~hard[mask][negative], dtype=np.float64))
        center_rows.append((center, 0.5 * (sensitivity + specificity)))
    intercept, slope = _calibration(y, p)
    return ProbabilityMetrics(
        rows[0].method_id, len(rows), brier, log_loss, intercept, slope,
        float(np.mean([value for _, value in center_rows], dtype=np.float64)),
        tuple(center_rows),
    )


__all__ = (
    "build_case_directional_oracles",
    "build_static_directional_oracles",
    "calibration_parameters",
    "score_identification_metrics",
    "score_probability_metrics",
)
