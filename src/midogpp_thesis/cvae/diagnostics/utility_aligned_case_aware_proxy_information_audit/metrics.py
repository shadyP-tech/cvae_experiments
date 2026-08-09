"""Nested-query metrics and nine-outer-center inference summaries."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    CENTERS,
    CONTROL_FAMILY_IDS,
    FAMILY_IDS,
    OUTER_INFERENCE_UNIT_COUNT,
    PRIMARY_RESPONSE_NAME,
    RESPONSE_NAMES,
    SCREENING_FAMILY_IDS,
    STUDENT_T_975_DF8,
    CaseAwareCrossfitResult,
    FamilySummaryRow,
    OuterMetricRow,
    QueryMetricRow,
)
from .family_designs import PROXY_FAMILY_SPECS


_TOLERANCE = 1.0e-12


def summarize_crossfit(
    crossfit: CaseAwareCrossfitResult,
) -> tuple[
    tuple[QueryMetricRow, ...],
    tuple[OuterMetricRow, ...],
    tuple[FamilySummaryRow, ...],
]:
    """Summarize without promoting query/candidate rows to inference units."""

    if not isinstance(crossfit, CaseAwareCrossfitResult):
        raise ProtocolError("Metrics require a typed case-aware crossfit result.")
    if crossfit.family_ids != FAMILY_IDS or crossfit.response_names != RESPONSE_NAMES:
        raise ProtocolError("Audit summaries require all predeclared families/responses.")
    queries = _query_rows(crossfit)
    outers = _outer_rows(crossfit, queries)
    families = _family_rows(outers)
    return queries, outers, families


summarize_proxy_information = summarize_crossfit


def _query_rows(crossfit: CaseAwareCrossfitResult) -> tuple[QueryMetricRow, ...]:
    grouped: dict[tuple[str, str, str, str], list[object]] = {}
    for row in crossfit.predictions:
        grouped.setdefault(
            (
                row.family_id,
                row.response_name,
                row.outer_target_id,
                row.query_id,
            ),
            [],
        ).append(row)
    output: list[QueryMetricRow] = []
    for family_id in FAMILY_IDS:
        for response_name in RESPONSE_NAMES:
            for outer in CENTERS:
                for query in (value for value in CENTERS if value != outer):
                    rows = tuple(
                        sorted(
                            grouped.get(
                                (family_id, response_name, outer, query), []
                            ),
                            key=lambda value: value.candidate_source,
                        )
                    )
                    expected_candidates = len(CENTERS) - 2
                    if (
                        len(rows) != expected_candidates
                        or len({row.candidate_source for row in rows})
                        != expected_candidates
                    ):
                        raise ProtocolError("Query metric candidate geometry drifted.")
                    predicted = np.asarray(
                        [row.predicted_delta for row in rows], dtype=np.float64
                    )
                    observed = np.asarray(
                        [row.observed_delta for row in rows], dtype=np.float64
                    )
                    predicted_top = _top_indices(predicted)
                    oracle_top = _top_indices(observed)
                    selected_index = min(predicted_top)
                    exact_top1 = float(
                        len(predicted_top) == 1
                        and len(oracle_top) == 1
                        and predicted_top == oracle_top
                    )
                    tie_aware = float(bool(set(predicted_top).intersection(oracle_top)))
                    correlation = _spearman(predicted, observed)
                    response_range = float(np.max(observed) - np.min(observed))
                    regret = (
                        0.0
                        if response_range <= _TOLERANCE
                        else float(
                            (np.max(observed) - observed[selected_index])
                            / response_range
                        )
                    )
                    pairwise = _pairwise_accuracy(predicted, observed)
                    rmse = float(
                        np.sqrt(
                            np.mean(
                                (predicted - observed) ** 2,
                                dtype=np.float64,
                            )
                        )
                    )
                    values = {
                        "family_id": family_id,
                        "response_name": response_name,
                        "outer_target_id": outer,
                        "query_id": query,
                        "candidate_count": len(rows),
                        "exact_top1": exact_top1,
                        "tie_aware_top1": tie_aware,
                        "spearman": 0.0 if correlation is None else correlation,
                        "spearman_defined": correlation is not None,
                        "normalized_oracle_regret": regret,
                        "pairwise_accuracy": pairwise,
                        "rmse": rmse,
                    }
                    unhashed = {
                        "schema_version": "midogpp_stage90_case_aware_query_metrics_v1",
                        **values,
                        "inference_unit": "descriptive_query_nested_within_outer_H",
                        "candidate_rows_are_inference_units": False,
                        "response_is_primary": response_name
                        == PRIMARY_RESPONSE_NAME,
                        "smooth_response_is_diagnostic_only": response_name
                        != PRIMARY_RESPONSE_NAME,
                    }
                    output.append(
                        QueryMetricRow(
                            **values,  # type: ignore[arg-type]
                            row_hash=canonical_sha256(unhashed),
                        )
                    )
    return tuple(output)


def _outer_rows(
    crossfit: CaseAwareCrossfitResult,
    query_rows: tuple[QueryMetricRow, ...],
) -> tuple[OuterMetricRow, ...]:
    output: list[OuterMetricRow] = []
    for family_id in FAMILY_IDS:
        for response_name in RESPONSE_NAMES:
            for outer in CENTERS:
                queries = tuple(
                    row
                    for row in query_rows
                    if row.family_id == family_id
                    and row.response_name == response_name
                    and row.outer_target_id == outer
                )
                candidates = tuple(
                    row
                    for row in crossfit.predictions
                    if row.family_id == family_id
                    and row.response_name == response_name
                    and row.outer_target_id == outer
                )
                if len(queries) != len(CENTERS) - 1 or len(candidates) != (
                    (len(CENTERS) - 1) * (len(CENTERS) - 2)
                ):
                    raise ProtocolError("Outer-center metric coverage drifted.")
                predicted = np.asarray(
                    [row.predicted_delta for row in candidates], dtype=np.float64
                )
                observed = np.asarray(
                    [row.observed_delta for row in candidates], dtype=np.float64
                )
                values = {
                    "family_id": family_id,
                    "response_name": response_name,
                    "outer_target_id": outer,
                    "query_count": len(queries),
                    "mean_exact_top1": _mean(row.exact_top1 for row in queries),
                    "mean_tie_aware_top1": _mean(
                        row.tie_aware_top1 for row in queries
                    ),
                    "mean_spearman": _mean(row.spearman for row in queries),
                    "mean_normalized_oracle_regret": _mean(
                        row.normalized_oracle_regret for row in queries
                    ),
                    "mean_pairwise_accuracy": _mean(
                        row.pairwise_accuracy for row in queries
                    ),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                (predicted - observed) ** 2,
                                dtype=np.float64,
                            )
                        )
                    ),
                }
                unhashed = {
                    "schema_version": "midogpp_stage90_case_aware_outer_metrics_v1",
                    **values,
                    "inference_unit": "outer_target_center",
                    "query_rows_are_nested_descriptive_units": True,
                    "candidate_and_seed_rows_are_not_inference_units": True,
                    "response_is_primary": response_name
                    == PRIMARY_RESPONSE_NAME,
                    "smooth_response_is_diagnostic_only": response_name
                    != PRIMARY_RESPONSE_NAME,
                }
                output.append(
                    OuterMetricRow(
                        **values,  # type: ignore[arg-type]
                        row_hash=canonical_sha256(unhashed),
                    )
                )
    return tuple(output)


def _family_rows(
    outer_rows: tuple[OuterMetricRow, ...],
) -> tuple[FamilySummaryRow, ...]:
    preliminary: dict[tuple[str, str], dict[str, object]] = {}
    for family_id in FAMILY_IDS:
        for response_name in RESPONSE_NAMES:
            rows = tuple(
                row
                for row in outer_rows
                if row.family_id == family_id
                and row.response_name == response_name
            )
            if tuple(row.outer_target_id for row in rows) != CENTERS:
                raise ProtocolError(
                    "Family inference must use exactly nine outer-center units."
                )
            spearman = np.asarray(
                [row.mean_spearman for row in rows], dtype=np.float64
            )
            regret = np.asarray(
                [row.mean_normalized_oracle_regret for row in rows],
                dtype=np.float64,
            )
            pairwise = np.asarray(
                [row.mean_pairwise_accuracy for row in rows], dtype=np.float64
            )
            spearman_ci = _student_t_ci95(spearman)
            regret_ci = _student_t_ci95(regret)
            pairwise_ci = _student_t_ci95(pairwise)
            preliminary[(family_id, response_name)] = {
                "family_id": family_id,
                "response_name": response_name,
                "family_role": PROXY_FAMILY_SPECS[family_id].family_role,
                "predictor_count": PROXY_FAMILY_SPECS[family_id].predictor_count,
                "outer_count": len(rows),
                "mean_exact_top1": _mean(row.mean_exact_top1 for row in rows),
                "mean_tie_aware_top1": _mean(
                    row.mean_tie_aware_top1 for row in rows
                ),
                "mean_spearman": float(np.mean(spearman, dtype=np.float64)),
                "spearman_ci95_lower": spearman_ci[0],
                "spearman_ci95_upper": spearman_ci[1],
                "mean_normalized_oracle_regret": float(
                    np.mean(regret, dtype=np.float64)
                ),
                "regret_ci95_lower": regret_ci[0],
                "regret_ci95_upper": regret_ci[1],
                "mean_pairwise_accuracy": float(
                    np.mean(pairwise, dtype=np.float64)
                ),
                "pairwise_ci95_lower": pairwise_ci[0],
                "pairwise_ci95_upper": pairwise_ci[1],
                "mean_rmse": _mean(row.rmse for row in rows),
            }

    control_regrets = {
        family_id: float(
            preliminary[(family_id, PRIMARY_RESPONSE_NAME)][
                "mean_normalized_oracle_regret"
            ]
        )
        for family_id in CONTROL_FAMILY_IDS
    }
    output: list[FamilySummaryRow] = []
    for family_id in FAMILY_IDS:
        for response_name in RESPONSE_NAMES:
            values = preliminary[(family_id, response_name)]
            eligible = (
                response_name == PRIMARY_RESPONSE_NAME
                and family_id in SCREENING_FAMILY_IDS
            )
            beats_controls = bool(
                eligible
                and all(
                    float(values["mean_normalized_oracle_regret"]) < control
                    for control in control_regrets.values()
                )
            )
            screening_passed = bool(
                eligible
                and float(values["spearman_ci95_lower"]) > 0.0
                and float(values["pairwise_ci95_lower"]) > 0.5
                and float(values["regret_ci95_upper"]) < 0.5
                and beats_controls
            )
            row_values = {
                **values,
                "beats_all_controls": beats_controls,
                "screening_eligible": eligible,
                "screening_passed": screening_passed,
            }
            unhashed = {
                "schema_version": "midogpp_stage90_case_aware_family_summary_v1",
                **row_values,
                "inference_unit": "outer_target_center",
                "outer_center_count": OUTER_INFERENCE_UNIT_COUNT,
                "response_is_primary": response_name == PRIMARY_RESPONSE_NAME,
                "smooth_response_is_diagnostic_only": (
                    response_name != PRIMARY_RESPONSE_NAME
                ),
                "screening_gate_may_authorize_policy": False,
            }
            output.append(
                FamilySummaryRow(
                    **row_values,  # type: ignore[arg-type]
                    row_hash=canonical_sha256(unhashed),
                )
            )
    return tuple(output)


def _student_t_ci95(values: np.ndarray) -> tuple[float, float]:
    if (
        values.shape != (OUTER_INFERENCE_UNIT_COUNT,)
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("Inference requires exactly nine finite outer values.")
    mean = float(np.mean(values, dtype=np.float64))
    standard_deviation = float(np.std(values, ddof=1))
    margin = (
        STUDENT_T_975_DF8
        * standard_deviation
        / math.sqrt(float(OUTER_INFERENCE_UNIT_COUNT))
    )
    return mean - margin, mean + margin


def _top_indices(values: np.ndarray) -> tuple[int, ...]:
    maximum = float(np.max(values))
    return tuple(
        index
        for index, value in enumerate(values)
        if abs(float(value) - maximum) <= _TOLERANCE
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - float(np.mean(left_rank, dtype=np.float64))
    right_centered = right_rank - float(np.mean(right_rank, dtype=np.float64))
    denominator = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and abs(
            float(values[order[end]] - values[order[cursor]])
        ) <= _TOLERANCE:
            end += 1
        ranks[order[cursor:end]] = 0.5 * float(cursor + end - 1)
        cursor = end
    return ranks


def _pairwise_accuracy(predicted: np.ndarray, observed: np.ndarray) -> float:
    scores: list[float] = []
    for left, right in combinations(range(len(predicted)), 2):
        predicted_sign = _sign(float(predicted[left] - predicted[right]))
        observed_sign = _sign(float(observed[left] - observed[right]))
        if predicted_sign == observed_sign:
            scores.append(1.0)
        elif predicted_sign == 0 or observed_sign == 0:
            scores.append(0.5)
        else:
            scores.append(0.0)
    expected_pairs = len(predicted) * (len(predicted) - 1) // 2
    if len(scores) != expected_pairs or not scores:
        raise ProtocolError("Pairwise accuracy candidate geometry drifted.")
    return float(np.mean(np.asarray(scores, dtype=np.float64), dtype=np.float64))


def _sign(value: float) -> int:
    if value > _TOLERANCE:
        return 1
    if value < -_TOLERANCE:
        return -1
    return 0


def _mean(values: object) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)  # type: ignore[arg-type]
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("Metric mean requires finite nonempty values.")
    return float(np.mean(array, dtype=np.float64))


__all__ = ("summarize_crossfit", "summarize_proxy_information")
