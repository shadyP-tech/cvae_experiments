"""Exact-terminal fixed-bank selection and outer-center inference metrics."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXACT_BACC_DELTA,
    GLOBAL_SOURCE_EXACT_CONTROL,
    OUTER_INFERENCE_UNIT_COUNT,
    PRIMARY_R_FAMILY_ID,
    STUDENT_T_975_DF8,
    candidate_sources,
)
from .metric_contracts import FamilySummaryRow, OuterMetricRow, QueryMetricRow
from .model_contracts import ExactCrossfitResult, family_spec


_TOLERANCE = 1.0e-12


def summarize_exact_crossfit(
    crossfit: ExactCrossfitResult,
) -> tuple[
    tuple[QueryMetricRow, ...],
    tuple[OuterMetricRow, ...],
    tuple[FamilySummaryRow, ...],
]:
    """Evaluate selected exact utility and paired target-specific ``R-G``."""

    if not isinstance(crossfit, ExactCrossfitResult):
        raise ProtocolError("Exact metrics require a typed exact crossfit result.")
    if GLOBAL_SOURCE_EXACT_CONTROL not in crossfit.family_ids:
        raise ProtocolError("Exact metrics require the faithful global-source G arm.")
    queries = _query_metrics(crossfit)
    outers = _outer_metrics(crossfit.family_ids, queries)
    families = _family_metrics(crossfit.family_ids, queries, outers)
    return queries, outers, families


def _query_metrics(crossfit: ExactCrossfitResult) -> tuple[QueryMetricRow, ...]:
    grouped: dict[tuple[str, str, str], list[object]] = {}
    for row in crossfit.predictions:
        if row.response_name != EXACT_BACC_DELTA:
            raise ProtocolError("Smooth rows entered exact metrics.")
        grouped.setdefault(
            (row.family_id, row.outer_target_id, row.query_id), []
        ).append(row)
    global_by_query: dict[tuple[str, str], tuple[object, ...]] = {}
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            global_by_query[(outer, query)] = _ordered_candidates(
                grouped[(GLOBAL_SOURCE_EXACT_CONTROL, outer, query)], outer, query
            )
    output: list[QueryMetricRow] = []
    for family_id in crossfit.family_ids:
        for outer in CENTERS:
            for query in (value for value in CENTERS if value != outer):
                rows = _ordered_candidates(
                    grouped[(family_id, outer, query)], outer, query
                )
                global_rows = global_by_query[(outer, query)]
                observed = np.asarray(
                    [row.observed_delta for row in rows], dtype=np.float64
                )
                global_observed = np.asarray(
                    [row.observed_delta for row in global_rows], dtype=np.float64
                )
                if not np.allclose(observed, global_observed, atol=0.0, rtol=0.0):
                    raise ProtocolError("Observed exact response changed across families.")
                predicted = np.asarray(
                    [row.predicted_delta for row in rows], dtype=np.float64
                )
                global_predicted = np.asarray(
                    [row.predicted_delta for row in global_rows], dtype=np.float64
                )
                selected_index = min(_top_indices(predicted))
                global_index = min(_top_indices(global_predicted))
                oracle_top = _top_indices(observed)
                predicted_top = _top_indices(predicted)
                response_range = float(np.max(observed) - np.min(observed))
                selected_regret = _regret(
                    observed, selected_index, response_range
                )
                global_regret = _regret(observed, global_index, response_range)
                correlation = _spearman(predicted, observed)
                output.append(
                    QueryMetricRow(
                        family_id=family_id,
                        outer_target_id=outer,
                        query_id=query,
                        candidate_count=len(rows),
                        selected_source=rows[selected_index].candidate_source,
                        global_selected_source=(
                            global_rows[global_index].candidate_source
                        ),
                        selected_exact_gain=float(observed[selected_index]),
                        global_selected_exact_gain=float(observed[global_index]),
                        r_minus_g_exact_gain=float(
                            observed[selected_index] - observed[global_index]
                        ),
                        exact_top1=float(
                            len(predicted_top) == 1
                            and len(oracle_top) == 1
                            and predicted_top == oracle_top
                        ),
                        tie_aware_top1=(
                            len(set(predicted_top).intersection(oracle_top))
                            / float(len(predicted_top))
                        ),
                        spearman=0.0 if correlation is None else correlation,
                        spearman_defined=correlation is not None,
                        pairwise_accuracy=_pairwise_accuracy(predicted, observed),
                        normalized_oracle_regret=selected_regret,
                        global_normalized_oracle_regret=global_regret,
                        regret_minus_g=selected_regret - global_regret,
                    )
                )
    return tuple(output)


def _outer_metrics(
    family_ids: tuple[str, ...], query_rows: tuple[QueryMetricRow, ...]
) -> tuple[OuterMetricRow, ...]:
    output: list[OuterMetricRow] = []
    for family_id in family_ids:
        for outer in CENTERS:
            rows = tuple(
                row
                for row in query_rows
                if row.family_id == family_id and row.outer_target_id == outer
            )
            if len(rows) != len(CENTERS) - 1:
                raise ProtocolError("Outer exact metric query coverage drifted.")
            selected = tuple(row.selected_source for row in rows)
            output.append(
                OuterMetricRow(
                    family_id=family_id,
                    outer_target_id=outer,
                    query_count=len(rows),
                    mean_selected_exact_gain=_mean(
                        row.selected_exact_gain for row in rows
                    ),
                    mean_global_selected_exact_gain=_mean(
                        row.global_selected_exact_gain for row in rows
                    ),
                    mean_r_minus_g_exact_gain=_mean(
                        row.r_minus_g_exact_gain for row in rows
                    ),
                    mean_exact_top1=_mean(row.exact_top1 for row in rows),
                    mean_tie_aware_top1=_mean(
                        row.tie_aware_top1 for row in rows
                    ),
                    mean_spearman=_mean(row.spearman for row in rows),
                    mean_pairwise_accuracy=_mean(
                        row.pairwise_accuracy for row in rows
                    ),
                    mean_normalized_oracle_regret=_mean(
                        row.normalized_oracle_regret for row in rows
                    ),
                    mean_global_normalized_oracle_regret=_mean(
                        row.global_normalized_oracle_regret for row in rows
                    ),
                    mean_regret_minus_g=_mean(row.regret_minus_g for row in rows),
                    source_max_selection_share=_max_share(selected),
                    source_selection_entropy=_normalized_entropy(selected),
                )
            )
    return tuple(output)


def _family_metrics(
    family_ids: tuple[str, ...],
    query_rows: tuple[QueryMetricRow, ...],
    outer_rows: tuple[OuterMetricRow, ...],
) -> tuple[FamilySummaryRow, ...]:
    output: list[FamilySummaryRow] = []
    for family_id in family_ids:
        rows = tuple(row for row in outer_rows if row.family_id == family_id)
        queries = tuple(row for row in query_rows if row.family_id == family_id)
        if len(rows) != OUTER_INFERENCE_UNIT_COUNT or len(queries) != 72:
            raise ProtocolError("Family exact metric coverage drifted.")
        selected_gain = _array(row.mean_selected_exact_gain for row in rows)
        r_minus_g = _array(row.mean_r_minus_g_exact_gain for row in rows)
        spearman = _array(row.mean_spearman for row in rows)
        pairwise = _array(row.mean_pairwise_accuracy for row in rows)
        regret = _array(row.mean_normalized_oracle_regret for row in rows)
        regret_minus_g = _array(row.mean_regret_minus_g for row in rows)
        selected_ci = _student_t_ci95(selected_gain)
        r_minus_g_ci = _student_t_ci95(r_minus_g)
        spearman_ci = _student_t_ci95(spearman)
        pairwise_ci = _student_t_ci95(pairwise)
        regret_ci = _student_t_ci95(regret)
        regret_minus_g_ci = _student_t_ci95(regret_minus_g)
        eligible = family_id == PRIMARY_R_FAMILY_ID
        passed = bool(
            eligible
            and selected_ci[0] > 0.0
            and r_minus_g_ci[0] > 0.0
            and spearman_ci[0] > 0.0
            and pairwise_ci[0] > 0.5
            and regret_minus_g_ci[1] < 0.0
        )
        selections = tuple(row.selected_source for row in queries)
        spec = family_spec(family_id)
        output.append(
            FamilySummaryRow(
                family_id=family_id,
                scientific_role=spec.scientific_role,
                local_predictor_count=len(spec.predictor_names),
                source_effects_included=spec.source_effects_included,
                outer_count=len(rows),
                mean_selected_exact_gain=float(np.mean(selected_gain)),
                selected_gain_ci95_lower=selected_ci[0],
                selected_gain_ci95_upper=selected_ci[1],
                mean_r_minus_g_exact_gain=float(np.mean(r_minus_g)),
                r_minus_g_ci95_lower=r_minus_g_ci[0],
                r_minus_g_ci95_upper=r_minus_g_ci[1],
                mean_exact_top1=_mean(row.mean_exact_top1 for row in rows),
                mean_tie_aware_top1=_mean(
                    row.mean_tie_aware_top1 for row in rows
                ),
                mean_spearman=float(np.mean(spearman)),
                spearman_ci95_lower=spearman_ci[0],
                spearman_ci95_upper=spearman_ci[1],
                mean_pairwise_accuracy=float(np.mean(pairwise)),
                pairwise_ci95_lower=pairwise_ci[0],
                pairwise_ci95_upper=pairwise_ci[1],
                mean_normalized_oracle_regret=float(np.mean(regret)),
                regret_ci95_lower=regret_ci[0],
                regret_ci95_upper=regret_ci[1],
                mean_regret_minus_g=float(np.mean(regret_minus_g)),
                regret_minus_g_ci95_lower=regret_minus_g_ci[0],
                regret_minus_g_ci95_upper=regret_minus_g_ci[1],
                source_selection_counts=tuple(
                    (source, selections.count(source)) for source in CENTERS
                ),
                source_max_selection_share=_max_share(selections),
                source_selection_entropy=_normalized_entropy(selections),
                publication_gate_eligible=eligible,
                exact_gate_passed=passed,
            )
        )
    return tuple(output)


def _ordered_candidates(rows: list[object], outer: str, query: str) -> tuple[object, ...]:
    by_source = {row.candidate_source: row for row in rows}
    sources = candidate_sources(outer, query)
    if len(by_source) != len(sources) or set(by_source) != set(sources):
        raise ProtocolError("Query candidate prediction geometry drifted.")
    return tuple(by_source[source] for source in sources)


def _top_indices(values: np.ndarray) -> tuple[int, ...]:
    maximum = float(np.max(values))
    return tuple(
        index
        for index, value in enumerate(values)
        if abs(float(value) - maximum) <= _TOLERANCE
    )


def _regret(observed: np.ndarray, selected: int, response_range: float) -> float:
    if response_range <= _TOLERANCE:
        return 0.0
    return float((np.max(observed) - observed[selected]) / response_range)


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - float(np.mean(left_rank))
    right_centered = right_rank - float(np.mean(right_rank))
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
    values: list[float] = []
    for left, right in combinations(range(len(predicted)), 2):
        predicted_sign = _sign(float(predicted[left] - predicted[right]))
        observed_sign = _sign(float(observed[left] - observed[right]))
        if predicted_sign == observed_sign:
            values.append(1.0)
        elif predicted_sign == 0 or observed_sign == 0:
            values.append(0.5)
        else:
            values.append(0.0)
    if not values:
        raise ProtocolError("Pairwise accuracy requires multiple candidates.")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _sign(value: float) -> int:
    if value > _TOLERANCE:
        return 1
    if value < -_TOLERANCE:
        return -1
    return 0


def _student_t_ci95(values: np.ndarray) -> tuple[float, float]:
    if values.shape != (OUTER_INFERENCE_UNIT_COUNT,) or not np.isfinite(values).all():
        raise ProtocolError("Inference requires exactly nine finite outer values.")
    mean = float(np.mean(values, dtype=np.float64))
    margin = (
        STUDENT_T_975_DF8
        * float(np.std(values, ddof=1))
        / math.sqrt(float(OUTER_INFERENCE_UNIT_COUNT))
    )
    return mean - margin, mean + margin


def _array(values: object) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)  # type: ignore[arg-type]
    if result.shape != (OUTER_INFERENCE_UNIT_COUNT,) or not np.isfinite(result).all():
        raise ProtocolError("Outer inference array drifted.")
    return result


def _mean(values: object) -> float:
    result = np.asarray(tuple(values), dtype=np.float64)  # type: ignore[arg-type]
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ProtocolError("Metric mean requires finite nonempty values.")
    return float(np.mean(result, dtype=np.float64))


def _max_share(values: tuple[str, ...]) -> float:
    return max(values.count(source) for source in CENTERS) / float(len(values))


def _normalized_entropy(values: tuple[str, ...]) -> float:
    probabilities = np.asarray(
        [values.count(source) / float(len(values)) for source in CENTERS],
        dtype=np.float64,
    )
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log(positive)) / math.log(len(CENTERS)))


__all__ = ("summarize_exact_crossfit",)
