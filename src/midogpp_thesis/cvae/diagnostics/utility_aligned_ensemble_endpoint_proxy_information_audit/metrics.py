"""Query metrics, outer-``H`` summaries, and the diagnostic screening gate."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    ABSOLUTE_SHIFT_CONTROL,
    CENTERS,
    CONTROL_FAMILY_IDS,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
    EXPECTED_OUTER_METRIC_ROW_COUNT_PER_FAMILY,
    EXPECTED_QUERY_METRIC_ROW_COUNT_PER_FAMILY,
    EXPERIMENT_ID,
    FAMILY_IDS,
    METADATA_ONLY_CONTROL,
    OUTER_INFERENCE_UNIT_COUNT,
    SCREENING_FAMILY_IDS,
    STUDENT_T_975_DF8,
    FamilySummaryRow,
    OuterMetricRow,
    ProxyCrossfitResult,
    ProxyFeatureRow,
    ProxyInformationAuditResult,
    ProxyUtilityRow,
    QueryMetricRow,
)
from .crossfit import crossfit_proxy_families
from .proxy_features import PROXY_FAMILY_SPECS, build_proxy_feature_surface


_TOLERANCE = 1.0e-12


def summarize_proxy_information_audit(
    crossfit: ProxyCrossfitResult,
) -> ProxyInformationAuditResult:
    """Summarize candidate predictions without treating them as inference units."""

    if not isinstance(crossfit, ProxyCrossfitResult):
        raise ProtocolError("Proxy metrics require a typed crossfit result.")
    query_rows = _query_metric_rows(crossfit)
    outer_rows = _outer_metric_rows(crossfit, query_rows)
    summaries = _family_summary_rows(outer_rows)
    informative = tuple(
        row.family_id for row in summaries if row.screening_passed
    )
    gate_passed = bool(informative)
    unhashed = _audit_result_payload(
        crossfit,
        query_rows,
        outer_rows,
        summaries,
        proxy_information_gate_passed=gate_passed,
        informative_family_ids=informative,
    )
    return ProxyInformationAuditResult(
        crossfit=crossfit,
        query_metrics=query_rows,
        outer_metrics=outer_rows,
        family_summaries=summaries,
        proxy_information_gate_passed=gate_passed,
        informative_family_ids=informative,
        result_hash=canonical_sha256(unhashed),
    )


def run_proxy_information_audit(
    proxy_rows: Sequence[ProxyFeatureRow | Mapping[str, object]],
    utility_rows: Sequence[ProxyUtilityRow | object],
) -> ProxyInformationAuditResult:
    """Build, cross-fit, evaluate, and serialize the pure scientific audit."""

    feature_surface = build_proxy_feature_surface(proxy_rows)
    crossfit = crossfit_proxy_families(feature_surface, utility_rows)
    return summarize_proxy_information_audit(crossfit)


def _query_metric_rows(
    crossfit: ProxyCrossfitResult,
) -> tuple[QueryMetricRow, ...]:
    by_group: dict[tuple[str, str, str], list[object]] = {}
    for row in crossfit.predictions:
        by_group.setdefault(
            (row.family_id, row.outer_target_id, row.query_id), []
        ).append(row)
    expected_count = len(FAMILY_IDS) * EXPECTED_QUERY_METRIC_ROW_COUNT_PER_FAMILY
    if len(by_group) != expected_count:
        raise ProtocolError("Proxy query-metric group coverage drifted.")
    output: list[QueryMetricRow] = []
    for family_id in FAMILY_IDS:
        for outer in CENTERS:
            for query in (center for center in CENTERS if center != outer):
                rows = tuple(
                    sorted(
                        by_group[(family_id, outer, query)],
                        key=lambda row: row.candidate_source,
                    )
                )
                if len(rows) != 7 or len({row.candidate_source for row in rows}) != 7:
                    raise ProtocolError("Proxy query metric requires seven candidates.")
                sources = tuple(row.candidate_source for row in rows)
                predicted = np.asarray(
                    [row.predicted_utility_delta for row in rows], dtype=np.float64
                )
                observed = np.asarray(
                    [row.observed_utility_delta for row in rows], dtype=np.float64
                )
                if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
                    raise ProtocolError("Proxy query metric input is non-finite.")
                predicted_top = tuple(
                    source
                    for source, value in zip(sources, predicted, strict=True)
                    if abs(float(value) - float(np.max(predicted))) <= _TOLERANCE
                )
                oracle = tuple(
                    source
                    for source, value in zip(sources, observed, strict=True)
                    if abs(float(value) - float(np.max(observed))) <= _TOLERANCE
                )
                selected = min(predicted_top)
                selected_index = sources.index(selected)
                exact_top1 = float(
                    len(predicted_top) == 1
                    and len(oracle) == 1
                    and predicted_top == oracle
                )
                tie_aware_top1 = float(bool(set(predicted_top).intersection(oracle)))
                correlation = _spearman(predicted, observed)
                regret_denominator = float(np.max(observed) - np.min(observed))
                regret = (
                    0.0
                    if regret_denominator <= _TOLERANCE
                    else float(
                        (np.max(observed) - observed[selected_index])
                        / regret_denominator
                    )
                )
                pairwise = _pairwise_accuracy(predicted, observed)
                intercept, slope, slope_defined = _calibration(predicted, observed)
                rmse = float(
                    np.sqrt(np.mean((predicted - observed) ** 2, dtype=np.float64))
                )
                unhashed = {
                    "schema_version": "midogpp_stage90_proxy_information_query_metrics_v1",
                    "family_id": family_id,
                    "outer_target_id": outer,
                    "query_id": query,
                    "candidate_count": len(rows),
                    "exact_top1": exact_top1,
                    "tie_aware_top1": tie_aware_top1,
                    "spearman": 0.0 if correlation is None else correlation,
                    "spearman_defined": correlation is not None,
                    "normalized_oracle_regret": regret,
                    "pairwise_accuracy": pairwise,
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "calibration_slope_defined": slope_defined,
                    "rmse": rmse,
                    "selected_source": selected,
                    "oracle_sources": list(oracle),
                    "predicted_top_sources": list(predicted_top),
                    "inference_unit": "descriptive_query_nested_within_outer_H",
                    "technical_seed_rows_are_independent_observations": False,
                }
                output.append(
                    QueryMetricRow(
                        family_id=family_id,
                        outer_target_id=outer,
                        query_id=query,
                        candidate_count=len(rows),
                        exact_top1=exact_top1,
                        tie_aware_top1=tie_aware_top1,
                        spearman=0.0 if correlation is None else correlation,
                        spearman_defined=correlation is not None,
                        normalized_oracle_regret=regret,
                        pairwise_accuracy=pairwise,
                        calibration_intercept=intercept,
                        calibration_slope=slope,
                        calibration_slope_defined=slope_defined,
                        rmse=rmse,
                        selected_source=selected,
                        oracle_sources=oracle,
                        predicted_top_sources=predicted_top,
                        row_hash=canonical_sha256(unhashed),
                    )
                )
    return tuple(output)


def _outer_metric_rows(
    crossfit: ProxyCrossfitResult,
    query_rows: tuple[QueryMetricRow, ...],
) -> tuple[OuterMetricRow, ...]:
    output: list[OuterMetricRow] = []
    for family_id in FAMILY_IDS:
        for outer in CENTERS:
            selected_queries = tuple(
                row
                for row in query_rows
                if row.family_id == family_id and row.outer_target_id == outer
            )
            candidate_rows = tuple(
                row
                for row in crossfit.predictions
                if row.family_id == family_id and row.outer_target_id == outer
            )
            if len(selected_queries) != 8 or len(candidate_rows) != 56:
                raise ProtocolError("Proxy outer metric coverage drifted.")
            predicted = np.asarray(
                [row.predicted_utility_delta for row in candidate_rows],
                dtype=np.float64,
            )
            observed = np.asarray(
                [row.observed_utility_delta for row in candidate_rows],
                dtype=np.float64,
            )
            intercept, slope, slope_defined = _calibration(predicted, observed)
            rmse = float(
                np.sqrt(np.mean((predicted - observed) ** 2, dtype=np.float64))
            )
            values = {
                "mean_exact_top1": _mean(row.exact_top1 for row in selected_queries),
                "mean_tie_aware_top1": _mean(
                    row.tie_aware_top1 for row in selected_queries
                ),
                # Undefined query ranks contribute zero, conservatively, and
                # remain separately countable through the defined count.
                "mean_spearman": _mean(row.spearman for row in selected_queries),
                "mean_normalized_oracle_regret": _mean(
                    row.normalized_oracle_regret for row in selected_queries
                ),
                "mean_pairwise_accuracy": _mean(
                    row.pairwise_accuracy for row in selected_queries
                ),
            }
            unhashed = {
                "schema_version": "midogpp_stage90_proxy_information_outer_metrics_v1",
                "family_id": family_id,
                "outer_target_id": outer,
                "query_count": len(selected_queries),
                **values,
                "defined_spearman_query_count": sum(
                    row.spearman_defined for row in selected_queries
                ),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "calibration_slope_defined": slope_defined,
                "rmse": rmse,
                "inference_unit": "outer_target_center",
                "query_rows_are_nested_descriptive_units": True,
                "technical_seed_rows_are_independent_observations": False,
            }
            output.append(
                OuterMetricRow(
                    family_id=family_id,
                    outer_target_id=outer,
                    query_count=len(selected_queries),
                    mean_exact_top1=values["mean_exact_top1"],
                    mean_tie_aware_top1=values["mean_tie_aware_top1"],
                    mean_spearman=values["mean_spearman"],
                    defined_spearman_query_count=sum(
                        row.spearman_defined for row in selected_queries
                    ),
                    mean_normalized_oracle_regret=values[
                        "mean_normalized_oracle_regret"
                    ],
                    mean_pairwise_accuracy=values["mean_pairwise_accuracy"],
                    calibration_intercept=intercept,
                    calibration_slope=slope,
                    calibration_slope_defined=slope_defined,
                    rmse=rmse,
                    row_hash=canonical_sha256(unhashed),
                )
            )
    if len(output) != len(FAMILY_IDS) * EXPECTED_OUTER_METRIC_ROW_COUNT_PER_FAMILY:
        raise ProtocolError("Proxy outer metric family coverage drifted.")
    return tuple(output)


def _family_summary_rows(
    outer_rows: tuple[OuterMetricRow, ...],
) -> tuple[FamilySummaryRow, ...]:
    preliminary: dict[str, dict[str, object]] = {}
    for family_id in FAMILY_IDS:
        rows = tuple(row for row in outer_rows if row.family_id == family_id)
        if tuple(row.outer_target_id for row in rows) != CENTERS:
            raise ProtocolError("Proxy family inference must use exactly nine outer H units.")
        spearman = np.asarray([row.mean_spearman for row in rows], dtype=np.float64)
        regret = np.asarray(
            [row.mean_normalized_oracle_regret for row in rows], dtype=np.float64
        )
        pairwise = np.asarray(
            [row.mean_pairwise_accuracy for row in rows], dtype=np.float64
        )
        spearman_ci = _student_t_ci95(spearman)
        regret_ci = _student_t_ci95(regret)
        pairwise_ci = _student_t_ci95(pairwise)
        preliminary[family_id] = {
            "family_id": family_id,
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
            "normalized_oracle_regret_ci95_lower": regret_ci[0],
            "normalized_oracle_regret_ci95_upper": regret_ci[1],
            "mean_pairwise_accuracy": float(
                np.mean(pairwise, dtype=np.float64)
            ),
            "pairwise_accuracy_ci95_lower": pairwise_ci[0],
            "pairwise_accuracy_ci95_upper": pairwise_ci[1],
            "mean_calibration_intercept": _mean(
                row.calibration_intercept for row in rows
            ),
            "mean_calibration_slope": _mean(
                row.calibration_slope for row in rows
            ),
            "mean_rmse": _mean(row.rmse for row in rows),
        }
    control_regrets = {
        family_id: float(preliminary[family_id]["mean_normalized_oracle_regret"])
        for family_id in CONTROL_FAMILY_IDS
    }
    output: list[FamilySummaryRow] = []
    for family_id in FAMILY_IDS:
        values = preliminary[family_id]
        eligible = family_id in SCREENING_FAMILY_IDS
        spearman_pass = bool(values["spearman_ci95_lower"] > 0.0)
        pairwise_pass = bool(values["pairwise_accuracy_ci95_lower"] > 0.5)
        regret_pass = bool(values["normalized_oracle_regret_ci95_upper"] < 0.5)
        beats_controls = bool(
            eligible
            and all(
                float(values["mean_normalized_oracle_regret"]) < control
                for control in control_regrets.values()
            )
        )
        screening_pass = bool(
            eligible
            and spearman_pass
            and pairwise_pass
            and regret_pass
            and beats_controls
        )
        row_values = {
            **values,
            "spearman_gate_passed": spearman_pass,
            "pairwise_gate_passed": pairwise_pass,
            "regret_gate_passed": regret_pass,
            "beats_all_regret_controls": beats_controls,
            "screening_eligible": eligible,
            "screening_passed": screening_pass,
        }
        unhashed = {
            "schema_version": "midogpp_stage90_proxy_information_family_summary_v1",
            **row_values,
            "inference_unit": "outer_target_center",
            "query_rows_are_nested_descriptive_units": True,
            "candidate_and_seed_rows_are_not_inference_units": True,
            "confidence_interval": "two_sided_student_t_95_percent_n9",
            "screening_gate_may_authorize_policy": False,
        }
        output.append(
            FamilySummaryRow(
                **row_values,  # type: ignore[arg-type]
                row_hash=canonical_sha256(unhashed),
            )
        )
    return tuple(output)


def _audit_result_payload(
    crossfit: ProxyCrossfitResult,
    query_rows: tuple[QueryMetricRow, ...],
    outer_rows: tuple[OuterMetricRow, ...],
    summaries: tuple[FamilySummaryRow, ...],
    *,
    proxy_information_gate_passed: bool,
    informative_family_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_stage90_proxy_information_audit_result_v1",
        "experiment_id": EXPERIMENT_ID,
        "feature_surface_hash": crossfit.feature_surface_hash,
        "utility_surface_hash": crossfit.utility_surface_hash,
        "crossfit_result_hash": crossfit.result_hash,
        "crossfit_fold_lock_hash": crossfit.fold_lock.lock_hash,
        "query_metric_row_count": len(query_rows),
        "outer_metric_row_count": len(outer_rows),
        "family_summary_row_count": len(summaries),
        "family_summary_row_hashes": [row.row_hash for row in summaries],
        "proxy_information_gate_passed": proxy_information_gate_passed,
        "informative_family_ids": list(informative_family_ids),
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "response_row_count": 504,
        "technical_seed_row_count": 4_536,
        "technical_seed_rows_are_independent_observations": False,
        "outer_target_centers_are_inference_units": True,
        "consumed_validation_data": True,
        "diagnostic_only": True,
        "screening_gate_may_authorize_policy": False,
        "routing_quality_claimed": False,
        "policy_update_authorized": False,
        "promotion_eligible": False,
    }


def _student_t_ci95(values: np.ndarray) -> tuple[float, float]:
    if values.shape != (OUTER_INFERENCE_UNIT_COUNT,) or not np.isfinite(values).all():
        raise ProtocolError("Proxy inference requires exactly nine finite outer-H values.")
    mean = float(np.mean(values, dtype=np.float64))
    standard_deviation = float(np.std(values, ddof=1))
    margin = STUDENT_T_975_DF8 * standard_deviation / math.sqrt(float(len(values)))
    return mean - margin, mean + margin


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
    if len(scores) != 21:
        raise ProtocolError("Proxy pairwise accuracy requires 21 candidate pairs.")
    return float(np.mean(np.asarray(scores, dtype=np.float64), dtype=np.float64))


def _calibration(
    predicted: np.ndarray, observed: np.ndarray
) -> tuple[float, float, bool]:
    mean_predicted = float(np.mean(predicted, dtype=np.float64))
    mean_observed = float(np.mean(observed, dtype=np.float64))
    centered = predicted - mean_predicted
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        return mean_observed, 0.0, False
    slope = float(np.dot(centered, observed - mean_observed) / denominator)
    intercept = mean_observed - slope * mean_predicted
    if not np.isfinite(intercept) or not np.isfinite(slope):
        raise ProtocolError("Proxy calibration is non-finite.")
    return intercept, slope, True


def _sign(value: float) -> int:
    if value > _TOLERANCE:
        return 1
    if value < -_TOLERANCE:
        return -1
    return 0


def _mean(values: object) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)  # type: ignore[arg-type]
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("Proxy metric mean requires finite nonempty values.")
    return float(np.mean(array, dtype=np.float64))


__all__ = (
    "run_proxy_information_audit",
    "summarize_proxy_information_audit",
)
