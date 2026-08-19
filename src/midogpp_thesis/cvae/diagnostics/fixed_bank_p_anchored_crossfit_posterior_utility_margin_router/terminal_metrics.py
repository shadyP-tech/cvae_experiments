"""Terminal utility and endpoint-oracle diagnostics for sealed probabilities."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    ENDPOINT_METHOD_IDS,
    HARD_THRESHOLD,
    LOG_LOSS_CLIP_EPSILON,
    PORTFOLIO_METHOD_ID,
)


_T_975_DF8 = 2.306004135204166


def score_methods(
    probabilities: Mapping[str, Mapping[str, Mapping[str, tuple[float, ...]]]],
    sample_ids: Mapping[str, Mapping[str, tuple[str, ...]]],
    labels: Mapping[tuple[str, str, str], int],
    *,
    method_order: Sequence[str],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, Mapping[str, Mapping[str, object]]],
]:
    denominators = _center_denominators(labels)
    reference = probabilities[PORTFOLIO_METHOD_ID]
    center_metrics: dict[str, dict[str, Mapping[str, object]]] = {}
    for method in method_order:
        center_metrics[method] = {
            center: _score_center(
                center,
                probabilities[method][center],
                reference[center],
                sample_ids[center],
                labels,
                denominators[center],
            )
            for center in CENTERS
        }
    p_bacc = {
        center: float(center_metrics[PORTFOLIO_METHOD_ID][center]["center_bacc"])
        for center in CENTERS
    }
    p_brier = {
        center: float(center_metrics[PORTFOLIO_METHOD_ID][center]["center_brier"])
        for center in CENTERS
    }
    p_log_loss = {
        center: float(center_metrics[PORTFOLIO_METHOD_ID][center]["center_log_loss"])
        for center in CENTERS
    }
    method_rows: list[Mapping[str, object]] = []
    center_rows: list[Mapping[str, object]] = []
    oracle_rows: list[Mapping[str, object]] = []
    for method in method_order:
        per_center = center_metrics[method]
        deltas = np.asarray(
            [float(per_center[c]["center_bacc"]) - p_bacc[c] for c in CENTERS],
            dtype=np.float64,
        )
        for center in CENTERS:
            center_rows.append(
                MappingProxyType(
                    {
                        **dict(per_center[center]),
                        "method_id": method,
                        "reference_method": PORTFOLIO_METHOD_ID,
                        "center_bacc_delta_vs_P": float(per_center[center]["center_bacc"])
                        - p_bacc[center],
                        "center_brier_delta_vs_P": float(
                            per_center[center]["center_brier"]
                        )
                        - p_brier[center],
                        "center_log_loss_delta_vs_P": float(
                            per_center[center]["center_log_loss"]
                        )
                        - p_log_loss[center],
                        "formal_claim_authorized": False,
                    }
                )
            )
        oracle = _oracle_diagnostics(
            method,
            probabilities,
            sample_ids,
            labels,
            denominators,
        )
        oracle_rows.extend(oracle["rows"])
        bacc = np.asarray([float(per_center[c]["center_bacc"]) for c in CENTERS])
        brier = np.asarray([float(per_center[c]["center_brier"]) for c in CENTERS])
        log_loss = np.asarray([float(per_center[c]["center_log_loss"]) for c in CENTERS])
        brier_deltas = np.asarray(
            [float(per_center[c]["center_brier"]) - p_brier[c] for c in CENTERS],
            dtype=np.float64,
        )
        log_loss_deltas = np.asarray(
            [float(per_center[c]["center_log_loss"]) - p_log_loss[c] for c in CENTERS],
            dtype=np.float64,
        )
        pooled = _pooled_metrics(per_center)
        mean_delta = float(np.mean(deltas, dtype=np.float64))
        se = float(np.std(deltas, ddof=1) / math.sqrt(len(CENTERS)))
        route_count = sum(int(per_center[c]["changed_case_count"]) for c in CENTERS)
        method_rows.append(
            MappingProxyType(
                {
                    "method_id": method,
                    "equal_center_bacc": float(np.mean(bacc, dtype=np.float64)),
                    "sample_pooled_bacc": pooled["sample_pooled_bacc"],
                    "global_brier": pooled["global_brier"],
                    "equal_center_brier": float(np.mean(brier, dtype=np.float64)),
                    "global_log_loss": pooled["global_log_loss"],
                    "equal_center_log_loss": float(np.mean(log_loss, dtype=np.float64)),
                    "mean_center_bacc_delta_vs_P": mean_delta,
                    "minimum_center_bacc_delta_vs_P": float(np.min(deltas)),
                    "maximum_center_bacc_delta_vs_P": float(np.max(deltas)),
                    "mean_center_brier_delta_vs_P": float(
                        np.mean(brier_deltas, dtype=np.float64)
                    ),
                    "maximum_center_brier_delta_vs_P": float(np.max(brier_deltas)),
                    "mean_center_log_loss_delta_vs_P": float(
                        np.mean(log_loss_deltas, dtype=np.float64)
                    ),
                    "maximum_center_log_loss_delta_vs_P": float(
                        np.max(log_loss_deltas)
                    ),
                    "positive_center_count": int(np.sum(deltas > 1.0e-12)),
                    "negative_center_count": int(np.sum(deltas < -1.0e-12)),
                    "zero_center_count": int(np.sum(np.abs(deltas) <= 1.0e-12)),
                    "descriptive_t8_lower": mean_delta - _T_975_DF8 * se,
                    "descriptive_t8_upper": mean_delta + _T_975_DF8 * se,
                    "descriptive_interval_has_no_nominal_coverage_claim": True,
                    "route_count": route_count,
                    "route_coverage": route_count
                    / sum(len(sample_ids[c]) for c in CENTERS),
                    "threshold_switch_count": sum(
                        int(per_center[c]["threshold_switch_count"]) for c in CENTERS
                    ),
                    "helpful_threshold_switch_count": sum(
                        int(per_center[c]["helpful_threshold_switch_count"]) for c in CENTERS
                    ),
                    "harmful_threshold_switch_count": sum(
                        int(per_center[c]["harmful_threshold_switch_count"]) for c in CENTERS
                    ),
                    "endpoint_oracle_top1_attainment_case_weighted": oracle["top1_case_weighted"],
                    "endpoint_oracle_top1_attainment_equal_center": oracle["top1_equal_center"],
                    "mean_endpoint_oracle_rank_case_weighted": oracle["rank_case_weighted"],
                    "mean_endpoint_oracle_rank_equal_center": oracle["rank_equal_center"],
                    "mean_normalized_endpoint_oracle_gap_case_weighted": oracle["gap_case_weighted"],
                    "mean_normalized_endpoint_oracle_gap_equal_center": oracle["gap_equal_center"],
                    "formal_claim_authorized": False,
                }
            )
        )
    frozen = MappingProxyType(
        {
            method: MappingProxyType(dict(rows))
            for method, rows in center_metrics.items()
        }
    )
    return tuple(method_rows), tuple(center_rows), tuple(oracle_rows), frozen


def _center_denominators(
    labels: Mapping[tuple[str, str, str], int]
) -> Mapping[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for center in CENTERS:
        values = [value for (observed, _case, _sample), value in labels.items() if observed == center]
        positive, negative = values.count(1), values.count(0)
        if not positive or not negative:
            raise ProtocolError("PUMR terminal center lacks both classes.")
        result[center] = (positive, negative)
    return MappingProxyType(result)


def _score_center(
    center: str,
    method: Mapping[str, tuple[float, ...]],
    reference: Mapping[str, tuple[float, ...]],
    case_samples: Mapping[str, tuple[str, ...]],
    labels: Mapping[tuple[str, str, str], int],
    denominators: tuple[int, int],
) -> Mapping[str, object]:
    y_values: list[int] = []
    probabilities: list[float] = []
    p_values: list[float] = []
    changed_cases = 0
    for case, samples in case_samples.items():
        current = tuple(float(value) for value in method[case])
        baseline = tuple(float(value) for value in reference[case])
        if len(current) != len(samples) or len(baseline) != len(samples):
            raise ProtocolError("PUMR terminal probability/sample alignment drifted.")
        changed_cases += int(any(abs(a - b) > 1.0e-15 for a, b in zip(current, baseline, strict=True)))
        y_values.extend(labels[(center, case, sample)] for sample in samples)
        probabilities.extend(current)
        p_values.extend(baseline)
    y = np.asarray(y_values, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    p = np.asarray(p_values, dtype=np.float64)
    hard, p_hard = probability >= HARD_THRESHOLD, p >= HARD_THRESHOLD
    n_positive, n_negative = denominators
    tp = int(np.sum((y == 1) & hard, dtype=np.int64))
    tn = int(np.sum((y == 0) & (~hard), dtype=np.int64))
    crossing = hard != p_hard
    clipped = np.clip(probability, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    losses = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    return MappingProxyType(
        {
            "target_center": center,
            "sample_count": len(y),
            "case_count": len(case_samples),
            "changed_case_count": changed_cases,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": int(np.sum((y == 0) & hard, dtype=np.int64)),
            "false_negative": int(np.sum((y == 1) & (~hard), dtype=np.int64)),
            "center_bacc": 0.5 * (tp / n_positive + tn / n_negative),
            "center_brier": float(np.mean((probability - y) ** 2, dtype=np.float64)),
            "center_log_loss": float(np.mean(losses, dtype=np.float64)),
            "threshold_switch_count": int(np.sum(crossing, dtype=np.int64)),
            "helpful_threshold_switch_count": int(np.sum(crossing & (hard == y), dtype=np.int64)),
            "harmful_threshold_switch_count": int(np.sum(crossing & (p_hard == y), dtype=np.int64)),
            "squared_error_sum": float(np.sum((probability - y) ** 2, dtype=np.float64)),
            "log_loss_sum": float(np.sum(losses, dtype=np.float64)),
        }
    )


def _pooled_metrics(per_center: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    positive = sum(int(per_center[c]["n_positive"]) for c in CENTERS)
    negative = sum(int(per_center[c]["n_negative"]) for c in CENTERS)
    count = sum(int(per_center[c]["sample_count"]) for c in CENTERS)
    return {
        "sample_pooled_bacc": 0.5
        * (
            sum(int(per_center[c]["true_positive"]) for c in CENTERS) / positive
            + sum(int(per_center[c]["true_negative"]) for c in CENTERS) / negative
        ),
        "global_brier": sum(float(per_center[c]["squared_error_sum"]) for c in CENTERS) / count,
        "global_log_loss": sum(float(per_center[c]["log_loss_sum"]) for c in CENTERS) / count,
    }


def _oracle_diagnostics(
    method: str,
    probabilities: Mapping[str, Mapping[str, Mapping[str, tuple[float, ...]]]],
    sample_ids: Mapping[str, Mapping[str, tuple[str, ...]]],
    labels: Mapping[tuple[str, str, str], int],
    denominators: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    rows: list[Mapping[str, object]] = []
    by_center: dict[str, list[tuple[float, float, float | None]]] = {
        center: [] for center in CENTERS
    }
    for center in CENTERS:
        n_positive, n_negative = denominators[center]
        for case, samples in sample_ids[center].items():
            y = np.asarray([labels[(center, case, sample)] for sample in samples], dtype=np.int8)
            endpoint_utility: dict[str, float] = {}
            for endpoint in ENDPOINT_METHOD_IDS:
                hard = np.asarray(probabilities[endpoint][center][case]) >= HARD_THRESHOLD
                endpoint_utility[endpoint] = 0.5 * (
                    np.sum((y == 1) & hard, dtype=np.int64) / n_positive
                    + np.sum((y == 0) & (~hard), dtype=np.int64) / n_negative
                )
            method_hard = np.asarray(probabilities[method][center][case]) >= HARD_THRESHOLD
            realized = 0.5 * (
                np.sum((y == 1) & method_hard, dtype=np.int64) / n_positive
                + np.sum((y == 0) & (~method_hard), dtype=np.int64) / n_negative
            )
            best, worst = max(endpoint_utility.values()), min(endpoint_utility.values())
            top1 = float(realized >= best - 1.0e-15)
            rank = 1.0 + sum(value > realized + 1.0e-15 for value in endpoint_utility.values())
            raw_regret = best - realized
            spread = best - worst
            degenerate = spread <= 1.0e-15
            gap = None if degenerate else raw_regret / spread
            outside_envelope = realized < worst - 1.0e-15 or realized > best + 1.0e-15
            by_center[center].append((top1, rank, gap))
            rows.append(
                MappingProxyType(
                    {
                        "method_id": method,
                        "target_center": center,
                        "case_id": case,
                        "method_case_contribution": float(realized),
                        "best_endpoint_case_contribution": float(best),
                        "worst_endpoint_case_contribution": float(worst),
                        "endpoint_oracle_top1_attained": bool(top1),
                        "endpoint_oracle_rank": rank,
                        "raw_endpoint_oracle_regret": float(raw_regret),
                        "endpoint_oracle_spread": float(spread),
                        "endpoint_oracle_spread_degenerate": degenerate,
                        "method_outside_endpoint_envelope": outside_envelope,
                        "normalized_endpoint_oracle_gap": (
                            None if gap is None else float(gap)
                        ),
                        "formal_claim_authorized": False,
                    }
                )
            )
    flat = [value for center in CENTERS for value in by_center[center]]
    return {
        "rows": tuple(rows),
        "top1_case_weighted": float(np.mean([row[0] for row in flat])),
        "top1_equal_center": float(np.mean([np.mean([row[0] for row in by_center[c]]) for c in CENTERS])),
        "rank_case_weighted": float(np.mean([row[1] for row in flat])),
        "rank_equal_center": float(np.mean([np.mean([row[1] for row in by_center[c]]) for c in CENTERS])),
        "gap_case_weighted": _optional_mean([row[2] for row in flat]),
        "gap_equal_center": _optional_mean(
            [_optional_mean([row[2] for row in by_center[c]]) for c in CENTERS]
        ),
    }


def _optional_mean(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined, dtype=np.float64)) if defined else None


__all__ = ("score_methods",)
