"""Exact aggregate and case-level terminal metrics for composed P-DCAPS outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from .....protocol import ProtocolError
from ...identity import METHOD_MENU, P_METHOD_ID
from ...label_firewall import TerminalLabelCapability
from ...method_controls import ComposedMethodPrediction


_HARD_THRESHOLD = 0.5
_LOG_EPSILON = 1.0e-12
_T_975_DF8 = 2.306004135204166


def score_composed_methods(
    compositions: Sequence[ComposedMethodPrediction],
    capabilities: Sequence[TerminalLabelCapability],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, dict[str, dict[str, object]]],
]:
    """Score the exact 9-center x 6-method frozen output rectangle."""

    rows = tuple(compositions)
    by_key = {
        (row.decision.outer_center, row.decision.method_id): row for row in rows
    }
    expected_keys = tuple(
        (center, method) for center in CENTERS for method in METHOD_MENU
    )
    if len(by_key) != len(rows) or tuple(by_key) != expected_keys:
        raise ProtocolError("P-DCAPS terminal composition inventory drifted.")
    capability_by_center = {row.center: row for row in capabilities}
    if tuple(capability_by_center) != CENTERS:
        raise ProtocolError("P-DCAPS terminal capability center inventory drifted.")

    truth_by_center: dict[str, dict[str, int]] = {}
    case_by_center_sample: dict[str, dict[str, str]] = {}
    for center in CENTERS:
        capability = capability_by_center[center]
        truth_by_center[center] = {
            row.sample_id: int(row.value) for row in capability.rows
        }
        case_by_center_sample[center] = {
            row.sample_id: row.case_id for row in capability.rows
        }

    center_metrics: dict[str, dict[str, dict[str, object]]] = {
        method: {} for method in METHOD_MENU
    }
    case_rows: list[dict[str, object]] = []
    for center in CENTERS:
        baseline = by_key[(center, P_METHOD_ID)].prediction
        baseline_by_sample = dict(
            zip(
                baseline.sample_ids,
                (float(value) for value in baseline.probabilities),
                strict=True,
            )
        )
        for method in METHOD_MENU:
            composition = by_key[(center, method)]
            prediction = composition.prediction
            if (
                prediction.sample_ids != baseline.sample_ids
                or set(prediction.sample_ids) != set(truth_by_center[center])
            ):
                raise ProtocolError(
                    "P-DCAPS terminal probability/label row order drifted."
                )
            metrics, per_case = _score_center(
                center=center,
                method=method,
                sample_ids=prediction.sample_ids,
                probabilities=prediction.probabilities,
                baseline_by_sample=baseline_by_sample,
                truth_by_sample=truth_by_center[center],
                case_by_sample=case_by_center_sample[center],
            )
            center_metrics[method][center] = metrics
            case_rows.extend(per_case)

    method_rows: list[dict[str, object]] = []
    center_rows: list[dict[str, object]] = []
    reference = center_metrics[P_METHOD_ID]
    for method in METHOD_MENU:
        metrics = center_metrics[method]
        deltas = np.asarray(
            [
                float(metrics[center]["center_bacc"])
                - float(reference[center]["center_bacc"])
                for center in CENTERS
            ],
            dtype=np.float64,
        )
        brier_deltas = np.asarray(
            [
                float(metrics[center]["center_brier"])
                - float(reference[center]["center_brier"])
                for center in CENTERS
            ],
            dtype=np.float64,
        )
        log_deltas = np.asarray(
            [
                float(metrics[center]["center_log_loss"])
                - float(reference[center]["center_log_loss"])
                for center in CENTERS
            ],
            dtype=np.float64,
        )
        for center in CENTERS:
            center_rows.append(
                {
                    **metrics[center],
                    "method_id": method,
                    "reference_method": P_METHOD_ID,
                    "center_bacc_delta_vs_P": (
                        float(metrics[center]["center_bacc"])
                        - float(reference[center]["center_bacc"])
                    ),
                    "center_brier_delta_vs_P": (
                        float(metrics[center]["center_brier"])
                        - float(reference[center]["center_brier"])
                    ),
                    "center_log_loss_delta_vs_P": (
                        float(metrics[center]["center_log_loss"])
                        - float(reference[center]["center_log_loss"])
                    ),
                    "formal_claim_authorized": False,
                }
            )
        pooled = _pooled_metrics(metrics)
        mean_delta = float(np.mean(deltas, dtype=np.float64))
        standard_error = float(
            np.std(deltas, ddof=1) / math.sqrt(len(CENTERS))
        )
        method_case_rows = [row for row in case_rows if row["method_id"] == method]
        method_rows.append(
            {
                "method_id": method,
                "equal_center_bacc": float(
                    np.mean(
                        [float(metrics[c]["center_bacc"]) for c in CENTERS],
                        dtype=np.float64,
                    )
                ),
                "sample_pooled_bacc": pooled["sample_pooled_bacc"],
                "global_brier": pooled["global_brier"],
                "equal_center_brier": float(
                    np.mean(
                        [float(metrics[c]["center_brier"]) for c in CENTERS],
                        dtype=np.float64,
                    )
                ),
                "global_log_loss": pooled["global_log_loss"],
                "equal_center_log_loss": float(
                    np.mean(
                        [float(metrics[c]["center_log_loss"]) for c in CENTERS],
                        dtype=np.float64,
                    )
                ),
                "mean_center_bacc_delta_vs_P": mean_delta,
                "minimum_center_bacc_delta_vs_P": float(np.min(deltas)),
                "maximum_center_bacc_delta_vs_P": float(np.max(deltas)),
                "mean_center_brier_delta_vs_P": float(
                    np.mean(brier_deltas, dtype=np.float64)
                ),
                "mean_center_log_loss_delta_vs_P": float(
                    np.mean(log_deltas, dtype=np.float64)
                ),
                "positive_center_count": int(np.sum(deltas > 1.0e-12)),
                "negative_center_count": int(np.sum(deltas < -1.0e-12)),
                "zero_center_count": int(np.sum(np.abs(deltas) <= 1.0e-12)),
                "descriptive_t8_lower": mean_delta - _T_975_DF8 * standard_error,
                "descriptive_t8_upper": mean_delta + _T_975_DF8 * standard_error,
                "descriptive_interval_has_no_nominal_coverage_claim": True,
                "route_count": sum(
                    int(metrics[c]["changed_case_count"]) for c in CENTERS
                ),
                "case_harm_count": sum(
                    int(bool(row["case_harmed_vs_P"])) for row in method_case_rows
                ),
                "case_harm_rate": (
                    sum(
                        int(bool(row["case_harmed_vs_P"]))
                        for row in method_case_rows
                    )
                    / len(method_case_rows)
                ),
                "formal_claim_authorized": False,
            }
        )
    return tuple(method_rows), tuple(center_rows), tuple(case_rows), center_metrics


def _score_center(
    *,
    center: str,
    method: str,
    sample_ids: Sequence[str],
    probabilities: object,
    baseline_by_sample: Mapping[str, float],
    truth_by_sample: Mapping[str, int],
    case_by_sample: Mapping[str, str],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    samples = tuple(str(value) for value in sample_ids)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(samples),) or not np.isfinite(probability).all():
        raise ProtocolError("P-DCAPS terminal probability vector drifted.")
    truth = np.asarray([truth_by_sample[value] for value in samples], dtype=np.int8)
    baseline = np.asarray([baseline_by_sample[value] for value in samples])
    positive = int(np.sum(truth == 1, dtype=np.int64))
    negative = int(np.sum(truth == 0, dtype=np.int64))
    if positive <= 0 or negative <= 0:
        raise ProtocolError("P-DCAPS terminal center lacks both classes.")
    hard = probability >= _HARD_THRESHOLD
    baseline_hard = baseline >= _HARD_THRESHOLD
    true_positive = int(np.sum((truth == 1) & hard, dtype=np.int64))
    true_negative = int(np.sum((truth == 0) & (~hard), dtype=np.int64))
    clipped = np.clip(probability, _LOG_EPSILON, 1.0 - _LOG_EPSILON)
    losses = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))

    positions_by_case: dict[str, list[int]] = defaultdict(list)
    for position, sample in enumerate(samples):
        positions_by_case[case_by_sample[sample]].append(position)
    case_rows: list[dict[str, object]] = []
    changed_cases = 0
    for case in sorted(positions_by_case):
        positions = np.asarray(positions_by_case[case], dtype=np.int64)
        current_errors = int(np.sum(hard[positions] != truth[positions]))
        baseline_errors = int(
            np.sum(baseline_hard[positions] != truth[positions])
        )
        changed = not np.array_equal(
            probability[positions].astype(np.float32),
            baseline[positions].astype(np.float32),
        )
        changed_cases += int(changed)
        case_rows.append(
            {
                "target_center": center,
                "case_id": case,
                "method_id": method,
                "sample_count": len(positions),
                "probability_changed_vs_P": changed,
                "threshold_error_delta_vs_P": current_errors - baseline_errors,
                "case_harmed_vs_P": current_errors > baseline_errors,
                "raw_labels_persisted": False,
                "formal_claim_authorized": False,
            }
        )
    crossing = hard != baseline_hard
    metrics = {
        "target_center": center,
        "sample_count": len(samples),
        "case_count": len(positions_by_case),
        "changed_case_count": changed_cases,
        "n_positive": positive,
        "n_negative": negative,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": int(np.sum((truth == 0) & hard, dtype=np.int64)),
        "false_negative": int(np.sum((truth == 1) & (~hard), dtype=np.int64)),
        "center_bacc": 0.5 * (
            true_positive / positive + true_negative / negative
        ),
        "center_brier": float(
            np.mean((probability - truth) ** 2, dtype=np.float64)
        ),
        "center_log_loss": float(np.mean(losses, dtype=np.float64)),
        "threshold_switch_count": int(np.sum(crossing, dtype=np.int64)),
        "helpful_threshold_switch_count": int(
            np.sum(crossing & (hard == truth), dtype=np.int64)
        ),
        "harmful_threshold_switch_count": int(
            np.sum(crossing & (baseline_hard == truth), dtype=np.int64)
        ),
        "squared_error_sum": float(
            np.sum((probability - truth) ** 2, dtype=np.float64)
        ),
        "log_loss_sum": float(np.sum(losses, dtype=np.float64)),
    }
    return metrics, tuple(case_rows)


def _pooled_metrics(
    per_center: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    positive = sum(int(per_center[c]["n_positive"]) for c in CENTERS)
    negative = sum(int(per_center[c]["n_negative"]) for c in CENTERS)
    count = sum(int(per_center[c]["sample_count"]) for c in CENTERS)
    return {
        "sample_pooled_bacc": 0.5
        * (
            sum(int(per_center[c]["true_positive"]) for c in CENTERS) / positive
            + sum(int(per_center[c]["true_negative"]) for c in CENTERS) / negative
        ),
        "global_brier": sum(
            float(per_center[c]["squared_error_sum"]) for c in CENTERS
        )
        / count,
        "global_log_loss": sum(
            float(per_center[c]["log_loss_sum"]) for c in CENTERS
        )
        / count,
    }


__all__ = ("score_composed_methods",)
