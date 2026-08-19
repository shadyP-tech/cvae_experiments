"""Terminal diagnostics for direct signed utility and stability transfer."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import directional_candidate
from .constants import (
    CENTERS,
    LOG_LOSS_CLIP_EPSILON,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PORTFOLIO_METHOD_ID,
)
from .engine import PreterminalResult
from .utility_contracts import UtilityDescriptor


def utility_information_diagnostics(
    preterminal: PreterminalResult,
    labels: Mapping[tuple[str, str, str], int],
    *,
    primary_mean_center_bacc_delta_vs_p: float,
    primary_mean_center_brier_delta_vs_p: float,
    primary_mean_center_log_loss_delta_vs_p: float,
    primary_helpful_switches: int,
    primary_harmful_switches: int,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    primary = {
        row.descriptor_hash: row
        for row in preterminal.utility_predictions_by_policy[MODEL_BASED_METHOD_ID]
    }
    blocked = {
        row.descriptor_hash: row
        for row in preterminal.utility_predictions_by_policy[PERMUTATION_METHOD_ID]
    }
    endpoint_by_key = {
        (center, row.case_id): row
        for center in CENTERS
        for row in preterminal.predictions_by_center[center]
    }
    selected = {
        (row.target_center, row.case_id, decision.direction): decision.selected_alternative
        for row in preterminal.composed_predictions_by_policy[MODEL_BASED_METHOD_ID]
        for decision in row.decisions
    }
    center_counts = {
        center: (
            sum(labels[key] == 1 for key in labels if key[0] == center),
            sum(labels[key] == 0 for key in labels if key[0] == center),
        )
        for center in CENTERS
    }
    rows: list[Mapping[str, object]] = []
    for center in CENTERS:
        for descriptor in preterminal.utility_descriptors_by_center[center]:
            prediction = primary[descriptor.descriptor_hash]
            control = blocked[descriptor.descriptor_hash]
            truth = _actual_utility(
                endpoint_by_key[(center, descriptor.case_id)],
                descriptor,
                labels,
                center_n_positive=center_counts[center][0],
                center_n_negative=center_counts[center][1],
            )
            rows.append(
                MappingProxyType(
                    {
                        "target_center": center,
                        "case_id": descriptor.case_id,
                        "alternative": descriptor.alternative,
                        "direction": descriptor.direction,
                        "crossing_count": descriptor.crossing_count,
                        "structural_zero": descriptor.crossing_count == 0,
                        "selected_by_primary": selected[
                            (center, descriptor.case_id, descriptor.direction)
                        ]
                        == descriptor.alternative,
                        "actual_bacc_contribution_delta": truth[0],
                        "actual_brier_contribution_delta": truth[1],
                        "actual_log_loss_contribution_delta": truth[2],
                        "predicted_robust_bacc_contribution_delta": prediction.robust(
                            "bacc_contribution_delta"
                        ),
                        "predicted_robust_brier_contribution_delta": prediction.robust(
                            "brier_contribution_delta"
                        ),
                        "predicted_robust_log_loss_contribution_delta": prediction.robust(
                            "log_loss_contribution_delta"
                        ),
                        "bacc_residual_scale": prediction.scale(
                            "bacc_contribution_delta"
                        ),
                        "bacc_positive_delete_fraction": prediction.fraction(
                            "bacc_contribution_delta"
                        ),
                        "brier_safe_delete_fraction": prediction.fraction(
                            "brier_contribution_delta"
                        ),
                        "log_loss_safe_delete_fraction": prediction.fraction(
                            "log_loss_contribution_delta"
                        ),
                        "blocked_predicted_robust_bacc_delta": control.robust(
                            "bacc_contribution_delta"
                        ),
                        "descriptor_hash": descriptor.descriptor_hash,
                        "prediction_hash": prediction.prediction_hash,
                        "raw_label_persisted": False,
                    }
                )
            )
    if len(rows) != 6 * sum(len(preterminal.predictions_by_center[c]) for c in CENTERS):
        raise ProtocolError("PDSUR terminal utility information surface is incomplete.")

    center_rows = tuple(
        MappingProxyType(_summarize_rows(tuple(row for row in rows if row["target_center"] == center), center=center))
        for center in CENTERS
    )
    summary = _summarize_rows(tuple(rows), center="ALL")
    selected_rows = tuple(row for row in rows if bool(row["selected_by_primary"]))
    selected_helpful = sum(
        float(row["actual_bacc_contribution_delta"]) > 0.0 for row in selected_rows
    )
    selected_harmful = sum(
        float(row["actual_bacc_contribution_delta"]) < 0.0 for row in selected_rows
    )
    proper_pass = (
        primary_mean_center_brier_delta_vs_p <= 0.0
        and primary_mean_center_log_loss_delta_vs_p <= 0.0
    )
    signal_pass = float(summary["bacc_spearman"]) > 0.0
    blocked_pass = float(summary["bacc_spearman"]) > float(
        summary["blocked_bacc_spearman"]
    )
    route_pass = selected_helpful >= selected_harmful
    bacc_pass = primary_mean_center_bacc_delta_vs_p > 0.0
    status = "PASS" if all((proper_pass, signal_pass, blocked_pass, route_pass, bacc_pass)) else "FAIL"
    if not selected_rows:
        bottleneck = "STABILITY_GUARD_ABSTAINS"
    elif not signal_pass:
        bottleneck = "NO_TRANSFERABLE_SIGNED_UTILITY_SIGNAL"
    elif not bacc_pass:
        bottleneck = "ACTION_RANKING_OR_COMPOSITION_FAILURE"
    elif not proper_pass:
        bottleneck = "PROBABILITY_CALIBRATION_FAILURE"
    elif not blocked_pass:
        bottleneck = "FEATURE_SIGNAL_DOES_NOT_BEAT_BLOCKED_CONTROL"
    else:
        bottleneck = "NO_DIAGNOSTIC_BOTTLENECK"
    gate = MappingProxyType(
        {
            "schema_version": "fixed_bank_pdsur_information_gate_v1",
            "status": status,
            "diagnosed_bottleneck": bottleneck,
            "primary_bacc_gain_positive": bacc_pass,
            "primary_proper_loss_safety_pass": proper_pass,
            "direct_signed_utility_rank_signal_positive": signal_pass,
            "direct_signed_utility_beats_blocked_control": blocked_pass,
            "selected_helpful_action_count": selected_helpful,
            "selected_harmful_action_count": selected_harmful,
            "selected_action_count": len(selected_rows),
            "selected_helpful_actions_not_fewer_than_harmful": route_pass,
            "primary_helpful_threshold_switch_count": primary_helpful_switches,
            "primary_harmful_threshold_switch_count": primary_harmful_switches,
            "terminal_information_may_change_same_surface_routes": False,
            "nominal_inference_claimed": False,
            "fresh_evidence": False,
        }
    )
    return tuple(rows), center_rows, gate


def _actual_utility(
    endpoint: object,
    descriptor: UtilityDescriptor,
    labels: Mapping[tuple[str, str, str], int],
    *,
    center_n_positive: int,
    center_n_negative: int,
) -> tuple[float, float, float]:
    portfolio = np.asarray(endpoint.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64)
    composed, mask = directional_candidate(
        endpoint, descriptor.alternative, descriptor.direction
    )
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    y = np.asarray(
        [labels[(endpoint.center, endpoint.case_id, sample)] for sample in endpoint.sample_ids],
        dtype=np.int8,
    )
    p_hard = portfolio >= 0.5
    hard = composed >= 0.5
    positive = y == 1
    negative = ~positive
    bacc = 0.5 * (
        float(np.sum(hard[positive].astype(np.int8) - p_hard[positive].astype(np.int8), dtype=np.int64))
        / center_n_positive
        + float(np.sum((~hard[negative]).astype(np.int8) - (~p_hard[negative]).astype(np.int8), dtype=np.int64))
        / center_n_negative
    )
    center_n_total = center_n_positive + center_n_negative
    brier = float(
        np.sum((composed - y) ** 2 - (portfolio - y) ** 2, dtype=np.float64)
        / center_n_total
    )
    p = np.clip(portfolio, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    q = np.clip(composed, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    log = float(
        np.sum(
            -(y * np.log(q) + (1 - y) * np.log1p(-q))
            + (y * np.log(p) + (1 - y) * np.log1p(-p)),
            dtype=np.float64,
        )
        / center_n_total
    )
    return bacc, brier, log


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    left = _rank(np.asarray(x, dtype=np.float64))
    right = _rank(np.asarray(y, dtype=np.float64))
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _summarize_rows(
    rows: Sequence[Mapping[str, object]], *, center: str
) -> dict[str, object]:
    actual = [float(row["actual_bacc_contribution_delta"]) for row in rows]
    predicted = [float(row["predicted_robust_bacc_contribution_delta"]) for row in rows]
    blocked = [float(row["blocked_predicted_robust_bacc_delta"]) for row in rows]
    nonzero = [index for index, row in enumerate(rows) if not bool(row["structural_zero"])]
    actual_nz = [actual[index] for index in nonzero]
    predicted_nz = [predicted[index] for index in nonzero]
    blocked_nz = [blocked[index] for index in nonzero]
    return {
        "target_center": center,
        "utility_row_count": len(rows),
        "structural_zero_count": sum(bool(row["structural_zero"]) for row in rows),
        "selected_action_count": sum(bool(row["selected_by_primary"]) for row in rows),
        "bacc_spearman": _spearman(predicted_nz, actual_nz) if nonzero else 0.0,
        "blocked_bacc_spearman": _spearman(blocked_nz, actual_nz) if nonzero else 0.0,
        "bacc_mean_absolute_error": float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual)), dtype=np.float64)),
        "response_sign_agreement": float(
            np.mean(
                np.sign(np.asarray(predicted_nz)) == np.sign(np.asarray(actual_nz)),
                dtype=np.float64,
            )
        ) if nonzero else 1.0,
        "count_by_alternative": dict(sorted(Counter(str(row["alternative"]) for row in rows).items())),
        "count_by_direction": dict(sorted(Counter(str(row["direction"]) for row in rows).items())),
        "confidence_bound_claimed": False,
    }


__all__ = ("utility_information_diagnostics",)
