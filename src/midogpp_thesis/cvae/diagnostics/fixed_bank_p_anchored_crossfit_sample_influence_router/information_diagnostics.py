"""Terminal-only sample-influence and donor-veto diagnostics."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .calibration import directional_candidate
from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    LOG_LOSS_CLIP_EPSILON,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
)
from .engine import PreterminalResult
from .utility_contracts import UtilityDescriptor


def sample_influence_information_diagnostics(
    preterminal: PreterminalResult,
    labels: Mapping[tuple[str, str, str], int],
    *,
    primary_mean_center_bacc_delta_vs_p: float,
    primary_minimum_center_bacc_delta_vs_p: float,
    primary_mean_center_brier_delta_vs_p: float,
    primary_mean_center_log_loss_delta_vs_p: float,
    primary_helpful_switches: int,
    primary_harmful_switches: int,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    """Open truth only to diagnose already sealed scores and routes."""

    primary = {
        row.descriptor_hash: row
        for row in preterminal.sample_influence_predictions_by_control[
            PRIMARY_FINGERPRINT_CONTROL_ID
        ]
    }
    blocked = {
        row.descriptor_hash: row
        for row in preterminal.sample_influence_predictions_by_control[
            BLOCKED_FINGERPRINT_CONTROL_ID
        ]
    }
    donor = {row.descriptor_hash: row for row in preterminal.donor_veto_predictions}
    primary_posteriors = {
        (row.target_center, row.case_id): row
        for row in preterminal.target_posterior_predictions_by_control[
            PRIMARY_FINGERPRINT_CONTROL_ID
        ]
    }
    blocked_posteriors = {
        (row.target_center, row.case_id): row
        for row in preterminal.target_posterior_predictions_by_control[
            BLOCKED_FINGERPRINT_CONTROL_ID
        ]
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
            sum(value == 1 for key, value in labels.items() if key[0] == center),
            sum(value == 0 for key, value in labels.items() if key[0] == center),
        )
        for center in CENTERS
    }

    action_rows: list[Mapping[str, object]] = []
    sample_rows: list[Mapping[str, object]] = []
    for center in CENTERS:
        for descriptor in preterminal.utility_descriptors_by_center[center]:
            key = (center, descriptor.case_id)
            target = primary[descriptor.descriptor_hash]
            control = blocked[descriptor.descriptor_hash]
            donor_prediction = donor[descriptor.descriptor_hash]
            truth = _actual_utility(
                endpoint_by_key[key],
                descriptor,
                labels,
                center_n_positive=center_counts[center][0],
                center_n_negative=center_counts[center][1],
            )
            action_rows.append(
                MappingProxyType(
                    {
                        "row_type": "action",
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
                        "target_influence_score": target.target_score,
                        "blocked_target_influence_score": control.target_score,
                        "donor_robust_bacc_delta": donor_prediction.robust(
                            "bacc_contribution_delta"
                        ),
                        "donor_robust_brier_delta": donor_prediction.robust(
                            "brier_contribution_delta"
                        ),
                        "donor_robust_log_loss_delta": donor_prediction.robust(
                            "log_loss_contribution_delta"
                        ),
                        "target_score_positive": target.target_score > 0.0,
                        "donor_dual_veto_pass": (
                            donor_prediction.robust("bacc_contribution_delta") > 0.0
                            and donor_prediction.robust("brier_contribution_delta")
                            <= 0.0
                            and donor_prediction.robust("log_loss_contribution_delta")
                            <= 0.0
                        ),
                        "descriptor_hash": descriptor.descriptor_hash,
                        "target_influence_hash": target.influence_hash,
                        "donor_veto_prediction_hash": donor_prediction.prediction_hash,
                        "raw_label_persisted": False,
                    }
                )
            )
            eta = dict(
                zip(
                    primary_posteriors[key].sample_ids,
                    primary_posteriors[key].natural_probabilities,
                    strict=True,
                )
            )
            blocked_eta = dict(
                zip(
                    blocked_posteriors[key].sample_ids,
                    blocked_posteriors[key].natural_probabilities,
                    strict=True,
                )
            )
            for sample_id in descriptor.crossing_sample_ids:
                label = labels[(center, descriptor.case_id, sample_id)]
                helpful = label if descriptor.direction == "zero_to_one" else 1 - label
                probability = (
                    eta[sample_id]
                    if descriptor.direction == "zero_to_one"
                    else 1.0 - eta[sample_id]
                )
                blocked_probability = (
                    blocked_eta[sample_id]
                    if descriptor.direction == "zero_to_one"
                    else 1.0 - blocked_eta[sample_id]
                )
                sample_rows.append(
                    MappingProxyType(
                        {
                            "target_center": center,
                            "actual_helpful": int(helpful),
                            "predicted_help_probability": float(probability),
                            "blocked_help_probability": float(blocked_probability),
                        }
                    )
                )
    expected_actions = 6 * sum(
        len(preterminal.predictions_by_center[center]) for center in CENTERS
    )
    if len(action_rows) != expected_actions:
        raise ProtocolError("PCSI terminal action-information surface is incomplete.")

    center_rows = tuple(
        MappingProxyType(
            _summarize_center(
                center,
                tuple(row for row in action_rows if row["target_center"] == center),
                tuple(row for row in sample_rows if row["target_center"] == center),
            )
        )
        for center in CENTERS
    )
    overall = _summarize_center("ALL", tuple(action_rows), tuple(sample_rows))
    defined_center_aucs = [
        float(row["crossing_sample_auc"])
        for row in center_rows
        if row["crossing_sample_auc"] is not None
    ]
    defined_blocked_aucs = [
        float(row["blocked_crossing_sample_auc"])
        for row in center_rows
        if row["blocked_crossing_sample_auc"] is not None
    ]
    equal_center_auc = (
        float(np.mean(defined_center_aucs, dtype=np.float64))
        if defined_center_aucs
        else 0.5
    )
    equal_center_blocked_auc = (
        float(np.mean(defined_blocked_aucs, dtype=np.float64))
        if defined_blocked_aucs
        else 0.5
    )
    selected_rows = tuple(row for row in action_rows if row["selected_by_primary"])
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
    center_safety_pass = primary_minimum_center_bacc_delta_vs_p >= -1.0e-15
    sample_signal_pass = len(defined_center_aucs) >= 7 and equal_center_auc >= 0.55
    blocked_pass = equal_center_auc >= equal_center_blocked_auc + 0.02
    calibration_pass = float(overall["crossing_sample_brier"]) < float(
        overall["crossing_sample_prevalence_brier"]
    )
    action_rank_pass = float(overall["action_bacc_spearman"]) > 0.0
    route_pass = selected_helpful >= selected_harmful
    bacc_pass = primary_mean_center_bacc_delta_vs_p > 0.0
    status = "PASS" if all(
        (
            proper_pass,
            center_safety_pass,
            sample_signal_pass,
            blocked_pass,
            calibration_pass,
            action_rank_pass,
            route_pass,
            bacc_pass,
        )
    ) else "FAIL"
    if not sample_signal_pass:
        bottleneck = "TARGET_LOCAL_HELPFULNESS_RANK_SIGNAL_WEAK"
    elif not blocked_pass:
        bottleneck = "PHYSICAL_FINGERPRINT_DOES_NOT_BEAT_BLOCKED_CONTROL"
    elif not calibration_pass:
        bottleneck = "TARGET_LOCAL_HELPFULNESS_CALIBRATION_FAILURE"
    elif not action_rank_pass:
        bottleneck = "SAMPLE_SIGNAL_DOES_NOT_AGGREGATE_TO_ACTION_RANKING"
    elif not bacc_pass or not center_safety_pass:
        bottleneck = "ACTION_SELECTION_OR_CENTER_STABILITY_FAILURE"
    elif not proper_pass:
        bottleneck = "PROBABILITY_LOSS_SAFETY_FAILURE"
    else:
        bottleneck = "NO_DIAGNOSTIC_BOTTLENECK"
    gate = MappingProxyType(
        {
            "schema_version": "fixed_bank_pcsi_information_gate_v1",
            "status": status,
            "diagnosed_bottleneck": bottleneck,
            "primary_bacc_gain_positive": bacc_pass,
            "primary_minimum_center_bacc_nonnegative": center_safety_pass,
            "primary_proper_loss_safety_pass": proper_pass,
            "crossing_sample_equal_center_auc": equal_center_auc,
            "blocked_crossing_sample_equal_center_auc": equal_center_blocked_auc,
            "crossing_sample_auc_defined_center_count": len(defined_center_aucs),
            "crossing_sample_auc_at_least_0_55": sample_signal_pass,
            "crossing_sample_auc_beats_blocked_by_0_02": blocked_pass,
            "crossing_sample_brier_beats_prevalence": calibration_pass,
            "action_bacc_spearman_positive": action_rank_pass,
            "selected_helpful_action_count": selected_helpful,
            "selected_harmful_action_count": selected_harmful,
            "selected_action_count": len(selected_rows),
            "selected_helpful_actions_not_fewer_than_harmful": route_pass,
            "primary_helpful_threshold_switch_count": primary_helpful_switches,
            "primary_harmful_threshold_switch_count": primary_harmful_switches,
            "terminal_information_may_change_same_surface_routes": False,
            "rows_are_not_independent_inference_units": True,
            "nominal_inference_claimed": False,
            "fresh_evidence": False,
        }
    )
    # Per-crossing helpfulness bits are deliberately not returned or persisted:
    # given the direction, they would reveal the terminal sample label.  Only
    # center-level AUC/Brier summaries leave this function.
    return tuple(action_rows), center_rows, gate


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
        float(
            np.sum(
                hard[positive].astype(np.int8) - p_hard[positive].astype(np.int8),
                dtype=np.int64,
            )
        )
        / center_n_positive
        + float(
            np.sum(
                (~hard[negative]).astype(np.int8)
                - (~p_hard[negative]).astype(np.int8),
                dtype=np.int64,
            )
        )
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
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    n_positive = int(np.sum(y == 1, dtype=np.int64))
    n_negative = int(np.sum(y == 0, dtype=np.int64))
    if not n_positive or not n_negative:
        return None
    ranks = _rank(score) + 1.0
    return float(
        (np.sum(ranks[y == 1], dtype=np.float64) - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _summarize_center(
    center: str,
    action_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    nonzero = tuple(row for row in action_rows if not row["structural_zero"])
    actual = [float(row["actual_bacc_contribution_delta"]) for row in nonzero]
    score = [float(row["target_influence_score"]) for row in nonzero]
    blocked_score = [float(row["blocked_target_influence_score"]) for row in nonzero]
    helpful = [int(row["actual_helpful"]) for row in sample_rows]
    probabilities = [float(row["predicted_help_probability"]) for row in sample_rows]
    blocked_probabilities = [float(row["blocked_help_probability"]) for row in sample_rows]
    prevalence = float(np.mean(helpful, dtype=np.float64)) if helpful else 0.0
    return {
        "target_center": center,
        "action_row_count": len(action_rows),
        "structural_zero_count": sum(bool(row["structural_zero"]) for row in action_rows),
        "selected_action_count": sum(bool(row["selected_by_primary"]) for row in action_rows),
        "crossing_sample_row_count": len(sample_rows),
        "crossing_sample_auc": _auc(helpful, probabilities),
        "blocked_crossing_sample_auc": _auc(helpful, blocked_probabilities),
        "crossing_sample_brier": float(
            np.mean(
                (np.asarray(probabilities) - np.asarray(helpful)) ** 2,
                dtype=np.float64,
            )
        ) if helpful else 0.0,
        "crossing_sample_prevalence_brier": prevalence * (1.0 - prevalence),
        "action_bacc_spearman": _spearman(score, actual),
        "blocked_action_bacc_spearman": _spearman(blocked_score, actual),
        "action_sign_agreement": float(
            np.mean(
                np.sign(np.asarray(score)) == np.sign(np.asarray(actual)),
                dtype=np.float64,
            )
        ) if actual else 1.0,
        "count_by_alternative": dict(
            sorted(Counter(str(row["alternative"]) for row in action_rows).items())
        ),
        "count_by_direction": dict(
            sorted(Counter(str(row["direction"]) for row in action_rows).items())
        ),
        "confidence_bound_claimed": False,
    }


# Compatibility alias for callers that still use the old descriptive name.
utility_information_diagnostics = sample_influence_information_diagnostics


__all__ = (
    "sample_influence_information_diagnostics",
    "utility_information_diagnostics",
)
