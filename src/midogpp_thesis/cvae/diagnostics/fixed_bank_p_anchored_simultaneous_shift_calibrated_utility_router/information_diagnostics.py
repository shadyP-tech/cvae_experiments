"""Terminal-only posterior-utility ranking and calibration diagnostics."""

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
    COMPOSED_POLICY_IDS,
    LOG_LOSS_CLIP_EPSILON,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
)
from .engine import PreterminalResult
from .utility_contracts import UtilityDescriptor


def posterior_utility_information_diagnostics(
    preterminal: PreterminalResult,
    labels: Mapping[tuple[str, str, str], int],
    *,
    method_metrics: Mapping[str, Mapping[str, object]],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    """Diagnose sealed posterior scores after terminal labels are opened."""

    primary_metric = method_metrics[MODEL_BASED_METHOD_ID]
    primary = {
        row.descriptor_hash: row
        for row in preterminal.posterior_utility_predictions_by_control[
            PRIMARY_FINGERPRINT_CONTROL_ID
        ]
    }
    blocked = {
        row.descriptor_hash: row
        for row in preterminal.posterior_utility_predictions_by_control[
            BLOCKED_FINGERPRINT_CONTROL_ID
        ]
    }
    certificates = {
        row.descriptor_hash: row
        for row in preterminal.utility_certificates_by_policy[
            MODEL_BASED_METHOD_ID
        ]
    }
    ensembles = {
        control: {
            (row.target_center, row.held_case_id): row
            for row in preterminal.route_posterior_ensembles_by_control[control]
        }
        for control in (
            PRIMARY_FINGERPRINT_CONTROL_ID,
            BLOCKED_FINGERPRINT_CONTROL_ID,
        )
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
            score = primary[descriptor.descriptor_hash]
            control = blocked[descriptor.descriptor_hash]
            certificate = certificates[descriptor.descriptor_hash]
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
                        "posterior_robust_bacc_lower": score.robust_bacc_lower,
                        "posterior_robust_brier_upper": score.robust_brier_upper,
                        "posterior_robust_log_loss_upper": score.robust_log_loss_upper,
                        "simultaneous_lower_bacc_delta": (
                            certificate.lower_bacc_delta
                        ),
                        "simultaneous_upper_brier_delta": (
                            certificate.upper_brier_delta
                        ),
                        "simultaneous_upper_log_loss_delta": (
                            certificate.upper_log_loss_delta
                        ),
                        "descriptor_shift": certificate.descriptor_shift,
                        "fold_instability": certificate.fold_instability,
                        "shift_inflation": certificate.shift_inflation,
                        "envelope_radius": certificate.envelope_radius,
                        "certificate_admissible": certificate.admissible,
                        "blocked_robust_bacc_lower": control.robust_bacc_lower,
                        "route_oof_auc": score.oof_auc,
                        "route_oof_brier_skill": score.oof_brier_skill,
                        "route_reliability_pass": score.reliability_pass,
                        "proper_safety_pass": score.proper_safe,
                        "descriptor_hash": descriptor.descriptor_hash,
                        "posterior_utility_hash": score.utility_hash,
                        "raw_label_persisted": False,
                    }
                )
            )
            eta_by_control: dict[str, dict[str, float]] = {}
            for control_id in ensembles:
                ensemble = ensembles[control_id][key]
                mean_eta = np.mean(
                    np.asarray(
                        ensemble.held_natural_probabilities_by_fold,
                        dtype=np.float64,
                    ),
                    axis=0,
                    dtype=np.float64,
                )
                eta_by_control[control_id] = dict(
                    zip(ensemble.held_sample_ids, mean_eta, strict=True)
                )
            for sample_id in descriptor.crossing_sample_ids:
                label = labels[(center, descriptor.case_id, sample_id)]
                helpful = label if descriptor.direction == "zero_to_one" else 1 - label
                primary_eta = eta_by_control[PRIMARY_FINGERPRINT_CONTROL_ID][sample_id]
                blocked_eta = eta_by_control[BLOCKED_FINGERPRINT_CONTROL_ID][sample_id]
                sample_rows.append(
                    MappingProxyType(
                        {
                            "target_center": center,
                            "actual_helpful": int(helpful),
                            "predicted_help_probability": float(
                                primary_eta
                                if descriptor.direction == "zero_to_one"
                                else 1.0 - primary_eta
                            ),
                            "blocked_help_probability": float(
                                blocked_eta
                                if descriptor.direction == "zero_to_one"
                                else 1.0 - blocked_eta
                            ),
                        }
                    )
                )
    expected_actions = 6 * sum(
        len(preterminal.predictions_by_center[center]) for center in CENTERS
    )
    if len(action_rows) != expected_actions:
        raise ProtocolError("PSSCUR terminal action-information surface is incomplete.")

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
    defined_auc = [
        float(row["crossing_sample_auc"])
        for row in center_rows
        if row["crossing_sample_auc"] is not None
    ]
    defined_blocked_auc = [
        float(row["blocked_crossing_sample_auc"])
        for row in center_rows
        if row["blocked_crossing_sample_auc"] is not None
    ]
    equal_auc = float(np.mean(defined_auc)) if defined_auc else 0.5
    equal_blocked_auc = (
        float(np.mean(defined_blocked_auc)) if defined_blocked_auc else 0.5
    )
    equal_envelope_coverage = float(
        np.mean(
            [
                float(row["simultaneous_joint_coverage_rate"])
                for row in center_rows
            ],
            dtype=np.float64,
        )
    )
    selected_rows = tuple(row for row in action_rows if row["selected_by_primary"])
    selected_helpful = sum(
        float(row["actual_bacc_contribution_delta"]) > 0.0 for row in selected_rows
    )
    selected_harmful = sum(
        float(row["actual_bacc_contribution_delta"]) < 0.0 for row in selected_rows
    )
    active_centers = len({str(row["target_center"]) for row in selected_rows})
    mean_bacc = float(primary_metric["mean_center_bacc_delta_vs_P"])
    minimum_bacc = float(primary_metric["minimum_center_bacc_delta_vs_P"])
    proper_pass = (
        float(primary_metric["mean_center_brier_delta_vs_P"]) <= 0.0
        and float(primary_metric["mean_center_log_loss_delta_vs_P"]) <= 0.0
        and float(primary_metric["maximum_center_brier_delta_vs_P"]) <= 0.0
        and float(primary_metric["maximum_center_log_loss_delta_vs_P"]) <= 0.0
    )
    control_best = max(
        float(method_metrics[method]["mean_center_bacc_delta_vs_P"])
        for method in COMPOSED_POLICY_IDS
        if method != MODEL_BASED_METHOD_ID
    )
    passes = {
        "primary_bacc_gain_positive": mean_bacc > 0.0,
        "primary_minimum_center_bacc_nonnegative": minimum_bacc >= -1.0e-15,
        "primary_proper_loss_safety_pass": proper_pass,
        "primary_beats_all_router_controls": mean_bacc > control_best + 1.0e-15,
        "crossing_sample_signal_pass": len(defined_auc) >= 7 and equal_auc >= 0.55,
        "blocked_control_separation_pass": equal_auc >= equal_blocked_auc + 0.02,
        "crossing_sample_calibration_pass": float(overall["crossing_sample_brier"])
        < float(overall["crossing_sample_prevalence_brier"]),
        "action_rank_pass": float(overall["action_bacc_spearman"]) > 0.0,
        "simultaneous_envelope_diagnostic_pass": (
            equal_envelope_coverage >= 0.75
        ),
        "route_precision_pass": selected_helpful > selected_harmful,
        "route_support_pass": active_centers >= 2,
    }
    status = "PASS" if all(passes.values()) else "FAIL"
    if not passes["crossing_sample_signal_pass"]:
        bottleneck = "TARGET_LOCAL_POSTERIOR_SIGNAL_WEAK"
    elif not passes["blocked_control_separation_pass"]:
        bottleneck = "PHYSICAL_FINGERPRINT_DOES_NOT_BEAT_BLOCKED_CONTROL"
    elif not passes["crossing_sample_calibration_pass"]:
        bottleneck = "TARGET_LOCAL_POSTERIOR_CALIBRATION_FAILURE"
    elif not passes["action_rank_pass"]:
        bottleneck = "POSTERIOR_UTILITY_DOES_NOT_RANK_REALIZED_ACTIONS"
    elif not passes["simultaneous_envelope_diagnostic_pass"]:
        bottleneck = "SIMULTANEOUS_ENVELOPE_UNDERCOVERS_DONOR_SHIFT"
    elif not passes["route_precision_pass"] or not passes["route_support_pass"]:
        bottleneck = "ENVELOPE_SELECTION_TOO_SPARSE_OR_IMPRECISE"
    elif not passes["primary_bacc_gain_positive"] or not passes[
        "primary_minimum_center_bacc_nonnegative"
    ]:
        bottleneck = "FROZEN_ROUTING_FAILS_TO_IMPROVE_P_STABLY"
    elif not passes["primary_proper_loss_safety_pass"]:
        bottleneck = "PROBABILITY_LOSS_SAFETY_FAILURE"
    elif not passes["primary_beats_all_router_controls"]:
        bottleneck = "ENVELOPE_OR_SHIFT_FILTER_ADDS_NO_VALUE"
    else:
        bottleneck = "NO_DIAGNOSTIC_BOTTLENECK"
    gate = MappingProxyType(
        {
            "schema_version": "fixed_bank_psscur_information_gate_v1",
            "status": status,
            "diagnosed_bottleneck": bottleneck,
            **passes,
            "primary_mean_center_bacc_delta_vs_P": mean_bacc,
            "best_control_mean_center_bacc_delta_vs_P": control_best,
            "crossing_sample_equal_center_auc": equal_auc,
            "blocked_crossing_sample_equal_center_auc": equal_blocked_auc,
            "crossing_sample_auc_defined_center_count": len(defined_auc),
            "simultaneous_envelope_equal_center_coverage": (
                equal_envelope_coverage
            ),
            "selected_helpful_action_count": selected_helpful,
            "selected_harmful_action_count": selected_harmful,
            "selected_action_count": len(selected_rows),
            "selected_active_center_count": active_centers,
            "authorized_primary_envelope_center_count": sum(
                preterminal.envelope_calibrations[
                    (center, PRIMARY_FINGERPRINT_CONTROL_ID)
                ].authorized
                for center in CENTERS
            ),
            "nonvacuous_primary_envelope_center_count": sum(
                preterminal.envelope_calibrations[
                    (center, PRIMARY_FINGERPRINT_CONTROL_ID)
                ].selected_action_count
                > 0
                for center in CENTERS
            ),
            "terminal_information_may_change_same_surface_routes": False,
            "rows_are_not_independent_inference_units": True,
            "nominal_inference_claimed": False,
            "fresh_evidence": False,
        }
    )
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
    p_hard, hard = portfolio >= 0.5, composed >= 0.5
    positive, negative = y == 1, y == 0
    bacc = 0.5 * (
        float(np.sum(hard[positive].astype(np.int8) - p_hard[positive].astype(np.int8)))
        / center_n_positive
        + float(
            np.sum(
                (~hard[negative]).astype(np.int8)
                - (~p_hard[negative]).astype(np.int8)
            )
        )
        / center_n_negative
    )
    total = center_n_positive + center_n_negative
    brier = float(np.sum((composed - y) ** 2 - (portfolio - y) ** 2) / total)
    p = np.clip(portfolio, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    q = np.clip(composed, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    log_loss = float(
        np.sum(
            -(y * np.log(q) + (1 - y) * np.log1p(-q))
            + (y * np.log(p) + (1 - y) * np.log1p(-p))
        )
        / total
    )
    return bacc, brier, log_loss


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
    left, right = _rank(np.asarray(x)), _rank(np.asarray(y))
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y, score = np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)
    positive, negative = int(np.sum(y == 1)), int(np.sum(y == 0))
    if not positive or not negative:
        return None
    ranks = _rank(score) + 1.0
    return float(
        (np.sum(ranks[y == 1]) - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _summarize_center(
    center: str,
    action_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    nonzero = tuple(row for row in action_rows if not row["structural_zero"])
    actual = [float(row["actual_bacc_contribution_delta"]) for row in nonzero]
    score = [float(row["simultaneous_lower_bacc_delta"]) for row in nonzero]
    blocked_score = [float(row["blocked_robust_bacc_lower"]) for row in nonzero]
    helpful = [int(row["actual_helpful"]) for row in sample_rows]
    probabilities = [float(row["predicted_help_probability"]) for row in sample_rows]
    blocked_probabilities = [float(row["blocked_help_probability"]) for row in sample_rows]
    prevalence = float(np.mean(helpful)) if helpful else 0.0
    joint_coverage = [
        float(row["actual_bacc_contribution_delta"])
        >= float(row["simultaneous_lower_bacc_delta"]) - 1.0e-12
        and float(row["actual_brier_contribution_delta"])
        <= float(row["simultaneous_upper_brier_delta"]) + 1.0e-12
        and float(row["actual_log_loss_contribution_delta"])
        <= float(row["simultaneous_upper_log_loss_delta"]) + 1.0e-12
        for row in nonzero
    ]
    return {
        "target_center": center,
        "action_row_count": len(action_rows),
        "structural_zero_count": sum(bool(row["structural_zero"]) for row in action_rows),
        "selected_action_count": sum(bool(row["selected_by_primary"]) for row in action_rows),
        "reliable_action_count": sum(bool(row["route_reliability_pass"]) for row in action_rows),
        "crossing_sample_row_count": len(sample_rows),
        "crossing_sample_auc": _auc(helpful, probabilities),
        "blocked_crossing_sample_auc": _auc(helpful, blocked_probabilities),
        "crossing_sample_brier": float(
            np.mean((np.asarray(probabilities) - np.asarray(helpful)) ** 2)
        )
        if helpful
        else 0.0,
        "crossing_sample_prevalence_brier": prevalence * (1.0 - prevalence),
        "action_bacc_spearman": _spearman(score, actual),
        "blocked_action_bacc_spearman": _spearman(blocked_score, actual),
        "action_sign_agreement": float(
            np.mean(np.sign(np.asarray(score)) == np.sign(np.asarray(actual)))
        )
        if actual
        else 1.0,
        "simultaneous_joint_coverage_rate": (
            float(np.mean(joint_coverage)) if joint_coverage else 1.0
        ),
        "simultaneous_joint_violation_count": sum(not value for value in joint_coverage),
        "certificate_admissible_count": sum(
            bool(row["certificate_admissible"]) for row in nonzero
        ),
        "mean_shift_inflation": (
            float(
                np.mean(
                    [float(row["shift_inflation"]) for row in nonzero],
                    dtype=np.float64,
                )
            )
            if nonzero
            else 1.0
        ),
        "count_by_alternative": dict(
            sorted(Counter(str(row["alternative"]) for row in action_rows).items())
        ),
        "count_by_direction": dict(
            sorted(Counter(str(row["direction"]) for row in action_rows).items())
        ),
        "finite_sample_coverage_claimed": False,
    }


__all__ = ("posterior_utility_information_diagnostics",)
