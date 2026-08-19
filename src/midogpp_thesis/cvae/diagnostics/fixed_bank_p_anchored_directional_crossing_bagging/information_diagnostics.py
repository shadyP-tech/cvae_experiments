"""Terminal-only tests of crossing signal, donor stability, and selectivity."""

from __future__ import annotations

from collections import Counter
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    HARD_THRESHOLD,
    LOG_LOSS_CLIP_EPSILON,
    MODEL_BASED_METHOD_ID,
    PERMUTATION_METHOD_ID,
    PORTFOLIO_METHOD_ID,
)
from .engine import PreterminalResult


def crossing_information_diagnostics(
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
        for row in preterminal.crossing_predictions_by_policy[MODEL_BASED_METHOD_ID]
    }
    blocked = {
        row.descriptor_hash: row
        for row in preterminal.crossing_predictions_by_policy[PERMUTATION_METHOD_ID]
    }
    endpoint_by_case = {
        center: {row.case_id: row for row in preterminal.predictions_by_center[center]}
        for center in CENTERS
    }
    rows: list[Mapping[str, object]] = []
    for center in CENTERS:
        for descriptor in preterminal.crossing_descriptors_by_center[center]:
            if descriptor.descriptor_hash not in primary or descriptor.descriptor_hash not in blocked:
                raise ProtocolError("PDCB crossing information surface is incomplete.")
            endpoint = endpoint_by_case[center][descriptor.case_id]
            index = endpoint.sample_ids.index(descriptor.sample_id)
            y = labels[(center, descriptor.case_id, descriptor.sample_id)]
            p_value = float(endpoint.probabilities[PORTFOLIO_METHOD_ID][index])
            alternative_value = float(endpoint.probabilities[descriptor.alternative][index])
            p_hard = p_value >= HARD_THRESHOLD
            alternative_hard = alternative_value >= HARD_THRESHOLD
            if p_hard == alternative_hard:
                raise ProtocolError("PDCB terminal information row is not a crossing.")
            truth = int(alternative_hard == bool(y))
            prediction = primary[descriptor.descriptor_hash]
            control = blocked[descriptor.descriptor_hash]
            deletion_values = np.asarray(
                [value for _center, value in prediction.deletion_probabilities],
                dtype=np.float64,
            )
            rows.append(
                MappingProxyType(
                    {
                        "target_center": center,
                        "case_id": descriptor.case_id,
                        "sample_id": descriptor.sample_id,
                        "alternative": descriptor.alternative,
                        "direction": descriptor.direction,
                        "descriptor_hash": descriptor.descriptor_hash,
                        "helpful": truth,
                        "full_probability": prediction.full_probability,
                        "robust_probability": prediction.robust_probability,
                        "blocked_robust_probability": control.robust_probability,
                        "positive_deletion_fraction": prediction.positive_deletion_fraction,
                        "full_delete_sign_agreement": prediction.full_delete_sign_agreement,
                        "raw_weight": prediction.raw_weight,
                        "delete_probability_mad": prediction.deletion_mad,
                        "delete_probability_iqr": prediction.deletion_iqr,
                        "delete_probability_sd": float(np.std(deletion_values, ddof=0)),
                        "terminal_information_only": True,
                        "may_change_same_surface_route": False,
                    }
                )
            )
    if not rows:
        center_rows = tuple(_score_center(center, ()) for center in CENTERS)
        criteria = {
            "at_least_seven_auc_defined_centers": False,
            "equal_center_auc_at_least_0_55": False,
            "auc_exceeds_blocked_control_by_0_02": False,
            "pooled_spearman_above_0_05": False,
            "crossing_brier_below_prevalence_baseline": False,
            "primary_mean_center_bacc_delta_positive": (
                primary_mean_center_bacc_delta_vs_p > 0.0
            ),
            "primary_equal_center_brier_nonworse_than_P": (
                primary_mean_center_brier_delta_vs_p <= 0.0
            ),
            "primary_equal_center_log_loss_nonworse_than_P": (
                primary_mean_center_log_loss_delta_vs_p <= 0.0
            ),
            "primary_helpful_switches_not_fewer_than_harmful": (
                primary_helpful_switches >= primary_harmful_switches
            ),
        }
        summary = MappingProxyType(
            {
                "schema_version": "fixed_bank_pdcb_information_gate_v1",
                "status": "FAIL",
                "crossing_row_count": 0,
                "helpful_prevalence": None,
                "valid_center_auc_count": 0,
                "equal_center_auc": None,
                "equal_center_blocked_auc": None,
                "auc_margin_vs_blocked": None,
                "pooled_spearman": None,
                "crossing_brier": None,
                "blocked_crossing_brier": None,
                "prevalence_baseline_brier": None,
                "crossing_log_loss": None,
                "mean_delete_probability_sd": None,
                "mean_delete_probability_mad": None,
                "mean_delete_probability_iqr": None,
                "mean_full_delete_sign_agreement": None,
                "positive_weight_fraction": 0.0,
                "primary_mean_center_bacc_delta_vs_P": (
                    primary_mean_center_bacc_delta_vs_p
                ),
                "primary_mean_center_brier_delta_vs_P": (
                    primary_mean_center_brier_delta_vs_p
                ),
                "primary_mean_center_log_loss_delta_vs_P": (
                    primary_mean_center_log_loss_delta_vs_p
                ),
                "primary_proper_loss_safety_pass": (
                    primary_mean_center_brier_delta_vs_p <= 0.0
                    and primary_mean_center_log_loss_delta_vs_p <= 0.0
                ),
                "crossing_count_by_center": {},
                "crossing_count_by_case": {},
                "crossing_count_by_alternative": {},
                "crossing_count_by_direction": {},
                "criteria": criteria,
                "diagnosed_bottleneck": "NO_ACTIONABLE_CROSSINGS_P_FALLBACK",
                "structural_no_crossing_route_is_exact_P": True,
                "terminal_information_only": True,
                "same_surface_policy_change_authorized": False,
                "fresh_routing_claim_authorized": False,
            }
        )
        return (), center_rows, summary
    center_rows = tuple(_score_center(center, [row for row in rows if row["target_center"] == center]) for center in CENTERS)
    valid_auc = [float(row["auc"]) for row in center_rows if row["auc"] is not None]
    valid_blocked_auc = [
        float(row["blocked_auc"]) for row in center_rows if row["blocked_auc"] is not None
    ]
    truth = np.asarray([int(row["helpful"]) for row in rows], dtype=np.float64)
    probability = np.asarray([float(row["robust_probability"]) for row in rows])
    blocked_probability = np.asarray([float(row["blocked_robust_probability"]) for row in rows])
    prevalence = float(np.mean(truth, dtype=np.float64))
    baseline_brier = prevalence * (1.0 - prevalence)
    brier = float(np.mean((probability - truth) ** 2, dtype=np.float64))
    blocked_brier = float(np.mean((blocked_probability - truth) ** 2, dtype=np.float64))
    clipped = np.clip(probability, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON)
    log_loss = float(
        np.mean(-(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped)))
    )
    equal_center_auc = float(np.mean(valid_auc)) if valid_auc else None
    equal_center_blocked_auc = (
        float(np.mean(valid_blocked_auc)) if valid_blocked_auc else None
    )
    spearman = _spearman(probability, truth)
    auc_margin = (
        None
        if equal_center_auc is None or equal_center_blocked_auc is None
        else equal_center_auc - equal_center_blocked_auc
    )
    criteria = {
        "at_least_seven_auc_defined_centers": len(valid_auc) >= 7,
        "equal_center_auc_at_least_0_55": equal_center_auc is not None and equal_center_auc >= 0.55,
        "auc_exceeds_blocked_control_by_0_02": auc_margin is not None and auc_margin >= 0.02,
        "pooled_spearman_above_0_05": spearman is not None and spearman > 0.05,
        "crossing_brier_below_prevalence_baseline": brier < baseline_brier,
        "primary_mean_center_bacc_delta_positive": (
            primary_mean_center_bacc_delta_vs_p > 0.0
        ),
        "primary_equal_center_brier_nonworse_than_P": (
            primary_mean_center_brier_delta_vs_p <= 0.0
        ),
        "primary_equal_center_log_loss_nonworse_than_P": (
            primary_mean_center_log_loss_delta_vs_p <= 0.0
        ),
        "primary_helpful_switches_not_fewer_than_harmful": primary_helpful_switches >= primary_harmful_switches,
    }
    gate_pass = all(criteria.values())
    positive_weight_fraction = float(
        np.mean([float(row["raw_weight"]) > 0.0 for row in rows], dtype=np.float64)
    )
    mean_delete_sd = float(
        np.mean([float(row["delete_probability_sd"]) for row in rows], dtype=np.float64)
    )
    count_by_center = Counter(str(row["target_center"]) for row in rows)
    count_by_alternative = Counter(str(row["alternative"]) for row in rows)
    count_by_direction = Counter(str(row["direction"]) for row in rows)
    count_by_case = Counter(
        f"{row['target_center']}::{row['case_id']}" for row in rows
    )
    bottleneck = _classify_bottleneck(
        equal_center_auc=equal_center_auc,
        auc_margin=auc_margin,
        brier=brier,
        baseline_brier=baseline_brier,
        mean_delete_sd=mean_delete_sd,
        positive_weight_fraction=positive_weight_fraction,
        utility_delta=primary_mean_center_bacc_delta_vs_p,
        brier_delta=primary_mean_center_brier_delta_vs_p,
        log_loss_delta=primary_mean_center_log_loss_delta_vs_p,
    )
    summary = MappingProxyType(
        {
            "schema_version": "fixed_bank_pdcb_information_gate_v1",
            "status": "PASS" if gate_pass else "FAIL",
            "crossing_row_count": len(rows),
            "helpful_prevalence": prevalence,
            "valid_center_auc_count": len(valid_auc),
            "equal_center_auc": equal_center_auc,
            "equal_center_blocked_auc": equal_center_blocked_auc,
            "auc_margin_vs_blocked": auc_margin,
            "pooled_spearman": spearman,
            "crossing_brier": brier,
            "blocked_crossing_brier": blocked_brier,
            "prevalence_baseline_brier": baseline_brier,
            "crossing_log_loss": log_loss,
            "mean_delete_probability_sd": mean_delete_sd,
            "mean_delete_probability_mad": float(
                np.mean([float(row["delete_probability_mad"]) for row in rows])
            ),
            "mean_delete_probability_iqr": float(
                np.mean([float(row["delete_probability_iqr"]) for row in rows])
            ),
            "mean_full_delete_sign_agreement": float(
                np.mean([float(row["full_delete_sign_agreement"]) for row in rows])
            ),
            "positive_weight_fraction": positive_weight_fraction,
            "primary_mean_center_bacc_delta_vs_P": (
                primary_mean_center_bacc_delta_vs_p
            ),
            "primary_mean_center_brier_delta_vs_P": (
                primary_mean_center_brier_delta_vs_p
            ),
            "primary_mean_center_log_loss_delta_vs_P": (
                primary_mean_center_log_loss_delta_vs_p
            ),
            "primary_proper_loss_safety_pass": (
                primary_mean_center_brier_delta_vs_p <= 0.0
                and primary_mean_center_log_loss_delta_vs_p <= 0.0
            ),
            "crossing_count_by_center": dict(sorted(count_by_center.items())),
            "crossing_count_by_case": dict(sorted(count_by_case.items())),
            "crossing_count_by_alternative": dict(sorted(count_by_alternative.items())),
            "crossing_count_by_direction": dict(sorted(count_by_direction.items())),
            "criteria": criteria,
            "diagnosed_bottleneck": bottleneck,
            "terminal_information_only": True,
            "same_surface_policy_change_authorized": False,
            "fresh_routing_claim_authorized": False,
        }
    )
    return tuple(rows), center_rows, summary


def _score_center(center: str, rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not rows:
        return MappingProxyType(
            {"target_center": center, "row_count": 0, "auc": None, "blocked_auc": None}
        )
    truth = np.asarray([int(row["helpful"]) for row in rows], dtype=np.int8)
    probability = np.asarray([float(row["robust_probability"]) for row in rows])
    blocked = np.asarray([float(row["blocked_robust_probability"]) for row in rows])
    return MappingProxyType(
        {
            "target_center": center,
            "row_count": len(rows),
            "positive_count": int(np.sum(truth == 1, dtype=np.int64)),
            "negative_count": int(np.sum(truth == 0, dtype=np.int64)),
            "auc": _auc(probability, truth),
            "blocked_auc": _auc(blocked, truth),
            "brier": float(np.mean((probability - truth) ** 2, dtype=np.float64)),
            "mean_delete_probability_sd": float(
                np.mean([float(row["delete_probability_sd"]) for row in rows])
            ),
            "positive_weight_fraction": float(
                np.mean([float(row["raw_weight"]) > 0.0 for row in rows])
            ),
            "formal_claim_authorized": False,
        }
    )


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = int(np.sum(labels == 1, dtype=np.int64))
    negative = int(np.sum(labels == 0, dtype=np.int64))
    if not positive or not negative:
        return None
    ranks = _average_ranks(scores)
    return float((np.sum(ranks[labels == 1]) - positive * (positive + 1) / 2) / (positive * negative))


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2:
        return None
    x, y = _average_ranks(first), _average_ranks(second)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    return None if denominator <= 0.0 else float(np.sum(x * y) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _classify_bottleneck(
    *,
    equal_center_auc: float | None,
    auc_margin: float | None,
    brier: float,
    baseline_brier: float,
    mean_delete_sd: float,
    positive_weight_fraction: float,
    utility_delta: float,
    brier_delta: float,
    log_loss_delta: float,
) -> str:
    if equal_center_auc is None or equal_center_auc < 0.55 or brier >= baseline_brier:
        return "CROSSING_PROXY_SIGNAL_BOTTLENECK"
    if auc_margin is None or auc_margin < 0.02:
        return "FEATURE_SIGNAL_NOT_DISTINGUISHABLE_FROM_BLOCKED_CONTROL"
    if mean_delete_sd > 0.10:
        return "DONOR_TRANSFER_INSTABILITY_BOTTLENECK"
    if positive_weight_fraction < 0.05:
        return "P_ANCHOR_SELECTIVITY_BOTTLENECK"
    if utility_delta <= 0.0:
        return "CROSSING_PROBABILITY_TO_UTILITY_ALIGNMENT_BOTTLENECK"
    if brier_delta > 0.0 or log_loss_delta > 0.0:
        return "COMPOSED_PROPER_LOSS_SAFETY_BOTTLENECK"
    return "NO_SINGLE_DOMINANT_BOTTLENECK_ON_CONSUMED_SURFACE"


__all__ = ("crossing_information_diagnostics",)
