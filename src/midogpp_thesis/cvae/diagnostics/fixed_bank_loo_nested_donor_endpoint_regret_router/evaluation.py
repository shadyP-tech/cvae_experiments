"""Terminal-only utility, proper-loss, oracle, and selection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CLAIM_ROLE,
    ENDPOINT_METHOD_IDS,
    ENDPOINT_ORDER,
    HARD_THRESHOLD,
    LOG_LOSS_CLIP_EPSILON,
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .contracts import BinaryLabel, EndpointCasePrediction, RouteDecision
from .engine import PreterminalResult
from .hashing import canonical_hash


_T_975_DF8 = 2.306004135204166


@dataclass(frozen=True)
class TerminalEvaluation:
    method_metrics: tuple[Mapping[str, object], ...]
    center_contrasts: tuple[Mapping[str, object], ...]
    case_oracle_rows: tuple[Mapping[str, object], ...]
    selection_control: Mapping[str, object]
    diagnostic_summary: Mapping[str, object]
    terminal_seal: Mapping[str, object]
    capability_report: Mapping[str, object]
    evaluation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = self.to_payload(include_hash=False)
        object.__setattr__(self, "evaluation_hash", canonical_hash(payload))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "fixed_bank_nested_regret_terminal_evaluation_v1",
            "method_metrics": [dict(row) for row in self.method_metrics],
            "center_contrasts": [dict(row) for row in self.center_contrasts],
            "case_oracle_rows": [dict(row) for row in self.case_oracle_rows],
            "selection_control": dict(self.selection_control),
            "diagnostic_summary": dict(self.diagnostic_summary),
            "terminal_seal": dict(self.terminal_seal),
            "capability_report": dict(self.capability_report),
        }
        return {**payload, "evaluation_hash": self.evaluation_hash} if include_hash else payload


def evaluate_terminal(preterminal: PreterminalResult) -> TerminalEvaluation:
    """Open terminal labels only after the global pre-evaluation barrier."""

    terminal_labels = preterminal.label_firewall.open_terminal_labels()
    label_map = {row.key: row.value for row in terminal_labels}
    expected = {
        (center, case_id, sample_id)
        for center in CENTERS
        for sample_id, case_id in zip(
            preterminal.surface.centers[center].sample_ids,
            preterminal.surface.centers[center].case_ids,
            strict=True,
        )
    }
    if set(label_map) != expected or len(label_map) != len(terminal_labels):
        raise ProtocolError("Terminal label capability does not match the sealed surface.")

    prediction_maps = {
        center: {
            row.case_id: row for row in preterminal.predictions_by_center[center]
        }
        for center in CENTERS
    }
    method_case_selections: dict[str, dict[tuple[str, str], str]] = {
        endpoint: {
            (center, case): endpoint
            for center in CENTERS
            for case in prediction_maps[center]
        }
        for endpoint in ENDPOINT_METHOD_IDS
    }
    decision_lookup: dict[str, dict[tuple[str, str], RouteDecision]] = {}
    for policy_id, decisions in preterminal.decisions_by_policy.items():
        lookup = {(row.target_center, row.case_id): row for row in decisions}
        decision_lookup[policy_id] = lookup
        method_case_selections[policy_id] = {
            key: row.selected_method for key, row in lookup.items()
        }

    denominators = _center_denominators(label_map)
    method_rows: list[Mapping[str, object]] = []
    center_rows: list[Mapping[str, object]] = []
    oracle_rows: list[Mapping[str, object]] = []
    metrics_by_method: dict[str, Mapping[str, object]] = {}
    p_center_bacc: dict[str, float] = {}
    p_probabilities: dict[tuple[str, str], tuple[float, ...]] = {}
    for center in CENTERS:
        for case, prediction in prediction_maps[center].items():
            p_probabilities[(center, case)] = prediction.probabilities[
                PORTFOLIO_METHOD_ID
            ]

    center_metrics_by_method: dict[str, dict[str, Mapping[str, object]]] = {}
    for method_id, selections in method_case_selections.items():
        per_center: dict[str, Mapping[str, object]] = {}
        for center in CENTERS:
            row = _score_center(
                center,
                selections,
                prediction_maps[center],
                label_map,
                denominators[center],
                p_probabilities,
            )
            per_center[center] = row
            if method_id == PORTFOLIO_METHOD_ID:
                p_center_bacc[center] = float(row["center_bacc"])
        center_metrics_by_method[method_id] = per_center

    for method_id, per_center in center_metrics_by_method.items():
        contrasts = np.asarray(
            [
                float(per_center[center]["center_bacc"])
                - p_center_bacc[center]
                for center in CENTERS
            ],
            dtype=np.float64,
        )
        for center in CENTERS:
            row = dict(per_center[center])
            row.update(
                {
                    "method_id": method_id,
                    "reference_method": PORTFOLIO_METHOD_ID,
                    "center_delta_vs_P": float(row["center_bacc"])
                    - p_center_bacc[center],
                    "formal_claim_authorized": False,
                }
            )
            center_rows.append(MappingProxyType(row))
        bacc_values = np.asarray(
            [float(per_center[center]["center_bacc"]) for center in CENTERS],
            dtype=np.float64,
        )
        brier_values = np.asarray(
            [float(per_center[center]["center_brier"]) for center in CENTERS],
            dtype=np.float64,
        )
        loss_values = np.asarray(
            [float(per_center[center]["center_log_loss"]) for center in CENTERS],
            dtype=np.float64,
        )
        pooled = _pooled_metrics(per_center)
        mean_delta = float(np.mean(contrasts, dtype=np.float64))
        delta_se = float(np.std(contrasts, ddof=1) / math.sqrt(len(CENTERS)))
        decisions = decision_lookup.get(method_id, {})
        oracle = _oracle_diagnostics(
            method_id,
            method_case_selections[method_id],
            decisions,
            prediction_maps,
            label_map,
            denominators,
        )
        oracle_rows.extend(oracle["rows"])
        method_row = MappingProxyType(
            {
                "method_id": method_id,
                "equal_center_bacc": float(np.mean(bacc_values, dtype=np.float64)),
                "sample_pooled_bacc": pooled["sample_pooled_bacc"],
                "global_brier": pooled["global_brier"],
                "equal_center_brier": float(np.mean(brier_values, dtype=np.float64)),
                "global_log_loss": pooled["global_log_loss"],
                "equal_center_log_loss": float(np.mean(loss_values, dtype=np.float64)),
                "mean_center_delta_vs_P": mean_delta,
                "minimum_center_delta_vs_P": float(np.min(contrasts)),
                "maximum_center_delta_vs_P": float(np.max(contrasts)),
                "positive_center_count": int(np.sum(contrasts > 1.0e-12)),
                "negative_center_count": int(np.sum(contrasts < -1.0e-12)),
                "zero_center_count": int(np.sum(np.abs(contrasts) <= 1.0e-12)),
                "descriptive_t8_lower": mean_delta - _T_975_DF8 * delta_se,
                "descriptive_t8_upper": mean_delta + _T_975_DF8 * delta_se,
                "descriptive_interval_has_no_nominal_coverage_claim": True,
                "route_count": int(
                    sum(
                        selected != PORTFOLIO_METHOD_ID
                        for selected in method_case_selections[method_id].values()
                    )
                ),
                "route_coverage": float(
                    np.mean(
                        [
                            selected != PORTFOLIO_METHOD_ID
                            for selected in method_case_selections[method_id].values()
                        ],
                        dtype=np.float64,
                    )
                ),
                "helpful_threshold_switch_count": int(
                    sum(int(per_center[c]["helpful_threshold_switch_count"]) for c in CENTERS)
                ),
                "harmful_threshold_switch_count": int(
                    sum(int(per_center[c]["harmful_threshold_switch_count"]) for c in CENTERS)
                ),
                "endpoint_oracle_top1_agreement_case_weighted": oracle[
                    "top1_agreement_case_weighted"
                ],
                "endpoint_oracle_top1_agreement_equal_center": oracle[
                    "top1_agreement_equal_center"
                ],
                "mean_endpoint_oracle_rank_case_weighted": oracle[
                    "mean_rank_case_weighted"
                ],
                "mean_endpoint_oracle_rank_equal_center": oracle[
                    "mean_rank_equal_center"
                ],
                "mean_normalized_endpoint_oracle_gap_case_weighted": oracle[
                    "mean_normalized_gap_case_weighted"
                ],
                "mean_normalized_endpoint_oracle_gap_equal_center": oracle[
                    "mean_normalized_gap_equal_center"
                ],
                "predicted_regret_spearman_candidate_cases": oracle["regret_spearman"],
                "publication_status": PUBLICATION_STATUS,
                "formal_claim_authorized": False,
            }
        )
        method_rows.append(method_row)
        metrics_by_method[method_id] = method_row

    selection_control = _selection_aware_center_sign_flip(
        center_metrics_by_method,
        tuple(policy.policy_id for policy in preterminal.policy_menu),
    )
    primary = metrics_by_method[MODEL_BASED_METHOD_ID]
    summary = MappingProxyType(
        {
            "primary_method_id": MODEL_BASED_METHOD_ID,
            "primary_equal_center_bacc": primary["equal_center_bacc"],
            "primary_mean_center_delta_vs_P": primary["mean_center_delta_vs_P"],
            "primary_route_count": primary["route_count"],
            "primary_harmful_threshold_switch_count": primary[
                "harmful_threshold_switch_count"
            ],
            "ltt_authorized_target_center_count": sum(
                row.selected_policy_id is not None
                for row in preterminal.ltt_authorizations
            ),
            "claim_role": CLAIM_ROLE,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "promotion_eligible": False,
            "may_feed_another_experiment": False,
        }
    )
    terminal_label_hash = canonical_hash(
        [[*key, label_map[key]] for key in sorted(label_map)]
    )
    terminal_payload = {
        "schema_version": "fixed_bank_nested_regret_terminal_seal_v1",
        "aggregate_preterminal_seal_hash": preterminal.aggregate_seal[
            "aggregate_seal_hash"
        ],
        "terminal_label_identity_and_value_hash": terminal_label_hash,
        "method_metrics_hash": canonical_hash([dict(row) for row in method_rows]),
        "center_contrasts_hash": canonical_hash([dict(row) for row in center_rows]),
        "case_oracle_rows_hash": canonical_hash([dict(row) for row in oracle_rows]),
        "selection_control_hash": canonical_hash(selection_control),
        "raw_labels_persisted": False,
        "fresh_evidence": False,
    }
    terminal_seal = MappingProxyType(
        {**terminal_payload, "terminal_seal_hash": canonical_hash(terminal_payload)}
    )
    capability = MappingProxyType(preterminal.label_firewall.report_payload())
    return TerminalEvaluation(
        tuple(method_rows),
        tuple(center_rows),
        tuple(oracle_rows),
        MappingProxyType(selection_control),
        summary,
        terminal_seal,
        capability,
    )


def _center_denominators(
    labels: Mapping[tuple[str, str, str], int]
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for center in CENTERS:
        values = [value for (observed, _case, _sample), value in labels.items() if observed == center]
        positive = values.count(1)
        negative = values.count(0)
        if not positive or not negative:
            raise ProtocolError("Terminal center lacks both classes.")
        result[center] = (positive, negative)
    return result


def _score_center(
    center: str,
    selections: Mapping[tuple[str, str], str],
    predictions: Mapping[str, EndpointCasePrediction],
    labels: Mapping[tuple[str, str, str], int],
    denominators: tuple[int, int],
    p_probabilities: Mapping[tuple[str, str], tuple[float, ...]],
) -> Mapping[str, object]:
    y_values: list[int] = []
    probabilities: list[float] = []
    reference: list[float] = []
    for case, prediction in predictions.items():
        selected = selections[(center, case)]
        y_values.extend(labels[(center, case, sample)] for sample in prediction.sample_ids)
        probabilities.extend(prediction.probabilities[selected])
        reference.extend(p_probabilities[(center, case)])
    y = np.asarray(y_values, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    p = np.asarray(reference, dtype=np.float64)
    hard = probability >= HARD_THRESHOLD
    p_hard = p >= HARD_THRESHOLD
    n_positive, n_negative = denominators
    tp = int(np.sum((y == 1) & hard, dtype=np.int64))
    tn = int(np.sum((y == 0) & (~hard), dtype=np.int64))
    crossing = hard != p_hard
    helpful = crossing & (hard == y)
    harmful = crossing & (p_hard == y)
    clipped = np.clip(
        probability, LOG_LOSS_CLIP_EPSILON, 1.0 - LOG_LOSS_CLIP_EPSILON
    )
    return {
        "target_center": center,
        "sample_count": len(y),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": int(np.sum((y == 0) & hard, dtype=np.int64)),
        "false_negative": int(np.sum((y == 1) & (~hard), dtype=np.int64)),
        "center_bacc": 0.5 * (tp / n_positive + tn / n_negative),
        "center_brier": float(np.mean((probability - y) ** 2, dtype=np.float64)),
        "center_log_loss": float(
            np.mean(
                -(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped)),
                dtype=np.float64,
            )
        ),
        "threshold_switch_count": int(np.sum(crossing, dtype=np.int64)),
        "helpful_threshold_switch_count": int(np.sum(helpful, dtype=np.int64)),
        "harmful_threshold_switch_count": int(np.sum(harmful, dtype=np.int64)),
        "probability_sum": float(np.sum(probability, dtype=np.float64)),
        "squared_error_sum": float(np.sum((probability - y) ** 2, dtype=np.float64)),
        "log_loss_sum": float(
            np.sum(
                -(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped)),
                dtype=np.float64,
            )
        ),
    }


def _pooled_metrics(
    per_center: Mapping[str, Mapping[str, object]]
) -> dict[str, float]:
    tp = sum(int(per_center[c]["true_positive"]) for c in CENTERS)
    tn = sum(int(per_center[c]["true_negative"]) for c in CENTERS)
    positive = sum(int(per_center[c]["n_positive"]) for c in CENTERS)
    negative = sum(int(per_center[c]["n_negative"]) for c in CENTERS)
    count = sum(int(per_center[c]["sample_count"]) for c in CENTERS)
    return {
        "sample_pooled_bacc": 0.5 * (tp / positive + tn / negative),
        "global_brier": sum(float(per_center[c]["squared_error_sum"]) for c in CENTERS) / count,
        "global_log_loss": sum(float(per_center[c]["log_loss_sum"]) for c in CENTERS) / count,
    }


def _oracle_diagnostics(
    method_id: str,
    selections: Mapping[tuple[str, str], str],
    decisions: Mapping[tuple[str, str], RouteDecision],
    predictions_by_center: Mapping[str, Mapping[str, EndpointCasePrediction]],
    labels: Mapping[tuple[str, str, str], int],
    denominators: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    rows: list[Mapping[str, object]] = []
    predicted: list[float] = []
    realized: list[float] = []
    for center in CENTERS:
        n_positive, n_negative = denominators[center]
        case_count = len(predictions_by_center[center])
        for case, prediction in predictions_by_center[center].items():
            y = np.asarray(
                [labels[(center, case, sample)] for sample in prediction.sample_ids],
                dtype=np.int8,
            )
            contributions: dict[str, float] = {}
            for endpoint in ENDPOINT_METHOD_IDS:
                hard = np.asarray(prediction.probabilities[endpoint]) >= HARD_THRESHOLD
                contributions[endpoint] = 0.5 * (
                    np.sum((y == 1) & hard, dtype=np.int64) / n_positive
                    + np.sum((y == 0) & (~hard), dtype=np.int64) / n_negative
                )
            oracle = min(
                ENDPOINT_METHOD_IDS,
                key=lambda endpoint: (
                    -contributions[endpoint],
                    0 if endpoint == PORTFOLIO_METHOD_ID else 1,
                    ENDPOINT_ORDER[endpoint],
                ),
            )
            selected = selections[(center, case)]
            selected_value = contributions[selected]
            p_value = contributions[PORTFOLIO_METHOD_ID]
            oracle_value = contributions[oracle]
            headroom = oracle_value - p_value
            normalized_gap = (
                max(0.0, oracle_value - selected_value) / headroom
                if headroom > 1.0e-12
                else 0.0
            )
            rank = 1 + sum(
                value > selected_value + 1.0e-12
                for value in contributions.values()
            )
            decision = decisions.get((center, case))
            if decision is not None and decision.alternative != PORTFOLIO_METHOD_ID:
                predicted.append(decision.predicted_bacc_regret)
                realized.append(
                    case_count
                    * (
                        contributions[decision.alternative]
                        - contributions[PORTFOLIO_METHOD_ID]
                    )
                )
            rows.append(
                MappingProxyType(
                    {
                        "method_id": method_id,
                        "target_center": center,
                        "case_id": case,
                        "selected_endpoint": selected,
                        "oracle_endpoint": oracle,
                        "top1_agreement": selected == oracle,
                        "selected_endpoint_rank": rank,
                        "oracle_headroom_additive_bacc": headroom,
                        "normalized_oracle_gap": normalized_gap,
                        "formal_claim_authorized": False,
                    }
                )
            )
    center_rows = {
        center: tuple(row for row in rows if row["target_center"] == center)
        for center in CENTERS
    }
    return {
        "rows": rows,
        "top1_agreement_case_weighted": float(
            np.mean([row["top1_agreement"] for row in rows])
        ),
        "top1_agreement_equal_center": float(
            np.mean(
                [
                    np.mean([row["top1_agreement"] for row in center_rows[center]])
                    for center in CENTERS
                ]
            )
        ),
        "mean_rank_case_weighted": float(
            np.mean([row["selected_endpoint_rank"] for row in rows])
        ),
        "mean_rank_equal_center": float(
            np.mean(
                [
                    np.mean(
                        [row["selected_endpoint_rank"] for row in center_rows[center]]
                    )
                    for center in CENTERS
                ]
            )
        ),
        "mean_normalized_gap_case_weighted": float(
            np.mean([row["normalized_oracle_gap"] for row in rows])
        ),
        "mean_normalized_gap_equal_center": float(
            np.mean(
                [
                    np.mean(
                        [row["normalized_oracle_gap"] for row in center_rows[center]]
                    )
                    for center in CENTERS
                ]
            )
        ),
        "regret_spearman": _spearman(predicted, realized),
    }


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 3 or len(first) != len(second):
        return None
    x = _average_ranks(np.asarray(first, dtype=np.float64))
    y = _average_ranks(np.asarray(second, dtype=np.float64))
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _selection_aware_center_sign_flip(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    policy_ids: Sequence[str],
) -> dict[str, object]:
    policies = tuple(policy_ids)
    contrasts = np.asarray(
        [
            [
                float(metrics[policy][center]["center_bacc"])
                - float(metrics[PORTFOLIO_METHOD_ID][center]["center_bacc"])
                for center in CENTERS
            ]
            for policy in policies
        ],
        dtype=np.float64,
    )
    observed_by_policy = np.mean(contrasts, axis=1, dtype=np.float64)
    observed_max = float(np.max(observed_by_policy))
    maxima = np.asarray(
        [
            np.max(
                np.mean(
                    contrasts * np.asarray(signs, dtype=np.float64)[None, :],
                    axis=1,
                    dtype=np.float64,
                )
            )
            for signs in product((-1.0, 1.0), repeat=len(CENTERS))
        ],
        dtype=np.float64,
    )
    return {
        "schema_version": "fixed_bank_nested_regret_fixed_decision_sign_flip_v1",
        "fixed_decision_policy_menu": list(policies),
        "effective_units": "nine_centers",
        "replicate_count": len(maxima),
        "policy_identity_reselected_inside_every_replicate": True,
        "route_features_models_and_decisions_refit_inside_replicate": False,
        "observed_max_mean_center_delta": observed_max,
        "observed_best_policy_id": policies[int(np.argmax(observed_by_policy))],
        "exact_sign_flip_tail_probability": float(
            np.mean(maxima >= observed_max - 1.0e-15, dtype=np.float64)
        ),
        "case_independence_claimed": False,
        "full_selection_inference_claimed": False,
        "nominal_inference_claimed": False,
        "consumed_test_diagnostic_only": True,
    }


__all__ = ("TerminalEvaluation", "evaluate_terminal")
