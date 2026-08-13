"""Terminal-only exact-BACC scoring and directional oracle references."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    B_ACTION_ID,
    CENTERS,
    DESCRIPTIVE_METHOD_IDS,
    METHOD_IDS,
    PRE_TERMINAL_METHOD_IDS,
    PRIMARY_METHOD_ID,
    TIE_TOLERANCE,
    U_ACTION_ID,
    a1_action_id,
    candidate_sources,
)
from .hashing import canonical_hash
from .probability_surfaces import ProbabilityIndex, hard_prediction
from .products import MethodPrediction
from .reports import seal_payload


_T8_975 = 2.306004135204166


def evaluate_terminal(
    *,
    probability_surface: object,
    method_predictions: Sequence[MethodPrediction],
    descriptive_predictions: Sequence[MethodPrediction],
    decisions: Sequence[object],
    aggregate_plan_decision_seal_hash: str,
    terminal_labels: Sequence[object],
    **_unused: object,
) -> Mapping[str, object]:
    """Score only after the caller has crossed the aggregate label barrier."""

    labels = _label_map(terminal_labels)
    index = ProbabilityIndex(probability_surface)
    preterminal = tuple(method_predictions)
    descriptive = tuple(descriptive_predictions)
    expected_preterminal = len(labels) * len(PRE_TERMINAL_METHOD_IDS)
    if (
        len(preterminal) != expected_preterminal
        or len(descriptive) != len(labels) * len(DESCRIPTIVE_METHOD_IDS)
    ):
        raise ProtocolError("Case-directional terminal prediction topology drifted.")
    static = _oracle_predictions(index, labels, case_conditional=False)
    case = _oracle_predictions(index, labels, case_conditional=True)
    all_predictions = (*preterminal, *static, *case, *descriptive)
    method_order = (*METHOD_IDS, *DESCRIPTIVE_METHOD_IDS)
    if tuple(dict.fromkeys(row.method_id for row in all_predictions)) != method_order:
        raise ProtocolError("Case-directional terminal method order drifted.")
    case_rows = _case_confusions(all_predictions, labels, method_order)
    center_rows = _center_metrics(case_rows, method_order)
    method_rows = _method_metrics(center_rows, method_order)
    contrasts = _contrasts(center_rows)
    identification = _identification_rows(decisions, static, case)
    permutation = _feature_permutation_rows(method_rows, contrasts)
    bindings = {
        "aggregate_plan_decision_seal_hash": aggregate_plan_decision_seal_hash,
        "case_confusions_hash": canonical_hash(case_rows),
        "method_metrics_hash": canonical_hash(method_rows),
        "center_metrics_hash": canonical_hash(center_rows),
        "contrasts_hash": canonical_hash(contrasts),
        "router_identification_hash": canonical_hash(identification),
        "feature_permutation_summary_hash": canonical_hash(permutation),
    }
    seal = seal_payload(
        "fixed_bank_cdca_terminal_evaluation_seal_v1",
        bindings=bindings,
        method_ids=list(method_order),
        terminal_row_count=len(labels),
        exact_bacc_is_terminal_only=True,
        support_calibrated_proxy_is_not_predicted_held_case_bacc=True,
        raw_labels_persisted=False,
        consumed_test_diagnostic_only=True,
    )
    return {
        "case_confusions": case_rows,
        "method_metrics": method_rows,
        "center_metrics": center_rows,
        "contrasts": contrasts,
        "router_identification": identification,
        "feature_permutation_summary": permutation,
        "terminal_seal": seal,
        "descriptive_summary": {
            "primary_method_id": PRIMARY_METHOD_ID,
            "primary_bacc": next(
                row["equal_center_mean_bacc"]
                for row in method_rows
                if row["method_id"] == PRIMARY_METHOD_ID
            ),
            "all_results_descriptive_only": True,
        },
    }


def _label_map(rows: Sequence[object]) -> dict[tuple[str, str, str], int]:
    output: dict[tuple[str, str, str], int] = {}
    for row in rows:
        scope = str(getattr(row, "label_scope", ""))
        key = (
            str(getattr(row, "target_center")),
            str(getattr(row, "case_id")),
            str(getattr(row, "sample_id")),
        )
        value = int(getattr(row, "value"))
        if (
            not scope.startswith("terminal")
            or value not in (0, 1)
            or key in output
        ):
            raise ProtocolError("Case-directional terminal label scope drifted.")
        output[key] = value
    if len(output) != 9_928:
        raise ProtocolError("Case-directional terminal label coverage drifted.")
    return output


def _oracle_predictions(
    index: ProbabilityIndex,
    labels: Mapping[tuple[str, str, str], int],
    *,
    case_conditional: bool,
) -> tuple[MethodPrediction, ...]:
    method = "O_case_directional" if case_conditional else "O_directional_static"
    chosen: dict[tuple[str, str, str], str | None] = {}
    for center in CENTERS:
        case_ids = sorted({case for target, case, _ in labels if target == center})
        n_pos = sum(value == 1 for key, value in labels.items() if key[0] == center)
        n_neg = sum(value == 0 for key, value in labels.items() if key[0] == center)
        if not n_pos or not n_neg:
            raise ProtocolError("Case-directional terminal center is single-class.")
        scopes = case_ids if case_conditional else ["*"]
        for scope_case in scopes:
            cases = {scope_case} if case_conditional else set(case_ids)
            for direction in ("zero_to_one", "one_to_zero"):
                scores: list[tuple[str | None, float]] = [(None, 0.0)]
                for source in candidate_sources(center):
                    favorable = adverse = 0
                    for key, truth in labels.items():
                        target, case_id, sample_id = key
                        if target != center or case_id not in cases:
                            continue
                        base = index[(target, case_id, sample_id, B_ACTION_ID)]
                        action = index[(target, case_id, sample_id, a1_action_id(source))]
                        if direction == "zero_to_one":
                            flips = base.hard_prediction == 0 and action.hard_prediction == 1
                            favorable += int(flips and truth == 1)
                            adverse += int(flips and truth == 0)
                        else:
                            flips = base.hard_prediction == 1 and action.hard_prediction == 0
                            favorable += int(flips and truth == 0)
                            adverse += int(flips and truth == 1)
                    gain = (
                        0.5 * (favorable / n_pos - adverse / n_neg)
                        if direction == "zero_to_one"
                        else 0.5 * (favorable / n_neg - adverse / n_pos)
                    )
                    scores.append((source, gain))
                maximum = max(value for _, value in scores)
                source = next(
                    source
                    for source, value in scores
                    if maximum - value <= TIE_TOLERANCE
                )
                for case_id in cases:
                    chosen[(center, case_id, direction)] = source
    predictions: list[MethodPrediction] = []
    for target, case_id, sample_id in labels:
        base = index[(target, case_id, sample_id, B_ACTION_ID)]
        direction = "zero_to_one" if base.hard_prediction == 0 else "one_to_zero"
        source = chosen[(target, case_id, direction)]
        selected = base if source is None else index[
            (target, case_id, sample_id, a1_action_id(source))
        ]
        predictions.append(
            MethodPrediction(
                target,
                case_id,
                sample_id,
                method,
                selected.probability_mean,
                hard_prediction(selected.probability_mean),
                base.hard_prediction,
                source,
            )
        )
    return tuple(predictions)


def _case_confusions(
    predictions: Sequence[MethodPrediction],
    labels: Mapping[tuple[str, str, str], int],
    methods: Sequence[str],
) -> tuple[dict[str, object], ...]:
    rows = []
    for method in methods:
        method_rows = [row for row in predictions if row.method_id == method]
        for center in CENTERS:
            cases = sorted({case for target, case, _ in labels if target == center})
            for case_id in cases:
                selected = [
                    row
                    for row in method_rows
                    if row.target_center == center and row.case_id == case_id
                ]
                if not selected:
                    raise ProtocolError("Case-directional method misses a held case.")
                truth = [labels[(center, case_id, row.sample_id)] for row in selected]
                hard = [row.hard_prediction for row in selected]
                tp = sum(y == 1 and p == 1 for y, p in zip(truth, hard, strict=True))
                tn = sum(y == 0 and p == 0 for y, p in zip(truth, hard, strict=True))
                fp = sum(y == 0 and p == 1 for y, p in zip(truth, hard, strict=True))
                fn = sum(y == 1 and p == 0 for y, p in zip(truth, hard, strict=True))
                row = {
                    "target_center": center,
                    "case_id": case_id,
                    "method_id": method,
                    "n_positive": tp + fn,
                    "n_negative": tn + fp,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
                rows.append({**row, "row_hash": canonical_hash(row)})
    return tuple(rows)


def _center_metrics(
    cases: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> tuple[dict[str, object], ...]:
    rows = []
    for method in methods:
        for center in CENTERS:
            selected = [
                row
                for row in cases
                if row["method_id"] == method and row["target_center"] == center
            ]
            totals = {
                key: sum(int(row[key]) for row in selected)
                for key in ("n_positive", "n_negative", "tp", "tn", "fp", "fn")
            }
            if totals["n_positive"] <= 0 or totals["n_negative"] <= 0:
                raise ProtocolError("Case-directional pooled BACC denominator absent.")
            bacc = 0.5 * (
                totals["tp"] / totals["n_positive"]
                + totals["tn"] / totals["n_negative"]
            )
            row = {
                "target_center": center,
                "method_id": method,
                **totals,
                "exact_bacc": bacc,
            }
            rows.append({**row, "row_hash": canonical_hash(row)})
    return tuple(rows)


def _method_metrics(
    centers: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> tuple[dict[str, object], ...]:
    rows = []
    for method in methods:
        values = [
            float(row["exact_bacc"])
            for row in centers
            if row["method_id"] == method
        ]
        if len(values) != 9:
            raise ProtocolError("Case-directional method lacks nine center metrics.")
        row = {
            "method_id": method,
            "equal_center_mean_bacc": mean(values),
            "center_count": len(values),
            "descriptive_only": True,
        }
        rows.append({**row, "row_hash": canonical_hash(row)})
    return tuple(rows)


def _contrasts(
    centers: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    baselines = (
        "B",
        "U",
        "G_directional_matched",
        "CDCA_case_proxy_only",
        "CDCA_feature_block_permutation_descriptive",
    )
    lookup = {
        (str(row["method_id"]), str(row["target_center"])): float(row["exact_bacc"])
        for row in centers
    }
    rows = []
    for baseline in baselines:
        values = [
            lookup[(PRIMARY_METHOD_ID, center)] - lookup[(baseline, center)]
            for center in CENTERS
        ]
        estimate = mean(values)
        sd = stdev(values)
        se = sd / math.sqrt(9)
        row = {
            "method_id": PRIMARY_METHOD_ID,
            "baseline_id": baseline,
            "estimate": estimate,
            "ci_low": estimate - _T8_975 * se,
            "ci_high": estimate + _T8_975 * se,
            "center_estimates": values,
            "center_count": 9,
            "descriptive_only": True,
            "pass_gate_used": False,
        }
        rows.append({**row, "row_hash": canonical_hash(row)})
    return tuple(rows)


def _identification_rows(
    decisions: Sequence[object],
    static: Sequence[MethodPrediction],
    case: Sequence[MethodPrediction],
) -> tuple[dict[str, object], ...]:
    primary = [row for row in decisions if getattr(row, "method_id", None) == PRIMARY_METHOD_ID]
    rows = []
    for center in CENTERS:
        selected = [row for row in primary if getattr(row, "target_center") == center]
        source_slots = [
            source
            for row in selected
            for source in (
                getattr(row, "zero_to_one").selected_source,
                getattr(row, "one_to_zero").selected_source,
            )
        ]
        static_selected = [row for row in static if row.target_center == center]
        case_selected = [row for row in case if row.target_center == center]
        row = {
            "target_center": center,
            "route_count": len(selected),
            "direction_slot_count": len(source_slots),
            "off_selection_count": sum(source is None for source in source_slots),
            "off_selection_rate": (
                sum(source is None for source in source_slots) / len(source_slots)
                if source_slots
                else 0.0
            ),
            "static_oracle_nonbaseline_prediction_rate": mean(
                row.hard_prediction != row.baseline_hard_prediction
                for row in static_selected
            ),
            "case_oracle_nonbaseline_prediction_rate": mean(
                row.hard_prediction != row.baseline_hard_prediction
                for row in case_selected
            ),
            "descriptive_only": True,
        }
        rows.append({**row, "row_hash": canonical_hash(row)})
    return tuple(rows)


def _feature_permutation_rows(
    methods: Sequence[Mapping[str, object]],
    contrasts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    lookup = {
        str(row["method_id"]): float(row["equal_center_mean_bacc"])
        for row in methods
    }
    row = {
        "method_id": "CDCA_feature_block_permutation_descriptive",
        "primary_method_id": PRIMARY_METHOD_ID,
        "permuted_equal_center_mean_bacc": lookup[
            "CDCA_feature_block_permutation_descriptive"
        ],
        "primary_minus_permuted": lookup[PRIMARY_METHOD_ID]
        - lookup["CDCA_feature_block_permutation_descriptive"],
        "algorithm": "splitmix64_route_direction_candidate_block_permutation_v1",
        "seed": 20_260_814,
        "exchangeability_claimed": False,
        "p_value_computed": False,
        "descriptive_only": True,
    }
    return ({**row, "row_hash": canonical_hash(row)},)


__all__ = ("evaluate_terminal",)
