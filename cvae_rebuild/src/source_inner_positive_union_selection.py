from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import numpy as np

from downstream import PredictionBundle, predict_from_probabilities
from metrics import nanmean
from preservation_repair import _float
from component_union_tailrisk_anchored_mass_bagged import (
    HARM_GATED_PRIMARY_SELECTABLE_RULES,
    POSITIVE_UNION_RULE_ARITHMETIC,
    POSITIVE_UNION_RULE_BETA025,
    POSITIVE_UNION_RULE_BETA050,
    POSITIVE_UNION_RULE_BETA100,
    SourceInnerHarmGatedPositiveUnionConfig,
    SourceInnerPositiveUnionConfig,
    _binary_metrics_from_predictions,
    _positive_union_rule_beta,
)


def _safe_int(value: object, *, default: int) -> int:
    try:
        if value is None or str(value) == "":
            return int(default)
        return int(float(str(value)))
    except Exception:
        return int(default)


def _positive_union_metrics(
    rule: str,
    bundle: PredictionBundle,
    labels: Sequence[int],
    *,
    scope: str,
) -> dict[str, object]:
    preds = predict_from_probabilities(bundle.probabilities, classes=bundle.classes)
    metrics = _binary_metrics_from_predictions(labels, preds)
    return {
        "scope": str(scope),
        "rule": str(rule),
        "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
        "class_order": "|".join(str(value) for value in bundle.classes),
        **metrics,
    }


def _empty_positive_union_metric_row(rule: str, *, scope: str) -> dict[str, object]:
    return {
        "scope": str(scope),
        "rule": str(rule),
        "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
        "class_order": "",
        "n_eval": 0,
        "class0_support": 0,
        "class1_support": 0,
        "class0_predicted_count": 0,
        "class1_predicted_count": 0,
        "class0_error_count": 0,
        "class1_error_count": 0,
        "class0_recall": math.nan,
        "class1_recall": math.nan,
        "class0_specificity": math.nan,
        "class1_specificity": math.nan,
        "precision_class0": math.nan,
        "precision": math.nan,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "predicted_positive_rate": math.nan,
        "bacc": math.nan,
        "macro_f1": math.nan,
        "smoothed_class0_recall": math.nan,
        "smoothed_class1_recall": math.nan,
        "smoothed_min_class_recall": math.nan,
        "smoothed_precision_class0": math.nan,
        "smoothed_precision": math.nan,
        "smoothed_bacc": math.nan,
        "smoothed_macro_f1": math.nan,
    }


def _select_positive_union_rule(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    source_rows: Sequence[Mapping[str, object]],
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    rows = [dict(row) for row in source_rows]
    by_rule = {str(row["rule"]): row for row in rows}
    arithmetic = by_rule[POSITIVE_UNION_RULE_ARITHMETIC]
    positive_count = _safe_int(arithmetic.get("class1_support"), default=0)
    negative_count = _safe_int(arithmetic.get("class0_support"), default=0)
    if positive_count < cfg.min_source_inner_positive_count:
        for row in rows:
            row["source_inner_eligible"] = row["rule"] == POSITIVE_UNION_RULE_ARITHMETIC
            row["source_inner_ineligible_reason"] = "" if row["rule"] == POSITIVE_UNION_RULE_ARITHMETIC else "insufficient_source_inner_positive_count"
        selected = POSITIVE_UNION_RULE_ARITHMETIC
        return selected, rows, _positive_union_selection_row(
            cfg,
            selected_rule=selected,
            selected_row=by_rule[selected],
            positive_count=positive_count,
            negative_count=negative_count,
            selection_reason="insufficient_source_inner_positive_count",
        )

    arith_bacc = _float(arithmetic.get("smoothed_bacc"))
    arith_class0 = _float(arithmetic.get("smoothed_class0_recall"))
    arith_class1 = _float(arithmetic.get("smoothed_class1_recall"))
    arith_precision = _float(arithmetic.get("smoothed_precision"))
    arith_ppr = _float(arithmetic.get("predicted_positive_rate"))
    for row in rows:
        rule = str(row["rule"])
        reasons: list[str] = []
        eligible = True
        if rule != POSITIVE_UNION_RULE_ARITHMETIC:
            if _float(row.get("smoothed_bacc")) < arith_bacc - cfg.source_inner_bacc_noninferiority_margin:
                reasons.append("source_inner_bacc_inferior")
            if _float(row.get("smoothed_class0_recall")) < arith_class0 - cfg.source_inner_class0_recall_margin:
                reasons.append("source_inner_class0_recall_harm")
            if _float(row.get("predicted_positive_rate")) > arith_ppr + cfg.source_inner_predicted_positive_rate_delta:
                reasons.append("source_inner_predicted_positive_rate_inflation")
            if rule == POSITIVE_UNION_RULE_BETA100:
                if _float(row.get("smoothed_class1_recall")) <= arith_class1:
                    reasons.append("beta100_no_class1_recall_gain")
                if _float(row.get("smoothed_class0_recall")) < arith_class0 - cfg.beta100_class0_recall_margin:
                    reasons.append("beta100_class0_recall_harm")
                if _float(row.get("smoothed_precision")) < arith_precision - cfg.beta100_precision_margin:
                    reasons.append("beta100_precision_harm")
            eligible = not reasons
        row["source_inner_eligible"] = eligible
        row["source_inner_ineligible_reason"] = "|".join(reasons)

    eligible_rows = [row for row in rows if row.get("source_inner_eligible") is True]
    if not eligible_rows:
        selected = POSITIVE_UNION_RULE_ARITHMETIC
        selected_row = by_rule[selected]
        reason = "no_eligible_rule_fallback_arithmetic"
    else:
        selected_row = max(
            eligible_rows,
            key=lambda row: (
                _float(row.get("smoothed_min_class_recall")),
                _float(row.get("smoothed_bacc")),
                _float(row.get("smoothed_macro_f1")),
                -cfg.candidate_pooling_rules.index(str(row["rule"])),
            ),
        )
        selected = str(selected_row["rule"])
        reason = "source_inner_selected"
    return selected, rows, _positive_union_selection_row(
        cfg,
        selected_rule=selected,
        selected_row=selected_row,
        positive_count=positive_count,
        negative_count=negative_count,
        selection_reason=reason,
    )


def _select_harm_gated_positive_union_rule(
    cfg: SourceInnerHarmGatedPositiveUnionConfig,
    *,
    source_rows: Sequence[Mapping[str, object]],
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    rows = [dict(row) for row in source_rows]
    by_rule = {str(row["rule"]): row for row in rows}
    arithmetic = by_rule[POSITIVE_UNION_RULE_ARITHMETIC]
    positive_count = _safe_int(arithmetic.get("class1_support"), default=0)
    negative_count = _safe_int(arithmetic.get("class0_support"), default=0)
    if positive_count < cfg.min_source_inner_positive_count:
        for row in rows:
            row["primary_selectable_rule"] = str(row["rule"]) in cfg.primary_selectable_rules
            row["source_inner_eligible"] = row["rule"] == POSITIVE_UNION_RULE_ARITHMETIC
            row["source_inner_ineligible_reason"] = "" if row["rule"] == POSITIVE_UNION_RULE_ARITHMETIC else "insufficient_source_inner_positive_count"
        selected = POSITIVE_UNION_RULE_ARITHMETIC
        return selected, rows, _positive_union_selection_row(
            cfg,
            selected_rule=selected,
            selected_row=by_rule[selected],
            positive_count=positive_count,
            negative_count=negative_count,
            selection_reason="insufficient_source_inner_positive_count",
        )

    arith_bacc = _float(arithmetic.get("smoothed_bacc"))
    arith_class0 = _float(arithmetic.get("smoothed_class0_recall"))
    arith_precision = _float(arithmetic.get("smoothed_precision"))
    arith_ppr = _float(arithmetic.get("predicted_positive_rate"))
    for row in rows:
        rule = str(row["rule"])
        reasons: list[str] = []
        selectable = rule in cfg.primary_selectable_rules
        if not selectable:
            reasons.append("audit_only_not_primary_selectable")
        if rule != POSITIVE_UNION_RULE_ARITHMETIC:
            if _float(row.get("smoothed_bacc")) < arith_bacc - cfg.harm_gate_bacc_noninferiority_margin:
                reasons.append("source_inner_bacc_inferior")
            if rule == POSITIVE_UNION_RULE_BETA025:
                if _float(row.get("smoothed_class0_recall")) < arith_class0 - cfg.beta025_class0_recall_margin:
                    reasons.append("beta025_class0_recall_harm")
                if _float(row.get("predicted_positive_rate")) > arith_ppr + cfg.beta025_predicted_positive_rate_delta:
                    reasons.append("beta025_predicted_positive_rate_inflation")
            elif rule == POSITIVE_UNION_RULE_BETA050:
                if positive_count < cfg.beta050_min_source_inner_positive_count:
                    reasons.append("beta050_insufficient_source_inner_positive_count")
                if _float(row.get("smoothed_class0_recall")) < arith_class0 - cfg.beta050_class0_recall_margin:
                    reasons.append("beta050_class0_recall_harm")
                if _float(row.get("smoothed_precision")) < arith_precision - cfg.beta050_precision_margin:
                    reasons.append("beta050_precision_harm")
                if _float(row.get("predicted_positive_rate")) > arith_ppr + cfg.beta050_predicted_positive_rate_delta:
                    reasons.append("beta050_predicted_positive_rate_inflation")
        row["primary_selectable_rule"] = selectable
        row["source_inner_eligible"] = not reasons
        row["source_inner_ineligible_reason"] = "|".join(reasons)

    eligible_rows = [row for row in rows if row.get("source_inner_eligible") is True and str(row["rule"]) in cfg.primary_selectable_rules]
    if not eligible_rows:
        selected = POSITIVE_UNION_RULE_ARITHMETIC
        selected_row = by_rule[selected]
        reason = "no_eligible_rule_fallback_arithmetic"
    else:
        selected_row = max(
            eligible_rows,
            key=lambda row: (
                _float(row.get("smoothed_min_class_recall")),
                _float(row.get("smoothed_bacc")),
                _float(row.get("smoothed_macro_f1")),
                -cfg.primary_selectable_rules.index(str(row["rule"])),
            ),
        )
        selected = str(selected_row["rule"])
        reason = "source_inner_harm_gated_selected"
    selection = _positive_union_selection_row(
        cfg,
        selected_rule=selected,
        selected_row=selected_row,
        positive_count=positive_count,
        negative_count=negative_count,
        selection_reason=reason,
    )
    selection.update(
        {
            "beta050_min_source_inner_positive_count": cfg.beta050_min_source_inner_positive_count,
            "harm_gate_bacc_noninferiority_margin": cfg.harm_gate_bacc_noninferiority_margin,
            "beta025_class0_recall_margin": cfg.beta025_class0_recall_margin,
            "beta025_predicted_positive_rate_delta": cfg.beta025_predicted_positive_rate_delta,
            "beta050_class0_recall_margin": cfg.beta050_class0_recall_margin,
            "beta050_precision_margin": cfg.beta050_precision_margin,
            "beta050_predicted_positive_rate_delta": cfg.beta050_predicted_positive_rate_delta,
        }
    )
    return selected, rows, selection


def _positive_union_selection_row(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    selected_rule: str,
    selected_row: Mapping[str, object],
    positive_count: int,
    negative_count: int,
    selection_reason: str,
) -> dict[str, object]:
    return {
        "selected_rule": selected_rule,
        "selected_beta": "" if _positive_union_rule_beta(selected_rule) is None else _positive_union_rule_beta(selected_rule),
        "selection_reason": selection_reason,
        "source_inner_positive_count": int(positive_count),
        "source_inner_negative_count": int(negative_count),
        "min_source_inner_positive_count": cfg.min_source_inner_positive_count,
        "selected_source_inner_min_class_recall_smoothed": selected_row.get("smoothed_min_class_recall", math.nan),
        "selected_source_inner_bacc_smoothed": selected_row.get("smoothed_bacc", math.nan),
        "selected_source_inner_macro_f1_smoothed": selected_row.get("smoothed_macro_f1", math.nan),
        "selected_source_inner_precision_smoothed": selected_row.get("smoothed_precision", math.nan),
        "selected_source_inner_predicted_positive_rate": selected_row.get("predicted_positive_rate", math.nan),
        "selection_used_target_labels": False,
        "target_support_used": False,
    }


def _effective_threshold_for_rule(rule: str, n_seed_bundles: int) -> tuple[float, float]:
    beta = _positive_union_rule_beta(rule)
    if beta is None:
        identical = 0.5
        single = math.nan if n_seed_bundles > 2 else 1.0
        return identical, single
    identical = 1.0 - 0.5 ** (1.0 / (float(n_seed_bundles) * float(beta)))
    single = 1.0 - 0.5 ** (1.0 / float(beta))
    return identical, single


def _row_delta(row: Mapping[str, object], base: Mapping[str, object], key: str) -> float:
    left = _float(row.get(key))
    right = _float(base.get(key))
    return left - right if math.isfinite(left) and math.isfinite(right) else math.nan


def _positive_union_effective_threshold_rows(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    n_seed_bundles: int,
    source_rows: Mapping[str, Mapping[str, object]],
    target_rows: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    out = []
    arith_source = source_rows[POSITIVE_UNION_RULE_ARITHMETIC]
    arith_target = target_rows[POSITIVE_UNION_RULE_ARITHMETIC]
    for rule in cfg.candidate_pooling_rules:
        source = source_rows[rule]
        target = target_rows[rule]
        identical, single = _effective_threshold_for_rule(rule, n_seed_bundles)
        out.append(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "rule": rule,
                "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
                "n_seed_bundles": n_seed_bundles,
                "identical_seed_probability_needed_for_positive_flip": identical,
                "single_seed_probability_needed_for_positive_flip_if_other_seeds_zero": single,
                "source_inner_predicted_positive_rate": source.get("predicted_positive_rate", math.nan),
                "target_predicted_positive_rate": target.get("predicted_positive_rate", math.nan),
                "delta_predicted_positive_rate_vs_arithmetic": _row_delta(target, arith_target, "predicted_positive_rate"),
                "source_inner_delta_predicted_positive_rate_vs_arithmetic": _row_delta(source, arith_source, "predicted_positive_rate"),
                "class1_recall_delta": _row_delta(target, arith_target, "class1_recall"),
                "class0_recall_delta": _row_delta(target, arith_target, "class0_recall"),
                "precision_delta": _row_delta(target, arith_target, "precision"),
                "macro_f1_delta": _row_delta(target, arith_target, "macro_f1"),
                "bacc_delta": _row_delta(target, arith_target, "bacc"),
                "audit_only": True,
                "primary_adoption_eligible": False,
                "selection_used_target_labels": False,
                "target_eval_labels_used_for_audit_only": True,
            }
        )
    return out


def _positive_union_class_conditional_rows(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    target_rows: Mapping[str, Mapping[str, object]],
    selected_rule: str,
) -> list[dict[str, object]]:
    out = []
    for rule in cfg.candidate_pooling_rules:
        row = dict(target_rows[rule])
        row.update(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "selected_rule_for_cell": selected_rule,
                "is_selected_rule": rule == selected_rule,
                "audit_only": True,
                "primary_adoption_eligible": False,
                "selection_used_target_labels": False,
                "target_eval_labels_used_for_audit_only": True,
            }
        )
        out.append(row)
    return out


def _positive_union_harm_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    selected_rule: str,
    selected_bundle: PredictionBundle,
    arithmetic_bundle: PredictionBundle,
    eval_labels: Sequence[int],
) -> dict[str, object]:
    selected_preds = predict_from_probabilities(selected_bundle.probabilities, classes=selected_bundle.classes)
    arithmetic_preds = predict_from_probabilities(arithmetic_bundle.probabilities, classes=arithmetic_bundle.classes)
    selected_metrics = _binary_metrics_from_predictions(eval_labels, selected_preds)
    arithmetic_metrics = _binary_metrics_from_predictions(eval_labels, arithmetic_preds)
    negative_to_positive = sum(int(a) == 0 and int(s) == 1 for a, s in zip(arithmetic_preds, selected_preds))
    positive_to_negative = sum(int(a) == 1 and int(s) == 0 for a, s in zip(arithmetic_preds, selected_preds))
    selected_true_positive = _safe_int(selected_metrics.get("class1_support"), default=0) - _safe_int(selected_metrics.get("class1_error_count"), default=0)
    arithmetic_true_positive = _safe_int(arithmetic_metrics.get("class1_support"), default=0) - _safe_int(arithmetic_metrics.get("class1_error_count"), default=0)
    bacc_delta = _row_delta(selected_metrics, arithmetic_metrics, "bacc")
    return {
        "experiment_seed": experiment_seed,
        "heldout_center": heldout_center,
        "selected_rule": selected_rule,
        "precision_delta_vs_arithmetic": _row_delta(selected_metrics, arithmetic_metrics, "precision"),
        "specificity_delta_vs_arithmetic": _row_delta(selected_metrics, arithmetic_metrics, "class1_specificity"),
        "predicted_positive_rate_delta": _row_delta(selected_metrics, arithmetic_metrics, "predicted_positive_rate"),
        "false_positive_count_delta_vs_arithmetic": _safe_int(selected_metrics.get("false_positive_count"), default=0) - _safe_int(arithmetic_metrics.get("false_positive_count"), default=0),
        "true_positive_count_delta_vs_arithmetic": selected_true_positive - arithmetic_true_positive,
        "negative_to_positive_flip_count": negative_to_positive,
        "positive_to_negative_flip_count": positive_to_negative,
        "bacc_delta_vs_arithmetic": bacc_delta,
        "worst_per_center_regression": bacc_delta,
        "worst_seed_center_regression": bacc_delta,
        "tail_risk_transfer_flag": bool(math.isfinite(bacc_delta) and bacc_delta < -0.010),
        "audit_only": True,
        "primary_adoption_eligible": False,
        "selection_used_target_labels": False,
        "target_eval_labels_used_for_audit_only": True,
    }


def _positive_union_per_source_harm_rows(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    source_ids: Sequence[str],
    source_labels: Sequence[int],
    source_bundles_by_rule: Mapping[str, PredictionBundle],
) -> list[dict[str, object]]:
    out = []
    arithmetic_preds = predict_from_probabilities(
        source_bundles_by_rule[POSITIVE_UNION_RULE_ARITHMETIC].probabilities,
        classes=source_bundles_by_rule[POSITIVE_UNION_RULE_ARITHMETIC].classes,
    )
    preds_by_rule = {
        rule: predict_from_probabilities(bundle.probabilities, classes=bundle.classes)
        for rule, bundle in source_bundles_by_rule.items()
    }
    for source_center in sorted(set(str(value) for value in source_ids)):
        indices = [idx for idx, value in enumerate(source_ids) if str(value) == source_center]
        labels = [int(source_labels[idx]) for idx in indices]
        arithmetic_metrics = _binary_metrics_from_predictions(labels, [int(arithmetic_preds[idx]) for idx in indices])
        for rule in cfg.candidate_pooling_rules:
            preds = [int(preds_by_rule[rule][idx]) for idx in indices]
            metrics = _binary_metrics_from_predictions(labels, preds)
            bacc_delta = _row_delta(metrics, arithmetic_metrics, "bacc")
            class0_delta = _row_delta(metrics, arithmetic_metrics, "class0_recall")
            class1_delta = _row_delta(metrics, arithmetic_metrics, "class1_recall")
            precision_delta = _row_delta(metrics, arithmetic_metrics, "precision")
            ppr_delta = _row_delta(metrics, arithmetic_metrics, "predicted_positive_rate")
            out.append(
                {
                    "experiment_seed": experiment_seed,
                    "heldout_center": heldout_center,
                    "source_center": source_center,
                    "rule": rule,
                    "beta": "" if _positive_union_rule_beta(rule) is None else _positive_union_rule_beta(rule),
                    "source_inner_positive_count": metrics["class1_support"],
                    "source_inner_negative_count": metrics["class0_support"],
                    "bacc_delta_vs_arithmetic": bacc_delta,
                    "class0_recall_delta_vs_arithmetic": class0_delta,
                    "class1_recall_delta_vs_arithmetic": class1_delta,
                    "precision_delta_vs_arithmetic": precision_delta,
                    "predicted_positive_rate_delta_vs_arithmetic": ppr_delta,
                    "worst_per_source_harm_flag": bool(
                        (math.isfinite(bacc_delta) and bacc_delta < -cfg.source_inner_bacc_noninferiority_margin)
                        or (math.isfinite(class0_delta) and class0_delta < -cfg.source_inner_class0_recall_margin)
                        or (math.isfinite(precision_delta) and precision_delta < -cfg.beta100_precision_margin)
                        or (math.isfinite(ppr_delta) and ppr_delta > cfg.source_inner_predicted_positive_rate_delta)
                    ),
                    "audit_only": True,
                    "primary_adoption_eligible": False,
                    "selection_used_target_labels": False,
                }
            )
    return out


def _harm_gated_proxy_validity_rows(
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    primary_method: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in candidate_rows:
        groups.setdefault((str(row.get("experiment_seed")), str(row.get("heldout_center"))), []).append(row)
    out = []
    for (seed, center), rows in sorted(groups.items()):
        selectable = [
            row
            for row in rows
            if str(row.get("rule")) in HARM_GATED_PRIMARY_SELECTABLE_RULES
        ]
        if not selectable:
            continue
        selected = next((row for row in rows if str(row.get("is_selected_rule")) == "True" or row.get("is_selected_rule") is True), selectable[0])
        source_bacc_scores = {str(row.get("rule")): _float(row.get("source_inner_smoothed_bacc")) for row in selectable}
        source_harm_scores = {str(row.get("rule")): _harm_proxy_score(row, prefix="source_inner_") for row in selectable}
        target_bacc_scores = {str(row.get("rule")): _float(row.get("target_bacc")) for row in selectable}
        target_tail_scores = {
            str(row.get("rule")): min(_float(row.get("target_class0_recall")), _float(row.get("target_class1_recall")))
            for row in selectable
        }
        target_harm_scores = {str(row.get("rule")): _harm_proxy_score(row, prefix="target_") for row in selectable}
        selected_rule = str(selected.get("rule"))
        target_best_bacc_rule = _best_score_rule(target_bacc_scores)
        target_best_tail_rule = _best_score_rule(target_tail_scores)
        out.append(
            {
                "experiment_seed": seed,
                "heldout_center": center,
                "source_inner_rank_of_rules": json.dumps(_rank_rules(source_bacc_scores)),
                "target_rank_of_rules_by_BACC": json.dumps(_rank_rules(target_bacc_scores)),
                "target_rank_of_rules_by_tail_metric": json.dumps(_rank_rules(target_tail_scores)),
                "target_rank_of_rules_by_harm_metric": json.dumps(_rank_rules(target_harm_scores)),
                "selected_rule": selected_rule,
                "top1_rule_hit": selected_rule == target_best_bacc_rule,
                "oracle_gap_BACC": _score_gap(target_bacc_scores, selected_rule),
                "oracle_gap_tail_metric": _score_gap(target_tail_scores, selected_rule),
                "spearman_source_inner_vs_target_BACC": _spearman_for_rule_scores(source_bacc_scores, target_bacc_scores),
                "spearman_source_inner_vs_target_harm": _spearman_for_rule_scores(source_harm_scores, target_harm_scores),
                "primary_method": primary_method,
                "audit_only": True,
                "primary_adoption_eligible": False,
                "selection_used_target_labels": False,
                "target_eval_labels_used_for_audit_only": True,
            }
        )
    return out


def _harm_proxy_score(row: Mapping[str, object], *, prefix: str) -> float:
    class0 = _float(row.get(f"{prefix}smoothed_class0_recall", row.get(f"{prefix}class0_recall", math.nan)))
    precision = _float(row.get(f"{prefix}smoothed_precision", row.get(f"{prefix}precision", math.nan)))
    ppr = _float(row.get(f"{prefix}predicted_positive_rate", math.nan))
    values = [value for value in (class0, precision, -ppr) if math.isfinite(value)]
    return sum(values) if values else math.nan


def _rank_rules(scores: Mapping[str, float]) -> list[str]:
    finite = [(rule, value) for rule, value in scores.items() if math.isfinite(value)]
    return [rule for rule, _value in sorted(finite, key=lambda item: (-item[1], HARM_GATED_PRIMARY_SELECTABLE_RULES.index(item[0]) if item[0] in HARM_GATED_PRIMARY_SELECTABLE_RULES else 999))]


def _best_score_rule(scores: Mapping[str, float]) -> str:
    ranked = _rank_rules(scores)
    return ranked[0] if ranked else ""


def _score_gap(scores: Mapping[str, float], selected_rule: str) -> float:
    best = max((value for value in scores.values() if math.isfinite(value)), default=math.nan)
    selected = _float(scores.get(selected_rule, math.nan))
    return best - selected if math.isfinite(best) and math.isfinite(selected) else math.nan


def _spearman_for_rule_scores(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    rules = [rule for rule in HARM_GATED_PRIMARY_SELECTABLE_RULES if math.isfinite(_float(left.get(rule))) and math.isfinite(_float(right.get(rule)))]
    if len(rules) < 2:
        return math.nan
    left_ranks = _numeric_ranks([_float(left[rule]) for rule in rules])
    right_ranks = _numeric_ranks([_float(right[rule]) for rule in rules])
    return _pearson(left_ranks, right_ranks)


def _numeric_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = float(rank)
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if float(np.std(left_arr)) == 0.0 or float(np.std(right_arr)) == 0.0:
        return math.nan
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def _harm_gated_selected_rule_distribution_rows(
    selection_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [dict(row) for row in selection_rows]
    out = [_selected_rule_distribution_row(rows, scope="overall", scope_value="all")]
    centers = sorted({str(row.get("heldout_center")) for row in rows})
    seeds = sorted({str(row.get("experiment_seed")) for row in rows}, key=lambda value: int(value) if value.isdigit() else value)
    for center in centers:
        out.append(_selected_rule_distribution_row([row for row in rows if str(row.get("heldout_center")) == center], scope="heldout_center", scope_value=center))
    for seed in seeds:
        out.append(_selected_rule_distribution_row([row for row in rows if str(row.get("experiment_seed")) == seed], scope="experiment_seed", scope_value=seed))
    return out


def _selected_rule_distribution_row(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    scope_value: str,
) -> dict[str, object]:
    n = len(rows)
    counts: dict[str, int] = {}
    insufficient = 0
    for row in rows:
        selected = str(row.get("selected_rule", ""))
        counts[selected] = counts.get(selected, 0) + 1
        if row.get("selection_reason") == "insufficient_source_inner_positive_count":
            insufficient += 1
    fractions = {rule: float(count) / float(n) for rule, count in counts.items()} if n else {}
    return {
        "scope": scope,
        "scope_value": scope_value,
        "n_cells": n,
        "selected_rule_counts": json.dumps(counts, sort_keys=True),
        "selected_rule_fraction_by_center": json.dumps(fractions, sort_keys=True) if scope == "heldout_center" else "",
        "selected_rule_fraction_by_seed": json.dumps(fractions, sort_keys=True) if scope == "experiment_seed" else "",
        "beta050_selection_rate": float(counts.get(POSITIVE_UNION_RULE_BETA050, 0)) / float(n) if n else math.nan,
        "beta025_selection_rate": float(counts.get(POSITIVE_UNION_RULE_BETA025, 0)) / float(n) if n else math.nan,
        "arithmetic_fallback_rate": float(counts.get(POSITIVE_UNION_RULE_ARITHMETIC, 0)) / float(n) if n else math.nan,
        "insufficient_positive_count_rate": float(insufficient) / float(n) if n else math.nan,
        "audit_only": True,
        "primary_adoption_eligible": False,
    }


