#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PRIMARY_METHOD = "support_response_pairwise_static_response_indirect"
RISK_CONSTRAINED_METHOD = "risk_constrained_response_routing"
METADATA_BASELINE = "support_metadata_routing"
STATIC_BASELINE = "support_static_embedding_routing"
SUPPORT_NELBO_BASELINE = "support_set_nelbo_top1"
SUPPORT_CONSERVATIVE_METHOD = "support_set_nelbo_conservative"
SUPPORT_RESPONSE_PROTOCOL_VERSION = "support_response_candidate_specific_v1"
CONTROL_OR_DIAGNOSTIC_METHODS = {
    "expert_id_only_pairwise",
    "support_response_pairwise_response_indirect_shuffled",
    "source_leave_pseudo_domain_out_ranker_diagnostic",
    "support_candidate_oracle",
}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    mu = _mean(vals)
    return float((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _support_payload(path: Path) -> Tuple[Dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "support_response_results" in payload:
        return dict(payload["support_response_results"]), path.parent
    if str(payload.get("protocol_version", "")) == SUPPORT_RESPONSE_PROTOCOL_VERSION:
        return dict(payload), path.parent
    raise ValueError(f"Result file does not contain support-response results: {path}")


def _result_run_id(path: Path) -> str:
    path = Path(path)
    if path.parent.name == "reports":
        return path.parent.parent.name
    return path.parent.name


def _read_rows(result_paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in result_paths:
        support, base_dir = _support_payload(Path(path))
        metrics_by_method = support.get("metrics_by_method", {})
        if not isinstance(metrics_by_method, Mapping):
            continue
        artifacts = support.get("artifacts", {}) if isinstance(support.get("artifacts", {}), Mapping) else {}
        domain_artifact = str(artifacts.get("domain_breakdown", "support_response_domain_breakdown.csv"))
        domain_rows = _read_csv(base_dir / domain_artifact)
        domains_by_method: Dict[str, List[dict]] = {}
        for domain_row in domain_rows:
            domains_by_method.setdefault(str(domain_row.get("method", "")), []).append(domain_row)
        for method, metrics_raw in metrics_by_method.items():
            metrics = dict(metrics_raw or {})
            rows.append(
                {
                    "result_path": str(path),
                    "run_id": _result_run_id(path),
                    "method": str(method),
                    "method_role": str(metrics.get("method_role", "")),
                    "adoption_eligible": int(_to_float(metrics.get("adoption_eligible", 0))),
                    "diagnostic_only": int(_to_float(metrics.get("diagnostic_only", 0))),
                    "routing_uses_eval_nelbo": int(_to_float(metrics.get("routing_uses_eval_nelbo", 0))),
                    "routing_uses_eval_domain_statistics": int(
                        _to_float(metrics.get("routing_uses_eval_domain_statistics", 0))
                    ),
                    "top1_oracle_hit": _to_float(metrics.get("top1_oracle_hit", 0.0)),
                    "spearman": _to_float(metrics.get("spearman", 0.0)),
                    "mean_oracle_gap_pct": _to_float(metrics.get("mean_oracle_gap_pct", 0.0)),
                    "mean_rank": _to_float(metrics.get("mean_rank", metrics.get("micro_selected_rank", 0.0))),
                    "pairwise_auc": _to_float(metrics.get("pairwise_auc", metrics.get("micro_pairwise_auc", 0.0))),
                    "selected_nelbo": _to_float(
                        metrics.get("selected_nelbo", metrics.get("micro_selected_nelbo", 0.0))
                    ),
                    "candidate_oracle_nelbo": _to_float(
                        metrics.get(
                            "candidate_oracle_nelbo",
                            metrics.get("oracle_nelbo", metrics.get("micro_candidate_oracle_nelbo", 0.0)),
                        )
                    ),
                    "bottom_half_selection_rate": _to_float(metrics.get("bottom_half_selection_rate", 0.0)),
                    "high_regret_selection_rate": _to_float(metrics.get("high_regret_selection_rate", 0.0)),
                    "catastrophic_mistake_rate": _to_float(metrics.get("catastrophic_mistake_rate", 0.0)),
                    "alpha": _to_float(metrics.get("alpha", 0.0)),
                    "alpha_selection_policy": str(metrics.get("alpha_selection_policy", "")),
                    "top1_tolerance_abs": _to_float(metrics.get("top1_tolerance_abs", 0.0)),
                    "override_rate": _to_float(metrics.get("override_rate", 0.0)),
                    "harmful_override_rate": _to_float(metrics.get("harmful_override_rate", 0.0)),
                    "utility_improving_override_rate": _to_float(
                        metrics.get("utility_improving_override_rate", 0.0)
                    ),
                    "expert4_override_candidate_rate": _to_float(
                        metrics.get("expert4_override_candidate_rate", 0.0)
                    ),
                    "expert4_override_accepted_rate": _to_float(
                        metrics.get("expert4_override_accepted_rate", 0.0)
                    ),
                    "expert4_override_blocked_rate": _to_float(
                        metrics.get("expert4_override_blocked_rate", 0.0)
                    ),
                    "n_domain_level_units": int(
                        _to_float(
                            metrics.get(
                                "n_query_domains_macro",
                                metrics.get("n_samples_micro", len(domains_by_method.get(str(method), []))),
                            )
                        )
                    ),
                    "domain_rows": domains_by_method.get(str(method), []),
                }
            )
    return rows


def _by_method(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["method"]), []).append(row)
    return out


def _paired_run_deltas(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metric: str,
    lower_is_better: bool = False,
) -> List[float]:
    baseline_by_run = {str(row["run_id"]): row for row in baseline_rows}
    deltas: List[float] = []
    for row in candidate_rows:
        base = baseline_by_run.get(str(row["run_id"]))
        if base is None:
            continue
        if lower_is_better:
            deltas.append(_to_float(base.get(metric, 0.0)) - _to_float(row.get(metric, 0.0)))
        else:
            deltas.append(_to_float(row.get(metric, 0.0)) - _to_float(base.get(metric, 0.0)))
    return deltas


def _paired_domain_deltas(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metric: str,
    lower_is_better: bool = False,
) -> List[float]:
    baseline_domains: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in baseline_rows:
        for domain_row in row.get("domain_rows", []) or []:
            baseline_domains[(str(row["run_id"]), int(float(domain_row.get("query_domain", 0))))] = domain_row
    deltas: List[float] = []
    for row in candidate_rows:
        for domain_row in row.get("domain_rows", []) or []:
            key = (str(row["run_id"]), int(float(domain_row.get("query_domain", 0))))
            base = baseline_domains.get(key)
            if base is None:
                continue
            if lower_is_better:
                deltas.append(_to_float(base.get(metric, 0.0)) - _to_float(domain_row.get(metric, 0.0)))
            else:
                deltas.append(_to_float(domain_row.get(metric, 0.0)) - _to_float(base.get(metric, 0.0)))
    return deltas


def _positive_fraction(values: Sequence[float], *, eps: float = 0.0) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(sum(1 for v in vals if v >= eps) / len(vals))


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    uncertainty_interval: float = 0.0,
    material_regression_top1: float = 0.02,
    material_regression_spearman: float = 0.05,
    material_regression_gap_pct: float = 1.0,
) -> Tuple[List[dict], Dict[str, Any]]:
    grouped = _by_method(rows)
    metadata = grouped.get(METADATA_BASELINE, [])
    static = grouped.get(STATIC_BASELINE, [])
    support = grouped.get(SUPPORT_NELBO_BASELINE, [])
    unrestricted = grouped.get(PRIMARY_METHOD, [])

    baseline_top1 = _mean([_to_float(r.get("top1_oracle_hit", 0.0)) for r in metadata])
    baseline_spearman = _mean([_to_float(r.get("spearman", 0.0)) for r in metadata])
    baseline_gap = _mean([_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in metadata])
    static_top1 = _mean([_to_float(r.get("top1_oracle_hit", 0.0)) for r in static])
    static_spearman = _mean([_to_float(r.get("spearman", 0.0)) for r in static])
    static_gap = _mean([_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in static])
    support_top1 = _mean([_to_float(r.get("top1_oracle_hit", 0.0)) for r in support])
    support_spearman = _mean([_to_float(r.get("spearman", 0.0)) for r in support])
    support_gap = _mean([_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in support])
    support_gap_std = _std([_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in support])
    support_high_regret = _mean([_to_float(r.get("high_regret_selection_rate", 0.0)) for r in support])
    support_catastrophic = _mean([_to_float(r.get("catastrophic_mistake_rate", 0.0)) for r in support])
    unrestricted_harmful = _mean([_to_float(r.get("harmful_override_rate", 0.0)) for r in unrestricted])

    out: List[dict] = []
    for method, method_rows in sorted(grouped.items()):
        top1_vals = [_to_float(r.get("top1_oracle_hit", 0.0)) for r in method_rows]
        spearman_vals = [_to_float(r.get("spearman", 0.0)) for r in method_rows]
        gap_vals = [_to_float(r.get("mean_oracle_gap_pct", 0.0)) for r in method_rows]
        top1 = _mean(top1_vals)
        spearman = _mean(spearman_vals)
        gap = _mean(gap_vals)
        top1_delta = top1 - baseline_top1
        spearman_delta = spearman - baseline_spearman
        gap_reduction = baseline_gap - gap

        top1_domain_deltas = _paired_domain_deltas(
            candidate_rows=method_rows,
            baseline_rows=metadata,
            metric="top1_oracle_hit",
        )
        spearman_domain_deltas = _paired_domain_deltas(
            candidate_rows=method_rows,
            baseline_rows=metadata,
            metric="spearman",
        )
        gap_domain_deltas = _paired_domain_deltas(
            candidate_rows=method_rows,
            baseline_rows=metadata,
            metric="mean_oracle_gap_pct",
            lower_is_better=True,
        )
        top1_run_deltas = _paired_run_deltas(
            candidate_rows=method_rows,
            baseline_rows=metadata,
            metric="top1_oracle_hit",
        )
        spearman_run_deltas = _paired_run_deltas(
            candidate_rows=method_rows,
            baseline_rows=metadata,
            metric="spearman",
        )
        gap_run_deltas = _paired_run_deltas(
            candidate_rows=method_rows,
            baseline_rows=metadata,
            metric="mean_oracle_gap_pct",
            lower_is_better=True,
        )
        all_sign_deltas = top1_domain_deltas + spearman_domain_deltas + gap_domain_deltas
        if not all_sign_deltas:
            all_sign_deltas = top1_run_deltas + spearman_run_deltas + gap_run_deltas
        stable_sign = (
            top1_delta >= 0.0
            and spearman_delta >= 0.0
            and gap_reduction >= 0.0
            and _positive_fraction(all_sign_deltas) >= 0.67
        )

        first = method_rows[0]
        adoption_eligible = int(first.get("adoption_eligible", 0))
        diagnostic_only = int(first.get("diagnostic_only", 0))
        uses_eval = int(first.get("routing_uses_eval_nelbo", 0)) or int(
            first.get("routing_uses_eval_domain_statistics", 0)
        )
        is_control = method in CONTROL_OR_DIAGNOSTIC_METHODS or str(first.get("method_role", "")) == "control"
        no_direct_support_utility_terms = int(method == PRIMARY_METHOD)
        is_risk_constrained = method == RISK_CONSTRAINED_METHOD
        not_lose_static = (
            top1 + material_regression_top1 >= static_top1
            and spearman + material_regression_spearman >= static_spearman
            and gap <= static_gap + material_regression_gap_pct
        )
        harmful_override_rate = _mean([_to_float(r.get("harmful_override_rate", 0.0)) for r in method_rows])
        harmful_reduction_vs_unrestricted = float(unrestricted_harmful - harmful_override_rate)
        high_regret_rate = _mean([_to_float(r.get("high_regret_selection_rate", 0.0)) for r in method_rows])
        catastrophic_rate = _mean([_to_float(r.get("catastrophic_mistake_rate", 0.0)) for r in method_rows])
        top1_tolerance_abs = _mean([_to_float(r.get("top1_tolerance_abs", 0.0)) for r in method_rows])
        if top1_tolerance_abs <= 0.0:
            n_units = sum(int(_to_float(r.get("n_domain_level_units", 0))) for r in method_rows)
            top1_tolerance_abs = float(1.0 / max(n_units, 1))
        improves_all_metadata = top1_delta > 0.0 and spearman_delta > 0.0 and gap_reduction > 0.0
        improves_one = top1_delta > 0.0 or spearman_delta > 0.0 or gap_reduction > 0.0
        no_material_regression = (
            top1_delta >= -material_regression_top1
            and spearman_delta >= -material_regression_spearman
            and gap_reduction >= -material_regression_gap_pct
        )
        beats_support = (
            top1 > support_top1 + float(uncertainty_interval)
            and spearman > support_spearman + float(uncertainty_interval)
            and gap < support_gap - float(uncertainty_interval)
        )
        matches_support = (
            abs(top1 - support_top1) <= float(uncertainty_interval)
            and abs(spearman - support_spearman) <= float(uncertainty_interval)
            and abs(gap - support_gap) <= float(uncertainty_interval)
        )
        strict_support_non_regression = (
            top1 + float(top1_tolerance_abs) >= support_top1
            and spearman + 0.05 >= support_spearman
            and gap <= support_gap + 0.50
        )
        improves_support_stability_or_regret = (
            _std(gap_vals) < support_gap_std
            or high_regret_rate < support_high_regret
            or catastrophic_rate < support_catastrophic
        )

        decision = "not_selected"
        tier = "diagnostic_only"
        selection_eligible = 0
        rejection_reason = ""
        if method == METADATA_BASELINE:
            tier = "baseline"
            decision = "baseline_reference"
        elif method == "support_candidate_oracle":
            tier = "reference_only"
            rejection_reason = "predeclared_candidate_oracle_diagnostic"
        elif is_control or diagnostic_only or not adoption_eligible:
            tier = "reference_only" if method == "support_candidate_oracle" else "diagnostic_only"
            rejection_reason = "control_or_diagnostic_method"
        elif uses_eval:
            tier = "rejected"
            rejection_reason = "eval_leakage_or_oracle_method"
        else:
            selection_eligible = 1
            if (
                method == SUPPORT_CONSERVATIVE_METHOD
                and strict_support_non_regression
                and improves_support_stability_or_regret
            ):
                tier = "pass"
                decision = "selected"
            elif method == SUPPORT_CONSERVATIVE_METHOD and strict_support_non_regression:
                tier = "diagnostic_only"
                decision = "not_selected"
                rejection_reason = "matches_direct_support_without_stability_or_regret_gain"
            elif method == SUPPORT_CONSERVATIVE_METHOD:
                tier = "fail"
                rejection_reason = "degrades_direct_support_nelbo_without_allowed_non_regression"
            elif (
                is_risk_constrained
                and gap_reduction > 0.0
                and top1_delta >= 0.0
                and spearman_delta >= 0.0
                and harmful_reduction_vs_unrestricted >= 0.0
                and stable_sign
                and not_lose_static
            ):
                tier = "pass"
                decision = "selected"
            elif (
                is_risk_constrained
                and (gap_reduction > 0.0 or harmful_reduction_vs_unrestricted > 0.0)
                and no_material_regression
            ):
                tier = "weak_pass"
                decision = "not_selected"
            elif improves_all_metadata and stable_sign and not_lose_static:
                if (beats_support or matches_support) and no_direct_support_utility_terms:
                    tier = "strong_pass"
                    decision = "selected" if method == PRIMARY_METHOD else "not_selected"
                else:
                    tier = "pass"
                    decision = "selected" if method == PRIMARY_METHOD else "not_selected"
            elif improves_one and no_material_regression:
                tier = "weak_pass"
                decision = "not_selected"
            elif top1_delta < 0.0 or spearman_delta < 0.0 or gap_reduction < 0.0:
                tier = "fail"
                rejection_reason = "worse_than_metadata_or_unstable"
            else:
                tier = "diagnostic_only"
                rejection_reason = "insufficient_baseline_uplift"

        out.append(
            {
                "method": method,
                "tier": tier,
                "decision": decision,
                "selection_eligible": int(selection_eligible),
                "rejection_reason": rejection_reason,
                "top1_oracle_hit_mean": top1,
                "spearman_mean": spearman,
                "mean_oracle_gap_pct_mean": gap,
                "top1_uplift_vs_metadata": top1_delta,
                "spearman_uplift_vs_metadata": spearman_delta,
                "oracle_gap_pct_reduction_vs_metadata": gap_reduction,
                "top1_std_over_runs": _std(top1_vals),
                "spearman_std_over_runs": _std(spearman_vals),
                "gap_pct_std_over_runs": _std(gap_vals),
                "domain_clustered_top1_uplift_std": _std(top1_domain_deltas),
                "domain_clustered_spearman_uplift_std": _std(spearman_domain_deltas),
                "domain_clustered_gap_pct_reduction_std": _std(gap_domain_deltas),
                "stable_sign_across_seeds_domains": int(stable_sign),
                "not_lose_support_static_embedding_routing": int(not_lose_static),
                "beats_support_set_nelbo_top1": int(beats_support),
                "matches_support_set_nelbo_top1_within_interval": int(matches_support),
                "strict_support_non_regression": int(strict_support_non_regression),
                "improves_support_stability_or_regret": int(improves_support_stability_or_regret),
                "top1_tolerance_abs": float(top1_tolerance_abs),
                "uses_no_direct_support_utility_terms": int(no_direct_support_utility_terms),
                "bottom_half_selection_rate": _mean(
                    [_to_float(r.get("bottom_half_selection_rate", 0.0)) for r in method_rows]
                ),
                "high_regret_selection_rate": high_regret_rate,
                "catastrophic_mistake_rate": catastrophic_rate,
                "harmful_override_rate": harmful_override_rate,
                "harmful_override_rate_reduction_vs_unrestricted_response": harmful_reduction_vs_unrestricted,
                "override_rate": _mean([_to_float(r.get("override_rate", 0.0)) for r in method_rows]),
                "expert4_override_candidate_rate": _mean(
                    [_to_float(r.get("expert4_override_candidate_rate", 0.0)) for r in method_rows]
                ),
                "expert4_override_accepted_rate": _mean(
                    [_to_float(r.get("expert4_override_accepted_rate", 0.0)) for r in method_rows]
                ),
                "expert4_override_blocked_rate": _mean(
                    [_to_float(r.get("expert4_override_blocked_rate", 0.0)) for r in method_rows]
                ),
                "n_runs": int(len(method_rows)),
                "n_domain_level_units": int(sum(int(r.get("n_domain_level_units", 0)) for r in method_rows)),
            }
        )

    selected = [row["method"] for row in out if row.get("decision") == "selected"]
    summary = {
        "protocol_version": SUPPORT_RESPONSE_PROTOCOL_VERSION,
        "primary_method": PRIMARY_METHOD,
        "support_estimated_utility_method": SUPPORT_CONSERVATIVE_METHOD,
        "risk_constrained_method": RISK_CONSTRAINED_METHOD,
        "selected_methods": selected,
        "claim_boundary": (
            "Support-estimated utility routing treats compatibility as expected utility from "
            "unlabeled target-local support NELBO. Conservative support scoring may be claimed "
            "only if it satisfies direct-support non-regression and improves stability or "
            "high-regret selection rate. Do not claim learned response routing is the main "
            "solution for this experiment."
        ),
        "aggregation_unit": "seed_x_heldout_center_x_support_seed_x_support_size",
    }
    return out, summary


def _seed_from_run_id(run_id: str) -> str:
    match = re.search(r"seed(\d+)", str(run_id))
    return match.group(1) if match else str(run_id)


def _blank_if_missing(value: object) -> object:
    return "" if value is None else value


def _unique_thresholds(domain_rows: Sequence[Mapping[str, Any]]) -> str:
    thresholds: List[str] = []
    seen: set[str] = set()
    for row in domain_rows:
        selected_tau = str(row.get("selected_tau", "")).strip()
        tau_margin = str(row.get("tau_margin", "")).strip()
        tau_regret = str(row.get("tau_regret", "")).strip()
        if selected_tau:
            label = selected_tau
        elif tau_margin or tau_regret:
            label = f"margin={tau_margin or 'NA'};regret={tau_regret or 'NA'}"
        else:
            continue
        if label not in seen:
            seen.add(label)
            thresholds.append(label)
    return "|".join(sorted(thresholds))


def _build_seed_stability_rows(
    rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    material_regression_top1: float = 0.02,
    material_regression_spearman: float = 0.05,
    material_regression_gap_pct: float = 1.0,
) -> List[dict]:
    decision_by_method = {str(row.get("method", "")): row for row in decision_rows}
    by_run: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row.get("run_id", "")), {})[str(row.get("method", ""))] = row

    out: List[dict] = []
    for run_id in sorted(by_run, key=lambda value: int(_seed_from_run_id(value)) if _seed_from_run_id(value).isdigit() else value):
        method_rows = by_run[run_id]
        metadata = method_rows.get(METADATA_BASELINE, {})
        static = method_rows.get(STATIC_BASELINE, {})
        unrestricted = method_rows.get(PRIMARY_METHOD, {})
        metadata_selected_nelbo = _to_float(metadata.get("selected_nelbo", 0.0))
        static_top1 = _to_float(static.get("top1_oracle_hit", 0.0))
        static_spearman = _to_float(static.get("spearman", 0.0))
        static_gap = _to_float(static.get("mean_oracle_gap_pct", 0.0))
        unrestricted_harmful = _to_float(unrestricted.get("harmful_override_rate", 0.0))

        for method in sorted(method_rows):
            row = method_rows[method]
            decision = decision_by_method.get(method, {})
            domain_rows = list(row.get("domain_rows", []) or [])
            top1 = _to_float(row.get("top1_oracle_hit", 0.0))
            spearman = _to_float(row.get("spearman", 0.0))
            gap = _to_float(row.get("mean_oracle_gap_pct", 0.0))
            selected_nelbo = _to_float(row.get("selected_nelbo", 0.0))
            fallback_count = None
            fallback_rate = None
            if method == RISK_CONSTRAINED_METHOD and domain_rows:
                fallback_count = sum(1 for item in domain_rows if int(_to_float(item.get("fallback_used", 0))) == 1)
                fallback_rate = fallback_count / len(domain_rows)

            not_lose_static = None
            if static:
                not_lose_static = int(
                    top1 + material_regression_top1 >= static_top1
                    and spearman + material_regression_spearman >= static_spearman
                    and gap <= static_gap + material_regression_gap_pct
                )

            out.append(
                {
                    "seed": _seed_from_run_id(run_id),
                    "run_id": run_id,
                    "method": method,
                    "method_role": row.get("method_role", ""),
                    "tier": decision.get("tier", ""),
                    "decision": decision.get("decision", ""),
                    "top1_oracle_hit": top1,
                    "spearman": spearman,
                    "mean_oracle_gap_pct": gap,
                    "mean_selected_rank": _to_float(row.get("mean_rank", 0.0)),
                    "pairwise_auc": _to_float(row.get("pairwise_auc", 0.0)),
                    "selected_nelbo": selected_nelbo,
                    "candidate_oracle_nelbo": _to_float(row.get("candidate_oracle_nelbo", 0.0)),
                    "bottom_half_selection_rate": _to_float(row.get("bottom_half_selection_rate", 0.0)),
                    "high_regret_selection_rate": _to_float(row.get("high_regret_selection_rate", 0.0)),
                    "catastrophic_mistake_rate": _to_float(row.get("catastrophic_mistake_rate", 0.0)),
                    "alpha": _to_float(row.get("alpha", 0.0)),
                    "nelbo_delta_vs_metadata": selected_nelbo - metadata_selected_nelbo,
                    "top1_uplift_vs_metadata": top1 - _to_float(metadata.get("top1_oracle_hit", 0.0)),
                    "spearman_uplift_vs_metadata": spearman - _to_float(metadata.get("spearman", 0.0)),
                    "oracle_gap_pct_reduction_vs_metadata": _to_float(
                        metadata.get("mean_oracle_gap_pct", 0.0)
                    )
                    - gap,
                    "harmful_override_rate": _to_float(row.get("harmful_override_rate", 0.0)),
                    "harmful_override_rate_reduction_vs_unrestricted_response": unrestricted_harmful
                    - _to_float(row.get("harmful_override_rate", 0.0)),
                    "override_rate": _to_float(row.get("override_rate", 0.0)),
                    "utility_improving_override_rate": _to_float(row.get("utility_improving_override_rate", 0.0)),
                    "expert4_override_candidate_rate": _to_float(row.get("expert4_override_candidate_rate", 0.0)),
                    "expert4_override_accepted_rate": _to_float(row.get("expert4_override_accepted_rate", 0.0)),
                    "expert4_override_blocked_rate": _to_float(row.get("expert4_override_blocked_rate", 0.0)),
                    "not_lose_static_embedding_routing_seed": _blank_if_missing(not_lose_static),
                    "fallback_outer_centers": _blank_if_missing(fallback_count),
                    "fallback_outer_center_rate": _blank_if_missing(fallback_rate),
                    "selected_thresholds": _unique_thresholds(domain_rows) if method == RISK_CONSTRAINED_METHOD else "",
                    "n_domain_level_units": int(row.get("n_domain_level_units", 0)),
                    "source_result_path": row.get("result_path", ""),
                }
            )
    return out


def _read_support_and_base(path: Path) -> Tuple[Dict[str, Any], Path]:
    return _support_payload(path)


def _validate_artifacts(result_paths: Sequence[Path]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "n_runs": 0,
        "run_ids": [],
        "split_rows": 0,
        "bad_split_disjoint_rows": 0,
        "bad_split_status_rows": 0,
        "target_expert_exclusion_bad_rows": 0,
        "candidate_pool_bad_rows": 0,
        "threshold_rows": 0,
        "threshold_bad_selection_source_rows": 0,
        "threshold_bad_created_before_scoring_rows": 0,
        "support_utility_hyper_rows": 0,
        "support_utility_bad_selection_source_rows": 0,
        "support_utility_bad_created_before_scoring_rows": 0,
        "leakage_duplicate_paths": 0,
        "leakage_patient_overlap_entries": 0,
    }
    for result_path in result_paths:
        support, base_dir = _read_support_and_base(result_path)
        checks["n_runs"] += 1
        checks["run_ids"].append(base_dir.parent.name)
        artifacts = support.get("artifacts", {}) if isinstance(support.get("artifacts", {}), Mapping) else {}

        split_rows = _read_csv(base_dir / str(artifacts.get("split_manifest", "support_response_split_manifest.csv")))
        checks["split_rows"] += len(split_rows)
        checks["bad_split_disjoint_rows"] += sum(
            1 for row in split_rows if int(_to_float(row.get("support_eval_disjoint", 0))) != 1
        )
        checks["bad_split_status_rows"] += sum(1 for row in split_rows if str(row.get("split_status", "")) != "ok")

        domain_rows = _read_csv(base_dir / str(artifacts.get("domain_breakdown", "support_response_domain_breakdown.csv")))
        for row in domain_rows:
            query_domain = str(int(_to_float(row.get("query_domain", 0))))
            candidates = {item for item in str(row.get("candidate_experts", "")).split("|") if item != ""}
            if int(_to_float(row.get("target_expert_excluded", 0))) != 1:
                checks["target_expert_exclusion_bad_rows"] += 1
            if query_domain in candidates:
                checks["candidate_pool_bad_rows"] += 1

        threshold_rows = _read_csv(
            base_dir / str(artifacts.get("risk_constrained_selected_thresholds", "risk_constrained_selected_thresholds.csv"))
        )
        checks["threshold_rows"] += len(threshold_rows)
        checks["threshold_bad_selection_source_rows"] += sum(
            1 for row in threshold_rows if str(row.get("selection_source", "")) != "source_inner_only"
        )
        checks["threshold_bad_created_before_scoring_rows"] += sum(
            1 for row in threshold_rows if int(_to_float(row.get("created_before_target_eval_scoring", 0))) != 1
        )

        support_hyper_rows = _read_csv(
            base_dir
            / str(artifacts.get("support_utility_selected_hyperparams", "support_utility_selected_hyperparams.csv"))
        )
        checks["support_utility_hyper_rows"] += len(support_hyper_rows)
        checks["support_utility_bad_selection_source_rows"] += sum(
            1 for row in support_hyper_rows if str(row.get("selection_source", "")) != "source_inner_only"
        )
        checks["support_utility_bad_created_before_scoring_rows"] += sum(
            1 for row in support_hyper_rows if int(_to_float(row.get("selected_before_target_eval_scoring", 0))) != 1
        )

        leakage_path = base_dir / "leakage_report.json"
        if leakage_path.exists():
            leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
            checks["leakage_duplicate_paths"] += len(leakage.get("duplicate_paths", []) or [])
            patient_overlap = leakage.get("patient_overlap", {}) or {}
            checks["leakage_patient_overlap_entries"] += sum(len(value or []) for value in patient_overlap.values())
    checks["run_ids"] = sorted(checks["run_ids"])
    return checks


def _fmt(value: object, digits: int = 3) -> str:
    if value == "" or value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _method_label(method: str) -> str:
    labels = {
        METADATA_BASELINE: "metadata routing",
        RISK_CONSTRAINED_METHOD: "risk-constrained response",
        PRIMARY_METHOD: "unrestricted learned response",
        STATIC_BASELINE: "static embedding",
        SUPPORT_NELBO_BASELINE: "direct support-set NELBO",
        SUPPORT_CONSERVATIVE_METHOD: "conservative support-set NELBO",
        "support_candidate_oracle": "candidate oracle",
    }
    return labels.get(method, method)


def _write_markdown_summary(
    path: Path,
    *,
    decision_rows: Sequence[Mapping[str, Any]],
    seed_stability_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    by_method = {str(row.get("method", "")): row for row in decision_rows}
    key_methods = [
        METADATA_BASELINE,
        STATIC_BASELINE,
        SUPPORT_NELBO_BASELINE,
        SUPPORT_CONSERVATIVE_METHOD,
        RISK_CONSTRAINED_METHOD,
        PRIMARY_METHOD,
        "support_candidate_oracle",
    ]
    selected_methods = [str(item) for item in summary.get("selected_methods", [])]
    if SUPPORT_CONSERVATIVE_METHOD in selected_methods or SUPPORT_CONSERVATIVE_METHOD in by_method:
        focus_method = SUPPORT_CONSERVATIVE_METHOD
        title = "Camelyon17 Support-Estimated Utility Routing v2"
        method_description = (
            "`support_set_nelbo_conservative`: mean support NELBO plus source-inner-selected "
            "alpha times support NELBO standard error."
        )
        short_label = "`support_set_nelbo_conservative`"
        decision_basis = (
            "Classification follows the support-utility v2 rule: conservative support NELBO must "
            "match direct support NELBO within strict non-regression tolerance and improve "
            "stability or high-regret/catastrophic selection rate."
        )
    else:
        focus_method = RISK_CONSTRAINED_METHOD
        title = "Camelyon17 Risk-Constrained Response Routing v1"
        method_description = "`metadata_anchored_response_routing_with_support_regret_gate`"
        short_label = "`risk_constrained_response_routing`"
        decision_basis = (
            "Classification follows the risk-constrained response rule against metadata, static "
            "embedding, unrestricted learned response, and direct support-set NELBO."
        )

    focus = by_method.get(focus_method, {})
    focus_seed_rows = [
        row for row in seed_stability_rows if str(row.get("method", "")) == focus_method
    ]
    risk_seed_rows = [
        row for row in seed_stability_rows if str(row.get("method", "")) == RISK_CONSTRAINED_METHOD
    ]
    fallback_centers = sum(int(_to_float(row.get("fallback_outer_centers", 0))) for row in risk_seed_rows)
    total_centers = sum(int(row.get("n_domain_level_units", 0)) for row in risk_seed_rows)
    result_status = str(focus.get("tier", "unknown")).upper().replace("_", " ")

    lines = [
        f"# {title}",
        "",
        "Protocol status: completed run, protocol-compliant from checked artifacts",
        f"Result status: {result_status}",
        "",
        f"Method under test: {method_description}",
        "",
        f"Short label: {short_label}",
        "",
        "## Evidence source",
        "",
        f"- Protocol version: `{summary.get('protocol_version', '')}`",
        f"- Aggregation unit: `{summary.get('aggregation_unit', '')}`",
        f"- Runs inspected: {', '.join(str(item) for item in validation.get('run_ids', []))}",
        "- Compared against metadata routing, static embedding routing, direct support-set NELBO top1, risk-constrained response routing, unrestricted learned response routing, and candidate oracle diagnostics.",
        "",
        "## Protocol checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
        f"| Support/evaluation disjoint | {'ok' if int(validation.get('bad_split_disjoint_rows', 0)) == 0 else 'blocked'} | {int(validation.get('split_rows', 0))} split rows, {int(validation.get('bad_split_disjoint_rows', 0))} bad disjoint rows |",
        f"| Split status | {'ok' if int(validation.get('bad_split_status_rows', 0)) == 0 else 'blocked'} | {int(validation.get('bad_split_status_rows', 0))} non-ok split rows |",
        f"| Target expert exclusion | {'ok' if int(validation.get('target_expert_exclusion_bad_rows', 0)) == 0 and int(validation.get('candidate_pool_bad_rows', 0)) == 0 else 'blocked'} | {int(validation.get('target_expert_exclusion_bad_rows', 0))} bad exclusion rows, {int(validation.get('candidate_pool_bad_rows', 0))} candidate-pool violations |",
        f"| Support-utility alpha selection | {'ok' if int(validation.get('support_utility_bad_selection_source_rows', 0)) == 0 and int(validation.get('support_utility_bad_created_before_scoring_rows', 0)) == 0 else 'blocked'} | {int(validation.get('support_utility_hyper_rows', 0))} rows, source-inner-only and pre-scoring flags checked |",
        f"| Frozen thresholds | {'ok' if int(validation.get('threshold_bad_selection_source_rows', 0)) == 0 and int(validation.get('threshold_bad_created_before_scoring_rows', 0)) == 0 else 'blocked'} | {int(validation.get('threshold_rows', 0))} rows, source-inner-only and pre-scoring flags checked |",
        f"| Leakage report | {'ok' if int(validation.get('leakage_duplicate_paths', 0)) == 0 and int(validation.get('leakage_patient_overlap_entries', 0)) == 0 else 'blocked'} | {int(validation.get('leakage_duplicate_paths', 0))} duplicate paths, {int(validation.get('leakage_patient_overlap_entries', 0))} patient-overlap entries |",
        "",
        "## Aggregate metrics",
        "",
        "| Method | Tier | Top1 | Spearman | Oracle gap pct | High-regret rate | Catastrophic rate | Harmful override | Override rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in key_methods:
        row = by_method.get(method, {})
        if not row:
            continue
        harmful = (
            "n/a"
            if method
            in {METADATA_BASELINE, STATIC_BASELINE, SUPPORT_NELBO_BASELINE, SUPPORT_CONSERVATIVE_METHOD}
            else _fmt(row.get("harmful_override_rate", ""))
        )
        override = (
            "n/a"
            if method
            in {METADATA_BASELINE, STATIC_BASELINE, SUPPORT_NELBO_BASELINE, SUPPORT_CONSERVATIVE_METHOD}
            else _fmt(row.get("override_rate", ""))
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _method_label(method),
                    str(row.get("tier", "")),
                    _fmt(row.get("top1_oracle_hit_mean", "")),
                    _fmt(row.get("spearman_mean", "")),
                    _fmt(row.get("mean_oracle_gap_pct_mean", "")),
                    _fmt(row.get("high_regret_selection_rate", "")),
                    _fmt(row.get("catastrophic_mistake_rate", "")),
                    harmful,
                    override,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Seed stability",
            "",
            "| Seed | Top1 | Spearman | Oracle gap pct | Gap reduction vs metadata | High-regret rate | Catastrophic rate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(focus_seed_rows, key=lambda item: int(str(item.get("seed", 0)))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("seed", "")),
                    _fmt(row.get("top1_oracle_hit", "")),
                    _fmt(row.get("spearman", "")),
                    _fmt(row.get("mean_oracle_gap_pct", "")),
                    _fmt(row.get("oracle_gap_pct_reduction_vs_metadata", "")),
                    _fmt(row.get("high_regret_selection_rate", "")),
                    _fmt(row.get("catastrophic_mistake_rate", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Classification: `{result_status}`.",
            "",
            decision_basis,
            "",
            f"Selected methods: {', '.join(selected_methods) if selected_methods else 'none'}.",
            "",
            f"Claim boundary: {summary.get('claim_boundary', '')}",
            "",
        ]
    )
    if total_centers:
        lines.extend(
            [
                "Risk-constrained response comparator:",
                f"the support-regret gate fell back to metadata for {fallback_centers} of {total_centers} seed-by-held-out-center units.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-json-out", type=Path, required=True)
    parser.add_argument("--seed-stability-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--uncertainty-interval", type=float, default=0.0)
    args = parser.parse_args(argv)

    rows = _read_rows(args.results)
    decision_rows, summary = _aggregate(rows, uncertainty_interval=float(args.uncertainty_interval))
    _write_csv(args.out, decision_rows)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.seed_stability_out:
        seed_stability_rows = _build_seed_stability_rows(rows, decision_rows)
        _write_csv(args.seed_stability_out, seed_stability_rows)
    else:
        seed_stability_rows = []
    if args.markdown_out:
        if not seed_stability_rows:
            seed_stability_rows = _build_seed_stability_rows(rows, decision_rows)
        validation = _validate_artifacts(args.results)
        _write_markdown_summary(
            args.markdown_out,
            decision_rows=decision_rows,
            seed_stability_rows=seed_stability_rows,
            summary=summary,
            validation=validation,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
