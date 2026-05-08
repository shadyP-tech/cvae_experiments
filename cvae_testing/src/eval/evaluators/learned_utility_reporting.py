from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_protocol import (
    _AGGREGATION_SOURCE,
    _CANDIDATE_EXPERT_ORDER,
    _CANDIDATE_POLICY,
    _LEARNED_PAIR_POLICY,
    _METRIC_AGGREGATION_POLICY,
    _MIN_CANDIDATES_FOR_RANK_METRICS,
    _ORACLE_POLICY,
    _PAIRWISE_AUC_NAN_POLICY,
    _PROTOCOL_VERSION,
    _SPEARMAN_NAN_POLICY,
    _aggregate_metrics_from_sample_rows,
    _domain_breakdown_rows,
    _method_protocol,
)


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else 0.0


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.std(vals)) if vals else 0.0


def _empirical_p_value(observed: float, null_values: Sequence[float], higher_is_better: bool) -> float:
    vals = [float(v) for v in null_values]
    if not vals:
        return 1.0
    if higher_is_better:
        count = sum(1 for v in vals if v >= float(observed))
    else:
        count = sum(1 for v in vals if v <= float(observed))
    return float((count + 1.0) / (len(vals) + 1.0))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_permutation_rows(
    *,
    permutation_sample_rows: Dict[Tuple[str, int], List[Dict[str, Any]]],
    aggregate_metrics: Callable[[Sequence[Dict[str, Any]]], Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    permutation_rows: List[Dict[str, Any]] = []
    for (null_type, rep), rows in sorted(permutation_sample_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        perm_metrics = aggregate_metrics(rows).get(str(null_type), {})
        permutation_rows.append(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "null_type": str(null_type),
                "repeat": int(rep),
                "method_role": "control",
                "adoption_eligible": 0,
                "diagnostic_only": 0,
                "top1_oracle_hit": float(perm_metrics.get("top1_oracle_hit", float("nan"))),
                "spearman": float(perm_metrics.get("spearman", float("nan"))),
                "mean_oracle_gap_pct": float(perm_metrics.get("mean_oracle_gap_pct", float("nan"))),
                "n_samples_micro": float(perm_metrics.get("n_samples_micro", 0.0)),
                "n_query_domains_macro": float(perm_metrics.get("n_query_domains_macro", 0.0)),
            }
        )
    return permutation_rows


def _summarize_permutation_nulls(
    *,
    permutation_rows: Sequence[Dict[str, Any]],
    baseline_top1: float,
    baseline_spearman: float,
    baseline_gap_pct: float,
    random_rank_gap: float,
    random_score_gap: float,
) -> Dict[str, Dict[str, Any]]:
    permutation_summary: Dict[str, Dict[str, Any]] = {}
    for null_type in sorted(set(str(r["null_type"]) for r in permutation_rows)):
        rows = [r for r in permutation_rows if str(r["null_type"]) == null_type]
        top1_vals = [float(r["top1_oracle_hit"]) for r in rows if np.isfinite(float(r["top1_oracle_hit"]))]
        spearman_vals = [float(r["spearman"]) for r in rows if np.isfinite(float(r["spearman"]))]
        gap_vals = [float(r["mean_oracle_gap_pct"]) for r in rows if np.isfinite(float(r["mean_oracle_gap_pct"]))]
        permutation_summary[null_type] = {
            "n_repeats": int(len(rows)),
            "top1_mean": _mean(top1_vals),
            "top1_std": _std(top1_vals),
            "spearman_mean": _mean(spearman_vals),
            "spearman_std": _std(spearman_vals),
            "mean_oracle_gap_pct_mean": _mean(gap_vals),
            "mean_oracle_gap_pct_std": _std(gap_vals),
            "p_value_vs_metadata_top1": _empirical_p_value(
                observed=baseline_top1,
                null_values=top1_vals,
                higher_is_better=True,
            ),
            "p_value_vs_metadata_spearman": _empirical_p_value(
                observed=baseline_spearman,
                null_values=spearman_vals,
                higher_is_better=True,
            ),
            "p_value_vs_metadata_gap_pct": _empirical_p_value(
                observed=baseline_gap_pct,
                null_values=gap_vals,
                higher_is_better=False,
            ),
            "delta_vs_metadata_top1": float(baseline_top1 - _mean(top1_vals)),
            "delta_vs_metadata_spearman": float(baseline_spearman - _mean(spearman_vals)),
            "gap_reduction_vs_null_pct": float(_mean(gap_vals) - baseline_gap_pct),
            "delta_vs_random_rank_floor_gap_pct": float(random_rank_gap - _mean(gap_vals)),
            "delta_vs_random_score_floor_gap_pct": float(random_score_gap - _mean(gap_vals)),
        }
    return permutation_summary


def _maybe_plot_hist_with_observed(
    *,
    out_path: Path,
    values: Sequence[float],
    observed: float,
    title: str,
    xlabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if vals.size == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30, alpha=0.7, color="#4C78A8", edgecolor="black")
    plt.axvline(float(observed), color="#D62728", linestyle="--", linewidth=2.0, label="observed")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _maybe_plot_overlay(
    *,
    out_path: Path,
    values_a: Sequence[float],
    label_a: str,
    values_b: Sequence[float],
    label_b: str,
    title: str,
    xlabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    a = np.asarray([float(v) for v in values_a], dtype=np.float64)
    b = np.asarray([float(v) for v in values_b], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(a, bins=40, alpha=0.55, density=True, label=str(label_a), color="#4C78A8")
    plt.hist(b, bins=40, alpha=0.55, density=True, label=str(label_b), color="#F58518")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _write_permutation_distribution_plots(
    *,
    reports_dir: Path,
    permutation_rows: Sequence[Dict[str, Any]],
    baseline_top1: float,
    baseline_spearman: float,
    baseline_gap_pct: float,
) -> List[str]:
    artifacts: List[str] = []
    for null_type in sorted(set(str(r["null_type"]) for r in permutation_rows)):
        rows = [r for r in permutation_rows if str(r["null_type"]) == null_type]
        for metric_name, observed_value, xlabel in [
            ("top1_oracle_hit", baseline_top1, "top1 oracle hit"),
            ("spearman", baseline_spearman, "spearman"),
            ("mean_oracle_gap_pct", baseline_gap_pct, "mean oracle gap percent"),
        ]:
            out_name = f"learned_utility_dist_{null_type}_{metric_name}.png"
            ok = _maybe_plot_hist_with_observed(
                out_path=reports_dir / out_name,
                values=[float(r[metric_name]) for r in rows],
                observed=float(observed_value),
                title=f"{null_type} distribution: {metric_name}",
                xlabel=xlabel,
            )
            if ok:
                artifacts.append(out_name)
    return artifacts


def _write_best_gap_overlay(
    *,
    reports_dir: Path,
    sample_rows: Sequence[Dict[str, Any]],
    uplift_reference_method: str,
    best_candidate_method: str,
) -> List[str]:
    if not best_candidate_method:
        return []
    baseline_vals = [
        float(r["oracle_gap_pct"])
        for r in sample_rows
        if str(r.get("method", "")) == str(uplift_reference_method)
    ]
    best_vals = [
        float(r["oracle_gap_pct"])
        for r in sample_rows
        if str(r.get("method", "")) == str(best_candidate_method)
    ]
    overlay_name = "learned_utility_overlay_gap_pct_baseline_vs_best.png"
    ok_overlay = _maybe_plot_overlay(
        out_path=reports_dir / overlay_name,
        values_a=baseline_vals,
        label_a=str(uplift_reference_method),
        values_b=best_vals,
        label_b=str(best_candidate_method),
        title="Oracle gap percent: baseline vs best learned",
        xlabel="oracle gap percent",
    )
    return [overlay_name] if ok_overlay else []


def _build_seed_gate_by_method(
    *,
    method_metrics: Dict[str, Dict[str, Any]],
    uplift_reference_method: str,
    strong_spearman_uplift: float,
    strong_top1_uplift: float,
    strong_gap_reduction: float,
    weak_spearman_uplift: float,
    weak_top1_uplift: float,
    weak_gap_reduction: float,
) -> Dict[str, Dict[str, Any]]:
    candidate_methods = sorted(
        method
        for method, metrics in method_metrics.items()
        if int(float(metrics.get("adoption_eligible", 0.0))) == 1
        and int(float(metrics.get("diagnostic_only", 0.0))) == 0
        and int(float(metrics.get("routing_uses_eval_nelbo", 0.0))) == 0
        and int(float(metrics.get("routing_uses_eval_domain_statistics", 0.0))) == 0
        and str(metrics.get("method_role", "")) == "learned"
    )

    baseline_metrics = method_metrics.get(uplift_reference_method, method_metrics.get("metadata_routing", {}))
    baseline_top1_gate = float(baseline_metrics.get("top1_oracle_hit", 0.0))
    baseline_spearman_gate = float(baseline_metrics.get("spearman", 0.0))
    baseline_gap_pct_gate = float(baseline_metrics.get("mean_oracle_gap_pct", 0.0))

    seed_gate_by_method: Dict[str, Dict[str, Any]] = {}
    for method in sorted(candidate_methods):
        mm = method_metrics.get(method, {})
        top1_uplift = float(mm.get("top1_oracle_hit", 0.0)) - baseline_top1_gate
        spearman_uplift = float(mm.get("spearman", 0.0)) - baseline_spearman_gate
        gap_pct_reduction = baseline_gap_pct_gate - float(mm.get("mean_oracle_gap_pct", 0.0))

        strong_pass = bool(
            spearman_uplift >= strong_spearman_uplift
            and top1_uplift >= strong_top1_uplift
            and gap_pct_reduction >= strong_gap_reduction
        )
        weak_pass = bool(
            spearman_uplift >= weak_spearman_uplift
            and top1_uplift >= weak_top1_uplift
            and gap_pct_reduction >= weak_gap_reduction
        )

        if strong_pass:
            tier = "strong_pass_seed"
        elif weak_pass:
            tier = "weak_pass_seed"
        else:
            tier = "fail_seed"

        seed_gate_by_method[method] = {
            "tier": str(tier),
            "uplift_reference_method": str(uplift_reference_method),
            "spearman_uplift": float(spearman_uplift),
            "top1_uplift": float(top1_uplift),
            "oracle_gap_pct_reduction": float(gap_pct_reduction),
            "strong_thresholds": {
                "spearman_uplift_min": float(strong_spearman_uplift),
                "top1_uplift_min": float(strong_top1_uplift),
                "oracle_gap_pct_reduction_min": float(strong_gap_reduction),
            },
            "weak_thresholds": {
                "spearman_uplift_min": float(weak_spearman_uplift),
                "top1_uplift_min": float(weak_top1_uplift),
                "oracle_gap_pct_reduction_min": float(weak_gap_reduction),
            },
        }
    return seed_gate_by_method


def _select_best_methods_by_gap(
    method_metrics: Dict[str, Dict[str, Any]],
    *,
    uplift_reference_method: str,
) -> Tuple[str, str]:
    best_candidate_method = ""
    adoption_methods = sorted(
        method
        for method, metrics in method_metrics.items()
        if str(method) != str(uplift_reference_method)
        and int(float(metrics.get("adoption_eligible", 0.0))) == 1
        and int(float(metrics.get("diagnostic_only", 0.0))) == 0
        and int(float(metrics.get("routing_uses_eval_nelbo", 0.0))) == 0
        and int(float(metrics.get("routing_uses_eval_domain_statistics", 0.0))) == 0
    )
    if adoption_methods:
        best_candidate_method = str(
            min(
                adoption_methods,
                key=lambda m: float(method_metrics.get(m, {}).get("mean_oracle_gap_pct", 1e12)),
            )
        )

    best_diagnostic_method = ""
    diagnostic_methods = sorted(
        method
        for method, metrics in method_metrics.items()
        if int(float(metrics.get("diagnostic_only", 0.0))) == 1
    )
    if diagnostic_methods:
        best_diagnostic_method = str(
            min(
                diagnostic_methods,
                key=lambda m: float(method_metrics.get(m, {}).get("mean_oracle_gap_pct", 1e12)),
            )
        )
    return best_candidate_method, best_diagnostic_method


def _build_hybrid_summary_rows(
    *,
    hybrid_method_meta: Dict[str, Dict[str, Any]],
    method_metrics: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hybrid_summary_rows: List[Dict[str, Any]] = []
    for method_name, meta in sorted(hybrid_method_meta.items()):
        metrics = method_metrics.get(method_name, {})
        hybrid_summary_rows.append(
            {
                "protocol_version": _PROTOCOL_VERSION,
                "method": str(method_name),
                "alpha": float(meta["alpha"]),
                "normalization_policy": str(meta["normalization_policy"]),
                "method_role": "diagnostic",
                "adoption_eligible": 0,
                "diagnostic_only": 1,
                "routing_uses_eval_domain_statistics": 1,
                "top1_oracle_hit": float(metrics.get("top1_oracle_hit", float("nan"))),
                "mean_rank": float(metrics.get("mean_rank", float("nan"))),
                "mean_oracle_gap": float(metrics.get("mean_oracle_gap", float("nan"))),
                "mean_oracle_gap_pct": float(metrics.get("mean_oracle_gap_pct", float("nan"))),
                "pairwise_auc": float(metrics.get("pairwise_auc", float("nan"))),
                "spearman": float(metrics.get("spearman", float("nan"))),
            }
        )
    return hybrid_summary_rows


def _build_method_summary_rows(method_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in sorted(method_metrics.items()):
        rows.append(
            {
                "protocol_version": str(metrics.get("protocol_version", _PROTOCOL_VERSION)),
                "method": str(method),
                "method_role": str(metrics.get("method_role", _method_protocol(method).method_role)),
                "adoption_eligible": int(float(metrics.get("adoption_eligible", 0.0))),
                "diagnostic_only": int(float(metrics.get("diagnostic_only", 0.0))),
                "routing_uses_query_features": int(float(metrics.get("routing_uses_query_features", 0.0))),
                "routing_uses_eval_domain_statistics": int(
                    float(metrics.get("routing_uses_eval_domain_statistics", 0.0))
                ),
                "routing_uses_eval_nelbo": int(float(metrics.get("routing_uses_eval_nelbo", 0.0))),
                "top1_oracle_hit": float(metrics.get("top1_oracle_hit", float("nan"))),
                "mean_rank": float(metrics.get("mean_rank", float("nan"))),
                "mean_oracle_gap": float(metrics.get("mean_oracle_gap", float("nan"))),
                "mean_oracle_gap_pct": float(metrics.get("mean_oracle_gap_pct", float("nan"))),
                "pairwise_auc": float(metrics.get("pairwise_auc", float("nan"))),
                "spearman": float(metrics.get("spearman", float("nan"))),
                "selected_nelbo": float(metrics.get("selected_nelbo", float("nan"))),
                "candidate_oracle_nelbo": float(metrics.get("candidate_oracle_nelbo", float("nan"))),
                "n_samples_micro": float(metrics.get("n_samples_micro", 0.0)),
                "n_query_domains_macro": float(metrics.get("n_query_domains_macro", 0.0)),
                "n_valid_spearman_samples": float(metrics.get("n_valid_spearman_samples", 0.0)),
                "n_valid_auc_samples": float(metrics.get("n_valid_auc_samples", 0.0)),
            }
        )
    return rows


def _build_hybrid_diagnostics(
    *,
    hybrid_summary_rows: Sequence[Dict[str, Any]],
    method_metrics: Dict[str, Dict[str, Any]],
    primary_norm_policy: str,
    sensitivity_norm_policy: str,
    run_sensitivity: bool,
    min_rank_improvement_abs: float,
    min_gap_pct_improvement_abs: float,
    max_top1_drop_abs: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    hybrid_best_by_policy: Dict[str, Dict[str, Any]] = {}
    hybrid_acceptance: Dict[str, Any] = {}
    if not hybrid_summary_rows:
        return hybrid_best_by_policy, hybrid_acceptance

    by_policy: Dict[str, List[Dict[str, Any]]] = {}
    for row in hybrid_summary_rows:
        by_policy.setdefault(str(row["normalization_policy"]), []).append(dict(row))
    for policy, rows in by_policy.items():
        ordered = sorted(
            rows,
            key=lambda r: (
                float(r["mean_oracle_gap_pct"]),
                float(r["mean_rank"]),
                -float(r["top1_oracle_hit"]),
            ),
        )
        hybrid_best_by_policy[policy] = {
            "best_method": str(ordered[0]["method"]),
            "best_alpha": float(ordered[0]["alpha"]),
            "ranking": ordered,
        }

    metadata_metrics = method_metrics.get("metadata_routing", {})
    baseline_top1 = float(metadata_metrics.get("top1_oracle_hit", 0.0))
    baseline_rank = float(metadata_metrics.get("mean_rank", 0.0))
    baseline_gap_pct = float(metadata_metrics.get("mean_oracle_gap_pct", 0.0))

    def _with_deltas(row: Dict[str, Any]) -> Dict[str, Any]:
        cand_top1 = float(row.get("top1_oracle_hit", 0.0))
        cand_rank = float(row.get("mean_rank", 0.0))
        cand_gap_pct = float(row.get("mean_oracle_gap_pct", 0.0))
        top1_delta = cand_top1 - baseline_top1
        rank_improvement = baseline_rank - cand_rank
        gap_pct_improvement = baseline_gap_pct - cand_gap_pct
        non_inferior_top1 = top1_delta >= -float(max_top1_drop_abs)
        efficacy_ok = (
            (rank_improvement >= float(min_rank_improvement_abs))
            or (gap_pct_improvement >= float(min_gap_pct_improvement_abs))
        )
        return {
            **row,
            "delta_vs_metadata_top1": float(top1_delta),
            "improvement_vs_metadata_mean_rank": float(rank_improvement),
            "improvement_vs_metadata_mean_oracle_gap_pct": float(gap_pct_improvement),
            "non_inferior_top1": bool(non_inferior_top1),
            "meets_effect_size_gate": bool(efficacy_ok),
            "passes_acceptance_gate": bool(non_inferior_top1 and efficacy_ok),
        }

    ranked_primary = hybrid_best_by_policy.get(primary_norm_policy, {}).get("ranking", [])
    ranked_sensitivity = hybrid_best_by_policy.get(sensitivity_norm_policy, {}).get("ranking", [])
    ranked_primary_delta = [_with_deltas(r) for r in ranked_primary]

    sensitivity_by_alpha: Dict[float, Dict[str, Any]] = {
        float(r.get("alpha", -1.0)): r for r in ranked_sensitivity
    }
    primary_best = ranked_primary_delta[0] if ranked_primary_delta else None
    sensitivity_match = None
    sensitivity_consistent = False
    if primary_best is not None and run_sensitivity:
        sensitivity_match = sensitivity_by_alpha.get(float(primary_best.get("alpha", -1.0)))
        if sensitivity_match is not None:
            sensitivity_consistent = bool(
                _with_deltas(sensitivity_match).get("passes_acceptance_gate", False)
            )

    hybrid_acceptance = {
        "thresholds": {
            "min_mean_rank_improvement_abs": float(min_rank_improvement_abs),
            "min_mean_oracle_gap_pct_improvement_abs": float(min_gap_pct_improvement_abs),
            "max_top1_drop_abs": float(max_top1_drop_abs),
        },
        "baseline_metadata": {
            "top1_oracle_hit": baseline_top1,
            "mean_rank": baseline_rank,
            "mean_oracle_gap_pct": baseline_gap_pct,
        },
        "primary_normalization_policy": str(primary_norm_policy),
        "best_primary": primary_best,
        "best_primary_sensitivity_match": sensitivity_match,
        "best_primary_passes_sensitivity_gate": bool(sensitivity_consistent),
        "primary_policy_ranking_with_deltas": ranked_primary_delta,
        "adoption_eligible": False,
        "not_adoption_eligible_reason": "hybrid methods use target evaluation-domain latent statistics in v2",
    }
    return hybrid_best_by_policy, hybrid_acceptance


def _finalize_learned_utility_outputs(
    *,
    reports_dir: Path,
    sample_rows: Sequence[Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    pair_training_rows: Sequence[Dict[str, Any]],
    proxy_diag_rows: Sequence[Dict[str, Any]],
    permutation_sample_rows: Dict[Tuple[str, int], List[Dict[str, Any]]],
    hybrid_method_meta: Dict[str, Dict[str, Any]],
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    save_distribution_plots: bool,
    uplift_reference_method: str,
    strong_spearman_uplift: float,
    strong_top1_uplift: float,
    strong_gap_reduction: float,
    weak_spearman_uplift: float,
    weak_top1_uplift: float,
    weak_gap_reduction: float,
    decision_policy_version: str,
    instability_std_threshold: float,
    top1_uplift_std_threshold: float,
    spearman_uplift_std_threshold: float,
    gap_pct_reduction_std_threshold: float,
    instability_sign_inconsistency_min_count: int,
    min_positive_fraction: float,
    ci_level: float,
    ci_bootstrap_reps: int,
    ci_bootstrap_seed: int,
    allow_missing_domain_breakdown_as_diagnostic: bool,
    hybrid_enabled: bool,
    tie_policy: str,
    primary_norm_policy: str,
    sensitivity_norm_policy: str,
    run_sensitivity: bool,
    min_rank_improvement_abs: float,
    min_gap_pct_improvement_abs: float,
    max_top1_drop_abs: float,
    enable_random_rank_floor: bool,
    enable_random_score_floor: bool,
    run_expert_label_permutation: bool,
    run_metadata_permutation: bool,
    permutation_repeats: int,
) -> Dict[str, Any]:
    method_metrics = _aggregate_metrics_from_sample_rows(sample_rows)
    domain_rows = _domain_breakdown_rows(sample_rows)

    permutation_rows = _build_permutation_rows(
        permutation_sample_rows=permutation_sample_rows,
        aggregate_metrics=_aggregate_metrics_from_sample_rows,
    )

    baseline_for_nulls = method_metrics.get("metadata_routing", {})
    baseline_top1 = float(baseline_for_nulls.get("top1_oracle_hit", 0.0))
    baseline_spearman = float(baseline_for_nulls.get("spearman", 0.0))
    baseline_gap_pct = float(baseline_for_nulls.get("mean_oracle_gap_pct", 0.0))
    random_rank_gap = float(method_metrics.get("random_rank_floor", {}).get("mean_oracle_gap_pct", 0.0))
    random_score_gap = float(method_metrics.get("random_score_floor", {}).get("mean_oracle_gap_pct", 0.0))
    permutation_summary = _summarize_permutation_nulls(
        permutation_rows=permutation_rows,
        baseline_top1=baseline_top1,
        baseline_spearman=baseline_spearman,
        baseline_gap_pct=baseline_gap_pct,
        random_rank_gap=random_rank_gap,
        random_score_gap=random_score_gap,
    )

    hybrid_summary_rows = _build_hybrid_summary_rows(
        hybrid_method_meta=hybrid_method_meta,
        method_metrics=method_metrics,
    )
    method_summary_rows = _build_method_summary_rows(method_metrics)

    _write_csv(reports_dir / "learned_utility_pair_predictions.csv", pair_rows)
    _write_csv(reports_dir / "learned_utility_sample_selections.csv", sample_rows)
    _write_csv(reports_dir / "learned_utility_domain_breakdown.csv", domain_rows)
    _write_csv(reports_dir / "learned_utility_pair_training_diagnostics.csv", pair_training_rows)
    _write_csv(reports_dir / "learned_utility_proxy_diagnostics.csv", proxy_diag_rows)
    _write_csv(reports_dir / "learned_utility_method_summary.csv", method_summary_rows)
    if permutation_rows:
        _write_csv(reports_dir / "learned_utility_permutation_nulls.csv", permutation_rows)

    diagnostic_plot_artifacts: List[str] = []
    if save_distribution_plots and permutation_rows:
        diagnostic_plot_artifacts.extend(
            _write_permutation_distribution_plots(
                reports_dir=reports_dir,
                permutation_rows=permutation_rows,
                baseline_top1=baseline_top1,
                baseline_spearman=baseline_spearman,
                baseline_gap_pct=baseline_gap_pct,
            )
        )

    seed_gate_by_method = _build_seed_gate_by_method(
        method_metrics=method_metrics,
        uplift_reference_method=str(uplift_reference_method),
        strong_spearman_uplift=strong_spearman_uplift,
        strong_top1_uplift=strong_top1_uplift,
        strong_gap_reduction=strong_gap_reduction,
        weak_spearman_uplift=weak_spearman_uplift,
        weak_top1_uplift=weak_top1_uplift,
        weak_gap_reduction=weak_gap_reduction,
    )
    best_candidate_method, best_diagnostic_method = _select_best_methods_by_gap(
        method_metrics,
        uplift_reference_method=str(uplift_reference_method),
    )

    if save_distribution_plots and best_candidate_method:
        diagnostic_plot_artifacts.extend(
            _write_best_gap_overlay(
                reports_dir=reports_dir,
                sample_rows=sample_rows,
                uplift_reference_method=str(uplift_reference_method),
                best_candidate_method=str(best_candidate_method),
            )
        )

    hybrid_best_by_policy, hybrid_acceptance = _build_hybrid_diagnostics(
        hybrid_summary_rows=hybrid_summary_rows,
        method_metrics=method_metrics,
        primary_norm_policy=primary_norm_policy,
        sensitivity_norm_policy=sensitivity_norm_policy,
        run_sensitivity=run_sensitivity,
        min_rank_improvement_abs=min_rank_improvement_abs,
        min_gap_pct_improvement_abs=min_gap_pct_improvement_abs,
        max_top1_drop_abs=max_top1_drop_abs,
    )
    if hybrid_summary_rows:
        _write_csv(reports_dir / "learned_utility_hybrid_alpha_summary.csv", hybrid_summary_rows)

    return {
        "metrics_by_method": method_metrics,
        "artifacts": {
            "pair_predictions": "learned_utility_pair_predictions.csv",
            "sample_selections": "learned_utility_sample_selections.csv",
            "domain_breakdown": "learned_utility_domain_breakdown.csv",
            "pair_training_diagnostics": "learned_utility_pair_training_diagnostics.csv",
            "proxy_diagnostics": "learned_utility_proxy_diagnostics.csv",
            "method_summary": "learned_utility_method_summary.csv",
            "permutation_nulls": "learned_utility_permutation_nulls.csv" if permutation_rows else "",
            "diagnostic_plots": diagnostic_plot_artifacts,
            "hybrid_alpha_summary": "learned_utility_hybrid_alpha_summary.csv" if hybrid_summary_rows else "",
        },
        "protocol_version": _PROTOCOL_VERSION,
        "protocol_contract": {
            "protocol_version": _PROTOCOL_VERSION,
            "candidate_policy": _CANDIDATE_POLICY,
            "candidate_expert_order": _CANDIDATE_EXPERT_ORDER,
            "oracle_policy": _ORACLE_POLICY,
            "learned_pair_policy": _LEARNED_PAIR_POLICY,
            "metric_aggregation_policy": _METRIC_AGGREGATION_POLICY,
            "aggregation_source": _AGGREGATION_SOURCE,
            "global_oracle_used_for_metrics": False,
            "metrics_comparable_to_previous_protocol": False,
            "previous_protocol_invalidated_by_target_candidate_leakage": True,
            "spearman_nan_policy": _SPEARMAN_NAN_POLICY,
            "pairwise_auc_nan_policy": _PAIRWISE_AUC_NAN_POLICY,
            "min_candidates_for_rank_metrics": _MIN_CANDIDATES_FOR_RANK_METRICS,
        },
        "compatibility_protocol": {
            "uplift_reference_method": str(uplift_reference_method),
            "floors": {
                "random_rank_floor_enabled": bool(enable_random_rank_floor),
                "random_score_floor_enabled": bool(enable_random_score_floor),
            },
            "permutation_tests": {
                "expert_label_permutation": bool(run_expert_label_permutation),
                "metadata_permutation": bool(run_metadata_permutation),
                "repeats": int(permutation_repeats),
                "summary": permutation_summary,
            },
            "gate": {
                "decision_policy_version": str(decision_policy_version),
                "seed_level": seed_gate_by_method,
                "strong": {
                    "spearman_uplift_min": float(strong_spearman_uplift),
                    "top1_uplift_min": float(strong_top1_uplift),
                    "oracle_gap_pct_reduction_min": float(strong_gap_reduction),
                },
                "weak": {
                    "spearman_uplift_min": float(weak_spearman_uplift),
                    "top1_uplift_min": float(weak_top1_uplift),
                    "oracle_gap_pct_reduction_min": float(weak_gap_reduction),
                },
                "instability": {
                    "std_threshold": float(instability_std_threshold),
                    "top1_uplift_std_threshold": float(top1_uplift_std_threshold),
                    "spearman_uplift_std_threshold": float(spearman_uplift_std_threshold),
                    "gap_pct_reduction_std_threshold": float(gap_pct_reduction_std_threshold),
                    "sign_inconsistency_min_count": int(instability_sign_inconsistency_min_count),
                    "min_positive_fraction": float(min_positive_fraction),
                    "ci_level": float(ci_level),
                    "ci_bootstrap_reps": int(ci_bootstrap_reps),
                    "ci_bootstrap_seed": int(ci_bootstrap_seed),
                    "allow_missing_domain_breakdown_as_diagnostic": bool(
                        allow_missing_domain_breakdown_as_diagnostic
                    ),
                    "note": "Instability is evaluated across seeds in aggregated decision-table stage.",
                },
            },
            "best_candidate_method_by_gap_pct": str(best_candidate_method),
            "best_diagnostic_method_by_gap_pct": str(best_diagnostic_method),
        },
        "hybrid_diagnostics": {
            "enabled": bool(hybrid_enabled),
            "tie_policy": str(tie_policy),
            "best_by_normalization_policy": hybrid_best_by_policy,
            "acceptance": hybrid_acceptance,
        },
        "n_samples": int(sample_domains.shape[0]),
        "n_experts": int(len(expert_domains)),
        "expert_domains": [int(v) for v in expert_domains],
    }
