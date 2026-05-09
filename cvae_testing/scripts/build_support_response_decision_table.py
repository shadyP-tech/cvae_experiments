#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
                    "run_id": str(path.parent.name),
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
                "uses_no_direct_support_utility_terms": int(no_direct_support_utility_terms),
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
        "risk_constrained_method": RISK_CONSTRAINED_METHOD,
        "selected_methods": selected,
        "claim_boundary": (
            "Risk-constrained response routing is a metadata-anchored learned-response proposal "
            "with a support-NELBO regret gate. Do not claim learned response routing beats metadata "
            "unless the completed run satisfies the predeclared result decision rule."
        ),
        "aggregation_unit": "seed_x_heldout_center_x_support_seed_x_support_size",
    }
    return out, summary


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
    parser.add_argument("--uncertainty-interval", type=float, default=0.0)
    args = parser.parse_args(argv)

    rows = _read_rows(args.results)
    decision_rows, summary = _aggregate(rows, uncertainty_interval=float(args.uncertainty_interval))
    _write_csv(args.out, decision_rows)
    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
