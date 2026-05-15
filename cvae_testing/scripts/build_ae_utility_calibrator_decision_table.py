#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


PRIMARY_METHOD = "ae_utility_calibrated_safe_override_v1"
PRIMARY_METHOD_V11 = "ae_utility_calibrated_precision_lcb_safe_override_v11"
PRIMARY_METHOD_V2 = "ae_utility_calibrated_consensus_safe_override_v2"
AE_ARGMIN_METHOD = "ae_argmin_zscore"
METADATA_METHOD = "metadata_routing"
GLOBAL_BASELINES = {
    "metadata_routing",
    "metadata_ae_residual_safe_override_v1",
    "pairwise_ranker_ae_combined",
    "ae_first_margin_gated_v1",
}
REQUIRED_METHODS = set(GLOBAL_BASELINES) | {
    PRIMARY_METHOD,
    PRIMARY_METHOD_V11,
    PRIMARY_METHOD_V2,
    AE_ARGMIN_METHOD,
    "ae_metadata_utility_calibrated_safe_override_v1",
    "ae_combined_utility_calibrated_safe_override_v1",
    "ae_metadata_utility_calibrated_consensus_safe_override_v2",
    "ae_combined_utility_calibrated_consensus_safe_override_v2",
    "ae_utility_pairwise_ranker_diagnostic_v1",
}
THRESHOLDS = {
    "top1_drop_abs_max": 0.02,
    "spearman_drop_abs_max": 0.03,
    "gap_pct_degradation_pp_max": 1.0,
    "min_active_override_rate_for_weak_pass": 0.10,
    "min_active_override_rate_for_pass": 0.20,
    "min_selected_override_precision_for_weak_pass": 0.50,
    "min_selected_override_precision_for_pass": 0.50,
}


def _read_manifest(path: Path) -> List[Path]:
    with path.open("r", encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip() and not line.startswith("#")]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(rows: Sequence[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_float(row.get(key), default) for row in rows]
    return float(mean(vals)) if vals else float(default)


def _dataset_from_path(path: Path) -> str:
    text = str(path).lower()
    if "breakhis" in text:
        return "breakhis"
    if "camelyon17" in text:
        return "camelyon17"
    return "unknown"


def _load_run_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy_rows = _read_csv(path.parent / "ae_utility_calibrator_policy_audit.csv")
    by_policy = {(row.get("method", ""), row.get("fold_query_domain", "")): row for row in policy_rows}
    dataset = _dataset_from_path(path)
    rows: List[Dict[str, Any]] = []
    for method, metrics in sorted((payload.get("metrics_by_method", {}) or {}).items()):
        if method not in REQUIRED_METHODS:
            continue
        row = {
            "dataset": dataset,
            "result_path": str(path),
            "method": str(method),
            "top1_oracle_hit": _float(metrics.get("macro_top1_oracle_hit_by_query_domain", metrics.get("top1_oracle_hit"))),
            "raw_spearman": _float(metrics.get("macro_spearman_by_query_domain", metrics.get("spearman"))),
            "mean_oracle_gap_pct": _float(
                metrics.get("macro_oracle_gap_pct_by_query_domain", metrics.get("mean_oracle_gap_pct"))
            ),
            "active_override_rate": _float(metrics.get("active_override_rate")),
            "selected_override_precision": _float(metrics.get("selected_override_precision"), float("nan")),
            "net_gain_vs_ae_argmin": _float(metrics.get("net_gain_vs_ae_argmin")),
            "harmful_vs_ae_argmin_rate": _float(metrics.get("harmful_vs_ae_argmin_rate")),
            "improving_vs_ae_argmin_rate": _float(metrics.get("improving_vs_ae_argmin_rate")),
            "diagnostic_only": int(_float(metrics.get("diagnostic_only"))),
            "adoption_eligible": int(_float(metrics.get("adoption_eligible"))),
        }
        if method in {PRIMARY_METHOD, PRIMARY_METHOD_V11, PRIMARY_METHOD_V2}:
            matching = [r for (m, _q), r in by_policy.items() if m == method]
            if matching:
                row["raw_predicted_delta_spearman_non_anchor"] = _mean(
                    matching, "raw_predicted_delta_spearman_non_anchor", default=float("nan")
                )
                row["override_capture_rate"] = _mean(matching, "override_capture_rate", default=float("nan"))
                row["oracle_improvable_query_rate"] = _mean(matching, "oracle_improvable_query_rate", default=float("nan"))
                row["captured_oracle_headroom_rate"] = _mean(matching, "captured_oracle_headroom_rate", default=float("nan"))
                row["abstention_rate"] = _mean(matching, "abstention_rate", default=float("nan"))
                row["abstention_correct_rate"] = _mean(matching, "abstention_correct_rate", default=float("nan"))
                row["strict_improvement_precision"] = _mean(
                    matching, "strict_improvement_precision", default=float("nan")
                )
                row["safe_override_precision"] = _mean(matching, "safe_override_precision", default=float("nan"))
                row["harmful_override_rate"] = _mean(matching, "harmful_override_rate", default=float("nan"))
                row["active_override_rate_heldout"] = _mean(matching, "active_override_rate_heldout", default=float("nan"))
                row["precision_lcb_selected_config_used"] = int(
                    any(str(r.get("selection_status")) == "precision_lcb_selected" for r in matching)
                )
        rows.append(row)
    return rows


def _domain_non_degradation_ok(paths: Sequence[Path], *, dataset: str, method: str, baseline: str) -> bool:
    ok = True
    for path in paths:
        if _dataset_from_path(path) != dataset:
            continue
        rows = _read_csv(path.parent / "learned_utility_domain_breakdown.csv")
        by_key = {(r.get("method", ""), r.get("query_domain", "")): r for r in rows}
        for _m, domain in sorted(k for k in by_key if k[0] == baseline):
            base = by_key.get((baseline, domain))
            cand = by_key.get((method, domain))
            if not base or not cand:
                ok = False
                continue
            if _float(base.get("top1_oracle_hit")) - _float(cand.get("top1_oracle_hit")) > THRESHOLDS["top1_drop_abs_max"]:
                ok = False
            if _float(base.get("spearman")) - _float(cand.get("spearman")) > THRESHOLDS["spearman_drop_abs_max"]:
                ok = False
            if _float(cand.get("mean_oracle_gap_pct")) - _float(base.get("mean_oracle_gap_pct")) > THRESHOLDS["gap_pct_degradation_pp_max"]:
                ok = False
    return ok


def _aggregate(rows: Sequence[Dict[str, Any]], paths: Sequence[Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    verdicts: Dict[str, str] = {}
    for dataset in sorted(set(str(r["dataset"]) for r in rows)):
        dataset_rows = [r for r in rows if str(r["dataset"]) == dataset]
        by_method = {method: [r for r in dataset_rows if r["method"] == method] for method in REQUIRED_METHODS}
        if by_method.get(PRIMARY_METHOD_V11, []):
            primary_method = PRIMARY_METHOD_V11
        elif by_method.get(PRIMARY_METHOD_V2, []):
            primary_method = PRIMARY_METHOD_V2
        else:
            primary_method = PRIMARY_METHOD
        primary = by_method.get(primary_method, [])
        ae_argmin = by_method.get(AE_ARGMIN_METHOD, [])
        metadata = by_method.get(METADATA_METHOD, [])
        if not primary or not ae_argmin or not metadata:
            verdicts[f"{dataset}_local_ae_calibration_verdict"] = "REJECTED"
            verdicts[f"{dataset}_global_adoption_verdict"] = "REJECTED"
            continue

        p_gap = _mean(primary, "mean_oracle_gap_pct")
        p_top1 = _mean(primary, "top1_oracle_hit")
        p_spearman = _mean(primary, "raw_spearman")
        a_gap = _mean(ae_argmin, "mean_oracle_gap_pct")
        a_top1 = _mean(ae_argmin, "top1_oracle_hit")
        a_spearman = _mean(ae_argmin, "raw_spearman")
        m_gap = _mean(metadata, "mean_oracle_gap_pct")
        m_top1 = _mean(metadata, "top1_oracle_hit")
        m_spearman = _mean(metadata, "raw_spearman")
        active = _mean(primary, "active_override_rate")
        precision = _mean(primary, "selected_override_precision", default=float("nan"))
        harmful = _mean(primary, "harmful_vs_ae_argmin_rate")
        improving = _mean(primary, "improving_vs_ae_argmin_rate")
        net_gain = _mean(primary, "net_gain_vs_ae_argmin")
        no_ae_degrade = (
            a_top1 - p_top1 <= THRESHOLDS["top1_drop_abs_max"]
            and a_spearman - p_spearman <= THRESHOLDS["spearman_drop_abs_max"]
            and p_gap - a_gap <= THRESHOLDS["gap_pct_degradation_pp_max"]
        )
        no_meta_degrade = (
            m_top1 - p_top1 <= THRESHOLDS["top1_drop_abs_max"]
            and m_spearman - p_spearman <= THRESHOLDS["spearman_drop_abs_max"]
            and p_gap - m_gap <= THRESHOLDS["gap_pct_degradation_pp_max"]
        )
        domain_ok = _domain_non_degradation_ok(paths, dataset=dataset, method=primary_method, baseline=AE_ARGMIN_METHOD)
        if primary_method == PRIMARY_METHOD_V11:
            v1_rows = by_method.get(PRIMARY_METHOD, [])
            if not v1_rows:
                local_pass = False
                local_weak = False
            else:
                v1_gap = _mean(v1_rows, "mean_oracle_gap_pct")
                v1_top1 = _mean(v1_rows, "top1_oracle_hit")
                v1_spearman = _mean(v1_rows, "raw_spearman")
                v1_strict_precision = _mean(v1_rows, "strict_improvement_precision", default=float("nan"))
                v1_harmful = _mean(v1_rows, "harmful_override_rate", default=float("nan"))
                p_strict_precision = _mean(primary, "strict_improvement_precision", default=float("nan"))
                p_harmful = _mean(primary, "harmful_override_rate", default=float("nan"))
                precision_used = any(int(_float(row.get("precision_lcb_selected_config_used"))) == 1 for row in primary)
                no_v1_degrade = (
                    v1_top1 - p_top1 <= THRESHOLDS["top1_drop_abs_max"]
                    and v1_spearman - p_spearman <= THRESHOLDS["spearman_drop_abs_max"]
                    and p_gap - v1_gap <= THRESHOLDS["gap_pct_degradation_pp_max"]
                )
                active_ok = active >= 0.05
                precision_improved = (
                    math.isfinite(p_strict_precision)
                    and (not math.isfinite(v1_strict_precision) or p_strict_precision > v1_strict_precision)
                )
                harm_reduced = (
                    math.isfinite(p_harmful)
                    and (not math.isfinite(v1_harmful) or p_harmful < v1_harmful)
                )
                utility_ok = (p_gap < v1_gap) or ((p_gap - v1_gap) < 0.25 and harm_reduced)
                seed_non_degrade_count = sum(1 for row in primary if _float(row.get("mean_oracle_gap_pct")) <= v1_gap + 1.0)
                local_pass = (
                    utility_ok
                    and no_v1_degrade
                    and precision_improved
                    and harm_reduced
                    and active_ok
                    and seed_non_degrade_count >= 2
                    and precision_used
                )
                local_weak = (
                    no_v1_degrade
                    and precision_improved
                    and harm_reduced
                    and active_ok
                    and precision_used
                )
        elif primary_method == PRIMARY_METHOD_V2:
            v1_rows = by_method.get(PRIMARY_METHOD, [])
            v1_gap_reduction = a_gap - _mean(v1_rows, "mean_oracle_gap_pct") if v1_rows else 0.0
            v2_gap_reduction = a_gap - p_gap
            retained_v1_gain = (
                v2_gap_reduction >= 0.80 * v1_gap_reduction
                if v1_gap_reduction > 0.0
                else v2_gap_reduction > 0.0
            )
            v1_precision = _mean(v1_rows, "selected_override_precision", default=float("nan")) if v1_rows else float("nan")
            v1_harmful = _mean(v1_rows, "harmful_vs_ae_argmin_rate", default=float("nan")) if v1_rows else float("nan")
            precision_or_harm_ok = (
                (v1_rows and math.isfinite(v1_precision) and precision >= v1_precision)
                or (v1_rows and math.isfinite(v1_harmful) and harmful <= v1_harmful)
                or not v1_rows
            )
            local_pass = (
                (p_gap < a_gap or p_top1 > a_top1)
                and no_ae_degrade
                and retained_v1_gain
                and precision_or_harm_ok
                and precision >= THRESHOLDS["min_selected_override_precision_for_pass"]
                and net_gain >= 0.0
                and domain_ok
            )
        else:
            local_pass = (
                p_gap < a_gap
                and p_top1 >= a_top1
                and p_spearman >= a_spearman
                and p_gap < m_gap
                and active >= THRESHOLDS["min_active_override_rate_for_pass"]
                and precision > THRESHOLDS["min_selected_override_precision_for_pass"]
                and harmful <= improving
                and domain_ok
            )
        if primary_method != PRIMARY_METHOD_V11:
            local_weak = (
                (p_gap < a_gap or p_top1 > a_top1)
                and no_ae_degrade
                and no_meta_degrade
                and net_gain > 0.0
                and active >= THRESHOLDS["min_active_override_rate_for_weak_pass"]
                and precision >= THRESHOLDS["min_selected_override_precision_for_weak_pass"]
                and harmful <= improving
            )
        if local_pass:
            local_verdict = "PASS"
        elif local_weak:
            local_verdict = "WEAK PASS"
        elif p_gap < a_gap or net_gain > 0.0 or active > 0.0:
            local_verdict = "DIAGNOSTIC ONLY"
        else:
            local_verdict = "FAIL"

        best_global_gap = min(_mean(by_method.get(method, []), "mean_oracle_gap_pct", default=float("inf")) for method in GLOBAL_BASELINES)
        global_verdict = local_verdict if local_pass and p_gap <= best_global_gap else "DIAGNOSTIC ONLY"
        verdicts[f"{dataset}_local_ae_calibration_verdict"] = local_verdict
        verdicts[f"{dataset}_global_adoption_verdict"] = global_verdict

        for method in sorted(set(str(r["method"]) for r in dataset_rows)):
            method_rows = [r for r in dataset_rows if str(r["method"]) == method]
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_runs": len(method_rows),
                    "top1_oracle_hit": _mean(method_rows, "top1_oracle_hit"),
                    "raw_spearman": _mean(method_rows, "raw_spearman"),
                    "mean_oracle_gap_pct": _mean(method_rows, "mean_oracle_gap_pct"),
                    "active_override_rate": _mean(method_rows, "active_override_rate"),
                    "selected_override_precision": _mean(method_rows, "selected_override_precision", default=float("nan")),
                    "strict_improvement_precision": _mean(
                        method_rows, "strict_improvement_precision", default=float("nan")
                    ),
                    "safe_override_precision": _mean(method_rows, "safe_override_precision", default=float("nan")),
                    "harmful_override_rate": _mean(method_rows, "harmful_override_rate", default=float("nan")),
                    "net_gain_vs_ae_argmin": _mean(method_rows, "net_gain_vs_ae_argmin"),
                    "harmful_vs_ae_argmin_rate": _mean(method_rows, "harmful_vs_ae_argmin_rate"),
                    "improving_vs_ae_argmin_rate": _mean(method_rows, "improving_vs_ae_argmin_rate"),
                    "local_ae_calibration_verdict": local_verdict if method == primary_method else "",
                    "global_adoption_verdict": global_verdict if method == primary_method else "",
                }
            )

    local_values = [v for k, v in verdicts.items() if k.endswith("_local_ae_calibration_verdict")]
    global_values = [v for k, v in verdicts.items() if k.endswith("_global_adoption_verdict")]
    verdicts["cross_dataset_local_ae_calibration_verdict"] = (
        "PASS" if local_values and all(v == "PASS" for v in local_values)
        else "WEAK PASS" if local_values and all(v in {"PASS", "WEAK PASS"} for v in local_values)
        else "DIAGNOSTIC ONLY" if any(v in {"PASS", "WEAK PASS", "DIAGNOSTIC ONLY"} for v in local_values)
        else "FAIL"
    )
    verdicts["cross_dataset_global_adoption_verdict"] = (
        "PASS" if global_values and all(v == "PASS" for v in global_values)
        else "DIAGNOSTIC ONLY" if any(v in {"PASS", "WEAK PASS", "DIAGNOSTIC ONLY"} for v in global_values)
        else "FAIL"
    )
    return out, {"thresholds": THRESHOLDS, "verdicts": verdicts}


def _write_md(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# AE Utility Calibrator Decision Table\n\n")
        for key, verdict in summary.get("verdicts", {}).items():
            f.write(f"- `{key}`: `{verdict}`\n")
        f.write("\n| dataset | method | top1 | spearman | gap pct | active override | precision | local verdict | global verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                "| {dataset} | {method} | {top1_oracle_hit:.4f} | {raw_spearman:.4f} | "
                "{mean_oracle_gap_pct:.4f} | {active_override_rate:.4f} | "
                "{selected_override_precision:.4f} | {local_ae_calibration_verdict} | "
                "{global_adoption_verdict} |\n".format(**row)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AE utility calibrator decision artifacts.")
    parser.add_argument("--manifest", type=Path, default=Path("results/comparison_tables/ae_utility_calibrator_run_manifest.txt"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/comparison_tables/ae_utility_calibrator_decision_table.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/comparison_tables/ae_utility_calibrator_decision_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/summaries/ae_utility_calibrator_decision_table.md"))
    args = parser.parse_args()

    paths = _read_manifest(args.manifest)
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_run_rows(path))
    if not rows:
        raise RuntimeError("No AE utility calibrator rows were found")
    out_rows, summary = _aggregate(rows, paths)
    _write_csv(args.output_csv, out_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_md(args.output_md, out_rows, summary)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
