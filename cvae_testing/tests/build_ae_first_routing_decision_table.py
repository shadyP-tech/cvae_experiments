#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


PRIMARY_METHOD = "ae_first_margin_gated_v1"
METADATA_METHOD = "metadata_routing"
SOURCE_PRIOR_METHOD = "source_prior_fallback"
AE_ARGMIN_METHOD = "ae_argmin_zscore"
METADATA_AE_RESIDUAL_METHOD = "metadata_ae_residual_safe_override_v1"
REQUIRED_METHODS = {
    METADATA_METHOD,
    SOURCE_PRIOR_METHOD,
    AE_ARGMIN_METHOD,
    METADATA_AE_RESIDUAL_METHOD,
    "random_rank_floor",
    "random_score_floor",
    "candidate_oracle_routing",
    PRIMARY_METHOD,
}
THRESHOLDS = {
    "top1_degradation_abs_max": 0.02,
    "raw_spearman_degradation_abs_max": 0.03,
    "mean_oracle_gap_pct_degradation_pp_max": 1.0,
    "min_ae_coverage_rate_for_weak_pass": 0.10,
    "min_ae_coverage_rate_for_pass": 0.20,
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


def _dataset_from_path(path: Path) -> str:
    text = str(path).lower()
    if "breakhis" in text:
        return "breakhis"
    if "camelyon17" in text:
        return "camelyon17"
    return "unknown"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(rows: Sequence[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_float(row.get(key), default) for row in rows]
    return float(mean(vals)) if vals else float(default)


def _load_run_rows(result_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    dataset = _dataset_from_path(result_path)
    rows: List[Dict[str, Any]] = []
    for method, metrics in sorted((payload.get("metrics_by_method", {}) or {}).items()):
        if str(method) not in REQUIRED_METHODS:
            continue
        rows.append(
            {
                "dataset": dataset,
                "result_path": str(result_path),
                "method": str(method),
                "top1_oracle_hit": _float(metrics.get("macro_top1_oracle_hit_by_query_domain", metrics.get("top1_oracle_hit"))),
                "raw_predicted_delta_spearman": _float(
                    metrics.get("macro_spearman_by_query_domain", metrics.get("spearman"))
                ),
                "mean_oracle_gap_pct": _float(
                    metrics.get("macro_oracle_gap_pct_by_query_domain", metrics.get("mean_oracle_gap_pct"))
                ),
                "mean_oracle_gap": _float(metrics.get("macro_oracle_gap_by_query_domain", metrics.get("mean_oracle_gap"))),
                "metadata_relative_gain": _float(metrics.get("metadata_relative_gain")),
                "source_prior_relative_gain": _float(metrics.get("source_prior_relative_gain")),
                "harmful_vs_metadata_rate": _float(metrics.get("harmful_vs_metadata_rate")),
                "improving_vs_metadata_rate": _float(metrics.get("improving_vs_metadata_rate")),
                "harmful_vs_source_prior_rate": _float(metrics.get("harmful_vs_source_prior_rate")),
                "improving_vs_source_prior_rate": _float(metrics.get("improving_vs_source_prior_rate")),
                "ae_coverage_rate": _float(metrics.get("ae_coverage_rate")),
                "fallback_rate": _float(metrics.get("fallback_rate")),
                "diagnostic_only": int(_float(metrics.get("diagnostic_only"))),
                "adoption_eligible": int(_float(metrics.get("adoption_eligible"))),
            }
        )
    return rows


def _domain_non_degradation_ok(result_paths: Sequence[Path], *, dataset: str, method: str) -> bool:
    ok = True
    for result_path in result_paths:
        if _dataset_from_path(result_path) != dataset:
            continue
        rows = _read_csv(result_path.parent / "learned_utility_domain_breakdown.csv")
        by_key = {(r.get("method", ""), r.get("query_domain", "")): r for r in rows}
        domains = sorted(q for m, q in by_key if m == METADATA_METHOD)
        for domain in domains:
            base = by_key.get((METADATA_METHOD, domain))
            cand = by_key.get((method, domain))
            if not base or not cand:
                ok = False
                continue
            top1_delta = _float(cand.get("top1_oracle_hit")) - _float(base.get("top1_oracle_hit"))
            spearman_delta = _float(cand.get("spearman")) - _float(base.get("spearman"))
            gap_delta = _float(cand.get("mean_oracle_gap_pct")) - _float(base.get("mean_oracle_gap_pct"))
            if top1_delta < -THRESHOLDS["top1_degradation_abs_max"]:
                ok = False
            if spearman_delta < -THRESHOLDS["raw_spearman_degradation_abs_max"]:
                ok = False
            if gap_delta > THRESHOLDS["mean_oracle_gap_pct_degradation_pp_max"]:
                ok = False
    return ok


def _margin_positive_checks(result_paths: Sequence[Path], *, dataset: str) -> int:
    checks = 0
    rows: List[Dict[str, str]] = []
    for result_path in result_paths:
        if _dataset_from_path(result_path) == dataset:
            rows.extend(
                row
                for row in _read_csv(result_path.parent / "learned_utility_sample_selections.csv")
                if row.get("method") == PRIMARY_METHOD
            )
    if not rows:
        return 0
    margins = [_float(r.get("ae_margin")) for r in rows]
    gains = [_float(r.get("metadata_relative_gain")) for r in rows]
    if len(margins) >= 2:
        m_bar = mean(margins)
        g_bar = mean(gains)
        cov = sum((m - m_bar) * (g - g_bar) for m, g in zip(margins, gains))
        var_m = sum((m - m_bar) ** 2 for m in margins)
        var_g = sum((g - g_bar) ** 2 for g in gains)
        if var_m > 0.0 and var_g > 0.0 and cov / ((var_m * var_g) ** 0.5) > 0.0:
            checks += 1

    bin_rows: List[Dict[str, str]] = []
    for result_path in result_paths:
        if _dataset_from_path(result_path) == dataset:
            bin_rows.extend(_read_csv(result_path.parent / "ae_first_margin_bins.csv"))
    nonempty = [r for r in bin_rows if int(_float(r.get("n_samples"))) > 0]
    if len(nonempty) >= 2:
        first = nonempty[0]
        last = nonempty[-1]
        if _float(last.get("harmful_vs_metadata_rate")) < _float(first.get("harmful_vs_metadata_rate")):
            checks += 1
        if _float(last.get("mean_oracle_gap_pct")) < _float(first.get("mean_oracle_gap_pct")):
            checks += 1
        if _float(last.get("top1_oracle_hit")) > _float(first.get("top1_oracle_hit")):
            checks += 1
    return checks


def _aggregate(rows: Sequence[Dict[str, Any]], result_paths: Sequence[Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    verdicts: Dict[str, str] = {}
    for dataset in sorted(set(str(r["dataset"]) for r in rows)):
        dataset_rows = [r for r in rows if str(r["dataset"]) == dataset]
        by_method = {m: [r for r in dataset_rows if r["method"] == m] for m in REQUIRED_METHODS}
        metadata = by_method.get(METADATA_METHOD, [])
        primary = by_method.get(PRIMARY_METHOD, [])
        if not metadata or not primary:
            verdicts[f"{dataset}_ae_first_verdict"] = "REJECTED"
            continue
        metadata_top1 = _mean(metadata, "top1_oracle_hit")
        metadata_spearman = _mean(metadata, "raw_predicted_delta_spearman")
        metadata_gap = _mean(metadata, "mean_oracle_gap_pct")
        primary_top1 = _mean(primary, "top1_oracle_hit")
        primary_spearman = _mean(primary, "raw_predicted_delta_spearman")
        primary_gap = _mean(primary, "mean_oracle_gap_pct")
        top1_delta = primary_top1 - metadata_top1
        spearman_delta = primary_spearman - metadata_spearman
        gap_reduction = metadata_gap - primary_gap
        coverage = _mean(primary, "ae_coverage_rate")
        harmful_meta = _mean(primary, "harmful_vs_metadata_rate")
        improving_meta = _mean(primary, "improving_vs_metadata_rate")
        metadata_gain = _mean(primary, "metadata_relative_gain")
        source_prior_gain = _mean(primary, "source_prior_relative_gain")
        source_prior_gap = _mean(by_method.get(SOURCE_PRIOR_METHOD, []), "mean_oracle_gap_pct")
        ae_argmin_gap = _mean(by_method.get(AE_ARGMIN_METHOD, []), "mean_oracle_gap_pct")
        residual_gap = _mean(by_method.get(METADATA_AE_RESIDUAL_METHOD, []), "mean_oracle_gap_pct")
        domain_ok = _domain_non_degradation_ok(result_paths, dataset=dataset, method=PRIMARY_METHOD)
        no_material = (
            top1_delta >= -THRESHOLDS["top1_degradation_abs_max"]
            and spearman_delta >= -THRESHOLDS["raw_spearman_degradation_abs_max"]
            and gap_reduction >= -THRESHOLDS["mean_oracle_gap_pct_degradation_pp_max"]
        )
        positive_margin_checks = _margin_positive_checks(result_paths, dataset=dataset)
        if (
            gap_reduction > 0.0
            and top1_delta >= 0.0
            and spearman_delta > 0.0
            and primary_gap < source_prior_gap
            and primary_gap < ae_argmin_gap
            and primary_gap <= residual_gap
            and domain_ok
            and coverage >= THRESHOLDS["min_ae_coverage_rate_for_pass"]
        ):
            verdict = "PASS"
        elif (
            primary_gap < ae_argmin_gap
            and primary_gap < source_prior_gap
            and no_material
            and harmful_meta <= improving_meta
            and metadata_gain >= 0.0
            and source_prior_gain > 0.0
            and coverage >= THRESHOLDS["min_ae_coverage_rate_for_weak_pass"]
            and positive_margin_checks >= 2
        ):
            verdict = "WEAK PASS"
        elif no_material or gap_reduction > 0.0 or positive_margin_checks > 0:
            verdict = "DIAGNOSTIC ONLY"
        else:
            verdict = "FAIL"
        verdicts[f"{dataset}_ae_first_verdict"] = verdict

        for method in sorted(set(str(r["method"]) for r in dataset_rows)):
            method_rows = [r for r in dataset_rows if str(r["method"]) == method]
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_runs": len(method_rows),
                    "top1_oracle_hit": _mean(method_rows, "top1_oracle_hit"),
                    "raw_predicted_delta_spearman": _mean(method_rows, "raw_predicted_delta_spearman"),
                    "mean_oracle_gap_pct": _mean(method_rows, "mean_oracle_gap_pct"),
                    "mean_oracle_gap": _mean(method_rows, "mean_oracle_gap"),
                    "metadata_relative_gain": _mean(method_rows, "metadata_relative_gain"),
                    "source_prior_relative_gain": _mean(method_rows, "source_prior_relative_gain"),
                    "harmful_vs_metadata_rate": _mean(method_rows, "harmful_vs_metadata_rate"),
                    "improving_vs_metadata_rate": _mean(method_rows, "improving_vs_metadata_rate"),
                    "ae_coverage_rate": _mean(method_rows, "ae_coverage_rate"),
                    "fallback_rate": _mean(method_rows, "fallback_rate"),
                    "positive_margin_checks": positive_margin_checks if method == PRIMARY_METHOD else "",
                    "no_material_degradation": int(no_material) if method == PRIMARY_METHOD else "",
                    "no_domain_material_degradation": int(domain_ok) if method == PRIMARY_METHOD else "",
                    "verdict": verdict if method == PRIMARY_METHOD else "",
                }
            )

    dataset_verdicts = [v for k, v in verdicts.items() if k.endswith("_ae_first_verdict")]
    if not dataset_verdicts:
        cross = "REJECTED"
    elif all(v == "PASS" for v in dataset_verdicts):
        cross = "PASS"
    elif all(v in {"PASS", "WEAK PASS"} for v in dataset_verdicts):
        cross = "WEAK PASS"
    elif any(v in {"PASS", "WEAK PASS", "DIAGNOSTIC ONLY"} for v in dataset_verdicts):
        cross = "DIAGNOSTIC ONLY"
    else:
        cross = "FAIL"
    verdicts["cross_dataset_verdict"] = cross
    return out, {"thresholds": THRESHOLDS, "verdicts": verdicts}


def _write_md(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# AE-First Routing Decision Table\n\n")
        for key, verdict in summary.get("verdicts", {}).items():
            f.write(f"- `{key}`: `{verdict}`\n")
        f.write("\n")
        f.write("| dataset | method | top1 | raw spearman | gap pct | metadata gain | source-prior gain | coverage | verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {dataset} | {method} | {top1_oracle_hit:.4f} | {raw_predicted_delta_spearman:.4f} | "
                "{mean_oracle_gap_pct:.4f} | {metadata_relative_gain:.4f} | "
                "{source_prior_relative_gain:.4f} | {ae_coverage_rate:.4f} | {verdict} |\n".format(**row)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AE-first routing decision artifacts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/comparison_tables/ae_first_run_manifest.txt"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/ae_first_routing_decision_table.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/comparison_tables/ae_first_routing_decision_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/ae_first_routing_decision_table.md"),
    )
    args = parser.parse_args()

    result_paths = _read_manifest(args.manifest)
    rows: List[Dict[str, Any]] = []
    for result_path in result_paths:
        rows.extend(_load_run_rows(result_path))
    if not rows:
        raise RuntimeError("No AE-first routing rows were found")
    out_rows, summary = _aggregate(rows, result_paths)
    _write_csv(args.output_csv, out_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_md(args.output_md, out_rows, summary)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
