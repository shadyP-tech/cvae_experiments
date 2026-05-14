#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


PRIMARY_METHOD = "metadata_ae_residual_safe_override_v1"
METADATA_METHOD = "metadata_routing"
REQUIRED_METHODS = {
    "metadata_routing",
    "random_rank_floor",
    "random_score_floor",
    "pairwise_ranker_metadata_only",
    "pairwise_ranker_latent_only",
    "pairwise_ranker_combined",
    "metadata_residual_thresholded_safe_v2",
    "ae_argmin_zscore",
    "ae_argmin_margin_gated",
    "pairwise_ranker_ae_only",
    "pairwise_ranker_ae_metadata",
    "pairwise_ranker_ae_combined",
    "candidate_oracle_routing",
    PRIMARY_METHOD,
}

THRESHOLDS = {
    "top1_degradation_abs_max": 0.02,
    "raw_spearman_degradation_abs_max": 0.03,
    "mean_oracle_gap_pct_degradation_pp_max": 1.0,
}


def _read_manifest(path: Path) -> List[Path]:
    rows: List[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows.append(Path(text))
    return rows


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
        for key in row.keys():
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _dataset_from_path(path: Path) -> str:
    low = str(path).lower()
    if "breakhis" in low:
        return "breakhis"
    if "camelyon17" in low:
        return "camelyon17"
    return "unknown"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(rows: Sequence[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [_float(r.get(key), default) for r in rows]
    return float(mean(vals)) if vals else float(default)


def _load_run_rows(result_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    dataset = _dataset_from_path(result_path)
    metrics_by_method = payload.get("metrics_by_method", {})
    rows: List[Dict[str, Any]] = []
    for method, metrics in sorted(metrics_by_method.items()):
        if str(method) not in REQUIRED_METHODS:
            continue
        rows.append(
            {
                "dataset": dataset,
                "result_path": str(result_path),
                "method": str(method),
                "top1_oracle_hit": _float(metrics.get("top1_oracle_hit")),
                "raw_predicted_delta_spearman": _float(
                    metrics.get("raw_predicted_delta_spearman", metrics.get("spearman"))
                ),
                "mean_oracle_gap_pct": _float(metrics.get("mean_oracle_gap_pct")),
                "mean_oracle_gap": _float(metrics.get("mean_oracle_gap")),
                "net_override_gain": _float(metrics.get("net_override_gain")),
                "harmful_override_rate": _float(metrics.get("harmful_override_rate")),
                "utility_improving_override_rate": _float(metrics.get("utility_improving_override_rate")),
                "override_rate": _float(metrics.get("override_rate")),
                "safe_fallback_rate": _float(metrics.get("safe_fallback_rate")),
                "diagnostic_only": int(_float(metrics.get("diagnostic_only"))),
                "adoption_eligible": int(_float(metrics.get("adoption_eligible"))),
            }
        )
    return rows


def _domain_non_degradation_ok(result_paths: Sequence[Path], *, method: str, dataset: str) -> bool:
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
    return bool(ok)


def _aggregate(rows: Sequence[Dict[str, Any]], result_paths: Sequence[Path]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    verdicts: Dict[str, str] = {}
    for dataset in sorted(set(str(r["dataset"]) for r in rows)):
        dataset_rows = [r for r in rows if str(r["dataset"]) == dataset]
        metadata = [r for r in dataset_rows if r["method"] == METADATA_METHOD]
        if not metadata:
            continue
        metadata_top1 = _mean(metadata, "top1_oracle_hit")
        metadata_spearman = _mean(metadata, "raw_predicted_delta_spearman")
        metadata_gap = _mean(metadata, "mean_oracle_gap_pct")
        for method in sorted(set(str(r["method"]) for r in dataset_rows)):
            method_rows = [r for r in dataset_rows if str(r["method"]) == method]
            top1 = _mean(method_rows, "top1_oracle_hit")
            spearman = _mean(method_rows, "raw_predicted_delta_spearman")
            gap = _mean(method_rows, "mean_oracle_gap_pct")
            top1_delta = top1 - metadata_top1
            spearman_delta = spearman - metadata_spearman
            gap_reduction = metadata_gap - gap
            net_gain = _mean(method_rows, "net_override_gain")
            harmful = _mean(method_rows, "harmful_override_rate")
            improving = _mean(method_rows, "utility_improving_override_rate")
            domain_ok = _domain_non_degradation_ok(result_paths, method=method, dataset=dataset)
            no_material = (
                top1_delta >= -THRESHOLDS["top1_degradation_abs_max"]
                and spearman_delta >= -THRESHOLDS["raw_spearman_degradation_abs_max"]
                and gap_reduction >= -THRESHOLDS["mean_oracle_gap_pct_degradation_pp_max"]
            )
            if method == PRIMARY_METHOD:
                improves_top1 = top1_delta > 0.0
                improves_spearman = spearman_delta > 0.0
                improves_gap = gap_reduction > 0.0
                safe_overrides = harmful <= improving
                if (
                    improves_top1
                    and improves_spearman
                    and improves_gap
                    and no_material
                    and safe_overrides
                    and net_gain >= 0.0
                    and domain_ok
                ):
                    verdict = "PASS"
                elif (
                    (improves_gap or improves_top1)
                    and top1_delta >= -THRESHOLDS["top1_degradation_abs_max"]
                    and spearman_delta >= -THRESHOLDS["raw_spearman_degradation_abs_max"]
                    and net_gain > 0.0
                    and safe_overrides
                    and domain_ok
                ):
                    verdict = "WEAK PASS"
                elif no_material or improves_gap or improves_top1 or spearman_delta > 0.0:
                    verdict = "DIAGNOSTIC ONLY"
                else:
                    verdict = "FAIL"
                verdicts[f"{dataset}_support_free_ae_verdict"] = verdict
            else:
                verdict = "DIAGNOSTIC_ONLY" if method == "candidate_oracle_routing" else ""
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_runs": len(method_rows),
                    "top1_oracle_hit": top1,
                    "raw_predicted_delta_spearman": spearman,
                    "mean_oracle_gap_pct": gap,
                    "mean_oracle_gap": _mean(method_rows, "mean_oracle_gap"),
                    "top1_delta_vs_metadata": top1_delta,
                    "raw_spearman_delta_vs_metadata": spearman_delta,
                    "oracle_gap_pct_reduction_vs_metadata": gap_reduction,
                    "override_rate": _mean(method_rows, "override_rate"),
                    "safe_fallback_rate": _mean(method_rows, "safe_fallback_rate"),
                    "net_override_gain": net_gain,
                    "harmful_override_rate": harmful,
                    "utility_improving_override_rate": improving,
                    "no_material_degradation": int(no_material),
                    "no_domain_material_degradation": int(domain_ok),
                    "verdict": verdict,
                }
            )

    dataset_verdict_values = [v for k, v in verdicts.items() if k.endswith("_support_free_ae_verdict")]
    if not dataset_verdict_values:
        cross = "REJECTED"
    elif all(v in {"PASS", "WEAK PASS"} for v in dataset_verdict_values):
        cross = "WEAK PASS" if "WEAK PASS" in dataset_verdict_values else "PASS"
    elif any(v in {"PASS", "WEAK PASS", "DIAGNOSTIC ONLY"} for v in dataset_verdict_values):
        cross = "DIAGNOSTIC ONLY"
    else:
        cross = "FAIL"
    verdicts["cross_dataset_verdict"] = cross
    return out, {"thresholds": THRESHOLDS, "verdicts": verdicts}


def _write_md(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Support-Free AE Routing Decision Table\n\n")
        for key, verdict in summary.get("verdicts", {}).items():
            f.write(f"- `{key}`: `{verdict}`\n")
        f.write("\n")
        f.write("| dataset | method | top1 | raw spearman | gap pct | net gain | harmful | improving | verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {dataset} | {method} | {top1_oracle_hit:.4f} | {raw_predicted_delta_spearman:.4f} | "
                "{mean_oracle_gap_pct:.4f} | {net_override_gain:.4f} | {harmful_override_rate:.4f} | "
                "{utility_improving_override_rate:.4f} | {verdict} |\n".format(**row)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build support-free AE routing decision artifacts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/comparison_tables/support_free_ae_run_manifest.txt"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/support_free_ae_routing_decision_table.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/comparison_tables/support_free_ae_routing_decision_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/support_free_ae_routing_decision_table.md"),
    )
    args = parser.parse_args()

    result_paths = _read_manifest(args.manifest)
    rows: List[Dict[str, Any]] = []
    for result_path in result_paths:
        rows.extend(_load_run_rows(result_path))
    if not rows:
        raise RuntimeError("No support-free AE routing rows were found")
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
