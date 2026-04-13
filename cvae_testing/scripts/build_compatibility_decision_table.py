#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0, 0.0
    mu = float(mean(clean))
    var = float(sum((v - mu) ** 2 for v in clean) / len(clean))
    return mu, math.sqrt(max(var, 0.0))


def _load_manifest(manifest_path: Path) -> List[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    paths: List[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {p}")
        paths.append(p)

    if not paths:
        raise RuntimeError("Manifest is empty; no result json files found.")
    return paths


def _seed_from_path(path: Path, fallback: int) -> int:
    text = str(path)
    m = re.search(r"seed(\d+)", text)
    if m is not None:
        return int(m.group(1))
    return int(fallback)


def _sign_inconsistency_count(values: Sequence[float]) -> int:
    pos = any(float(v) > 1e-12 for v in values)
    neg = any(float(v) < -1e-12 for v in values)
    return 1 if (pos and neg) else 0


def _tier(
    *,
    improving_seed_count: int,
    min_improving_seeds: int,
    spearman_uplift_mean: float,
    top1_uplift_mean: float,
    gap_reduction_mean: float,
    strong: Dict[str, float],
    weak: Dict[str, float],
    instability_breach: bool,
) -> str:
    if instability_breach:
        return "fail"

    strong_ok = (
        improving_seed_count >= int(min_improving_seeds)
        and spearman_uplift_mean >= float(strong["spearman_uplift_min"])
        and top1_uplift_mean >= float(strong["top1_uplift_min"])
        and gap_reduction_mean >= float(strong["oracle_gap_pct_reduction_min"])
    )
    if strong_ok:
        return "strong_pass"

    weak_ok = (
        improving_seed_count >= int(min_improving_seeds)
        and spearman_uplift_mean >= float(weak["spearman_uplift_min"])
        and top1_uplift_mean >= float(weak["top1_uplift_min"])
        and gap_reduction_mean >= float(weak["oracle_gap_pct_reduction_min"])
    )
    if weak_ok:
        return "weak_pass"

    return "fail"


def _read_rows(
    result_paths: Sequence[Path],
    uplift_reference_method: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, path in enumerate(result_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics_by_method = payload.get("metrics_by_method", {})
        if not isinstance(metrics_by_method, dict) or not metrics_by_method:
            continue

        seed = _seed_from_path(path, fallback=idx)
        baseline = metrics_by_method.get(uplift_reference_method, {})
        b_top1 = _to_float((baseline or {}).get("top1_oracle_hit", 0.0))
        b_spearman = _to_float((baseline or {}).get("spearman", 0.0))
        b_gap_pct = _to_float((baseline or {}).get("mean_oracle_gap_pct", 0.0))

        for method, m in metrics_by_method.items():
            mm = m or {}
            top1 = _to_float(mm.get("top1_oracle_hit", 0.0))
            spearman = _to_float(mm.get("spearman", 0.0))
            gap_pct = _to_float(mm.get("mean_oracle_gap_pct", 0.0))
            rows.append(
                {
                    "seed": int(seed),
                    "method": str(method),
                    "top1_oracle_hit": top1,
                    "spearman": spearman,
                    "mean_oracle_gap_pct": gap_pct,
                    "top1_uplift_vs_metadata": float(top1 - b_top1),
                    "spearman_uplift_vs_metadata": float(spearman - b_spearman),
                    "oracle_gap_pct_reduction_vs_metadata": float(b_gap_pct - gap_pct),
                    "source_json": str(path),
                }
            )
    return rows


def _aggregate(
    rows: Sequence[Dict[str, Any]],
    uplift_reference_method: str,
    min_improving_seeds: int,
    strong: Dict[str, float],
    weak: Dict[str, float],
    instability_std_threshold: float,
    instability_sign_inconsistency_min_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    non_selectable_methods = {
        str(uplift_reference_method),
        "oracle_routing",
        "random_rank_floor",
        "random_score_floor",
        "latent_wasserstein_routing",
    }

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    out_rows: List[Dict[str, Any]] = []
    for method in sorted(by_method.keys()):
        method_rows = sorted(by_method[method], key=lambda r: int(r["seed"]))
        seeds = [int(r["seed"]) for r in method_rows]

        top1_vals = [float(r["top1_oracle_hit"]) for r in method_rows]
        spearman_vals = [float(r["spearman"]) for r in method_rows]
        gap_vals = [float(r["mean_oracle_gap_pct"]) for r in method_rows]

        top1_uplifts = [float(r["top1_uplift_vs_metadata"]) for r in method_rows]
        spearman_uplifts = [float(r["spearman_uplift_vs_metadata"]) for r in method_rows]
        gap_reductions = [float(r["oracle_gap_pct_reduction_vs_metadata"]) for r in method_rows]

        top1_mean, top1_std = _mean_std(top1_vals)
        spearman_mean, spearman_std = _mean_std(spearman_vals)
        gap_mean, gap_std = _mean_std(gap_vals)

        top1_uplift_mean, top1_uplift_std = _mean_std(top1_uplifts)
        spearman_uplift_mean, spearman_uplift_std = _mean_std(spearman_uplifts)
        gap_reduction_mean, gap_reduction_std = _mean_std(gap_reductions)

        improving_seed_count = sum(
            1
            for i in range(len(method_rows))
            if top1_uplifts[i] > 0.0 and spearman_uplifts[i] > 0.0 and gap_reductions[i] > 0.0
        )

        std_breach = bool(
            top1_uplift_std > float(instability_std_threshold)
            or spearman_uplift_std > float(instability_std_threshold)
            or gap_reduction_std > float(instability_std_threshold)
        )
        sign_inconsistency_count = (
            _sign_inconsistency_count(top1_uplifts)
            + _sign_inconsistency_count(spearman_uplifts)
            + _sign_inconsistency_count(gap_reductions)
        )
        sign_breach = bool(sign_inconsistency_count >= int(instability_sign_inconsistency_min_count))
        instability_breach = bool(std_breach or sign_breach)

        is_hybrid_method = str(method).startswith("hybrid_alpha_")
        if method == str(uplift_reference_method):
            tier = "baseline"
        elif method in non_selectable_methods or is_hybrid_method:
            tier = "reference_only"
        else:
            tier = _tier(
                improving_seed_count=int(improving_seed_count),
                min_improving_seeds=int(min_improving_seeds),
                spearman_uplift_mean=float(spearman_uplift_mean),
                top1_uplift_mean=float(top1_uplift_mean),
                gap_reduction_mean=float(gap_reduction_mean),
                strong=strong,
                weak=weak,
                instability_breach=instability_breach,
            )

        out_rows.append(
            {
                "method": method,
                "n_seeds": int(len(seeds)),
                "seeds": ",".join(str(s) for s in seeds),
                "top1_oracle_hit_mean": float(top1_mean),
                "top1_oracle_hit_std": float(top1_std),
                "spearman_mean": float(spearman_mean),
                "spearman_std": float(spearman_std),
                "mean_oracle_gap_pct_mean": float(gap_mean),
                "mean_oracle_gap_pct_std": float(gap_std),
                "top1_uplift_vs_metadata_mean": float(top1_uplift_mean),
                "top1_uplift_vs_metadata_std": float(top1_uplift_std),
                "spearman_uplift_vs_metadata_mean": float(spearman_uplift_mean),
                "spearman_uplift_vs_metadata_std": float(spearman_uplift_std),
                "oracle_gap_pct_reduction_vs_metadata_mean": float(gap_reduction_mean),
                "oracle_gap_pct_reduction_vs_metadata_std": float(gap_reduction_std),
                "improving_seed_count": int(improving_seed_count),
                "instability_std_threshold": float(instability_std_threshold),
                "instability_sign_inconsistency_count": int(sign_inconsistency_count),
                "instability_sign_inconsistency_min_count": int(instability_sign_inconsistency_min_count),
                "instability_breach": int(instability_breach),
                "tier": str(tier),
            }
        )

    candidates = [
        r
        for r in out_rows
        if str(r["method"]) not in non_selectable_methods
        and not str(r["method"]).startswith("hybrid_alpha_")
    ]
    strong_candidates = [r for r in candidates if str(r["tier"]) == "strong_pass"]
    weak_candidates = [r for r in candidates if str(r["tier"]) == "weak_pass"]

    selected_method = str(uplift_reference_method)
    overall_tier = "fail"
    if strong_candidates:
        selected_method = str(
            max(strong_candidates, key=lambda r: float(r["oracle_gap_pct_reduction_vs_metadata_mean"]))["method"]
        )
        overall_tier = "strong_pass"
    elif weak_candidates:
        selected_method = str(
            max(weak_candidates, key=lambda r: float(r["oracle_gap_pct_reduction_vs_metadata_mean"]))["method"]
        )
        overall_tier = "weak_pass"

    for r in out_rows:
        if str(r["method"]) == selected_method:
            r["decision"] = "selected"
        elif str(r["method"]) == str(uplift_reference_method):
            r["decision"] = "baseline_reference"
        else:
            r["decision"] = "not_selected"

    summary = {
        "uplift_reference_method": str(uplift_reference_method),
        "overall_tier": str(overall_tier),
        "selected_method": str(selected_method),
        "min_improving_seeds": int(min_improving_seeds),
        "strong_thresholds": strong,
        "weak_thresholds": weak,
        "instability": {
            "std_threshold": float(instability_std_threshold),
            "sign_inconsistency_min_count": int(instability_sign_inconsistency_min_count),
        },
    }
    return out_rows, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Compatibility Decision Table\n\n")
        f.write("- uplift_reference_method: {}\n".format(summary["uplift_reference_method"]))
        f.write("- overall_tier: {}\n".format(summary["overall_tier"]))
        f.write("- selected_method: {}\n".format(summary["selected_method"]))
        f.write("- min_improving_seeds: {}\n".format(summary["min_improving_seeds"]))
        f.write(
            "- instability: std_threshold={} sign_inconsistency_min_count={}\n".format(
                summary["instability"]["std_threshold"],
                summary["instability"]["sign_inconsistency_min_count"],
            )
        )
        f.write("\n")
        f.write(
            "| method | decision | tier | n_seeds | top1 | spearman | mean_oracle_gap_pct | "
            "top1_uplift_vs_metadata | spearman_uplift_vs_metadata | gap_pct_reduction_vs_metadata | "
            "improving_seed_count | instability_breach |\n"
        )
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {} | {} | {} | {} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | "
                "{:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {:.4f} +- {:.4f} | {} | {} |\n".format(
                    r["method"],
                    r.get("decision", "not_selected"),
                    r["tier"],
                    r["n_seeds"],
                    float(r["top1_oracle_hit_mean"]),
                    float(r["top1_oracle_hit_std"]),
                    float(r["spearman_mean"]),
                    float(r["spearman_std"]),
                    float(r["mean_oracle_gap_pct_mean"]),
                    float(r["mean_oracle_gap_pct_std"]),
                    float(r["top1_uplift_vs_metadata_mean"]),
                    float(r["top1_uplift_vs_metadata_std"]),
                    float(r["spearman_uplift_vs_metadata_mean"]),
                    float(r["spearman_uplift_vs_metadata_std"]),
                    float(r["oracle_gap_pct_reduction_vs_metadata_mean"]),
                    float(r["oracle_gap_pct_reduction_vs_metadata_std"]),
                    int(r["improving_seed_count"]),
                    int(r["instability_breach"]),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compatibility decision table from learned utility run manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/comparison_tables/compatibility_run_manifest.txt"),
    )
    parser.add_argument("--uplift-reference-method", type=str, default="metadata_routing")
    parser.add_argument("--min-improving-seeds", type=int, default=2)
    parser.add_argument("--strong-spearman-uplift-min", type=float, default=0.05)
    parser.add_argument("--strong-top1-uplift-min", type=float, default=0.10)
    parser.add_argument("--strong-gap-pct-reduction-min", type=float, default=5.0)
    parser.add_argument("--weak-spearman-uplift-min", type=float, default=0.025)
    parser.add_argument("--weak-top1-uplift-min", type=float, default=0.05)
    parser.add_argument("--weak-gap-pct-reduction-min", type=float, default=2.5)
    parser.add_argument("--instability-std-threshold", type=float, default=0.05)
    parser.add_argument("--instability-sign-inconsistency-min-count", type=int, default=2)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/compatibility_decision_table.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/compatibility_decision_table.md"),
    )
    args = parser.parse_args()

    result_paths = _load_manifest(args.manifest)
    rows = _read_rows(result_paths, uplift_reference_method=str(args.uplift_reference_method))
    if not rows:
        raise RuntimeError("No rows could be read from result json files.")

    strong = {
        "spearman_uplift_min": float(args.strong_spearman_uplift_min),
        "top1_uplift_min": float(args.strong_top1_uplift_min),
        "oracle_gap_pct_reduction_min": float(args.strong_gap_pct_reduction_min),
    }
    weak = {
        "spearman_uplift_min": float(args.weak_spearman_uplift_min),
        "top1_uplift_min": float(args.weak_top1_uplift_min),
        "oracle_gap_pct_reduction_min": float(args.weak_gap_pct_reduction_min),
    }

    out_rows, summary = _aggregate(
        rows=rows,
        uplift_reference_method=str(args.uplift_reference_method),
        min_improving_seeds=int(args.min_improving_seeds),
        strong=strong,
        weak=weak,
        instability_std_threshold=float(args.instability_std_threshold),
        instability_sign_inconsistency_min_count=int(args.instability_sign_inconsistency_min_count),
    )

    _write_csv(args.output_csv, out_rows)
    _write_md(args.output_md, out_rows, summary)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
