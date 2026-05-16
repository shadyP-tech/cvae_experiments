#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Tuple

import yaml


def _load_manifest(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    out: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {p}")
        out.append(p)
    if not out:
        raise RuntimeError("Manifest is empty")
    return out


def _infer_seed_from_routing_json(path: Path) -> int:
    text = str(path)
    m = re.search(r"seed(\d+)", text)
    if m is not None:
        return int(m.group(1))

    run_root = path.parent.parent
    cfg_path = run_root / "config_resolved.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return int(cfg.get("seed", -1))

    raise RuntimeError(f"Unable to infer seed for report: {path}")


def _load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(payload.get("metrics", {}).get(key, default))


def _std_mean(payload: Dict[str, Any]) -> float:
    return float(
        payload.get("diagnostics", {})
        .get("expert_nelbo_std_per_query", {})
        .get("mean", 0.0)
    )


def _e1_relative_delta(cond: float, base: float) -> float:
    denom = max(abs(base), 1e-12)
    return float((cond - base) / denom)


def _e3_relative_gap_reduction(cond_gap: float, base_gap: float) -> float:
    denom = max(abs(base_gap), 1e-12)
    return float((base_gap - cond_gap) / denom)


def _pair_by_seed(baseline_paths: List[Path], conditioned_paths: List[Path]) -> List[Tuple[int, Path, Path]]:
    base_by_seed = {_infer_seed_from_routing_json(p): p for p in baseline_paths}
    cond_by_seed = {_infer_seed_from_routing_json(p): p for p in conditioned_paths}
    seeds = sorted(set(base_by_seed).intersection(cond_by_seed))
    if not seeds:
        raise RuntimeError("No overlapping seeds between baseline and conditioned manifests")
    return [(s, base_by_seed[s], cond_by_seed[s]) for s in seeds]


def _decision_label(
    e1_delta_median: float,
    top1_uplift_median: float,
    spearman_uplift_median: float,
    e2_pass: bool,
    e1_pass: bool,
    e3_pass: bool,
    std_collapse_flag: bool,
) -> str:
    if abs(e1_delta_median) < 0.01 and top1_uplift_median < 0.02 and spearman_uplift_median < 0.02:
        return "utility_flat_ranking_flat"
    if e1_pass and e2_pass and e3_pass:
        return "utility_up_ranking_up"
    if e1_pass and (top1_uplift_median >= 0.0 and spearman_uplift_median >= 0.0) and not e2_pass:
        return "utility_up_ranking_flat"
    if e1_pass and (top1_uplift_median < 0.0 or spearman_uplift_median < 0.0):
        return "utility_up_ranking_down"
    if std_collapse_flag:
        return "utility_up_ranking_down"
    return "mixed_signal"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build decision table for legacy conditioned routing protocol.")
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--conditioned-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_decision_table.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/comparison_tables/legacy_conditioning_decision_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/legacy_conditioning_decision_table.md"),
    )

    # Locked defaults from protocol
    parser.add_argument("--e1-median-relative-delta-max", type=float, default=-0.03)
    parser.add_argument("--e1-per-seed-relative-delta-max", type=float, default=-0.01)
    parser.add_argument("--e1-min-passing-seeds", type=int, default=2)
    parser.add_argument("--e2-top1-uplift-min", type=float, default=0.05)
    parser.add_argument("--e2-spearman-uplift-min", type=float, default=0.05)
    parser.add_argument("--e2-min-passing-seeds", type=int, default=2)
    parser.add_argument("--e2-backup-top1-min", type=float, default=0.08)
    parser.add_argument("--e2-backup-spearman-min", type=float, default=0.03)
    parser.add_argument("--e2-backup-spearman-max-exclusive", type=float, default=0.05)
    parser.add_argument("--e3-relative-gap-reduction-min", type=float, default=0.30)
    parser.add_argument("--e3-abs-normalized-gap-median-max", type=float, default=0.05)
    parser.add_argument("--std-collapse-relative-drop-min", type=float, default=0.20)
    parser.add_argument("--std-collapse-baseline-min", type=float, default=1e-4)
    args = parser.parse_args()

    baseline_paths = _load_manifest(args.baseline_manifest)
    conditioned_paths = _load_manifest(args.conditioned_manifest)
    paired = _pair_by_seed(baseline_paths, conditioned_paths)

    rows: List[Dict[str, Any]] = []
    e1_deltas: List[float] = []
    e2_top1_uplifts: List[float] = []
    e2_spearman_uplifts: List[float] = []
    e3_rel_gap_reductions: List[float] = []
    e3_abs_norm_gap_conditioned: List[float] = []
    std_rel_drops: List[float] = []

    e1_seed_pass_count = 0
    e2_seed_pass_count = 0

    for seed, base_path, cond_path in paired:
        base_payload = _load_payload(base_path)
        cond_payload = _load_payload(cond_path)

        base_best = _metric(base_payload, "best_expert_true_utility_nelbo")
        cond_best = _metric(cond_payload, "best_expert_true_utility_nelbo")
        e1_delta = _e1_relative_delta(cond_best, base_best)
        e1_deltas.append(e1_delta)

        base_top1 = _metric(base_payload, "top1_oracle_hit_true_utility")
        cond_top1 = _metric(cond_payload, "top1_oracle_hit_true_utility")
        top1_uplift = float(cond_top1 - base_top1)
        e2_top1_uplifts.append(top1_uplift)

        base_spearman = _metric(base_payload, "spearman_with_true_utility")
        cond_spearman = _metric(cond_payload, "spearman_with_true_utility")
        spearman_uplift = float(cond_spearman - base_spearman)
        e2_spearman_uplifts.append(spearman_uplift)

        base_gap = _metric(base_payload, "routed_to_global_gap")
        cond_gap = _metric(cond_payload, "routed_to_global_gap")
        e3_rel_gap_reduction = _e3_relative_gap_reduction(cond_gap, base_gap)
        e3_rel_gap_reductions.append(e3_rel_gap_reduction)
        e3_abs_norm_gap_conditioned.append(abs(_metric(cond_payload, "routed_to_global_gap_norm_abs_median")))

        base_std = _std_mean(base_payload)
        cond_std = _std_mean(cond_payload)
        std_rel_drop = float((base_std - cond_std) / max(abs(base_std), 1e-12))
        std_rel_drops.append(std_rel_drop)

        e1_seed_pass = bool(e1_delta <= args.e1_per_seed_relative_delta_max)
        e2_seed_pass = bool(top1_uplift > 0.0 and spearman_uplift > 0.0)
        e1_seed_pass_count += 1 if e1_seed_pass else 0
        e2_seed_pass_count += 1 if e2_seed_pass else 0

        rows.append(
            {
                "seed": int(seed),
                "baseline_report": str(base_path),
                "conditioned_report": str(cond_path),
                "e1_relative_delta": float(e1_delta),
                "e2_top1_uplift": float(top1_uplift),
                "e2_spearman_uplift": float(spearman_uplift),
                "e3_relative_gap_reduction": float(e3_rel_gap_reduction),
                "conditioned_abs_norm_gap_median": float(e3_abs_norm_gap_conditioned[-1]),
                "baseline_expert_nelbo_std_mean": float(base_std),
                "conditioned_expert_nelbo_std_mean": float(cond_std),
                "std_relative_drop": float(std_rel_drop),
                "e1_seed_pass": int(e1_seed_pass),
                "e2_seed_positive": int(e2_seed_pass),
            }
        )

    e1_median = float(median(e1_deltas))
    e2_top1_median = float(median(e2_top1_uplifts))
    e2_spearman_median = float(median(e2_spearman_uplifts))
    e3_rel_median = float(median(e3_rel_gap_reductions))
    e3_abs_norm_median = float(median(e3_abs_norm_gap_conditioned))
    std_rel_drop_median = float(median(std_rel_drops))
    baseline_std_median = float(median([r["baseline_expert_nelbo_std_mean"] for r in rows]))

    e1_pass = bool(
        e1_median <= args.e1_median_relative_delta_max
        and e1_seed_pass_count >= int(args.e1_min_passing_seeds)
    )

    e2_pass = bool(
        e2_top1_median >= args.e2_top1_uplift_min
        and e2_spearman_median >= args.e2_spearman_uplift_min
        and e2_seed_pass_count >= int(args.e2_min_passing_seeds)
    )
    e2_partial = bool(
        e2_top1_median >= args.e2_backup_top1_min
        and e2_spearman_median >= args.e2_backup_spearman_min
        and e2_spearman_median < args.e2_backup_spearman_max_exclusive
    )

    e3_pass = bool(
        e3_rel_median >= args.e3_relative_gap_reduction_min
        or e3_abs_norm_median <= args.e3_abs_normalized_gap_median_max
    )

    std_collapse_evaluable = bool(baseline_std_median >= args.std_collapse_baseline_min)
    std_collapse_flag = bool(std_collapse_evaluable and std_rel_drop_median >= args.std_collapse_relative_drop_min)

    label = _decision_label(
        e1_delta_median=e1_median,
        top1_uplift_median=e2_top1_median,
        spearman_uplift_median=e2_spearman_median,
        e2_pass=e2_pass,
        e1_pass=e1_pass,
        e3_pass=e3_pass,
        std_collapse_flag=std_collapse_flag,
    )

    summary = {
        "n_seeds": int(len(rows)),
        "seeds": [int(r["seed"]) for r in rows],
        "gates": {
            "e1": {
                "median_relative_delta": e1_median,
                "seed_pass_count": int(e1_seed_pass_count),
                "pass": e1_pass,
                "negative_delta_indicates_improvement": True,
            },
            "e2": {
                "median_top1_uplift": e2_top1_median,
                "median_spearman_uplift": e2_spearman_median,
                "seed_positive_count": int(e2_seed_pass_count),
                "pass": e2_pass,
                "partial_ranking_gain": e2_partial,
            },
            "e3": {
                "median_relative_gap_reduction": e3_rel_median,
                "median_abs_normalized_gap": e3_abs_norm_median,
                "pass": e3_pass,
            },
        },
        "uniformization": {
            "evaluable": bool(std_collapse_evaluable),
            "baseline_std_median": float(baseline_std_median),
            "std_relative_drop_median": float(std_rel_drop_median),
            "flag": bool(std_collapse_flag),
        },
        "taxonomy_label": str(label),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["seed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("w", encoding="utf-8") as f:
        f.write("# Legacy Conditioning Decision Summary\n\n")
        f.write(f"- n_seeds: {summary['n_seeds']}\n")
        f.write(f"- seeds: {summary['seeds']}\n")
        f.write(f"- taxonomy_label: {summary['taxonomy_label']}\n")
        f.write("\n")
        f.write("## Gate Outcomes\n\n")
        f.write(f"- E1 pass: {summary['gates']['e1']['pass']}\n")
        f.write(f"- E1 median_relative_delta: {summary['gates']['e1']['median_relative_delta']:.6f}\n")
        f.write(f"- E1 seed_pass_count: {summary['gates']['e1']['seed_pass_count']}\n")
        f.write(f"- E2 pass: {summary['gates']['e2']['pass']}\n")
        f.write(f"- E2 partial_ranking_gain: {summary['gates']['e2']['partial_ranking_gain']}\n")
        f.write(f"- E2 median_top1_uplift: {summary['gates']['e2']['median_top1_uplift']:.6f}\n")
        f.write(f"- E2 median_spearman_uplift: {summary['gates']['e2']['median_spearman_uplift']:.6f}\n")
        f.write(f"- E3 pass: {summary['gates']['e3']['pass']}\n")
        f.write(f"- E3 median_relative_gap_reduction: {summary['gates']['e3']['median_relative_gap_reduction']:.6f}\n")
        f.write(f"- E3 median_abs_normalized_gap: {summary['gates']['e3']['median_abs_normalized_gap']:.6f}\n")
        f.write("\n")
        f.write("## Uniformization\n\n")
        f.write(f"- evaluable: {summary['uniformization']['evaluable']}\n")
        f.write(f"- baseline_std_median: {summary['uniformization']['baseline_std_median']:.6f}\n")
        f.write(f"- std_relative_drop_median: {summary['uniformization']['std_relative_drop_median']:.6f}\n")
        f.write(f"- flag: {summary['uniformization']['flag']}\n")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
