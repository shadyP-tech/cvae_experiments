#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean_std(values: List[float]) -> Tuple[float, float]:
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
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {path}")
        paths.append(path)
    if not paths:
        raise RuntimeError("Manifest is empty; no run reports to aggregate.")
    return paths


def _read_hybrid_rows(report_paths: List[Path], variant: str, budget: str) -> List[Dict[str, object]]:
    per_key: Dict[Tuple[int, str], Dict[str, object]] = {}

    for csv_path in report_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("variant", "")).strip() != str(variant):
                    continue
                if str(row.get("budget", "")).strip() != str(budget):
                    continue

                seed = int(_to_float(row.get("seed", 0), 0.0))
                mode = str(row.get("aggregation_mode", "top1_hard")).strip()
                key = (seed, mode)
                payload = {
                    "seed": seed,
                    "aggregation_mode": mode,
                    "oracle_gap_pct": _to_float(row.get("oracle_gap_pct", 0.0)),
                    "top1_oracle_hit_score_level": _to_float(row.get("top1_oracle_hit_score_level", 0.0)),
                    "spearman_score_level": _to_float(row.get("spearman_score_level", 0.0)),
                    "mean_auroc": _to_float(row.get("mean_auroc", 0.0)),
                    "std_auroc": _to_float(row.get("std_auroc", 0.0)),
                    "auroc_delta_routed_vs_pooled": _to_float(row.get("auroc_delta_routed_vs_pooled", 0.0)),
                    "mean_bacc": _to_float(row.get("mean_bacc", 0.0)),
                    "std_bacc": _to_float(row.get("std_bacc", 0.0)),
                    "source_csv": str(csv_path),
                }
                per_key[key] = payload

    rows = list(per_key.values())
    if not rows:
        raise RuntimeError(
            f"No rows found after filtering for variant={variant} and budget={budget}."
        )
    return rows


def _read_metadata_baseline(path: Path | None) -> Tuple[Dict[int, float], str]:
    if path is None:
        return {}, "variantB_top1_reference_fallback"

    if not path.exists():
        raise FileNotFoundError(f"metadata baseline csv not found: {path}")

    seed_to_auroc: Dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(_to_float(row.get("seed", 0), 0.0))
            if "mean_auroc" in row:
                auroc = _to_float(row.get("mean_auroc", 0.0))
            elif "auroc" in row:
                auroc = _to_float(row.get("auroc", 0.0))
            elif "metadata_top1_mean_auroc" in row:
                auroc = _to_float(row.get("metadata_top1_mean_auroc", 0.0))
            else:
                raise ValueError(
                    "metadata baseline csv must contain one of columns: mean_auroc, auroc, metadata_top1_mean_auroc"
                )
            seed_to_auroc[seed] = auroc

    return seed_to_auroc, str(path)


def _build_decision_rows(
    rows: List[Dict[str, object]],
    metadata_baseline_by_seed: Dict[int, float],
    min_delta_vs_variant_top1: float,
    max_delta_std_tolerance: float,
) -> List[Dict[str, object]]:
    by_mode: Dict[str, List[Dict[str, object]]] = {}
    top1_by_seed: Dict[int, float] = {}

    for row in rows:
        mode = str(row["aggregation_mode"])
        seed = int(row["seed"])
        by_mode.setdefault(mode, []).append(row)
        if mode == "top1_hard":
            top1_by_seed[seed] = float(row["mean_auroc"])

    if not top1_by_seed:
        raise RuntimeError("No top1_hard rows found; cannot compute variantB_top1_reference deltas.")

    for seed in sorted(top1_by_seed.keys()):
        if seed not in metadata_baseline_by_seed:
            metadata_baseline_by_seed[seed] = float(top1_by_seed[seed])

    decision_rows: List[Dict[str, object]] = []
    for mode in sorted(by_mode.keys()):
        mode_rows = sorted(by_mode[mode], key=lambda r: int(r["seed"]))

        seeds: List[int] = []
        oracle_gap_pct_vals: List[float] = []
        top1_vals: List[float] = []
        spearman_vals: List[float] = []
        mean_auroc_vals: List[float] = []
        delta_vs_variant_top1_vals: List[float] = []
        delta_vs_metadata_vals: List[float] = []
        delta_vs_pooled_vals: List[float] = []

        for row in mode_rows:
            seed = int(row["seed"])
            seeds.append(seed)
            oracle_gap_pct_vals.append(float(row["oracle_gap_pct"]))
            top1_vals.append(float(row["top1_oracle_hit_score_level"]))
            spearman_vals.append(float(row["spearman_score_level"]))
            mean_auroc_vals.append(float(row["mean_auroc"]))
            delta_vs_pooled_vals.append(float(row["auroc_delta_routed_vs_pooled"]))

            if seed not in top1_by_seed:
                raise RuntimeError(f"Missing top1_hard reference for seed {seed}")
            delta_vs_variant_top1_vals.append(float(row["mean_auroc"]) - float(top1_by_seed[seed]))
            delta_vs_metadata_vals.append(float(row["mean_auroc"]) - float(metadata_baseline_by_seed[seed]))

        oracle_gap_mean, oracle_gap_std = _mean_std(oracle_gap_pct_vals)
        top1_mean, top1_std = _mean_std(top1_vals)
        spearman_mean, spearman_std = _mean_std(spearman_vals)
        auroc_mean, auroc_std = _mean_std(mean_auroc_vals)
        d_var_mean, d_var_std = _mean_std(delta_vs_variant_top1_vals)
        d_meta_mean, d_meta_std = _mean_std(delta_vs_metadata_vals)
        d_pool_mean, d_pool_std = _mean_std(delta_vs_pooled_vals)

        passes_delta = bool(d_var_mean >= float(min_delta_vs_variant_top1))
        passes_stability = bool(d_var_std <= float(max_delta_std_tolerance))
        decision_rows.append(
            {
                "aggregation_mode": mode,
                "n_seeds": len(seeds),
                "seeds": ",".join(str(s) for s in seeds),
                "oracle_gap_pct_mean": oracle_gap_mean,
                "oracle_gap_pct_std": oracle_gap_std,
                "top1_oracle_hit_score_level_mean": top1_mean,
                "top1_oracle_hit_score_level_std": top1_std,
                "spearman_score_level_mean": spearman_mean,
                "spearman_score_level_std": spearman_std,
                "mean_auroc": auroc_mean,
                "std_auroc": auroc_std,
                "auroc_delta_vs_variantB_top1_reference_mean": d_var_mean,
                "auroc_delta_vs_variantB_top1_reference_std": d_var_std,
                "min_delta_vs_variantB_top1_required": float(min_delta_vs_variant_top1),
                "max_delta_std_tolerance": float(max_delta_std_tolerance),
                "passes_delta_threshold": int(passes_delta),
                "passes_stability_tolerance": int(passes_stability),
                "auroc_delta_vs_metadata_top1_baseline_mean": d_meta_mean,
                "auroc_delta_vs_metadata_top1_baseline_std": d_meta_std,
                "auroc_delta_vs_pooled_baseline_mean": d_pool_mean,
                "auroc_delta_vs_pooled_baseline_std": d_pool_std,
            }
        )

    # Add a compact recommendation marker based on threshold + tolerance.
    candidates = [
        r
        for r in decision_rows
        if str(r.get("aggregation_mode", "")) != "top1_hard"
        and int(r.get("passes_delta_threshold", 0)) == 1
        and int(r.get("passes_stability_tolerance", 0)) == 1
    ]
    selected_mode = None
    if candidates:
        selected_mode = str(
            max(
                candidates,
                key=lambda r: float(r.get("auroc_delta_vs_variantB_top1_reference_mean", 0.0)),
            )["aggregation_mode"]
        )

    for r in decision_rows:
        mode = str(r.get("aggregation_mode", ""))
        if selected_mode is not None and mode == selected_mode:
            r["decision"] = "select"
        elif mode == "top1_hard" and selected_mode is None:
            r["decision"] = "keep_top1_stop"
        else:
            r["decision"] = "not_selected"

    return decision_rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No decision rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: List[Dict[str, object]], metadata_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        selected = [r for r in rows if str(r.get("decision", "")) == "select"]
        selected_mode = str(selected[0]["aggregation_mode"]) if selected else "top1_hard"
        decision_policy = "select_mode" if selected else "keep_top1_stop"
        threshold = float(rows[0].get("min_delta_vs_variantB_top1_required", 0.0)) if rows else 0.0
        tolerance = float(rows[0].get("max_delta_std_tolerance", 0.0)) if rows else 0.0

        f.write("# Aggregation Decision Table\n\n")
        f.write("- Primary scope: variant B, budget_1.0x, seeds 42/43/44\n")
        f.write(f"- metadata_top1_baseline source: {metadata_source}\n")
        f.write("- variantB_top1_reference: aggregation_mode=top1_hard\n")
        f.write(f"- decision_policy: {decision_policy} (selected_mode={selected_mode})\n")
        f.write(
            f"- stop_rule: require delta_vs_variantB_top1_mean >= {threshold:.4f} and delta_std <= {tolerance:.4f}\n"
        )
        f.write("\n")
        f.write("| mode | decision | n_seeds | oracle_gap_pct | top1_score | spearman_score | mean_auroc | std_auroc | delta_vs_variantB_top1 | delta_vs_metadata_top1 | delta_vs_pooled |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['aggregation_mode']} | {row.get('decision', 'not_selected')} | {row['n_seeds']} | "
                f"{float(row['oracle_gap_pct_mean']):.4f} +- {float(row['oracle_gap_pct_std']):.4f} | "
                f"{float(row['top1_oracle_hit_score_level_mean']):.4f} +- {float(row['top1_oracle_hit_score_level_std']):.4f} | "
                f"{float(row['spearman_score_level_mean']):.4f} +- {float(row['spearman_score_level_std']):.4f} | "
                f"{float(row['mean_auroc']):.4f} | {float(row['std_auroc']):.4f} | "
                f"{float(row['auroc_delta_vs_variantB_top1_reference_mean']):.4f} +- {float(row['auroc_delta_vs_variantB_top1_reference_std']):.4f} | "
                f"{float(row['auroc_delta_vs_metadata_top1_baseline_mean']):.4f} +- {float(row['auroc_delta_vs_metadata_top1_baseline_std']):.4f} | "
                f"{float(row['auroc_delta_vs_pooled_baseline_mean']):.4f} +- {float(row['auroc_delta_vs_pooled_baseline_std']):.4f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregation decision table from run manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/comparison_tables/aggregation_ablation_run_manifest.txt"),
    )
    parser.add_argument("--variant", type=str, default="B")
    parser.add_argument("--budget", type=str, default="budget_1.0x")
    parser.add_argument(
        "--metadata-baseline-csv",
        type=Path,
        default=None,
        help="Optional CSV with per-seed metadata baseline AUROC.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/comparison_tables/aggregation_decision_table.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/summaries/aggregation_decision_table.md"),
    )
    parser.add_argument(
        "--min-delta-vs-variant-top1",
        type=float,
        default=0.005,
        help="Minimum AUROC delta vs variantB top1 required to select a non-top1 aggregation mode.",
    )
    parser.add_argument(
        "--max-delta-std-tolerance",
        type=float,
        default=0.010,
        help="Maximum tolerated std of delta_vs_variantB_top1 for a selectable mode.",
    )
    args = parser.parse_args()

    report_paths = _load_manifest(args.manifest)
    rows = _read_hybrid_rows(report_paths=report_paths, variant=args.variant, budget=args.budget)
    metadata_baseline_by_seed, metadata_source = _read_metadata_baseline(args.metadata_baseline_csv)
    decision_rows = _build_decision_rows(
        rows=rows,
        metadata_baseline_by_seed=metadata_baseline_by_seed,
        min_delta_vs_variant_top1=float(args.min_delta_vs_variant_top1),
        max_delta_std_tolerance=float(args.max_delta_std_tolerance),
    )

    _write_csv(args.output_csv, decision_rows)
    _write_md(args.output_md, decision_rows, metadata_source=metadata_source)

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
