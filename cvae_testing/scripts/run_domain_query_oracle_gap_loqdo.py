#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.domain_query_oracle_gap import (
    DEFAULT_INTERPRETATION_THRESHOLDS,
    aggregate_domain_query_oracle_gap_rows,
    evaluate_domain_query_oracle_gap_for_run,
)
from src.eval.evaluators.support_set_calibration import SupportSetRunMeta, write_csv


def _resolve_run_dir(path: Path) -> Path:
    if path.is_dir() and (path / "config_resolved.yaml").exists():
        return path
    latest = path / "latest.txt"
    if path.is_dir() and latest.exists():
        run_id = latest.read_text(encoding="utf-8").strip()
        resolved = path / run_id
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve run directory from {path}. Expected a run dir with config_resolved.yaml "
        "or an experiment dir with latest.txt."
    )


def _default_experiment_dirs(root: Path) -> List[Path]:
    return [
        root / "outputs" / "breakhis" / "hybrid_ablation_extractor_resnet18_v1",
        root / "outputs" / "breakhis" / "hybrid_ablation_extractor_resnet50_v1",
        root / "outputs" / "breakhis" / "hybrid_ablation_extractor_dinov2_vitb14_v1",
    ]


def _load_run_config(run_dir: Path) -> Mapping[str, object]:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Config is not a dictionary: {config_path}")
    return cfg


def _run_meta(run_dir: Path, cfg: Mapping[str, object], variant: str) -> SupportSetRunMeta:
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment", {}), dict) else {}
    features = cfg.get("features", {}) if isinstance(cfg.get("features", {}), dict) else {}
    return SupportSetRunMeta(
        dataset_name=str(experiment.get("dataset_name", "unknown")),
        seed=int(cfg.get("seed", 0)),
        backbone_type=str(features.get("backbone_type", "unknown")),
        run_id=run_dir.name,
        variant=str(variant),
        run_dir=str(run_dir),
    )


def _as_abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _thresholds_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "low_normalized_gap": float(args.low_normalized_gap_threshold),
        "high_normalized_gap": float(args.high_normalized_gap_threshold),
        "high_normalized_entropy": float(args.high_normalized_entropy_threshold),
        "high_switch_rate": float(args.high_switch_rate_threshold),
        "high_margin_low_margin_share": float(args.high_margin_low_margin_share_threshold),
        "metadata_close_to_fixed_oracle": float(args.metadata_close_to_fixed_oracle_threshold),
    }


def _write_summary_markdown(
    *,
    path: Path,
    stats_rows: List[Mapping[str, object]],
    raw_out: Path,
    per_sample_out: Path,
    stats_out: Path,
    summary_json_out: Path,
    thresholds: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Fixed-Domain vs Per-Query Oracle Gap Under BreakHis LOQDO")
    lines.append("")
    lines.append("Thresholds in this report are descriptive heuristics, not statistical acceptance criteria.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Raw folds: `{raw_out}`")
    lines.append(f"- Per-sample rows: `{per_sample_out}`")
    lines.append(f"- Aggregate stats: `{stats_out}`")
    lines.append(f"- JSON summary: `{summary_json_out}`")
    lines.append("")
    lines.append("## Advisory Thresholds")
    lines.append("")
    lines.append("| threshold | value |")
    lines.append("|---|---:|")
    for key, value in thresholds.items():
        lines.append(f"| `{key}` | {float(value):.6g} |")
    lines.append("")
    lines.append("## Interpretation Table")
    lines.append("")
    lines.append("| Result pattern | Interpretation | Consequence |")
    lines.append("|---|---|---|")
    lines.append(
        "| Low gap, low entropy, low switch rate | One fixed expert is effectively optimal per target domain | "
        "Per-query routing is unnecessary; domain routing is enough in principle |"
    )
    lines.append(
        "| Low gap, high switch rate | Expert choices vary, but utility differences are small | "
        "Per-query routing may not improve NELBO much |"
    )
    lines.append(
        "| High gap, high entropy, high margin | Strong within-domain heterogeneity | "
        "Per-query compatibility learning or expert aggregation is justified |"
    )
    lines.append(
        "| High gap, low entropy | Variation is concentrated | Inspect class, subtype, patient, or sample-quality effects |"
    )
    lines.append(
        "| High gap, metadata close to fixed oracle | Metadata captures most fixed-expert signal | "
        "Learned methods must beat metadata stably |"
    )
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    if not stats_rows:
        lines.append("No aggregate rows were produced.")
    else:
        lines.append(
            "| dataset | backbone | variant | n_folds | norm_gap_mean | switch_mean | "
            "entropy_norm_mean | low_margin_share | pattern |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for row in stats_rows:
            lines.append(
                "| "
                f"{row.get('dataset_name', '')} | "
                f"{row.get('backbone_type', '')} | "
                f"{row.get('variant', '')} | "
                f"{int(float(row.get('n_folds', 0) or 0))} | "
                f"{float(row.get('normalized_fixed_to_query_oracle_gap_mean', 0.0) or 0.0):.4f} | "
                f"{float(row.get('per_query_oracle_switch_rate_mean', 0.0) or 0.0):.4f} | "
                f"{float(row.get('per_query_expert_entropy_normalized_mean', 0.0) or 0.0):.4f} | "
                f"{float(row.get('low_margin_share_mean', 0.0) or 0.0):.4f} | "
                f"{row.get('interpretation_pattern', '')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BreakHis LOQDO fixed-domain vs per-query oracle gap diagnostic.")
    parser.add_argument("--experiment-dirs", nargs="+", default=None)
    parser.add_argument("--variant", default="B")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--metadata-tau", type=float, default=100.0)
    parser.add_argument("--eps", type=float, default=1.0e-12)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1337)
    parser.add_argument("--low-margin-abs-threshold", type=float, default=1.0e-8)
    parser.add_argument("--low-margin-rel-threshold", type=float, default=0.05)
    parser.add_argument("--pair-table-dirname", default="domain_query_oracle_gap_loqdo")
    parser.add_argument(
        "--raw-out",
        default="results/comparison_tables/domain_query_oracle_gap_loqdo_breakhis_raw.csv",
    )
    parser.add_argument(
        "--per-sample-out",
        default="results/comparison_tables/domain_query_oracle_gap_loqdo_breakhis_per_sample.csv",
    )
    parser.add_argument(
        "--stats-out",
        default="results/comparison_tables/domain_query_oracle_gap_loqdo_breakhis_stats.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        default="results/comparison_tables/domain_query_oracle_gap_loqdo_breakhis_summary.json",
    )
    parser.add_argument(
        "--summary-md-out",
        default="results/summaries/domain_query_oracle_gap_loqdo_breakhis_summary.md",
    )
    parser.add_argument(
        "--low-normalized-gap-threshold",
        type=float,
        default=DEFAULT_INTERPRETATION_THRESHOLDS["low_normalized_gap"],
    )
    parser.add_argument(
        "--high-normalized-gap-threshold",
        type=float,
        default=DEFAULT_INTERPRETATION_THRESHOLDS["high_normalized_gap"],
    )
    parser.add_argument(
        "--high-normalized-entropy-threshold",
        type=float,
        default=DEFAULT_INTERPRETATION_THRESHOLDS["high_normalized_entropy"],
    )
    parser.add_argument(
        "--high-switch-rate-threshold",
        type=float,
        default=DEFAULT_INTERPRETATION_THRESHOLDS["high_switch_rate"],
    )
    parser.add_argument(
        "--high-margin-low-margin-share-threshold",
        type=float,
        default=DEFAULT_INTERPRETATION_THRESHOLDS["high_margin_low_margin_share"],
    )
    parser.add_argument(
        "--metadata-close-to-fixed-oracle-threshold",
        type=float,
        default=DEFAULT_INTERPRETATION_THRESHOLDS["metadata_close_to_fixed_oracle"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dirs = [Path(p) for p in args.experiment_dirs] if args.experiment_dirs else _default_experiment_dirs(PROJECT_ROOT)
    resolved_runs = [_resolve_run_dir(p if p.is_absolute() else PROJECT_ROOT / p) for p in experiment_dirs]
    thresholds = _thresholds_from_args(args)

    all_fold_rows: List[Mapping[str, object]] = []
    all_sample_rows: List[Mapping[str, object]] = []
    run_summaries: List[Dict[str, object]] = []
    for run_dir in resolved_runs:
        cfg = _load_run_config(run_dir)
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
        features_cfg = cfg.get("features", {}) if isinstance(cfg.get("features", {}), dict) else {}
        run_meta = _run_meta(run_dir, cfg, variant=str(args.variant))
        variant_name = str(args.variant).upper()
        variant_checkpoint = run_dir / "checkpoints" / f"hybrid_variant_{variant_name}.pt"
        expert_manifest = run_dir / "checkpoints" / "expert_checkpoints.json"
        fold_rows, sample_rows = evaluate_domain_query_oracle_gap_for_run(
            test_cache=run_dir / "embeddings" / "test.pt",
            variant_checkpoint=variant_checkpoint if variant_checkpoint.exists() else None,
            expert_manifest=expert_manifest if expert_manifest.exists() else None,
            hidden_dim=int(model_cfg.get("hidden_dim", 0)),
            latent_dim=int(model_cfg.get("latent_dim", 0)),
            metadata_constraint_cfg=model_cfg.get("metadata_constraint", {}) if isinstance(model_cfg, dict) else {},
            run_meta=run_meta,
            batch_size=int(args.batch_size),
            eps=float(args.eps),
            bootstrap_reps=int(args.bootstrap_reps),
            bootstrap_seed=int(args.bootstrap_seed),
            low_margin_abs_threshold=float(args.low_margin_abs_threshold),
            low_margin_rel_threshold=float(args.low_margin_rel_threshold),
            metadata_tau=float(args.metadata_tau),
        )
        all_fold_rows.extend(fold_rows)
        all_sample_rows.extend(sample_rows)

        report_dir = run_dir / "reports" / str(args.pair_table_dirname)
        write_csv(report_dir / "domain_query_oracle_gap_folds.csv", fold_rows)
        write_csv(report_dir / "domain_query_oracle_gap_per_sample.csv", sample_rows)
        run_summary = {
            "run_dir": str(run_dir),
            "dataset_name": run_meta.dataset_name,
            "seed": int(run_meta.seed),
            "backbone_type": run_meta.backbone_type,
            "variant": run_meta.variant,
            "embedding_dim": int(features_cfg.get("embedding_dim", 0)),
            "n_fold_rows": int(len(fold_rows)),
            "n_per_sample_rows": int(len(sample_rows)),
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
        run_summaries.append(run_summary)

    fold_rows_sorted = sorted(
        all_fold_rows,
        key=lambda r: (
            str(r.get("dataset_name", "")),
            str(r.get("backbone_type", "")),
            str(r.get("run_id", "")),
            int(float(r.get("target_domain", 0) or 0)),
        ),
    )
    sample_rows_sorted = sorted(
        all_sample_rows,
        key=lambda r: (
            str(r.get("dataset_name", "")),
            str(r.get("backbone_type", "")),
            str(r.get("run_id", "")),
            int(float(r.get("target_domain", 0) or 0)),
            int(float(r.get("sample_index", 0) or 0)),
        ),
    )
    stats_rows = aggregate_domain_query_oracle_gap_rows(
        fold_rows_sorted,
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
        interpretation_thresholds=thresholds,
    )

    raw_out = _as_abs(str(args.raw_out))
    per_sample_out = _as_abs(str(args.per_sample_out))
    stats_out = _as_abs(str(args.stats_out))
    summary_json_out = _as_abs(str(args.summary_json_out))
    summary_md_out = _as_abs(str(args.summary_md_out))

    write_csv(raw_out, fold_rows_sorted)
    write_csv(per_sample_out, sample_rows_sorted)
    write_csv(stats_out, stats_rows)
    summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    summary_json_out.write_text(
        json.dumps(
            {
                "protocol": {
                    "name": "domain_query_oracle_gap_loqdo",
                    "candidate_policy": "exclude_target_domain",
                    "oracle_definition": "fixed_domain_vs_per_query_min_nelbo",
                    "bootstrap_reps": int(args.bootstrap_reps),
                    "bootstrap_seed": int(args.bootstrap_seed),
                    "eps": float(args.eps),
                    "metadata_tau": float(args.metadata_tau),
                    "low_margin_abs_threshold": float(args.low_margin_abs_threshold),
                    "low_margin_rel_threshold": float(args.low_margin_rel_threshold),
                    "interpretation_thresholds_are_descriptive_heuristics": True,
                    "interpretation_thresholds": thresholds,
                },
                "runs": run_summaries,
                "raw_csv": str(raw_out),
                "per_sample_csv": str(per_sample_out),
                "stats_csv": str(stats_out),
                "summary_md": str(summary_md_out),
                "n_fold_rows": int(len(fold_rows_sorted)),
                "n_per_sample_rows": int(len(sample_rows_sorted)),
                "n_stats_rows": int(len(stats_rows)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary_markdown(
        path=summary_md_out,
        stats_rows=stats_rows,
        raw_out=raw_out,
        per_sample_out=per_sample_out,
        stats_out=stats_out,
        summary_json_out=summary_json_out,
        thresholds=thresholds,
    )
    print(f"Wrote raw rows: {raw_out}")
    print(f"Wrote per-sample rows: {per_sample_out}")
    print(f"Wrote stats rows: {stats_out}")
    print(f"Wrote summary JSON: {summary_json_out}")
    print(f"Wrote summary Markdown: {summary_md_out}")


if __name__ == "__main__":
    main()
