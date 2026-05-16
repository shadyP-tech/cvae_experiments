#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.support_set_calibration import (
    ResponseProxyBaseline,
    SupportSetRunMeta,
    aggregate_support_set_rows,
    evaluate_support_set_calibration_for_run,
    write_csv,
)


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


def _is_leakage_safe_response_row(row: Mapping[str, object], target_domain: int) -> bool:
    method = str(row.get("method", ""))
    if method.startswith("oracle_") or method.startswith("semi_oracle_"):
        return False
    if int(float(row.get("diagnostic_only", 0) or 0)) != 0:
        return False
    if int(float(row.get("control_only", 0) or 0)) != 0:
        return False
    if str(row.get("blocked_feature_terms", "")).strip():
        return False
    try:
        selected = int(float(row.get("selected_expert", -999999)))
        if selected == int(target_domain):
            return False
    except Exception:
        pass
    return True


def _load_response_proxy_lookup(path: Path | None) -> Dict[Tuple[str, int, str, str, int], ResponseProxyBaseline]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Response-proxy CSV not found: {path}")

    by_key: Dict[Tuple[str, int, str, str, int], List[Mapping[str, object]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            method = str(row.get("method", ""))
            regime = str(row.get("feature_regime", ""))
            if method == "metadata_routing":
                continue
            if "response" not in regime and str(row.get("response_feature_mode", "")) != "on":
                continue
            target = int(float(row.get("heldout_query_domain", row.get("target_domain", 0)) or 0))
            key = (
                str(row.get("dataset_name", "")),
                int(float(row.get("seed", 0) or 0)),
                str(row.get("backbone_type", "")),
                str(row.get("run_id", "")),
                int(target),
            )
            by_key.setdefault(key, []).append(row)

    out: Dict[Tuple[str, int, str, str, int], ResponseProxyBaseline] = {}
    for key, rows in by_key.items():
        target = int(key[-1])
        ordered = sorted(
            rows,
            key=lambda r: float(r.get("normalized_metadata_to_oracle_gap", r.get("metadata_to_oracle_gap", 1e12)) or 1e12),
        )
        row = ordered[0]
        adoption_flag = int(float(row.get("adoption_eligible", 0) or 0))
        adoption_eligible = 1 if adoption_flag == 1 and _is_leakage_safe_response_row(row, target) else 0
        selected_nelbo = float(row.get("selected_routing_nelbo", row.get("selected_eval_nelbo", 0.0)) or 0.0)
        oracle_nelbo = float(row.get("oracle_routing_nelbo", row.get("oracle_eval_nelbo", selected_nelbo)) or selected_nelbo)
        worst_nelbo = float(row.get("worst_routing_nelbo", max(selected_nelbo, oracle_nelbo)) or max(selected_nelbo, oracle_nelbo))
        norm_gap = float(row.get("normalized_metadata_to_oracle_gap", row.get("normalized_oracle_gap", 0.0)) or 0.0)
        gap = float(row.get("metadata_to_oracle_gap", row.get("oracle_gap", selected_nelbo - oracle_nelbo)) or 0.0)
        out[key] = ResponseProxyBaseline(
            selected_expert=int(float(row.get("selected_expert", -1) or -1)),
            oracle_expert=int(float(row.get("oracle_expert", -1) or -1)),
            selected_eval_nelbo=selected_nelbo,
            oracle_eval_nelbo=oracle_nelbo,
            worst_eval_nelbo=worst_nelbo,
            normalized_oracle_gap=norm_gap,
            oracle_gap=gap,
            top1_oracle_hit=float(row.get("top1_agreement_with_best_expert", row.get("top1_oracle_hit", 0.0)) or 0.0),
            spearman_support_vs_eval_utility=float(
                row.get("spearman_similarity_vs_neg_nelbo", row.get("spearman_support_vs_eval_utility", 0.0)) or 0.0
            ),
            adoption_eligible=adoption_eligible,
            baseline_available=1,
            source_method=str(row.get("method", "")),
        )
    return out


def _as_abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BreakHis LOQDO support-set utility calibration.")
    parser.add_argument("--experiment-dirs", nargs="+", default=None)
    parser.add_argument("--variant", default="B")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--metadata-tau", type=float, default=100.0)
    parser.add_argument("--support-sizes", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--support-seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument("--sampling-policies", nargs="+", default=["class_balanced", "random"])
    parser.add_argument("--topk-values", nargs="*", type=int, default=[])
    parser.add_argument("--softmax-temperatures", nargs="*", type=float, default=[])
    parser.add_argument("--response-proxy-raw", default=None)
    parser.add_argument("--pair-table-dirname", default="support_set_calibration_loqdo")
    parser.add_argument(
        "--raw-out",
        default="results/comparison_tables/support_set_calibration_loqdo_breakhis_raw.csv",
    )
    parser.add_argument(
        "--stats-out",
        default="results/comparison_tables/support_set_calibration_loqdo_breakhis_stats.csv",
    )
    parser.add_argument(
        "--summary-json-out",
        default="results/comparison_tables/support_set_calibration_loqdo_breakhis_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dirs = [Path(p) for p in args.experiment_dirs] if args.experiment_dirs else _default_experiment_dirs(PROJECT_ROOT)
    resolved_runs = [_resolve_run_dir(p if p.is_absolute() else PROJECT_ROOT / p) for p in experiment_dirs]
    response_lookup = _load_response_proxy_lookup(_as_abs(args.response_proxy_raw) if args.response_proxy_raw else None)

    all_rows: List[Mapping[str, object]] = []
    run_summaries: List[Dict[str, object]] = []
    for run_dir in resolved_runs:
        cfg = _load_run_config(run_dir)
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
        features_cfg = cfg.get("features", {}) if isinstance(cfg.get("features", {}), dict) else {}
        run_meta = _run_meta(run_dir, cfg, variant=str(args.variant))
        variant_name = str(args.variant).upper()
        variant_checkpoint = run_dir / "checkpoints" / f"hybrid_variant_{variant_name}.pt"
        expert_manifest = run_dir / "checkpoints" / "expert_checkpoints.json"
        rows = evaluate_support_set_calibration_for_run(
            test_cache=run_dir / "embeddings" / "test.pt",
            variant_checkpoint=variant_checkpoint if variant_checkpoint.exists() else None,
            expert_manifest=expert_manifest if expert_manifest.exists() else None,
            hidden_dim=int(model_cfg.get("hidden_dim", 0)),
            latent_dim=int(model_cfg.get("latent_dim", 0)),
            metadata_constraint_cfg=model_cfg.get("metadata_constraint", {}) if isinstance(model_cfg, dict) else {},
            run_meta=run_meta,
            support_sizes=[int(v) for v in args.support_sizes],
            support_seeds=[int(v) for v in args.support_seeds],
            sampling_policies=[str(v) for v in args.sampling_policies],
            batch_size=int(args.batch_size),
            metadata_tau=float(args.metadata_tau),
            topk_values=[int(v) for v in args.topk_values],
            softmax_temperatures=[float(v) for v in args.softmax_temperatures],
            response_proxy_lookup=response_lookup,
        )
        all_rows.extend(rows)

        report_dir = run_dir / "reports" / str(args.pair_table_dirname)
        write_csv(report_dir / "support_set_calibration_raw.csv", rows)
        run_summary = {
            "run_dir": str(run_dir),
            "dataset_name": run_meta.dataset_name,
            "seed": int(run_meta.seed),
            "backbone_type": run_meta.backbone_type,
            "variant": run_meta.variant,
            "embedding_dim": int(features_cfg.get("embedding_dim", 0)),
            "support_sizes": [int(v) for v in args.support_sizes],
            "support_seeds": [int(v) for v in args.support_seeds],
            "sampling_policies": [str(v) for v in args.sampling_policies],
            "n_rows": int(len(rows)),
            "response_proxy_rows_available": int(sum(1 for r in rows if str(r.get("method", "")) == "response_proxy_baseline")),
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
        run_summaries.append(run_summary)

    raw_rows = sorted(
        all_rows,
        key=lambda r: (
            str(r.get("dataset_name", "")),
            str(r.get("backbone_type", "")),
            str(r.get("run_id", "")),
            int(float(r.get("target_domain", 0) or 0)),
            int(float(r.get("support_seed", 0) or 0)),
            int(float(r.get("support_size_requested", 0) or 0)),
            str(r.get("sampling_policy", "")),
            str(r.get("method", "")),
        ),
    )
    stats_rows = aggregate_support_set_rows(raw_rows)

    raw_out = _as_abs(str(args.raw_out))
    stats_out = _as_abs(str(args.stats_out))
    summary_out = _as_abs(str(args.summary_json_out))
    write_csv(raw_out, raw_rows)
    write_csv(stats_out, stats_rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(
            {
                "protocol": {
                    "name": "support_set_utility_calibration_loqdo",
                    "candidate_policy": "exclude_target_domain",
                    "support_sizes": [int(v) for v in args.support_sizes],
                    "support_seeds": [int(v) for v in args.support_seeds],
                    "sampling_policies": [str(v) for v in args.sampling_policies],
                    "metadata_tau": float(args.metadata_tau),
                    "topk_values": [int(v) for v in args.topk_values],
                    "softmax_temperatures": [float(v) for v in args.softmax_temperatures],
                    "response_proxy_raw": str(args.response_proxy_raw or ""),
                },
                "runs": run_summaries,
                "raw_csv": str(raw_out),
                "stats_csv": str(stats_out),
                "n_raw_rows": int(len(raw_rows)),
                "n_stats_rows": int(len(stats_rows)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote raw rows: {raw_out}")
    print(f"Wrote stats rows: {stats_out}")
    print(f"Wrote summary: {summary_out}")


if __name__ == "__main__":
    main()
