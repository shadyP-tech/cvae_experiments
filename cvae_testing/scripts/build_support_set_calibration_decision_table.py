#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.support_set_calibration import global_calibration_error_bin10, write_csv


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"Raw CSV is empty: {path}")
    return rows


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(clean)) if clean else 0.0


def _std(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0
    mu = float(mean(clean))
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / len(clean))


def _matched_key(row: Mapping[str, object]) -> Tuple[str, str, str, str, str, str, str, str, str]:
    return (
        str(row.get("dataset_name", "")),
        str(row.get("backbone_type", "")),
        str(row.get("run_id", "")),
        str(row.get("target_domain", "")),
        str(row.get("support_seed", "")),
        str(row.get("support_size_requested", "")),
        str(row.get("sampling_policy", "")),
        str(row.get("sampling_policy_effective", "")),
        str(row.get("support_eval_split_id", "")),
    )


def _condition_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (
        str(row.get("dataset_name", "")),
        str(row.get("variant", "")),
        str(row.get("method", "")),
    )


def _safe_audit(row: Mapping[str, object]) -> bool:
    return bool(
        int(_to_float(row.get("target_expert_excluded", 0))) == 1
        and int(_to_float(row.get("support_eval_disjoint", 0))) == 1
        and int(_to_float(row.get("support_is_target_local", 0))) == 1
        and int(_to_float(row.get("routing_uses_eval_nelbo", 1))) == 0
        and int(_to_float(row.get("routing_uses_eval_indices", 1))) == 0
    )


def _paired_deltas(rows: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str, str, str, str, str, str, str], Dict[str, dict]] = {}
    for row in rows:
        if str(row.get("split_status", "")) != "ok":
            continue
        groups.setdefault(_matched_key(row), {})[str(row.get("method", ""))] = row

    out: List[dict] = []
    for key, by_method in sorted(groups.items()):
        support = by_method.get("support_set_calibration_top1")
        metadata = by_method.get("metadata_ordinal_baseline")
        embedding = by_method.get("static_embedding_baseline")
        if support is None or metadata is None or embedding is None:
            continue
        support_gap = _to_float(support.get("normalized_oracle_gap", 0.0))
        metadata_gap = _to_float(metadata.get("normalized_oracle_gap", 0.0))
        embedding_gap = _to_float(embedding.get("normalized_oracle_gap", 0.0))
        out.append(
            {
                "matched_key": "|".join(key),
                "dataset_name": support.get("dataset_name", ""),
                "backbone_type": support.get("backbone_type", ""),
                "run_id": support.get("run_id", ""),
                "target_domain": support.get("target_domain", ""),
                "support_seed": support.get("support_seed", ""),
                "support_size_requested": support.get("support_size_requested", ""),
                "sampling_policy": support.get("sampling_policy", ""),
                "sampling_policy_effective": support.get("sampling_policy_effective", ""),
                "support_gap": support_gap,
                "metadata_gap": metadata_gap,
                "embedding_gap": embedding_gap,
                "normalized_gap_reduction_vs_metadata": metadata_gap - support_gap,
                "normalized_gap_reduction_vs_static_embedding": embedding_gap - support_gap,
                "top1_delta_vs_metadata": _to_float(support.get("top1_oracle_hit", 0.0))
                - _to_float(metadata.get("top1_oracle_hit", 0.0)),
                "top1_delta_vs_static_embedding": _to_float(support.get("top1_oracle_hit", 0.0))
                - _to_float(embedding.get("top1_oracle_hit", 0.0)),
                "safe_audit": int(_safe_audit(support)),
            }
        )
    return out


def _non_degrading_k_trend(deltas: Sequence[dict], *, tolerance: float) -> Tuple[int, str]:
    by_k: Dict[int, List[float]] = {}
    for row in deltas:
        k = int(_to_float(row.get("support_size_requested", 0)))
        by_k.setdefault(k, []).append(float(row.get("support_gap", 0.0)))
    if len(by_k) < 2:
        return 0, json.dumps({str(k): _mean(v) for k, v in sorted(by_k.items())}, sort_keys=True)
    means = {int(k): _mean(v) for k, v in sorted(by_k.items())}
    smallest = min(means)
    largest = max(means)
    ok = means[largest] <= means[smallest] + float(tolerance)
    return int(ok), json.dumps({str(k): float(v) for k, v in means.items()}, sort_keys=True)


def build_decision_rows(
    rows: Sequence[dict],
    *,
    min_gap_reduction: float,
    max_top1_regression: float,
    min_positive_seed_fraction: float,
    k_trend_tolerance: float,
    min_backbones: int,
    min_model_seeds: int,
) -> Tuple[List[dict], List[dict], dict]:
    valid_rows = [r for r in rows if str(r.get("split_status", "")) == "ok"]
    deltas = _paired_deltas(valid_rows)

    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in valid_rows:
        groups.setdefault(_condition_key(row), []).append(row)

    decision_rows: List[dict] = []
    for key, vals in sorted(groups.items()):
        dataset_name, variant, method = key
        norm_gap = [_to_float(v.get("normalized_oracle_gap", 0.0)) for v in vals]
        top1 = [_to_float(v.get("top1_oracle_hit", 0.0)) for v in vals]
        spearman = [_to_float(v.get("spearman_support_vs_eval_utility", 0.0)) for v in vals]
        cal = [_to_float(v.get("calibration_mae", 0.0)) for v in vals]
        method_deltas = [
            d
            for d in deltas
            if str(d.get("dataset_name", "")) == dataset_name
            and method == "support_set_calibration_top1"
        ]
        meta_reductions = [_to_float(d.get("normalized_gap_reduction_vs_metadata", 0.0)) for d in method_deltas]
        emb_reductions = [_to_float(d.get("normalized_gap_reduction_vs_static_embedding", 0.0)) for d in method_deltas]
        top1_meta = [_to_float(d.get("top1_delta_vs_metadata", 0.0)) for d in method_deltas]
        top1_emb = [_to_float(d.get("top1_delta_vs_static_embedding", 0.0)) for d in method_deltas]
        safe_audit = all(int(_to_float(v.get("target_expert_excluded", 0))) == 1 for v in vals) and all(
            _safe_audit(v) or int(_to_float(v.get("diagnostic_only", 0))) == 1 for v in vals
        )
        positive_seed_fraction = 0.0
        if method_deltas:
            by_seed: Dict[str, List[float]] = {}
            for d in method_deltas:
                by_seed.setdefault(str(d.get("support_seed", "")), []).append(
                    min(
                        _to_float(d.get("normalized_gap_reduction_vs_metadata", 0.0)),
                        _to_float(d.get("normalized_gap_reduction_vs_static_embedding", 0.0)),
                    )
                )
            seed_pass = [1 for values in by_seed.values() if _mean(values) >= float(min_gap_reduction)]
            positive_seed_fraction = float(len(seed_pass) / max(len(by_seed), 1))
        k_trend_ok, gap_by_k_json = _non_degrading_k_trend(method_deltas, tolerance=float(k_trend_tolerance))

        adoption_gate = 0
        if method == "support_set_calibration_top1":
            run_seeds = {str(v.get("seed", "")) for v in vals}
            backbones = {str(v.get("backbone_type", "")) for v in vals}
            enough_coverage = len(run_seeds) >= int(min_model_seeds) and len(backbones) >= int(min_backbones)
            adoption_gate = int(
                _mean(meta_reductions) >= float(min_gap_reduction)
                and _mean(emb_reductions) >= float(min_gap_reduction)
                and _mean(top1_meta) >= -float(max_top1_regression)
                and _mean(top1_emb) >= -float(max_top1_regression)
                and positive_seed_fraction >= float(min_positive_seed_fraction)
                and bool(k_trend_ok)
                and bool(enough_coverage)
                and bool(safe_audit)
            )

        decision_rows.append(
            {
                "dataset_name": dataset_name,
                "backbone_type": "all",
                "variant": variant,
                "method": method,
                "n_rows": int(len(vals)),
                "n_backbones": int(len(set(str(v.get("backbone_type", "")) for v in vals))),
                "n_model_seeds": int(len(set(str(v.get("seed", "")) for v in vals))),
                "adoption_eligible": int(max(int(_to_float(v.get("adoption_eligible", 0))) for v in vals)),
                "diagnostic_only": int(max(int(_to_float(v.get("diagnostic_only", 0))) for v in vals)),
                "exploratory_only": int(max(int(_to_float(v.get("exploratory_only", 0))) for v in vals)),
                "normalized_oracle_gap_mean": _mean(norm_gap),
                "normalized_oracle_gap_std": _std(norm_gap),
                "top1_oracle_hit_mean": _mean(top1),
                "spearman_support_vs_eval_utility_mean": _mean(spearman),
                "calibration_mae_mean": _mean(cal),
                "global_calibration_error_bin10": global_calibration_error_bin10(vals),
                "normalized_gap_reduction_vs_metadata_mean": _mean(meta_reductions),
                "normalized_gap_reduction_vs_static_embedding_mean": _mean(emb_reductions),
                "top1_delta_vs_metadata_mean": _mean(top1_meta),
                "top1_delta_vs_static_embedding_mean": _mean(top1_emb),
                "positive_support_seed_fraction": positive_seed_fraction,
                "non_degrading_k_trend": int(k_trend_ok),
                "normalized_gap_by_k_json": gap_by_k_json,
                "safe_audit": int(safe_audit),
                "adoption_gate_pass": int(adoption_gate),
            }
        )

    summary = {
        "primary_metric": "normalized_oracle_gap",
        "secondary_metric": "top1_oracle_hit",
        "diagnostic_metric": "spearman_support_vs_eval_utility",
        "n_input_rows": int(len(rows)),
        "n_valid_rows": int(len(valid_rows)),
        "n_paired_delta_rows": int(len(deltas)),
        "thresholds": {
            "min_gap_reduction": float(min_gap_reduction),
            "max_top1_regression": float(max_top1_regression),
            "min_positive_seed_fraction": float(min_positive_seed_fraction),
            "k_trend_tolerance": float(k_trend_tolerance),
            "min_backbones": int(min_backbones),
            "min_model_seeds": int(min_model_seeds),
        },
    }
    return decision_rows, deltas, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build support-set calibration LOQDO decision table.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/comparison_tables/support_set_calibration_loqdo_breakhis_decision.csv"),
    )
    parser.add_argument(
        "--paired-out",
        type=Path,
        default=Path("results/comparison_tables/support_set_calibration_loqdo_breakhis_paired_deltas.csv"),
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=Path("results/comparison_tables/support_set_calibration_loqdo_breakhis_decision_summary.json"),
    )
    parser.add_argument("--min-gap-reduction", type=float, default=0.0)
    parser.add_argument("--max-top1-regression", type=float, default=0.0)
    parser.add_argument("--min-positive-seed-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--k-trend-tolerance", type=float, default=0.02)
    parser.add_argument("--min-backbones", type=int, default=2)
    parser.add_argument("--min-model-seeds", type=int, default=2)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    rows = _read_csv(_resolve(args.raw))
    decision_rows, paired_rows, summary = build_decision_rows(
        rows,
        min_gap_reduction=float(args.min_gap_reduction),
        max_top1_regression=float(args.max_top1_regression),
        min_positive_seed_fraction=float(args.min_positive_seed_fraction),
        k_trend_tolerance=float(args.k_trend_tolerance),
        min_backbones=int(args.min_backbones),
        min_model_seeds=int(args.min_model_seeds),
    )
    out = _resolve(args.out)
    paired_out = _resolve(args.paired_out)
    summary_out = _resolve(args.summary_json_out)
    write_csv(out, decision_rows)
    write_csv(paired_out, paired_rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote decision table: {out}")
    print(f"Wrote paired deltas: {paired_out}")
    print(f"Wrote summary: {summary_out}")


if __name__ == "__main__":
    main()
