from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict


def _mean(xs: list[float]) -> float:
    clean = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(clean) / len(clean) if clean else 0.0


def _std(xs: list[float]) -> float:
    clean = [float(x) for x in xs if math.isfinite(float(x))]
    if len(clean) <= 1:
        return 0.0
    mu = _mean(clean)
    var = sum((v - mu) ** 2 for v in clean) / float(len(clean))
    return math.sqrt(max(var, 0.0))


def write_hybrid_compact_reports(reports_dir: Path, hybrid_results: Dict[str, object]) -> None:
    rows = []
    variants = hybrid_results.get("variants", {})
    global_baselines = hybrid_results.get("global_baselines", {})
    backbone_type = str(hybrid_results.get("backbone_type", "unknown"))
    dataset_name = str(hybrid_results.get("dataset_name", "unknown"))
    seed = int(hybrid_results.get("seed", 0))

    for variant_name, payload in variants.items():
        routing_stats = payload.get("routing_statistics", {})
        routing_metrics = payload.get("routing_metrics", {})
        aggregation_policy = payload.get("aggregation_policy", {})
        sharpness = routing_stats.get("compatibility_sharpness_nelbo", {})
        downstream = payload.get("downstream_utility", {})

        metadata_nelbo = float(routing_metrics.get("metadata_routing_nelbo", 0.0))
        oracle_nelbo = float(routing_metrics.get("oracle_routing_nelbo", 0.0))
        metadata_to_oracle_gap = float(routing_metrics.get("metadata_to_oracle_gap", 0.0))
        metadata_to_oracle_gap_pct = float(routing_metrics.get("metadata_to_oracle_gap_pct", 0.0))
        if metadata_to_oracle_gap_pct == 0.0 and abs(oracle_nelbo) > 1e-8:
            metadata_to_oracle_gap_pct = 100.0 * metadata_to_oracle_gap / abs(oracle_nelbo)

        for budget_key, by_domain in downstream.items():
            auroc_real = []
            auroc_random = []
            auroc_pooled = []
            auroc_routed = []
            bacc_real = []
            bacc_random = []
            bacc_pooled = []
            bacc_routed = []

            for _, domain_payload in by_domain.items():
                m = domain_payload.get("metrics", {})
                auroc_real.append(float(m.get("real_only", {}).get("auroc", 0.0)))
                auroc_random.append(float(m.get("real_plus_random_synthetic", {}).get("auroc", 0.0)))
                auroc_pooled.append(float(m.get("real_plus_pooled_synthetic", {}).get("auroc", 0.0)))
                auroc_routed.append(float(m.get("real_plus_routed_synthetic", {}).get("auroc", 0.0)))

                bacc_real.append(float(m.get("real_only", {}).get("balanced_accuracy", 0.0)))
                bacc_random.append(float(m.get("real_plus_random_synthetic", {}).get("balanced_accuracy", 0.0)))
                bacc_pooled.append(float(m.get("real_plus_pooled_synthetic", {}).get("balanced_accuracy", 0.0)))
                bacc_routed.append(float(m.get("real_plus_routed_synthetic", {}).get("balanced_accuracy", 0.0)))

            row = {
                "dataset_name": dataset_name,
                "seed": seed,
                "backbone_type": backbone_type,
                "variant": str(variant_name),
                "aggregation_mode": str(aggregation_policy.get("mode", "top1_hard")),
                "aggregation_topk": int(aggregation_policy.get("topk_k", 2)),
                "aggregation_temperature": float(aggregation_policy.get("temperature", 1.0)),
                "budget": str(budget_key),
                "metadata_nelbo": metadata_nelbo,
                "oracle_nelbo": oracle_nelbo,
                "metadata_to_oracle_gap": metadata_to_oracle_gap,
                "oracle_gap_pct": metadata_to_oracle_gap_pct,
                "spearman_similarity_vs_neg_nelbo": float(
                    routing_stats.get("spearman_similarity_vs_neg_nelbo", 0.0)
                ),
                "spearman_score_level": float(
                    routing_stats.get("spearman_similarity_vs_neg_nelbo", 0.0)
                ),
                "top1_agreement_with_best_expert": float(
                    routing_stats.get("top1_agreement_with_best_expert", 0.0)
                ),
                "top1_oracle_hit_score_level": float(
                    routing_stats.get("top1_agreement_with_best_expert", 0.0)
                ),
                "mean_rank_metadata_selected": float(
                    routing_stats.get("mean_rank_of_metadata_selected_expert", 0.0)
                ),
                "compat_diagonal_mean": float(sharpness.get("diagonal_mean", 0.0)),
                "compat_offdiagonal_mean": float(sharpness.get("offdiagonal_mean", 0.0)),
                "compat_offdiagonal_std": float(sharpness.get("offdiagonal_std", 0.0)),
                "compat_diagonal_offdiagonal_gap": float(sharpness.get("diagonal_offdiagonal_gap", 0.0)),
                "compat_diagonal_gap_ratio": float(sharpness.get("diagonal_gap_ratio", 0.0)),
                "compat_diagonal_margin": float(sharpness.get("diagonal_margin_to_best_offdiagonal", 0.0)),
                "legacy_global_nelbo": float(global_baselines.get("legacy_global_nelbo", 0.0)),
                "hybrid_pooled_global_nelbo": float(global_baselines.get("hybrid_pooled_global_nelbo", 0.0)),
                "auroc_real_only": _mean(auroc_real),
                "auroc_real_plus_random": _mean(auroc_random),
                "auroc_real_plus_pooled": _mean(auroc_pooled),
                "auroc_real_plus_routed": _mean(auroc_routed),
                "mean_auroc": _mean(auroc_routed),
                "std_auroc": _std(auroc_routed),
                "auroc_delta_routed_vs_real": _mean(auroc_routed) - _mean(auroc_real),
                "auroc_delta_routed_vs_random": _mean(auroc_routed) - _mean(auroc_random),
                "auroc_delta_routed_vs_pooled": _mean(auroc_routed) - _mean(auroc_pooled),
                "bacc_real_only": _mean(bacc_real),
                "bacc_real_plus_random": _mean(bacc_random),
                "bacc_real_plus_pooled": _mean(bacc_pooled),
                "bacc_real_plus_routed": _mean(bacc_routed),
                "mean_bacc": _mean(bacc_routed),
                "std_bacc": _std(bacc_routed),
                "bacc_delta_routed_vs_real": _mean(bacc_routed) - _mean(bacc_real),
                "bacc_delta_routed_vs_random": _mean(bacc_routed) - _mean(bacc_random),
                "bacc_delta_routed_vs_pooled": _mean(bacc_routed) - _mean(bacc_pooled),
            }
            rows.append(row)

    if not rows:
        return

    csv_path = reports_dir / "hybrid_variant_comparison.csv"
    md_path = reports_dir / "hybrid_variant_comparison.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Hybrid Ablation Compact Comparison\\n\\n")
        f.write("| dataset | seed | backbone_type | variant | aggregation_mode | budget | oracle_gap_pct | spearman | top1 | mean_rank | mean_auroc | std_auroc | auroc routed-real | auroc routed-random | auroc routed-pooled |\\n")
        f.write("|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\\n")
        for r in rows:
            f.write(
                f"| {r['dataset_name']} | {r['seed']} | {r['backbone_type']} | {r['variant']} | {r['aggregation_mode']} | {r['budget']} | {r['oracle_gap_pct']:.2f} | {r['spearman_score_level']:.3f} | "
                f"{r['top1_agreement_with_best_expert']:.3f} | {r['mean_rank_metadata_selected']:.2f} | "
                f"{r['mean_auroc']:.4f} | {r['std_auroc']:.4f} | "
                f"{r['auroc_delta_routed_vs_real']:.4f} | {r['auroc_delta_routed_vs_random']:.4f} | "
                f"{r['auroc_delta_routed_vs_pooled']:.4f} |\\n"
            )

        f.write("\\n")
        f.write("## Global Baselines\\n\\n")
        f.write(
            f"- legacy_global_nelbo: {float(global_baselines.get('legacy_global_nelbo', 0.0)):.2f}\\n"
        )
        f.write(
            f"- hybrid_pooled_global_nelbo: {float(global_baselines.get('hybrid_pooled_global_nelbo', 0.0)):.2f}\\n"
        )
