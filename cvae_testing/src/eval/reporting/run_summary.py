from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def write_run_summary(reports_dir: Path, mode: str, payload: Dict[str, Any]) -> None:
    summary = {
        "mode": mode,
        "artifacts": payload,
    }

    with (reports_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (reports_dir / "run_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Run Summary\n\n")
        f.write(f"- mode: {mode}\n")

        if mode == "legacy_routed_cvae":
            metrics = payload.get("routing_metrics", {})
            f.write("\n## Routing Metrics\n\n")
            f.write(f"- hard_metadata_routing_nelbo: {float(metrics.get('hard_metadata_routing_nelbo', 0.0)):.4f}\n")
            f.write(f"- global_cvae_nelbo: {float(metrics.get('global_cvae_nelbo', 0.0)):.4f}\n")
            f.write(f"- oracle_expert_nelbo: {float(metrics.get('oracle_expert_nelbo', 0.0)):.4f}\n")
            f.write(f"- routing_selection_accuracy: {float(metrics.get('routing_selection_accuracy', 0.0)):.4f}\n")

        elif mode == "hybrid_ablation":
            baselines = payload.get("global_baselines", {})
            variants = payload.get("variants", {})
            f.write("\n## Hybrid Baselines\n\n")
            f.write(f"- dataset_name: {str(payload.get('dataset_name', 'unknown'))}\n")
            f.write(f"- seed: {int(payload.get('seed', 0))}\n")
            f.write(f"- backbone_type: {str(payload.get('backbone_type', 'resnet18'))}\n")
            f.write(f"- embedding_dim: {int(payload.get('embedding_dim', 0))}\n")
            f.write(f"- legacy_global_nelbo: {float(baselines.get('legacy_global_nelbo', 0.0)):.4f}\n")
            f.write(f"- hybrid_pooled_global_nelbo: {float(baselines.get('hybrid_pooled_global_nelbo', 0.0)):.4f}\n")
            f.write(f"- n_variants: {len(variants)}\n")

        elif mode == "latent_compatibility":
            f.write("\n## Latent Compatibility Artifacts\n\n")
            f.write(f"- routing_artifact: {payload.get('routing_artifact', '')}\n")
            f.write(f"- gaussian_stats_artifact: {payload.get('gaussian_stats_artifact', '')}\n")
            f.write(f"- correlation_artifact: {payload.get('correlation_artifact', '')}\n")
            f.write(f"- report_artifact: {payload.get('report_artifact', '')}\n")

        elif mode == "learned_utility_routing":
            f.write("\n## Learned Utility Routing\n\n")
            f.write(f"- status: {str(payload.get('status', 'unknown'))}\n")
            f.write(f"- protocol_lock_artifact: {str(payload.get('protocol_lock_artifact', ''))}\n")
            f.write(f"- results_artifact: {str(payload.get('results_artifact', ''))}\n")
            predictors = payload.get("predictors", [])
            if isinstance(predictors, list):
                f.write(f"- predictors: {', '.join(str(p) for p in predictors)}\n")
            metrics_by_method = payload.get("metrics_by_method", {})
            if isinstance(metrics_by_method, dict) and metrics_by_method:
                f.write("\n## Metrics By Method\n\n")
                for method, m in metrics_by_method.items():
                    if not isinstance(m, dict):
                        continue
                    f.write(f"- {method}: top1={float(m.get('top1_oracle_hit', 0.0)):.4f}, ")
                    f.write(f"gap_pct={float(m.get('mean_oracle_gap_pct', 0.0)):.4f}, ")
                    f.write(f"spearman={float(m.get('spearman', 0.0)):.4f}\n")
