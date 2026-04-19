from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.engine.contracts import RunContext
from src.eval.evaluators import compute_hybrid_matrices_and_routing, evaluate_downstream_utility, evaluate_global_baselines
from src.eval.reporting.hybrid_compact import write_hybrid_compact_reports
from src.eval.reporting.run_summary import write_run_summary
from src.experiments.base import BaseExperiment
from src.train.hybrid_ablation import train_hybrid_pooled_baseline, train_hybrid_variant


def _validate_hybrid_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hybrid_cfg = cfg.get("hybrid", {})
    variants = [str(v) for v in hybrid_cfg.get("variants", ["A", "B", "C"])]
    budgets = [float(v) for v in hybrid_cfg.get("synthetic_budgets", [1.0, 0.5])]

    allowed_variants = {"A", "B", "C"}
    if not variants:
        raise ValueError("hybrid.variants must not be empty")
    unknown = sorted(set(variants) - allowed_variants)
    if unknown:
        raise ValueError(f"hybrid.variants contains unsupported values: {unknown}")

    if any(b <= 0 for b in budgets):
        raise ValueError("hybrid.synthetic_budgets must contain only positive values")

    budget_set = {round(b, 6) for b in budgets}
    if 1.0 not in budget_set or 0.5 not in budget_set:
        raise ValueError(
            "hybrid.synthetic_budgets must include the locked protocol budgets 1.0 and 0.5"
        )

    aggregation_mode = str(hybrid_cfg.get("aggregation_mode", "top1_hard")).strip().lower()
    allowed_modes = {"top1_hard", "topk2_uniform", "topk3_uniform", "soft_all_softmax"}
    if aggregation_mode not in allowed_modes:
        raise ValueError(
            f"hybrid.aggregation_mode must be one of {sorted(allowed_modes)}, got: {aggregation_mode}"
        )

    topk = int(hybrid_cfg.get("topk_k", 2))
    if topk <= 0:
        raise ValueError("hybrid.topk_k must be > 0")

    aggregation_temperature = float(hybrid_cfg.get("aggregation_temperature", 1.0))
    if aggregation_temperature <= 0:
        raise ValueError("hybrid.aggregation_temperature must be > 0")

    if aggregation_mode == "topk2_uniform" and topk != 2:
        raise ValueError("hybrid.topk_k must be 2 when hybrid.aggregation_mode is 'topk2_uniform'")
    if aggregation_mode == "topk3_uniform" and topk != 3:
        raise ValueError("hybrid.topk_k must be 3 when hybrid.aggregation_mode is 'topk3_uniform'")

    return {
        "variants": variants,
        "budgets": budgets,
        "projection_dim": int(hybrid_cfg.get("projection_dim", cfg["features"]["embedding_dim"])),
        "head_hidden_dim": int(hybrid_cfg.get("head_hidden_dim", 256)),
        "cvae_hidden_dim": int(hybrid_cfg.get("cvae_hidden_dim", cfg["model"]["hidden_dim"])),
        "aggregation_mode": aggregation_mode,
        "topk_k": topk,
        "aggregation_temperature": aggregation_temperature,
    }


class HybridAblationExperiment(BaseExperiment):
    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        variants = [str(v) for v in cfg.get("hybrid", {}).get("variants", ["A", "B", "C"])]
        # Base runner contributes 5 steps; hybrid contributes 5 + 3 per variant.
        return 10 + (3 * len(variants))

    def run(
        self,
        cfg: Dict[str, Any],
        run_ctx: RunContext,
        cache_paths: Dict[str, Path],
        global_ckpt: Path,
        progress: Any,
        resume_checkpoints_dir: Path | None = None,
    ) -> None:
        hybrid_validated = _validate_hybrid_config(cfg)
        progress.advance("hybrid config validated")
        domains = [int(d) for d in cfg["data"]["magnifications"]]
        projection_dim = hybrid_validated["projection_dim"]
        head_hidden_dim = hybrid_validated["head_hidden_dim"]
        cvae_hidden_dim = hybrid_validated["cvae_hidden_dim"]
        variants = hybrid_validated["variants"]
        budgets = hybrid_validated["budgets"]
        aggregation_mode = hybrid_validated["aggregation_mode"]
        aggregation_topk = hybrid_validated["topk_k"]
        aggregation_temperature = hybrid_validated["aggregation_temperature"]

        pooled_info = train_hybrid_pooled_baseline(
            train_cache=cache_paths["train"],
            val_cache=cache_paths["val"],
            out_dir=run_ctx.checkpoints_dir,
            domains=domains,
            projection_dim=projection_dim,
            head_hidden_dim=head_hidden_dim,
            cvae_hidden_dim=cvae_hidden_dim,
            latent_dim=int(cfg["model"]["latent_dim"]),
            lr=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            patience=int(cfg["training"]["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            seed=int(cfg["seed"]),
            model_name="hybrid_pooled_baseline",
            resume_from=(resume_checkpoints_dir / "hybrid_pooled_baseline.pt") if resume_checkpoints_dir is not None else None,
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
        )
        progress.advance("hybrid pooled baseline trained")

        hybrid_results: Dict[str, object] = {
            "mode": "hybrid_ablation",
            "dataset_name": str(cfg.get("experiment", {}).get("dataset_name", "unknown")),
            "seed": int(cfg["seed"]),
            "backbone_type": str(cfg.get("features", {}).get("backbone_type", "resnet18")),
            "embedding_dim": int(cfg.get("features", {}).get("embedding_dim", 0)),
            "variants": {},
            "legacy_global_checkpoint": str(global_ckpt),
            "hybrid_pooled_checkpoint": str(pooled_info["checkpoint"]),
            "synthetic_budgets": budgets,
        }

        baseline_metrics = evaluate_global_baselines(
            test_cache=cache_paths["test"],
            legacy_global_checkpoint=Path(global_ckpt),
            legacy_hidden_dim=int(cfg["model"]["hidden_dim"]),
            legacy_latent_dim=int(cfg["model"]["latent_dim"]),
            pooled_checkpoint=Path(pooled_info["checkpoint"]),
        )
        hybrid_results["global_baselines"] = baseline_metrics
        progress.advance("global baseline metrics computed")

        for variant in variants:
            model_name = f"hybrid_variant_{variant}"
            info = train_hybrid_variant(
                train_cache=cache_paths["train"],
                val_cache=cache_paths["val"],
                out_dir=run_ctx.checkpoints_dir,
                domains=domains,
                projection_dim=projection_dim,
                head_hidden_dim=head_hidden_dim,
                cvae_hidden_dim=cvae_hidden_dim,
                latent_dim=int(cfg["model"]["latent_dim"]),
                lr=float(cfg["training"]["learning_rate"]),
                epochs=int(cfg["training"]["epochs"]),
                patience=int(cfg["training"]["patience"]),
                batch_size=int(cfg["training"]["batch_size"]),
                seed=int(cfg["seed"]),
                variant=variant,
                model_name=model_name,
                resume_from=(resume_checkpoints_dir / f"{model_name}.pt") if resume_checkpoints_dir is not None else None,
                metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
            )
            progress.advance(f"variant {variant} trained")

            matrix_and_routing = compute_hybrid_matrices_and_routing(
                test_cache=cache_paths["test"],
                hybrid_checkpoint=Path(info["checkpoint"]),
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                temperature=float(cfg["routing"]["temperature"]),
                seed=int(cfg["seed"]),
                similarity_matrix=cfg.get("routing", {}).get("similarity_matrix"),
            )
            progress.advance(f"variant {variant} routing and matrix evaluated")

            downstream = evaluate_downstream_utility(
                train_cache=cache_paths["train"],
                test_cache=cache_paths["test"],
                hybrid_checkpoint=Path(info["checkpoint"]),
                pooled_checkpoint=Path(pooled_info["checkpoint"]),
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                temperature=float(cfg["routing"]["temperature"]),
                seed=int(cfg["seed"]),
                budget_multipliers=budgets,
                aggregation_mode=str(aggregation_mode),
                aggregation_topk=int(aggregation_topk),
                aggregation_temperature=float(aggregation_temperature),
            )
            progress.advance(f"variant {variant} downstream utility evaluated")

            hybrid_results["variants"][variant] = {
                "checkpoint": info["checkpoint"],
                "history": info["history"],
                "compatibility_matrix": matrix_and_routing["compatibility_matrix"],
                "routing_matrix": matrix_and_routing["routing_matrix"],
                "routing_statistics": matrix_and_routing["routing_statistics"],
                "routing_metrics": matrix_and_routing["routing_metrics"],
                "aggregation_policy": {
                    "mode": str(aggregation_mode),
                    "topk_k": int(aggregation_topk),
                    "temperature": float(aggregation_temperature),
                },
                "downstream_utility": downstream,
                "global_baselines": baseline_metrics,
            }

            with (run_ctx.reports_dir / f"hybrid_variant_{variant}_report.json").open("w", encoding="utf-8") as f:
                json.dump(hybrid_results["variants"][variant], f, indent=2)

        with (run_ctx.reports_dir / "hybrid_summary.json").open("w", encoding="utf-8") as f:
            json.dump(hybrid_results, f, indent=2)
        progress.advance("hybrid summary written")

        write_hybrid_compact_reports(run_ctx.reports_dir, hybrid_results)
        progress.advance("hybrid compact reports written")

        write_run_summary(
            reports_dir=run_ctx.reports_dir,
            mode="hybrid_ablation",
            payload={
                "dataset_name": hybrid_results.get("dataset_name", ""),
                "seed": hybrid_results.get("seed", 0),
                "backbone_type": hybrid_results.get("backbone_type", ""),
                "embedding_dim": hybrid_results.get("embedding_dim", 0),
                "global_baselines": hybrid_results.get("global_baselines", {}),
                "variants": hybrid_results.get("variants", {}),
                "summary_artifact": "hybrid_summary.json",
                "compact_artifact": "hybrid_variant_comparison.csv",
            },
        )

        progress.close()

        print("Hybrid ablation experiment complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
        print("Latest run pointer:", run_ctx.latest_file)
