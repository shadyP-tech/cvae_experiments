from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.engine.contracts import RunContext
from src.eval.evaluators.learned_utility import evaluate_learned_utility_loqdo
from src.eval.reporting.run_summary import write_run_summary
from src.experiments.base import BaseExperiment
from src.train.train_experts import train_domain_experts


def _as_str_list(values: object, default: list[str]) -> list[str]:
    if values is None:
        return list(default)
    if not isinstance(values, list):
        raise ValueError("Expected a list value for learned_utility setting")
    out = [str(v).strip() for v in values]
    if any(not v for v in out):
        raise ValueError("List settings in learned_utility cannot contain empty values")
    return out


class LearnedUtilityRoutingExperiment(BaseExperiment):
    """Phase-1 implementation scaffold for learned utility routing.

    This mode currently validates protocol locks and emits run artifacts
    that make subsequent implementation phases reproducible.
    """

    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        # Base runner contributes 5 stages; this mode adds 3 stages.
        return 8

    def run(
        self,
        cfg: Dict[str, Any],
        run_ctx: RunContext,
        cache_paths: Dict[str, Path],
        global_ckpt: Path,
        progress: Any,
        resume_checkpoints_dir: Path | None = None,
    ) -> None:
        _ = global_ckpt
        learned_cfg = cfg.get("learned_utility", {})
        if not isinstance(learned_cfg, dict):
            raise ValueError("learned_utility must be a dictionary")

        predictors = _as_str_list(learned_cfg.get("predictors"), ["linear_regressor", "mlp_regressor"])
        tie_breakers = _as_str_list(
            learned_cfg.get("winner_rule", {}).get("tie_breakers"),
            ["top1_oracle_hit", "spearman_with_oracle"],
        )

        protocol_lock = {
            "experiment_mode": "learned_utility_routing",
            "split_protocol": str(learned_cfg.get("split_protocol", "loqdo_query_domain")),
            "query_domain_field": str(learned_cfg.get("query_domain_field", "magnification")),
            "predictors": predictors,
            "target": {
                "name": str(learned_cfg.get("target", {}).get("name", "nelbo")),
                "normalization": str(
                    learned_cfg.get("target", {}).get("normalization", "per_query_domain_zscore")
                ),
                "normalization_stats_source": str(
                    learned_cfg.get("target", {}).get("normalization_stats_source", "train_fold_only")
                ),
                "eval_scale": str(learned_cfg.get("target", {}).get("eval_scale", "raw_nelbo")),
            },
            "scoring": {
                "granularity": str(learned_cfg.get("scoring", {}).get("granularity", "sample_expert_pair")),
                "enforce_full_expert_scoring": bool(
                    learned_cfg.get("scoring", {}).get("enforce_full_expert_scoring", True)
                ),
                "pair_batch_size": int(learned_cfg.get("scoring", {}).get("pair_batch_size", 4096)),
            },
            "latent_comparator": {
                "primary": str(learned_cfg.get("latent_comparator", {}).get("primary", "wasserstein")),
                "diagnostics": _as_str_list(
                    learned_cfg.get("latent_comparator", {}).get("diagnostics"),
                    ["centroid", "gaussian_kl"],
                ),
            },
            "winner_rule": {
                "primary_metric": str(
                    learned_cfg.get("winner_rule", {}).get("primary_metric", "mean_oracle_gap_pct")
                ),
                "tie_breakers": tie_breakers,
                "mlp_min_improvement_abs_pct": float(
                    learned_cfg.get("winner_rule", {}).get("mlp_min_improvement_abs_pct", 1.0)
                ),
                "max_allowed_seed_regression_pct": float(
                    learned_cfg.get("winner_rule", {}).get("max_allowed_seed_regression_pct", 5.0)
                ),
            },
            "backbone_type": str(cfg.get("features", {}).get("backbone_type", "resnet50")),
            "status": "implementation_active",
        }

        lock_path = run_ctx.reports_dir / "learned_utility_protocol_lock.json"
        with lock_path.open("w", encoding="utf-8") as f:
            json.dump(protocol_lock, f, indent=2)
        progress.advance("learned utility protocol lock written")

        experts = train_domain_experts(
            train_cache=cache_paths["train"],
            val_cache=cache_paths["val"],
            out_dir=run_ctx.checkpoints_dir,
            domains=[int(v) for v in cfg["data"]["magnifications"]],
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            lr=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            patience=int(cfg["training"]["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            resume_from_dir=resume_checkpoints_dir,
        )
        progress.advance("domain experts trained for utility scoring")

        results = evaluate_learned_utility_loqdo(
            test_cache=cache_paths["test"],
            expert_checkpoints=experts,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            strategy=str(cfg["routing"]["strategy"]),
            tau=float(cfg["routing"]["tau"]),
            seed=int(cfg["seed"]),
            learned_cfg=learned_cfg,
            reports_dir=run_ctx.reports_dir,
        )
        with (run_ctx.reports_dir / "learned_utility_results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        progress.advance("learned utility LOQDO evaluation complete")

        write_run_summary(
            reports_dir=run_ctx.reports_dir,
            mode="learned_utility_routing",
            payload={
                "protocol_lock_artifact": str(lock_path.name),
                "predictors": predictors,
                "status": "phase2_evaluated",
                "results_artifact": "learned_utility_results.json",
                "metrics_by_method": results.get("metrics_by_method", {}),
            },
        )
        progress.advance("learned utility summary written")

        print("Learned utility routing evaluation complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
