from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.engine.contracts import RunContext
from src.eval.evaluators import compute_expert_domain_matrix, evaluate_expert_latent_usage, evaluate_routing
from src.eval.make_plots import generate_plots_from_reports
from src.eval.plots import plot_reconstruction_vs_magnification
from src.eval.reporting.run_summary import write_run_summary
from src.experiments.base import BaseExperiment
from src.train.train_experts import train_domain_experts


class LegacyRoutedExperiment(BaseExperiment):
    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        return 10

    def run(
        self,
        cfg: Dict[str, Any],
        run_ctx: RunContext,
        cache_paths: Dict[str, Path],
        global_ckpt: Path,
        progress: Any,
        resume_checkpoints_dir: Path | None = None,
    ) -> None:
        experts = train_domain_experts(
            train_cache=cache_paths["train"],
            val_cache=cache_paths["val"],
            out_dir=run_ctx.checkpoints_dir,
            domains=cfg["data"]["magnifications"],
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            lr=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            patience=int(cfg["training"]["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            resume_from_dir=resume_checkpoints_dir,
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
        )
        progress.advance("domain experts trained")

        matrix = compute_expert_domain_matrix(
            test_cache=cache_paths["test"],
            expert_checkpoints=experts,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
        )
        with (run_ctx.reports_dir / "expert_matrix.json").open("w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2)
        plot_reconstruction_vs_magnification(
            matrix["reconstruction_matrix"],
            run_ctx.reports_dir / "reconstruction_vs_magnification.png",
        )
        progress.advance("expert-domain matrix evaluated")

        routing_results = evaluate_routing(
            test_cache=cache_paths["test"],
            expert_checkpoints=experts,
            global_checkpoint=Path(global_ckpt),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            strategy=str(cfg["routing"]["strategy"]),
            tau=float(cfg["routing"]["tau"]),
            temperature=float(cfg["routing"]["temperature"]),
            seed=int(cfg["seed"]),
            similarity_matrix=cfg.get("routing", {}).get("similarity_matrix"),
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
        )
        with (run_ctx.reports_dir / "routing_results.json").open("w", encoding="utf-8") as f:
            json.dump(routing_results, f, indent=2)
        progress.advance("routing evaluation complete")

        latent_usage = evaluate_expert_latent_usage(
            test_cache=cache_paths["test"],
            expert_checkpoints=experts,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
        )
        with (run_ctx.reports_dir / "latent_usage_report.json").open("w", encoding="utf-8") as f:
            json.dump(latent_usage, f, indent=2)
        progress.advance("latent usage probes computed")

        write_run_summary(
            reports_dir=run_ctx.reports_dir,
            mode="legacy_routed_cvae",
            payload={
                "routing_metrics": routing_results.get("metrics", {}),
                "routing_artifact": "routing_results.json",
                "expert_matrix_artifact": "expert_matrix.json",
                "latent_usage_artifact": "latent_usage_report.json",
                "latent_usage_probe_accuracy_mean": float(
                    latent_usage.get("aggregate", {}).get("probe_accuracy_mean", 0.0)
                ),
            },
        )

        generated_plots_dir = generate_plots_from_reports(
            reports_dir=run_ctx.reports_dir,
            out_dir=run_ctx.plots_dir,
        )
        progress.advance("plots generated")
        progress.close()

        print("Experiment complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
        print("Plots:", generated_plots_dir)
        print("Latest run pointer:", run_ctx.latest_file)
