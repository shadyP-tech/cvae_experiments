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
        residual_routing_cfg = learned_cfg.get("residual_routing", {}) or {}
        support_response_cfg = learned_cfg.get("support_response_routing", {}) or {}
        support_response_pool_scope = "test_split"

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
            "hybrid_scoring": {
                "enabled": bool(learned_cfg.get("hybrid_scoring", {}).get("enabled", False)),
                "alphas": [
                    float(v) for v in learned_cfg.get("hybrid_scoring", {}).get("alphas", [i / 10.0 for i in range(11)])
                ],
                "latent_metric": str(learned_cfg.get("hybrid_scoring", {}).get("latent_metric", "wasserstein")),
                "normalization_primary": str(
                    learned_cfg.get("hybrid_scoring", {}).get("normalization_primary", "per_query_zscore")
                ),
                "normalization_sensitivity": str(
                    learned_cfg.get("hybrid_scoring", {}).get("normalization_sensitivity", "per_query_minmax")
                ),
                "run_sensitivity": bool(learned_cfg.get("hybrid_scoring", {}).get("run_sensitivity", True)),
                "tie_policy": str(learned_cfg.get("hybrid_scoring", {}).get("tie_policy", "stable_expert_index")),
                "acceptance": {
                    "min_mean_rank_improvement_abs": float(
                        learned_cfg.get("hybrid_scoring", {}).get("acceptance", {}).get("min_mean_rank_improvement_abs", 0.05)
                    ),
                    "min_mean_oracle_gap_pct_improvement_abs": float(
                        learned_cfg.get("hybrid_scoring", {})
                        .get("acceptance", {})
                        .get("min_mean_oracle_gap_pct_improvement_abs", 0.50)
                    ),
                    "max_top1_drop_abs": float(
                        learned_cfg.get("hybrid_scoring", {}).get("acceptance", {}).get("max_top1_drop_abs", 0.0)
                    ),
                },
            },
            "residual_routing": {
                "enabled": bool(residual_routing_cfg.get("enabled", False)),
                "residual_policy_version": str(
                    residual_routing_cfg.get(
                        "residual_policy_version",
                        "metadata_residual_v1",
                    )
                ),
                "models": _as_str_list(
                    residual_routing_cfg.get("models"),
                    ["ridge"],
                ),
                "thresholds": [
                    str(v)
                    for v in residual_routing_cfg.get(
                        "thresholds",
                        [0, 0.01, 0.05, 0.10, 0.25, 0.50, "inf"],
                    )
                ],
                "feature_sets": _as_str_list(
                    residual_routing_cfg.get("feature_sets"),
                    ["minimal", "latent", "calibrated"],
                ),
                "adoption_feature_sets": _as_str_list(
                    residual_routing_cfg.get("adoption_feature_sets"),
                    ["minimal", "latent"],
                ),
                "diagnostic_feature_sets": _as_str_list(
                    residual_routing_cfg.get("diagnostic_feature_sets"),
                    ["calibrated"],
                ),
                "allow_calibrated_adoption": bool(
                    residual_routing_cfg.get("allow_calibrated_adoption", False)
                ),
                "harmful_override_max": float(residual_routing_cfg.get("harmful_override_max", 0.05)),
                "gap_regression_max": float(residual_routing_cfg.get("gap_regression_max", 2.0)),
                "catastrophic_top1_floor": float(
                    residual_routing_cfg.get("catastrophic_top1_floor", -0.05)
                ),
                "selection_metric": str(
                    residual_routing_cfg.get(
                        "selection_metric",
                        "validation_safe_gap_then_top1",
                    )
                ),
                "unconstrained_reference_method": str(
                    residual_routing_cfg.get(
                        "unconstrained_reference_method",
                        "pairwise_ranker_metadata_only",
                    )
                ),
                "ridge_l2": float(residual_routing_cfg.get("ridge_l2", 1.0e-4)),
                "selection_discipline": (
                    "feature_set_variant_and_threshold_selected_by_inner_source_query_domain_loqdo"
                ),
                "target_scale": "delta_u_pct_for_training_and_thresholding_raw_nelbo_for_final_metrics",
            },
            "support_response_routing": {
                "enabled": bool(support_response_cfg.get("enabled", False)),
                "support_sizes": [
                    int(v) for v in support_response_cfg.get("support_sizes", [8, 16, 32])
                ],
                "support_seeds": [
                    int(v) for v in support_response_cfg.get("support_seeds", [17, 23])
                ],
                "sampling_policies": _as_str_list(
                    support_response_cfg.get("sampling_policies"),
                    ["random"],
                ),
                "feature_regimes": _as_str_list(
                    support_response_cfg.get("feature_regimes"),
                    ["static_response_indirect", "response_indirect_shuffled"],
                ),
                "primary_feature_regime": str(
                    support_response_cfg.get("primary_feature_regime", "static_response_indirect")
                ),
                "ranker": str(support_response_cfg.get("ranker", "linear_pairwise_ridge")),
                "ridge_l2": float(support_response_cfg.get("ridge_l2", 1.0e-3)),
                "num_response_repeats": int(support_response_cfg.get("num_response_repeats", 8)),
                "tie_policy": str(support_response_cfg.get("tie_policy", "stable_expert_index")),
                "domain_level_aggregation": bool(
                    support_response_cfg.get("domain_level_aggregation", True)
                ),
                "source_leave_pseudo_domain_out_diagnostic": bool(
                    support_response_cfg.get("source_leave_pseudo_domain_out_diagnostic", True)
                ),
                "support_utility": {
                    "enabled": bool((support_response_cfg.get("support_utility", {}) or {}).get("enabled", False)),
                    "alpha_grid": [
                        float(v)
                        for v in (support_response_cfg.get("support_utility", {}) or {}).get(
                            "alpha_grid",
                            [0.0, 0.5, 1.0, 1.5, 2.0],
                        )
                    ],
                    "alpha_selection_policy": str(
                        (support_response_cfg.get("support_utility", {}) or {}).get(
                            "alpha_selection_policy",
                            "source_inner_gap_min_with_non_regression",
                        )
                    ),
                    "require_unlabeled_support": bool(
                        (support_response_cfg.get("support_utility", {}) or {}).get(
                            "require_unlabeled_support",
                            True,
                        )
                    ),
                    "support_labels_used_for_routing": 0,
                },
                "scaler_fit_scope": "source_training_pairs_only",
                "ranker_model_selection_scope": "source_only_fixed_config",
                "score_direction": "predicted_mean_nelbo_lower_is_better",
                "evaluation_pool_scope": support_response_pool_scope,
                "evaluation_pool_note": "Support-response routing uses the test embedding split.",
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
            "compatibility_research": {
                "floors": {
                    "random_rank_floor": bool(
                        learned_cfg.get("compatibility_research", {})
                        .get("floors", {})
                        .get("random_rank_floor", True)
                    ),
                    "random_score_floor": bool(
                        learned_cfg.get("compatibility_research", {})
                        .get("floors", {})
                        .get("random_score_floor", True)
                    ),
                },
                "permutation_tests": {
                    "expert_label_permutation": bool(
                        learned_cfg.get("compatibility_research", {})
                        .get("permutation_tests", {})
                        .get("expert_label_permutation", True)
                    ),
                    "metadata_permutation": bool(
                        learned_cfg.get("compatibility_research", {})
                        .get("permutation_tests", {})
                        .get("metadata_permutation", True)
                    ),
                    "repeats": int(
                        learned_cfg.get("compatibility_research", {})
                        .get("permutation_tests", {})
                        .get("repeats", 200)
                    ),
                },
                "gate": {
                    "decision_policy_version": str(
                        learned_cfg.get("compatibility_research", {})
                        .get("gate", {})
                        .get("decision_policy_version", "sign_ci_v2")
                    ),
                    "uplift_reference_method": str(
                        learned_cfg.get("compatibility_research", {})
                        .get("gate", {})
                        .get("uplift_reference_method", "metadata_routing")
                    ),
                    "min_improving_seeds": int(
                        learned_cfg.get("compatibility_research", {})
                        .get("gate", {})
                        .get("min_improving_seeds", 2)
                    ),
                    "strong": {
                        "spearman_uplift_min": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("strong", {})
                            .get("spearman_uplift_min", 0.05)
                        ),
                        "top1_uplift_min": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("strong", {})
                            .get("top1_uplift_min", 0.10)
                        ),
                        "oracle_gap_pct_reduction_min": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("strong", {})
                            .get("oracle_gap_pct_reduction_min", 5.0)
                        ),
                    },
                    "weak": {
                        "spearman_uplift_min": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("weak", {})
                            .get("spearman_uplift_min", 0.025)
                        ),
                        "top1_uplift_min": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("weak", {})
                            .get("top1_uplift_min", 0.05)
                        ),
                        "oracle_gap_pct_reduction_min": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("weak", {})
                            .get("oracle_gap_pct_reduction_min", 2.5)
                        ),
                    },
                    "instability": {
                        "std_threshold": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("std_threshold", 0.05)
                        ),
                        "top1_uplift_std_threshold": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("top1_uplift_std_threshold", 0.05)
                        ),
                        "spearman_uplift_std_threshold": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("spearman_uplift_std_threshold", 0.05)
                        ),
                        "gap_pct_reduction_std_threshold": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("gap_pct_reduction_std_threshold", 3.0)
                        ),
                        "sign_inconsistency_min_count": int(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("sign_inconsistency_min_count", 2)
                        ),
                        "min_positive_fraction": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("min_positive_fraction", 0.67)
                        ),
                        "ci_level": float(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("ci_level", 0.95)
                        ),
                        "ci_bootstrap_reps": int(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("ci_bootstrap_reps", 10000)
                        ),
                        "ci_bootstrap_seed": int(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("ci_bootstrap_seed", 1337)
                        ),
                        "allow_missing_domain_breakdown_as_diagnostic": bool(
                            learned_cfg.get("compatibility_research", {})
                            .get("gate", {})
                            .get("instability", {})
                            .get("allow_missing_domain_breakdown_as_diagnostic", False)
                        ),
                    },
                },
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
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
        )
        progress.advance("domain experts trained for utility scoring")

        results = evaluate_learned_utility_loqdo(
            test_cache=cache_paths["test"],
            cache_paths=cache_paths,
            expert_checkpoints=experts,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            strategy=str(cfg["routing"]["strategy"]),
            tau=float(cfg["routing"]["tau"]),
            seed=int(cfg["seed"]),
            learned_cfg=learned_cfg,
            reports_dir=run_ctx.reports_dir,
            conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
            configured_domains=cfg.get("data", {}).get("magnifications", []),
            metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
            data_cfg=cfg.get("data", {}),
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
                "compatibility_protocol": results.get("compatibility_protocol", {}),
                "hybrid_diagnostics": results.get("hybrid_diagnostics", {}),
                "support_response_results": results.get("support_response_results", {}),
            },
        )
        progress.advance("learned utility summary written")

        print("Learned utility routing evaluation complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
