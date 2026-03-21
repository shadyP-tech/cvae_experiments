from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.engine.contracts import RunContext
from src.eval.evaluators import compute_expert_domain_matrix
from src.eval.evaluators.latent_compatibility import (
    build_metadata_score_matrix,
    build_metric_payload,
    compute_distance_utility_correlation,
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    compute_metric_utility_correlation,
    compute_proxy_oracle_alignment,
    distance_to_similarity,
    evaluate_routing_alignment,
    load_embeddings_with_domains,
    matrix_to_domain_dict,
    maybe_project_latent_2d,
    plot_composite_figure,
    plot_distance_vs_utility,
    plot_latent_map,
    plot_matrix_heatmap,
    verify_similarity_matrix,
)
from src.eval.reporting.run_summary import write_run_summary
from src.experiments.base import BaseExperiment
from src.train.train_experts import train_domain_experts


def _validate_latent_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    block = cfg.get("latent_compatibility", {})
    allowed_metrics = {"centroid", "wasserstein", "gaussian_kl"}
    metrics = [str(m) for m in block.get("metrics", ["centroid", "wasserstein", "gaussian_kl"])]
    if not metrics:
        raise ValueError("latent_compatibility.metrics must not be empty")
    unknown = sorted(set(metrics) - allowed_metrics)
    if unknown:
        raise ValueError(f"latent_compatibility.metrics contains unsupported values: {unknown}")

    similarity_transform = str(block.get("similarity_transform", "exp_neg")).strip()
    if similarity_transform != "exp_neg":
        raise ValueError("latent_compatibility.similarity_transform must be 'exp_neg'")

    splits = [str(s) for s in block.get("splits", ["test"])]
    allowed_splits = {"train", "val", "test"}
    if not splits or any(s not in allowed_splits for s in splits):
        raise ValueError(f"latent_compatibility.splits must be in {sorted(allowed_splits)}")

    verification = block.get("verification", {})
    wasserstein = block.get("wasserstein", {})
    similarity = block.get("similarity", {})
    empirical = block.get("empirical_utility", {})
    coverage_gates = block.get("coverage_gates", {})
    thresholds = block.get("acceptance_thresholds", {})
    strong = thresholds.get("strong", {}) if isinstance(thresholds, dict) else {}
    non_inferiority = thresholds.get("non_inferiority", {}) if isinstance(thresholds, dict) else {}

    out = {
        "metrics": metrics,
        "splits": splits,
        "similarity_transform": similarity_transform,
        "min_samples_per_domain": int(block.get("min_samples_per_domain", 50)),
        "covariance_regularization_lambda": float(block.get("covariance_regularization_lambda", 1e-4)),
        "evaluation_metrics": [
            str(m) for m in block.get("evaluation_metrics", ["top1_agreement", "mean_rank", "spearman_with_utility"])
        ],
        "verification": {
            "symmetry_atol": float(verification.get("symmetry_atol", 1e-6)),
            "symmetry_rtol": float(verification.get("symmetry_rtol", 1e-5)),
            "diag_opt_tol": float(verification.get("diag_opt_tol", 1e-6)),
        },
        "wasserstein": {
            "eigenvalue_floor": float(wasserstein.get("eigenvalue_floor", 1e-10)),
        },
        "similarity": {
            "scale_floor": float(similarity.get("scale_floor", 1e-8)),
            "scale_policy": str(similarity.get("scale_policy", block.get("scale_policy", "median_off_diagonal"))),
        },
        "umap": {
            "max_points": int(block.get("umap", {}).get("max_points", 5000)),
        },
        "composite_metric": str(block.get("composite_metric", "wasserstein")),
        "empirical_utility": {
            "enabled": bool(empirical.get("enabled", True)),
        },
        "persist_raw_and_transformed_scores": bool(block.get("persist_raw_and_transformed_scores", True)),
        "compute_oracle_alignment": bool(block.get("compute_oracle_alignment", True)),
        "include_metadata_oracle_proxy_table": bool(block.get("include_metadata_oracle_proxy_table", True)),
        "learned_comparison": {
            "strict_context_match": bool(block.get("learned_comparison", {}).get("strict_context_match", True)),
        },
        "coverage_gates": {
            "require_complete_loqdo_folds": bool(coverage_gates.get("require_complete_loqdo_folds", True)),
            "require_all_query_domains_present": bool(coverage_gates.get("require_all_query_domains_present", True)),
            "exclude_partial_metric_rows": bool(coverage_gates.get("exclude_partial_metric_rows", True)),
        },
        "acceptance_thresholds": {
            "enabled": bool(thresholds.get("enabled", True)) if isinstance(thresholds, dict) else True,
            "strong": {
                "spearman_uplift_gt": float(strong.get("spearman_uplift_gt", 0.10)),
                "top1_uplift_gte": float(strong.get("top1_uplift_gte", 0.25)),
                "oracle_gap_reduction_gt_pct": float(strong.get("oracle_gap_reduction_gt_pct", 10.0)),
                "min_backbone_fraction": float(strong.get("min_backbone_fraction", 0.67)),
                "disallow_any_backbone_guardrail_breach": bool(strong.get("disallow_any_backbone_guardrail_breach", True)),
            },
            "non_inferiority": {
                "max_oracle_gap_worsening_pct": float(non_inferiority.get("max_oracle_gap_worsening_pct", 5.0)),
            },
        },
    }

    if out["similarity"]["scale_policy"] != "median_off_diagonal":
        raise ValueError("latent_compatibility similarity scale policy must be 'median_off_diagonal'")
    if out["min_samples_per_domain"] <= 0:
        raise ValueError("latent_compatibility.min_samples_per_domain must be > 0")
    if out["covariance_regularization_lambda"] <= 0:
        raise ValueError("latent_compatibility.covariance_regularization_lambda must be > 0")
    if out["wasserstein"]["eigenvalue_floor"] <= 0:
        raise ValueError("latent_compatibility.wasserstein.eigenvalue_floor must be > 0")
    if out["similarity"]["scale_floor"] <= 0:
        raise ValueError("latent_compatibility.similarity.scale_floor must be > 0")
    if out["umap"]["max_points"] <= 0:
        raise ValueError("latent_compatibility.umap.max_points must be > 0")
    if out["acceptance_thresholds"]["strong"]["min_backbone_fraction"] <= 0:
        raise ValueError("latent_compatibility.acceptance_thresholds.strong.min_backbone_fraction must be > 0")

    return out


def _to_jsonable_gaussian_stats(
    domain_order: List[int],
    stats: Dict[int, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for d in domain_order:
        ds = stats[d]
        payload[f"{d}x"] = {
            "n_samples": int(ds.n_samples),
            "used_diagonal_covariance": bool(ds.used_diagonal_covariance),
            "mean": ds.mean.astype(float).tolist(),
            "covariance_diagonal": np.diag(ds.covariance).astype(float).tolist(),
        }
    return payload


def _utility_matrix_from_expert_matrix(domain_order: List[int], expert_matrix_report: Dict[str, Any]) -> np.ndarray:
    confidence = expert_matrix_report.get("confidence", {})
    matrix = np.zeros((len(domain_order), len(domain_order)), dtype=np.float64)
    for i, query_domain in enumerate(domain_order):
        for j, expert_domain in enumerate(domain_order):
            expert_key = f"{expert_domain}x"
            query_key = f"{query_domain}x"
            mean_nelbo = float(confidence.get(expert_key, {}).get(query_key, {}).get("mean", 0.0))
            matrix[i, j] = -mean_nelbo
    return matrix


def _write_metric_correlation_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "corr_with_utility", "corr_distance_with_utility"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_rows_csv(out_path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_coverage_status(
    backbone_type: str,
    domain_order: List[int],
    proxy_alignment_by_name: Dict[str, Dict[str, Any]],
    coverage_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    expected = [f"{d}x" for d in domain_order]
    expected_set = set(expected)
    all_present = True
    has_partial = False
    per_proxy: Dict[str, Any] = {}

    for proxy_name, payload in proxy_alignment_by_name.items():
        observed = [str(row.get("query_domain", "")) for row in payload.get("per_query", [])]
        observed_set = set(observed)
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        partial = bool(missing or extra or len(observed) != len(expected))
        all_present = all_present and (len(missing) == 0)
        has_partial = has_partial or partial
        per_proxy[proxy_name] = {
            "expected_query_domains": expected,
            "observed_query_domains": sorted(observed_set),
            "missing_query_domains": missing,
            "extra_query_domains": extra,
            "partial_rows": partial,
        }

    complete_folds = bool(all_present and not has_partial)
    valid = True
    if bool(coverage_cfg.get("require_complete_loqdo_folds", True)) and not complete_folds:
        valid = False
    if bool(coverage_cfg.get("require_all_query_domains_present", True)) and not all_present:
        valid = False
    if bool(coverage_cfg.get("exclude_partial_metric_rows", True)) and has_partial:
        valid = False

    return {
        "backbone_type": str(backbone_type),
        "status": "valid" if valid else "insufficient_backbone_coverage",
        "valid_for_tier": bool(valid),
        "all_loqdo_query_folds_present": bool(complete_folds),
        "all_expected_query_domains_present": bool(all_present),
        "has_partial_metric_rows": bool(has_partial),
        "details_by_proxy": per_proxy,
    }


def _relative_gap_change_pct(baseline_gap: float, candidate_gap: float) -> float:
    denom = max(abs(float(baseline_gap)), 1e-12)
    return float(((candidate_gap - baseline_gap) / denom) * 100.0)


def _build_acceptance_decision(
    metric_name: str,
    metadata_summary: Dict[str, Any],
    latent_summary: Dict[str, Any],
    coverage_status: Dict[str, Any],
    thresholds_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    strong_cfg = thresholds_cfg["strong"]
    guardrail_cfg = thresholds_cfg["non_inferiority"]

    spearman_uplift = float(latent_summary["mean_spearman"] - metadata_summary["mean_spearman"])
    top1_uplift = float(latent_summary["top1"] - metadata_summary["top1"])
    baseline_gap = float(metadata_summary["mean_oracle_gap"])
    latent_gap = float(latent_summary["mean_oracle_gap"])
    gap_reduction_pct = float((-_relative_gap_change_pct(baseline_gap, latent_gap)))
    gap_worsening_pct = float(_relative_gap_change_pct(baseline_gap, latent_gap))

    guardrail_breach = bool(
        spearman_uplift > 0.0
        and gap_worsening_pct > float(guardrail_cfg["max_oracle_gap_worsening_pct"])
    )

    valid_backbones = 1 if bool(coverage_status.get("valid_for_tier", False)) else 0
    improved_backbones = 1 if (spearman_uplift > 0 and top1_uplift > 0 and gap_reduction_pct > 0 and valid_backbones == 1) else 0
    backbone_fraction = float(improved_backbones / max(valid_backbones, 1))

    strong_checks = {
        "spearman_uplift_gt": bool(spearman_uplift > float(strong_cfg["spearman_uplift_gt"])),
        "top1_uplift_gte": bool(top1_uplift >= float(strong_cfg["top1_uplift_gte"])),
        "oracle_gap_reduction_gt_pct": bool(gap_reduction_pct > float(strong_cfg["oracle_gap_reduction_gt_pct"])),
        "min_backbone_fraction": bool(backbone_fraction >= float(strong_cfg["min_backbone_fraction"])),
        "no_backbone_guardrail_breach": bool(not guardrail_breach),
        "backbone_coverage_valid": bool(coverage_status.get("valid_for_tier", False)),
    }

    strong_ok = bool(all(strong_checks.values()))
    if not bool(strong_cfg.get("disallow_any_backbone_guardrail_breach", True)):
        strong_ok = bool(
            strong_checks["spearman_uplift_gt"]
            and strong_checks["top1_uplift_gte"]
            and strong_checks["oracle_gap_reduction_gt_pct"]
            and strong_checks["min_backbone_fraction"]
            and strong_checks["backbone_coverage_valid"]
        )

    any_primary_positive = bool(spearman_uplift > 0 or top1_uplift > 0 or gap_reduction_pct > 0)

    if not bool(coverage_status.get("valid_for_tier", False)):
        tier = "insufficient_backbone_coverage"
    elif strong_ok:
        tier = "strong_latent_superiority"
    elif guardrail_breach:
        tier = "structural_but_not_utility_improvement"
    elif any_primary_positive:
        tier = "partial_or_backbone_dependent_improvement"
    else:
        tier = "no_latent_superiority"

    return {
        "metric": str(metric_name),
        "tier": tier,
        "uplifts": {
            "mean_spearman": spearman_uplift,
            "top1": top1_uplift,
            "oracle_gap_reduction_pct": gap_reduction_pct,
            "oracle_gap_worsening_pct": gap_worsening_pct,
        },
        "guardrail": {
            "max_oracle_gap_worsening_pct": float(guardrail_cfg["max_oracle_gap_worsening_pct"]),
            "breach": guardrail_breach,
        },
        "strong_checks": strong_checks,
        "coverage_status": {
            "status": str(coverage_status.get("status", "insufficient_backbone_coverage")),
            "valid_for_tier": bool(coverage_status.get("valid_for_tier", False)),
        },
    }


class LatentCompatibilityExperiment(BaseExperiment):
    def estimate_total_steps(self, cfg: Dict[str, Any]) -> int:
        latent_cfg = _validate_latent_config(cfg)
        # Base runner contributes 5 stages before experiment.run() is called.
        # This experiment contributes 7 stages, plus 2 when empirical utility is enabled.
        steps = 12
        if bool(latent_cfg["empirical_utility"]["enabled"]):
            steps += 2
        return steps

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
        latent_cfg = _validate_latent_config(cfg)
        progress.advance("latent compatibility config validated")

        embeddings, sample_domains, _ = load_embeddings_with_domains(
            cache_paths=cache_paths,
            splits=latent_cfg["splits"],
        )
        progress.advance("latent embeddings loaded")

        domain_order, gaussian_stats, gaussian_warnings = compute_domain_gaussian_stats(
            embeddings=embeddings,
            domains=sample_domains,
            covariance_regularization_lambda=float(latent_cfg["covariance_regularization_lambda"]),
            min_samples_per_domain=int(latent_cfg["min_samples_per_domain"]),
        )
        progress.advance("domain gaussian summaries computed")

        distance_mats = compute_distance_matrices(
            domain_order=domain_order,
            stats=gaussian_stats,
            eigenvalue_floor=float(latent_cfg["wasserstein"]["eigenvalue_floor"]),
        )

        distance_name_map = {
            "centroid": "distance_centroid.npy",
            "wasserstein": "distance_wasserstein.npy",
            "gaussian_kl": "distance_kl_sym.npy",
        }
        for metric_name, distance in distance_mats.items():
            np.save(run_ctx.reports_dir / distance_name_map[metric_name], distance)
        progress.advance("distance matrices computed")

        similarity_name_map = {
            "centroid": "similarity_centroid.npy",
            "wasserstein": "similarity_wasserstein.npy",
            "gaussian_kl": "similarity_kl_sym.npy",
        }

        similarity_mats: Dict[str, np.ndarray] = {}
        scale_by_metric: Dict[str, float] = {}
        verification_by_metric: Dict[str, Any] = {}
        routing_by_metric: Dict[str, Any] = {}
        metric_payloads: Dict[str, Any] = {}

        for metric_name in latent_cfg["metrics"]:
            sim, scale = distance_to_similarity(
                distance=distance_mats[metric_name],
                scale_floor=float(latent_cfg["similarity"]["scale_floor"]),
            )
            similarity_mats[metric_name] = sim
            scale_by_metric[metric_name] = float(scale)
            np.save(run_ctx.reports_dir / similarity_name_map[metric_name], sim)

            if bool(latent_cfg["persist_raw_and_transformed_scores"]):
                metric_payloads[metric_name] = build_metric_payload(
                    metric_name=metric_name,
                    raw_distance_matrix=distance_mats[metric_name],
                    score_matrix=sim,
                    scale=float(scale),
                    domain_order=domain_order,
                )

            verification_by_metric[metric_name] = verify_similarity_matrix(
                matrix=sim,
                atol=float(latent_cfg["verification"]["symmetry_atol"]),
                rtol=float(latent_cfg["verification"]["symmetry_rtol"]),
                diag_opt_tol=float(latent_cfg["verification"]["diag_opt_tol"]),
                symmetric_expected=True,
            )

            routing_by_metric[metric_name] = evaluate_routing_alignment(
                domain_order=domain_order,
                similarity_matrix=sim,
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                similarity_lookup_matrix=cfg.get("routing", {}).get("similarity_matrix"),
            )
        progress.advance("similarity matrices, verification, and routing computed")

        utility_matrix: np.ndarray | None = None
        utility_report: Dict[str, Any] | None = None
        if bool(latent_cfg["empirical_utility"]["enabled"]):
            experts = train_domain_experts(
                train_cache=cache_paths["train"],
                val_cache=cache_paths["val"],
                out_dir=run_ctx.checkpoints_dir,
                domains=[int(d) for d in cfg["data"]["magnifications"]],
                hidden_dim=int(cfg["model"]["hidden_dim"]),
                latent_dim=int(cfg["model"]["latent_dim"]),
                lr=float(cfg["training"]["learning_rate"]),
                epochs=int(cfg["training"]["epochs"]),
                patience=int(cfg["training"]["patience"]),
                batch_size=int(cfg["training"]["batch_size"]),
                resume_from_dir=resume_checkpoints_dir,
            )
            progress.advance("empirical utility experts trained")

            utility_report = compute_expert_domain_matrix(
                test_cache=cache_paths["test"],
                expert_checkpoints=experts,
                hidden_dim=int(cfg["model"]["hidden_dim"]),
                latent_dim=int(cfg["model"]["latent_dim"]),
            )
            utility_matrix = _utility_matrix_from_expert_matrix(domain_order, utility_report)
            with (run_ctx.reports_dir / "expert_utility_matrix.json").open("w", encoding="utf-8") as f:
                json.dump(utility_report, f, indent=2)
            progress.advance("empirical utility matrix computed")

        metric_corr_rows: List[Dict[str, Any]] = []
        proxy_alignment_by_name: Dict[str, Dict[str, Any]] = {}
        acceptance_decisions: Dict[str, Any] = {}
        coverage_status: Dict[str, Any] | None = None
        for metric_name in latent_cfg["metrics"]:
            corr = 0.0
            corr_dist = 0.0
            if utility_matrix is not None:
                corr = compute_metric_utility_correlation(similarity_mats[metric_name], utility_matrix)
                corr_dist = compute_distance_utility_correlation(
                    distance_matrix=distance_mats[metric_name],
                    utility_matrix=utility_matrix,
                    off_diagonal_only=True,
                )
            routing_by_metric[metric_name]["spearman_with_utility"] = float(corr)
            routing_by_metric[metric_name]["spearman_distance_with_utility"] = float(corr_dist)
            metric_corr_rows.append(
                {
                    "metric": metric_name,
                    "corr_with_utility": f"{corr:.6f}",
                    "corr_distance_with_utility": f"{corr_dist:.6f}",
                }
            )

        if utility_matrix is not None and bool(latent_cfg["compute_oracle_alignment"]):
            metadata_scores = build_metadata_score_matrix(
                domain_order=domain_order,
                strategy=str(cfg["routing"]["strategy"]),
                tau=float(cfg["routing"]["tau"]),
                similarity_lookup_matrix=cfg.get("routing", {}).get("similarity_matrix"),
            )
            proxy_alignment_by_name["metadata"] = compute_proxy_oracle_alignment(
                proxy_name="metadata",
                score_matrix=metadata_scores,
                oracle_utility_matrix=utility_matrix,
                domain_order=domain_order,
            )
            for metric_name in latent_cfg["metrics"]:
                proxy_alignment_by_name[metric_name] = compute_proxy_oracle_alignment(
                    proxy_name=metric_name,
                    score_matrix=similarity_mats[metric_name],
                    oracle_utility_matrix=utility_matrix,
                    domain_order=domain_order,
                )

            coverage_status = _compute_coverage_status(
                backbone_type=str(cfg.get("features", {}).get("backbone_type", "unknown")),
                domain_order=domain_order,
                proxy_alignment_by_name=proxy_alignment_by_name,
                coverage_cfg=latent_cfg["coverage_gates"],
            )

            if bool(latent_cfg["acceptance_thresholds"]["enabled"]):
                metadata_summary = proxy_alignment_by_name["metadata"]
                for metric_name in latent_cfg["metrics"]:
                    acceptance_decisions[metric_name] = _build_acceptance_decision(
                        metric_name=metric_name,
                        metadata_summary=metadata_summary,
                        latent_summary=proxy_alignment_by_name[metric_name],
                        coverage_status=coverage_status,
                        thresholds_cfg=latent_cfg["acceptance_thresholds"],
                    )

        _write_metric_correlation_csv(run_ctx.reports_dir / "metric_correlation_table.csv", metric_corr_rows)

        if proxy_alignment_by_name:
            per_query_rows: List[Dict[str, Any]] = []
            for payload in proxy_alignment_by_name.values():
                per_query_rows.extend(payload.get("per_query", []))
            _write_rows_csv(
                run_ctx.reports_dir / "proxy_oracle_alignment_per_query.csv",
                fieldnames=[
                    "proxy",
                    "query_domain",
                    "selected_expert",
                    "oracle_best_expert",
                    "spearman_with_oracle",
                    "top1_oracle_hit",
                    "rank_of_oracle_best_in_proxy",
                    "selected_utility",
                    "oracle_best_utility",
                    "oracle_gap",
                    "oracle_gap_pct",
                ],
                rows=per_query_rows,
            )

            summary_payload = {
                k: {
                    "proxy": v["proxy"],
                    "n_queries": int(v["n_queries"]),
                    "mean_spearman": float(v["mean_spearman"]),
                    "std_spearman": float(v["std_spearman"]),
                    "top1": float(v["top1"]),
                    "mean_oracle_gap": float(v["mean_oracle_gap"]),
                    "mean_oracle_gap_pct": float(v["mean_oracle_gap_pct"]),
                }
                for k, v in proxy_alignment_by_name.items()
            }
            with (run_ctx.reports_dir / "proxy_oracle_alignment_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)

            _write_rows_csv(
                run_ctx.reports_dir / "routing_to_oracle_summary.csv",
                fieldnames=[
                    "proxy",
                    "n_queries",
                    "mean_spearman",
                    "std_spearman",
                    "top1",
                    "mean_oracle_gap",
                    "mean_oracle_gap_pct",
                ],
                rows=list(summary_payload.values()),
            )

            if coverage_status is not None:
                _write_rows_csv(
                    run_ctx.reports_dir / "backbone_coverage_eligibility.csv",
                    fieldnames=[
                        "backbone_type",
                        "status",
                        "valid_for_tier",
                        "all_loqdo_query_folds_present",
                        "all_expected_query_domains_present",
                        "has_partial_metric_rows",
                    ],
                    rows=[
                        {
                            "backbone_type": coverage_status["backbone_type"],
                            "status": coverage_status["status"],
                            "valid_for_tier": int(bool(coverage_status["valid_for_tier"])),
                            "all_loqdo_query_folds_present": int(bool(coverage_status["all_loqdo_query_folds_present"])),
                            "all_expected_query_domains_present": int(bool(coverage_status["all_expected_query_domains_present"])),
                            "has_partial_metric_rows": int(bool(coverage_status["has_partial_metric_rows"])),
                        }
                    ],
                )

            with (run_ctx.reports_dir / "acceptance_decision_summary.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backbone_type": str(cfg.get("features", {}).get("backbone_type", "unknown")),
                        "coverage": coverage_status,
                        "decisions": acceptance_decisions,
                    },
                    f,
                    indent=2,
                )

        for metric_name in latent_cfg["metrics"]:
            fname = {
                "centroid": "compatibility_heatmap_centroid.png",
                "wasserstein": "compatibility_heatmap_wasserstein.png",
                "gaussian_kl": "compatibility_heatmap_kl.png",
            }[metric_name]
            plot_matrix_heatmap(
                matrix=similarity_mats[metric_name],
                domain_order=domain_order,
                title=f"Latent Compatibility ({metric_name})",
                out_path=run_ctx.plots_dir / fname,
            )

        projection, sample_idxs, reducer_info = maybe_project_latent_2d(
            embeddings=embeddings,
            seed=int(cfg["seed"]),
            max_points=int(latent_cfg["umap"]["max_points"]),
        )
        sampled_domains = sample_domains[sample_idxs] if sample_idxs.size > 0 else np.empty((0,), dtype=np.int64)

        plot_latent_map(
            coords=projection,
            sample_domains=sampled_domains,
            domain_order=domain_order,
            out_path=run_ctx.plots_dir / "latent_umap.png",
            title=f"Latent Domain Map ({reducer_info.get('method', 'unknown')})",
        )

        if utility_matrix is not None:
            plot_matrix_heatmap(
                matrix=utility_matrix,
                domain_order=domain_order,
                title="Empirical Expert Utility (negative NELBO)",
                out_path=run_ctx.plots_dir / "expert_utility_heatmap.png",
                cmap="magma",
            )

            for metric_name in latent_cfg["metrics"]:
                scatter_name = {
                    "centroid": "distance_vs_utility_centroid.png",
                    "wasserstein": "distance_vs_utility_wasserstein.png",
                    "gaussian_kl": "distance_vs_utility_kl.png",
                }[metric_name]
                plot_distance_vs_utility(
                    distance_matrix=distance_mats[metric_name],
                    utility_matrix=utility_matrix,
                    domain_order=domain_order,
                    out_path=run_ctx.plots_dir / scatter_name,
                    title=f"Distance vs Utility ({metric_name})",
                    add_regression=True,
                    color_by_query=True,
                )

        composite_metric = str(latent_cfg["composite_metric"])
        if composite_metric not in similarity_mats:
            composite_metric = latent_cfg["metrics"][0]
        plot_composite_figure(
            coords=projection,
            sample_domains=sampled_domains,
            domain_order=domain_order,
            compatibility_matrix=similarity_mats[composite_metric],
            utility_matrix=utility_matrix,
            distance_matrix=distance_mats[composite_metric] if utility_matrix is not None else None,
            out_path=run_ctx.plots_dir / "latent_compatibility_composite.png",
        )
        progress.advance("plots generated")

        gaussian_json = {
            "domain_order": [f"{d}x" for d in domain_order],
            "covariance_regularization_lambda": float(latent_cfg["covariance_regularization_lambda"]),
            "min_samples_per_domain": int(latent_cfg["min_samples_per_domain"]),
            "warnings": gaussian_warnings,
            "domains": _to_jsonable_gaussian_stats(domain_order, gaussian_stats),
        }
        with (run_ctx.reports_dir / "latent_gaussian_stats.json").open("w", encoding="utf-8") as f:
            json.dump(gaussian_json, f, indent=2)

        routing_payload = {
            "domain_order": [f"{d}x" for d in domain_order],
            "scales": scale_by_metric,
            "verification": verification_by_metric,
            "routing": routing_by_metric,
            "reducer": reducer_info,
            "metric_payloads": metric_payloads,
            "distance_matrices": {
                k: matrix_to_domain_dict(domain_order, v) for k, v in distance_mats.items()
            },
            "similarity_matrices": {
                k: matrix_to_domain_dict(domain_order, v) for k, v in similarity_mats.items()
            },
        }
        with (run_ctx.reports_dir / "routing_agreement.json").open("w", encoding="utf-8") as f:
            json.dump(routing_payload, f, indent=2)

        with (run_ctx.reports_dir / "report.md").open("w", encoding="utf-8") as f:
            f.write("# Latent Compatibility Report\n\n")
            f.write(f"- domains: {', '.join(f'{d}x' for d in domain_order)}\n")
            f.write(f"- splits: {', '.join(latent_cfg['splits'])}\n")
            f.write(f"- covariance_regularization_lambda: {float(latent_cfg['covariance_regularization_lambda']):.1e}\n")
            f.write(f"- eigenvalue_floor: {float(latent_cfg['wasserstein']['eigenvalue_floor']):.1e}\n")
            f.write(f"- scale_floor: {float(latent_cfg['similarity']['scale_floor']):.1e}\n")
            f.write("\n## Routing Agreement\n\n")
            for metric_name in latent_cfg["metrics"]:
                r = routing_by_metric[metric_name]
                f.write(f"- {metric_name}: top1={float(r['top1_agreement']):.4f}, ")
                f.write(f"mean_rank={float(r['mean_rank']):.4f}, ")
                f.write(f"spearman_with_utility={float(r.get('spearman_with_utility', 0.0)):.4f}, ")
                f.write(f"spearman_distance_with_utility={float(r.get('spearman_distance_with_utility', 0.0)):.4f}\n")

            if proxy_alignment_by_name and bool(latent_cfg["include_metadata_oracle_proxy_table"]):
                f.write("\n## Proxy Quality vs Oracle\n\n")
                f.write("| proxy | mean_spearman | top1 | mean_oracle_gap | mean_oracle_gap_pct |\n")
                f.write("|---|---:|---:|---:|---:|\n")
                for name in ["metadata"] + [m for m in latent_cfg["metrics"] if m in proxy_alignment_by_name]:
                    p = proxy_alignment_by_name[name]
                    f.write(
                        f"| {name} | {float(p['mean_spearman']):.4f} | {float(p['top1']):.4f} | {float(p['mean_oracle_gap']):.6f} | {float(p['mean_oracle_gap_pct']):.2f} |\n"
                    )

                if acceptance_decisions:
                    f.write("\n## Acceptance Decisions\n\n")
                    for metric_name in latent_cfg["metrics"]:
                        if metric_name not in acceptance_decisions:
                            continue
                        d = acceptance_decisions[metric_name]
                        u = d["uplifts"]
                        f.write(
                            f"- {metric_name}: tier={d['tier']}, "
                            f"spearman_uplift={float(u['mean_spearman']):.4f}, "
                            f"top1_uplift={float(u['top1']):.4f}, "
                            f"oracle_gap_reduction_pct={float(u['oracle_gap_reduction_pct']):.2f}, "
                            f"guardrail_breach={bool(d['guardrail']['breach'])}\n"
                        )

                if coverage_status is not None:
                    f.write("\n## Backbone Coverage\n\n")
                    f.write(
                        f"- backbone={coverage_status['backbone_type']}, status={coverage_status['status']}, "
                        f"all_loqdo_query_folds_present={coverage_status['all_loqdo_query_folds_present']}, "
                        f"all_expected_query_domains_present={coverage_status['all_expected_query_domains_present']}, "
                        f"has_partial_metric_rows={coverage_status['has_partial_metric_rows']}\n"
                    )

            f.write("\n## Verification\n\n")
            for metric_name in latent_cfg["metrics"]:
                v = verification_by_metric[metric_name]
                f.write(
                    f"- {metric_name}: finite_ok={v['finite_ok']}, symmetry_ok={v['symmetry_ok']}, diagonal_optimality_ok={v['diagonal_optimality_ok']}\n"
                )

            if gaussian_warnings:
                f.write("\n## Warnings\n\n")
                for warning in gaussian_warnings:
                    f.write(f"- {warning}\n")

        write_run_summary(
            reports_dir=run_ctx.reports_dir,
            mode="latent_compatibility",
            payload={
                "routing_artifact": "routing_agreement.json",
                "gaussian_stats_artifact": "latent_gaussian_stats.json",
                "correlation_artifact": "metric_correlation_table.csv",
                "proxy_alignment_per_query_artifact": "proxy_oracle_alignment_per_query.csv",
                "proxy_alignment_summary_artifact": "proxy_oracle_alignment_summary.json",
                "routing_to_oracle_summary_artifact": "routing_to_oracle_summary.csv",
                "backbone_coverage_artifact": "backbone_coverage_eligibility.csv",
                "acceptance_decision_artifact": "acceptance_decision_summary.json",
                "report_artifact": "report.md",
            },
        )
        progress.advance("reports written")
        progress.close()

        print("Latent compatibility experiment complete.")
        print("Run directory:", run_ctx.run_root)
        print("Reports:", run_ctx.reports_dir)
        print("Plots:", run_ctx.plots_dir)
        print("Latest run pointer:", run_ctx.latest_file)
