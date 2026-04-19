from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
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
    evaluate_sample_level_routing_alignment,
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
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load
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

    routing_granularity = str(block.get("routing_granularity", "sample")).strip().lower()
    if routing_granularity not in {"domain", "sample"}:
        raise ValueError("latent_compatibility.routing_granularity must be one of ['domain', 'sample']")

    verification = block.get("verification", {})
    wasserstein = block.get("wasserstein", {})
    similarity = block.get("similarity", {})
    empirical = block.get("empirical_utility", {})
    sample_level = block.get("sample_level_routing", {})
    coverage_gates = block.get("coverage_gates", {})
    thresholds = block.get("acceptance_thresholds", {})
    failure_gates = block.get("step1_failure_gates", {})
    step2_oracle = block.get("step2_oracle_benchmark", {})
    strong = thresholds.get("strong", {}) if isinstance(thresholds, dict) else {}
    non_inferiority = thresholds.get("non_inferiority", {}) if isinstance(thresholds, dict) else {}

    out = {
        "metrics": metrics,
        "splits": splits,
        "routing_granularity": routing_granularity,
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
        "sample_level_routing": {
            "max_samples": int(sample_level.get("max_samples", 0)),
            "timing_every": int(sample_level.get("timing_every", 0)),
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
                "expected_backbones": [str(x) for x in strong.get("expected_backbones", [])],
                "disallow_any_backbone_guardrail_breach": bool(strong.get("disallow_any_backbone_guardrail_breach", True)),
            },
            "non_inferiority": {
                "max_oracle_gap_worsening_pct": float(non_inferiority.get("max_oracle_gap_worsening_pct", 5.0)),
            },
        },
        "step1_failure_gates": {
            "enabled": bool(failure_gates.get("enabled", True)),
            "overall_top1_lt": float(failure_gates.get("overall_top1_lt", 0.50)),
            "mean_oracle_gap_pct_gt": float(failure_gates.get("mean_oracle_gap_pct_gt", 10.0)),
            "per_domain_top1_lt": float(failure_gates.get("per_domain_top1_lt", 0.50)),
            "min_failing_query_domains": int(failure_gates.get("min_failing_query_domains", 2)),
        },
        "step2_oracle_benchmark": {
            "enabled": bool(step2_oracle.get("enabled", True)),
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
    if out["sample_level_routing"]["max_samples"] < 0:
        raise ValueError("latent_compatibility.sample_level_routing.max_samples must be >= 0")
    if out["sample_level_routing"]["timing_every"] < 0:
        raise ValueError("latent_compatibility.sample_level_routing.timing_every must be >= 0")
    if not (0.0 <= out["step1_failure_gates"]["overall_top1_lt"] <= 1.0):
        raise ValueError("latent_compatibility.step1_failure_gates.overall_top1_lt must be in [0, 1]")
    if out["step1_failure_gates"]["mean_oracle_gap_pct_gt"] < 0.0:
        raise ValueError("latent_compatibility.step1_failure_gates.mean_oracle_gap_pct_gt must be >= 0")
    if not (0.0 <= out["step1_failure_gates"]["per_domain_top1_lt"] <= 1.0):
        raise ValueError("latent_compatibility.step1_failure_gates.per_domain_top1_lt must be in [0, 1]")
    if out["step1_failure_gates"]["min_failing_query_domains"] < 1:
        raise ValueError("latent_compatibility.step1_failure_gates.min_failing_query_domains must be >= 1")

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

    expected_backbones = [str(x).strip().lower() for x in strong_cfg.get("expected_backbones", []) if str(x).strip()]
    current_backbone = str(coverage_status.get("backbone_type", "")).strip().lower()
    if expected_backbones:
        backbone_denominator = len(set(expected_backbones))
        current_backbone_is_expected = int(current_backbone in set(expected_backbones))
    else:
        backbone_denominator = 1
        current_backbone_is_expected = 1

    valid_backbones = 1 if bool(coverage_status.get("valid_for_tier", False)) and current_backbone_is_expected == 1 else 0
    improved_backbones = 1 if (spearman_uplift > 0 and top1_uplift > 0 and gap_reduction_pct > 0 and valid_backbones == 1) else 0
    backbone_fraction = float(improved_backbones / max(backbone_denominator, 1))

    strong_checks = {
        "spearman_uplift_gt": bool(spearman_uplift > float(strong_cfg["spearman_uplift_gt"])),
        "top1_uplift_gte": bool(top1_uplift >= float(strong_cfg["top1_uplift_gte"])),
        "oracle_gap_reduction_gt_pct": bool(gap_reduction_pct > float(strong_cfg["oracle_gap_reduction_gt_pct"])),
        "min_backbone_fraction": bool(backbone_fraction >= float(strong_cfg["min_backbone_fraction"])),
        "no_backbone_guardrail_breach": bool(not guardrail_breach),
        "backbone_coverage_valid": bool(coverage_status.get("valid_for_tier", False)),
        "current_backbone_in_expected_set": bool(current_backbone_is_expected == 1),
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
            "expected_backbones": expected_backbones,
            "current_backbone": current_backbone,
            "backbone_fraction_numerator": int(improved_backbones),
            "backbone_fraction_denominator": int(backbone_denominator),
            "backbone_fraction": float(backbone_fraction),
        },
    }


def _try_parse_domain_label(value: Any) -> int | None:
    try:
        s = str(value).strip().lower().replace("x", "")
        return int(s)
    except Exception:
        return None


def _build_learned_comparison_eligibility_rows(
    cfg: Dict[str, Any],
    domain_order: List[int],
    strict_context_match: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    source_path = Path("results/comparison_tables/learned_compatibility_loqdo_breakhis_raw.csv")
    if not source_path.exists():
        return [
            {
                "source_file": str(source_path),
                "status": "source_missing",
                "dataset_match": 0,
                "backbone_match": 0,
                "seed_match": 0,
                "variant_match": 0,
                "domain_set_match": 0,
                "split_semantics_match": 0,
                "eligible": 0,
                "reason": "learned comparison source CSV was not found",
            }
        ]

    dataset_name = str(cfg.get("experiment", {}).get("dataset_name", "")).strip().lower()
    backbone_type = str(cfg.get("features", {}).get("backbone_type", "")).strip().lower()
    seed = int(cfg.get("seed", 0))
    variant = str(cfg.get("experiment", {}).get("variant", "B")).strip().upper()
    expected_domains = set(int(d) for d in domain_order)

    with source_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            row_dataset = str(item.get("dataset_name", "")).strip().lower()
            row_backbone = str(item.get("backbone_type", "")).strip().lower()
            row_seed = int(float(item.get("seed", 0) or 0))
            row_variant = str(item.get("variant", "")).strip().upper()
            heldout = _try_parse_domain_label(item.get("heldout_query_domain", ""))

            dataset_match = int(row_dataset == dataset_name)
            backbone_match = int(row_backbone == backbone_type)
            seed_match = int(row_seed == seed)
            variant_match = int(row_variant == variant)
            domain_set_match = int(heldout in expected_domains if heldout is not None else False)
            split_semantics_match = int(heldout is not None)

            if strict_context_match:
                eligible = int(
                    dataset_match
                    and backbone_match
                    and seed_match
                    and variant_match
                    and domain_set_match
                    and split_semantics_match
                )
            else:
                eligible = int(dataset_match and backbone_match and domain_set_match and split_semantics_match)

            reasons: List[str] = []
            if not dataset_match:
                reasons.append("dataset_mismatch")
            if not backbone_match:
                reasons.append("backbone_mismatch")
            if not seed_match:
                reasons.append("seed_mismatch")
            if not variant_match:
                reasons.append("variant_mismatch")
            if not domain_set_match:
                reasons.append("domain_set_mismatch")
            if not split_semantics_match:
                reasons.append("split_semantics_mismatch")

            rows.append(
                {
                    "source_file": str(source_path),
                    "run_id": str(item.get("run_id", "")),
                    "method": str(item.get("method", "")),
                    "heldout_query_domain": str(item.get("heldout_query_domain", "")),
                    "status": "eligible" if eligible else "excluded",
                    "dataset_match": dataset_match,
                    "backbone_match": backbone_match,
                    "seed_match": seed_match,
                    "variant_match": variant_match,
                    "domain_set_match": domain_set_match,
                    "split_semantics_match": split_semantics_match,
                    "eligible": eligible,
                    "reason": "ok" if eligible else ";".join(reasons),
                }
            )

    if not rows:
        rows.append(
            {
                "source_file": str(source_path),
                "status": "empty_source",
                "dataset_match": 0,
                "backbone_match": 0,
                "seed_match": 0,
                "variant_match": 0,
                "domain_set_match": 0,
                "split_semantics_match": 0,
                "eligible": 0,
                "reason": "source CSV had no rows",
            }
        )
    return rows


def _build_step1_failure_gate_decision(
    metric_name: str,
    routing_summary: Dict[str, Any],
    metadata_oracle_summary: Dict[str, Any],
    gates_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    overall_top1 = float(routing_summary.get("top1_agreement", 0.0))
    mean_oracle_gap_pct = float(metadata_oracle_summary.get("mean_oracle_gap_pct", 0.0))

    per_domain_rows = routing_summary.get("per_domain_summary", [])
    per_domain_threshold = float(gates_cfg["per_domain_top1_lt"])
    failing_query_domains: List[str] = []
    for row in per_domain_rows:
        domain_top1 = float(row.get("top1_agreement", 1.0))
        if domain_top1 < per_domain_threshold:
            failing_query_domains.append(str(row.get("query_domain", "unknown")))

    checks = {
        "overall_top1_lt": bool(overall_top1 < float(gates_cfg["overall_top1_lt"])),
        "mean_oracle_gap_pct_gt": bool(mean_oracle_gap_pct > float(gates_cfg["mean_oracle_gap_pct_gt"])),
        "min_failing_query_domains": bool(len(failing_query_domains) >= int(gates_cfg["min_failing_query_domains"])),
    }

    return {
        "metric": str(metric_name),
        "routing_granularity": str(routing_summary.get("granularity", "unknown")),
        "failure_confirmed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "overall_top1_lt": float(gates_cfg["overall_top1_lt"]),
            "mean_oracle_gap_pct_gt": float(gates_cfg["mean_oracle_gap_pct_gt"]),
            "per_domain_top1_lt": float(gates_cfg["per_domain_top1_lt"]),
            "min_failing_query_domains": int(gates_cfg["min_failing_query_domains"]),
        },
        "observed": {
            "overall_top1": overall_top1,
            "mean_oracle_gap_pct": mean_oracle_gap_pct,
            "n_failing_query_domains": int(len(failing_query_domains)),
            "failing_query_domains": sorted(failing_query_domains),
        },
    }


def _load_expert_model(
    checkpoint_path: Path,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
    metadata_dim: int = 0,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
) -> CVAEExpert:
    model = CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        metadata_dim=int(metadata_dim),
        metadata_constraint_cfg=metadata_constraint_cfg,
        aux_metadata_dim=int(metadata_dim),
    ).to(device)
    model.load_state_dict(safe_torch_load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _compute_sample_expert_nelbo_matrix(
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    expert_checkpoints: Dict[str, str],
    domain_order: List[int],
    hidden_dim: int,
    latent_dim: int,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be [N, D], got shape={embeddings.shape}")
    if not domain_order:
        raise ValueError("domain_order must not be empty")

    input_dim = int(embeddings.shape[1])
    x_cpu = torch.from_numpy(embeddings.astype(np.float32, copy=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_vectors = None
    metadata_dim = 0
    if conditioning_enabled:
        configured_order = resolve_domain_order(configured_domains or domain_order)
        metadata_items = [{"magnification": int(v)} for v in sample_domains.tolist()]
        metadata_vectors = build_domain_one_hot(metadata_items, configured_order)
        metadata_dim = int(len(configured_order))

    rows: List[np.ndarray] = []
    with torch.no_grad():
        for expert_domain in domain_order:
            checkpoint_key = f"{int(expert_domain)}x"
            checkpoint_raw = expert_checkpoints.get(checkpoint_key)
            if checkpoint_raw is None:
                raise RuntimeError(f"Missing expert checkpoint for domain {checkpoint_key}")

            model = _load_expert_model(
                checkpoint_path=Path(checkpoint_raw),
                input_dim=input_dim,
                hidden_dim=int(hidden_dim),
                latent_dim=int(latent_dim),
                device=device,
                metadata_dim=metadata_dim,
                metadata_constraint_cfg=metadata_constraint_cfg,
            )
            parts: List[torch.Tensor] = []
            for i in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[i : i + int(batch_size)].to(device)
                mb = metadata_vectors[i : i + int(batch_size)].to(device) if metadata_vectors is not None else None
                recon, mu, logvar = model(xb, m=mb)
                prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=mb)
                rec, kl = elbo_components(
                    recon,
                    xb,
                    mu,
                    logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    kl_weight=kl_weight,
                )
                parts.append((rec + kl).detach().cpu())

            if not parts:
                rows.append(np.empty((0,), dtype=np.float64))
            else:
                rows.append(torch.cat(parts, dim=0).numpy().astype(np.float64, copy=False))

    return np.stack(rows, axis=0)


def _compute_sample_level_metadata_oracle_gap(
    sample_domains: np.ndarray,
    domain_order: List[int],
    strategy: str,
    tau: float,
    nelbo_by_expert_sample: np.ndarray,
    similarity_lookup_matrix: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    n_domains = len(domain_order)
    n_samples = int(sample_domains.shape[0])
    if nelbo_by_expert_sample.shape != (n_domains, n_samples):
        raise ValueError(
            "nelbo_by_expert_sample must have shape "
            f"({n_domains}, {n_samples}), got {nelbo_by_expert_sample.shape}"
        )

    metadata_scores_by_query: Dict[int, List[float]] = {}
    for query_domain in set(int(d) for d in sample_domains.tolist()):
        metadata_scores_by_query[query_domain] = [
            compute_similarity(
                {"magnification": int(query_domain)},
                {"magnification": int(expert_domain)},
                strategy=str(strategy),
                tau=float(tau),
                similarity_matrix=similarity_lookup_matrix,
            )
            for expert_domain in domain_order
        ]

    per_sample_rows: List[Dict[str, Any]] = []
    per_domain_rows: Dict[int, List[float]] = {int(d): [] for d in domain_order}
    gaps: List[float] = []
    gaps_pct: List[float] = []
    top1_hits = 0

    for idx in range(n_samples):
        query_domain = int(sample_domains[idx])
        metadata_scores = metadata_scores_by_query[query_domain]
        selected_idx = int(np.argmax(np.asarray(metadata_scores, dtype=np.float64)))

        nelbo_col = nelbo_by_expert_sample[:, idx]
        oracle_idx = int(np.argmin(nelbo_col))

        selected_nelbo = float(nelbo_col[selected_idx])
        oracle_nelbo = float(nelbo_col[oracle_idx])
        gap = float(selected_nelbo - oracle_nelbo)
        gap_pct = float((gap / max(abs(oracle_nelbo), 1e-12)) * 100.0)

        top1_hit = int(selected_idx == oracle_idx)
        top1_hits += top1_hit
        gaps.append(gap)
        gaps_pct.append(gap_pct)
        per_domain_rows.setdefault(query_domain, []).append(gap_pct)

        per_sample_rows.append(
            {
                "sample_index": int(idx),
                "query_domain": f"{query_domain}x",
                "selected_expert": f"{domain_order[selected_idx]}x",
                "oracle_best_expert": f"{domain_order[oracle_idx]}x",
                "top1_oracle_hit": int(top1_hit),
                "selected_nelbo": selected_nelbo,
                "oracle_nelbo": oracle_nelbo,
                "oracle_gap": gap,
                "oracle_gap_pct": gap_pct,
            }
        )

    per_domain_summary = [
        {
            "query_domain": f"{int(d)}x",
            "n_samples": int(len(per_domain_rows.get(int(d), []))),
            "mean_oracle_gap_pct": float(np.mean(per_domain_rows.get(int(d), [0.0]))),
        }
        for d in domain_order
    ]

    return {
        "proxy": "metadata_sample",
        "n_samples": int(n_samples),
        "top1": float(top1_hits / max(n_samples, 1)),
        "mean_oracle_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gaps_pct)) if gaps_pct else 0.0,
        "per_sample": per_sample_rows,
        "per_domain": per_domain_summary,
    }


def _compute_sample_level_proxy_oracle_summary(
    proxy_name: str,
    sample_domains: np.ndarray,
    domain_order: List[int],
    nelbo_by_expert_sample: np.ndarray,
    selected_expert_domains: List[int],
    sample_indices: List[int] | None = None,
) -> Dict[str, Any]:
    idx_by_domain = {int(d): i for i, d in enumerate(domain_order)}
    if sample_indices is None:
        sample_indices = list(range(len(selected_expert_domains)))

    if len(sample_indices) != len(selected_expert_domains):
        raise ValueError("sample_indices and selected_expert_domains must have the same length")

    per_sample_rows: List[Dict[str, Any]] = []
    gaps: List[float] = []
    gaps_pct: List[float] = []
    selected_nelbos: List[float] = []
    oracle_nelbos: List[float] = []
    top1_hits = 0
    per_query_gap_pct: Dict[int, List[float]] = {int(d): [] for d in domain_order}

    for local_idx, sample_idx in enumerate(sample_indices):
        query_domain = int(sample_domains[int(sample_idx)])
        selected_domain = int(selected_expert_domains[local_idx])
        if selected_domain not in idx_by_domain:
            raise RuntimeError(f"Selected expert domain {selected_domain} is not in domain_order")
        selected_idx = idx_by_domain[selected_domain]

        nelbo_col = nelbo_by_expert_sample[:, int(sample_idx)]
        oracle_idx = int(np.argmin(nelbo_col))
        oracle_domain = int(domain_order[oracle_idx])

        selected_nelbo = float(nelbo_col[selected_idx])
        oracle_nelbo = float(nelbo_col[oracle_idx])
        gap = float(selected_nelbo - oracle_nelbo)
        gap_pct = float((gap / max(abs(oracle_nelbo), 1e-12)) * 100.0)
        top1_hit = int(selected_idx == oracle_idx)

        top1_hits += top1_hit
        gaps.append(gap)
        gaps_pct.append(gap_pct)
        selected_nelbos.append(selected_nelbo)
        oracle_nelbos.append(oracle_nelbo)
        per_query_gap_pct.setdefault(query_domain, []).append(gap_pct)

        per_sample_rows.append(
            {
                "proxy": str(proxy_name),
                "sample_index": int(sample_idx),
                "query_domain": f"{query_domain}x",
                "selected_expert": f"{selected_domain}x",
                "oracle_best_expert": f"{oracle_domain}x",
                "top1_oracle_hit": int(top1_hit),
                "selected_nelbo": selected_nelbo,
                "oracle_nelbo": oracle_nelbo,
                "oracle_gap": gap,
                "oracle_gap_pct": gap_pct,
            }
        )

    n_samples = len(sample_indices)
    per_domain_summary = [
        {
            "proxy": str(proxy_name),
            "query_domain": f"{int(d)}x",
            "n_samples": int(len(per_query_gap_pct.get(int(d), []))),
            "mean_oracle_gap_pct": float(np.mean(per_query_gap_pct.get(int(d), [0.0]))),
        }
        for d in domain_order
    ]

    return {
        "proxy": str(proxy_name),
        "n_samples": int(n_samples),
        "top1": float(top1_hits / max(n_samples, 1)),
        "routed_nelbo_mean": float(np.mean(selected_nelbos)) if selected_nelbos else 0.0,
        "oracle_routed_nelbo_mean": float(np.mean(oracle_nelbos)) if oracle_nelbos else 0.0,
        "mean_oracle_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gaps_pct)) if gaps_pct else 0.0,
        "per_sample": per_sample_rows,
        "per_domain": per_domain_summary,
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
            if latent_cfg["routing_granularity"] == "sample":
                routing_by_metric[metric_name] = evaluate_sample_level_routing_alignment(
                    embeddings=embeddings,
                    sample_domains=sample_domains,
                    domain_order=domain_order,
                    stats=gaussian_stats,
                    strategy=str(cfg["routing"]["strategy"]),
                    tau=float(cfg["routing"]["tau"]),
                    eigenvalue_floor=float(latent_cfg["wasserstein"]["eigenvalue_floor"]),
                    similarity_lookup_matrix=cfg.get("routing", {}).get("similarity_matrix"),
                    max_samples=(
                        int(latent_cfg["sample_level_routing"]["max_samples"])
                        if int(latent_cfg["sample_level_routing"]["max_samples"]) > 0
                        else None
                    ),
                    timing_every=int(latent_cfg["sample_level_routing"]["timing_every"]),
                )
        progress.advance("similarity matrices, verification, and routing computed")

        utility_matrix: np.ndarray | None = None
        utility_report: Dict[str, Any] | None = None
        expert_checkpoints: Dict[str, str] = {}
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
                conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
                configured_domains=cfg.get("data", {}).get("magnifications", []),
                metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
            )
            expert_checkpoints = dict(experts)
            progress.advance("empirical utility experts trained")

            utility_report = compute_expert_domain_matrix(
                test_cache=cache_paths["test"],
                expert_checkpoints=experts,
                hidden_dim=int(cfg["model"]["hidden_dim"]),
                latent_dim=int(cfg["model"]["latent_dim"]),
                conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
                configured_domains=cfg.get("data", {}).get("magnifications", []),
                metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
            )
            utility_matrix = _utility_matrix_from_expert_matrix(domain_order, utility_report)
            with (run_ctx.reports_dir / "expert_utility_matrix.json").open("w", encoding="utf-8") as f:
                json.dump(utility_report, f, indent=2)
            progress.advance("empirical utility matrix computed")

        metric_corr_rows: List[Dict[str, Any]] = []
        proxy_alignment_by_name: Dict[str, Dict[str, Any]] = {}
        acceptance_decisions: Dict[str, Any] = {}
        step1_failure_gate_by_metric: Dict[str, Any] = {}
        metadata_sample_oracle_summary: Dict[str, Any] | None = None
        sample_level_oracle_benchmark_by_proxy: Dict[str, Dict[str, Any]] = {}
        sample_level_oracle_benchmark_per_sample_rows: List[Dict[str, Any]] = []
        sample_level_oracle_benchmark_per_domain_rows: List[Dict[str, Any]] = []
        nelbo_by_expert_sample: np.ndarray | None = None
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

            if bool(latent_cfg["step1_failure_gates"]["enabled"]) and latent_cfg["routing_granularity"] == "sample":
                if not expert_checkpoints:
                    raise RuntimeError(
                        "Step 1 sample-level failure gates require expert checkpoints. "
                        "Enable latent_compatibility.empirical_utility.enabled to compute per-sample oracle gaps."
                    )
                if nelbo_by_expert_sample is None:
                    nelbo_by_expert_sample = _compute_sample_expert_nelbo_matrix(
                        embeddings=embeddings,
                        sample_domains=sample_domains,
                        expert_checkpoints=expert_checkpoints,
                        domain_order=domain_order,
                        hidden_dim=int(cfg["model"]["hidden_dim"]),
                        latent_dim=int(cfg["model"]["latent_dim"]),
                        conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
                        configured_domains=cfg.get("data", {}).get("magnifications", []),
                        metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
                        batch_size=int(cfg["training"].get("batch_size", 2048)),
                    )
                metadata_sample_oracle_summary = _compute_sample_level_metadata_oracle_gap(
                    sample_domains=sample_domains,
                    domain_order=domain_order,
                    strategy=str(cfg["routing"]["strategy"]),
                    tau=float(cfg["routing"]["tau"]),
                    nelbo_by_expert_sample=nelbo_by_expert_sample,
                    similarity_lookup_matrix=cfg.get("routing", {}).get("similarity_matrix"),
                )

                metadata_summary = metadata_sample_oracle_summary
                for metric_name in latent_cfg["metrics"]:
                    step1_failure_gate_by_metric[metric_name] = _build_step1_failure_gate_decision(
                        metric_name=metric_name,
                        routing_summary=routing_by_metric[metric_name],
                        metadata_oracle_summary=metadata_summary,
                        gates_cfg=latent_cfg["step1_failure_gates"],
                    )

            if bool(latent_cfg["step2_oracle_benchmark"]["enabled"]) and latent_cfg["routing_granularity"] == "sample":
                if not expert_checkpoints:
                    raise RuntimeError(
                        "Step 2 oracle benchmark requires expert checkpoints. "
                        "Enable latent_compatibility.empirical_utility.enabled."
                    )
                if nelbo_by_expert_sample is None:
                    nelbo_by_expert_sample = _compute_sample_expert_nelbo_matrix(
                        embeddings=embeddings,
                        sample_domains=sample_domains,
                        expert_checkpoints=expert_checkpoints,
                        domain_order=domain_order,
                        hidden_dim=int(cfg["model"]["hidden_dim"]),
                        latent_dim=int(cfg["model"]["latent_dim"]),
                        conditioning_cfg=cfg.get("model", {}).get("conditioning", {}),
                        configured_domains=cfg.get("data", {}).get("magnifications", []),
                        metadata_constraint_cfg=cfg.get("model", {}).get("metadata_constraint", {}),
                        batch_size=int(cfg["training"].get("batch_size", 2048)),
                    )

                sample_rows_ref = routing_by_metric[latent_cfg["metrics"][0]].get("per_sample", [])
                sample_indices = [int(r["sample_index"]) for r in sample_rows_ref]

                metadata_selected_domains = [
                    int(str(r["metadata_selected_expert"]).replace("x", "")) for r in sample_rows_ref
                ]
                metadata_summary = _compute_sample_level_proxy_oracle_summary(
                    proxy_name="metadata",
                    sample_domains=sample_domains,
                    domain_order=domain_order,
                    nelbo_by_expert_sample=nelbo_by_expert_sample,
                    selected_expert_domains=metadata_selected_domains,
                    sample_indices=sample_indices,
                )
                sample_level_oracle_benchmark_by_proxy["metadata"] = metadata_summary
                sample_level_oracle_benchmark_per_sample_rows.extend(metadata_summary["per_sample"])
                sample_level_oracle_benchmark_per_domain_rows.extend(metadata_summary["per_domain"])

                for metric_name in latent_cfg["metrics"]:
                    metric_rows = routing_by_metric[metric_name].get("per_sample", [])
                    latent_selected_domains = [
                        int(str(r["latent_best_expert"]).replace("x", "")) for r in metric_rows
                    ]
                    metric_summary = _compute_sample_level_proxy_oracle_summary(
                        proxy_name=metric_name,
                        sample_domains=sample_domains,
                        domain_order=domain_order,
                        nelbo_by_expert_sample=nelbo_by_expert_sample,
                        selected_expert_domains=latent_selected_domains,
                        sample_indices=sample_indices,
                    )
                    sample_level_oracle_benchmark_by_proxy[metric_name] = metric_summary
                    sample_level_oracle_benchmark_per_sample_rows.extend(metric_summary["per_sample"])
                    sample_level_oracle_benchmark_per_domain_rows.extend(metric_summary["per_domain"])

                oracle_selected_domains = [
                    int(domain_order[int(np.argmin(nelbo_by_expert_sample[:, int(idx)]))]) for idx in sample_indices
                ]
                oracle_summary = _compute_sample_level_proxy_oracle_summary(
                    proxy_name="oracle",
                    sample_domains=sample_domains,
                    domain_order=domain_order,
                    nelbo_by_expert_sample=nelbo_by_expert_sample,
                    selected_expert_domains=oracle_selected_domains,
                    sample_indices=sample_indices,
                )
                sample_level_oracle_benchmark_by_proxy["oracle"] = oracle_summary
                sample_level_oracle_benchmark_per_sample_rows.extend(oracle_summary["per_sample"])
                sample_level_oracle_benchmark_per_domain_rows.extend(oracle_summary["per_domain"])

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

            if step1_failure_gate_by_metric:
                with (run_ctx.reports_dir / "step1_failure_gate_summary.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "routing_granularity": str(latent_cfg["routing_granularity"]),
                            "oracle_gap_source": "sample_level_nelbo",
                            "thresholds": latent_cfg["step1_failure_gates"],
                            "composite_metric": str(latent_cfg["composite_metric"]),
                            "metadata_sample_oracle_summary": metadata_sample_oracle_summary,
                            "by_metric": step1_failure_gate_by_metric,
                        },
                        f,
                        indent=2,
                    )
                _write_rows_csv(
                    run_ctx.reports_dir / "step1_failure_gate_summary.csv",
                    fieldnames=[
                        "metric",
                        "failure_confirmed",
                        "overall_top1",
                        "overall_top1_lt_threshold",
                        "mean_oracle_gap_pct",
                        "mean_oracle_gap_pct_gt_threshold",
                        "n_failing_query_domains",
                        "min_failing_query_domains_threshold",
                        "failing_query_domains",
                    ],
                    rows=[
                        {
                            "metric": str(metric_name),
                            "failure_confirmed": int(bool(payload["failure_confirmed"])),
                            "overall_top1": float(payload["observed"]["overall_top1"]),
                            "overall_top1_lt_threshold": float(payload["thresholds"]["overall_top1_lt"]),
                            "mean_oracle_gap_pct": float(payload["observed"]["mean_oracle_gap_pct"]),
                            "mean_oracle_gap_pct_gt_threshold": float(payload["thresholds"]["mean_oracle_gap_pct_gt"]),
                            "n_failing_query_domains": int(payload["observed"]["n_failing_query_domains"]),
                            "min_failing_query_domains_threshold": int(payload["thresholds"]["min_failing_query_domains"]),
                            "failing_query_domains": ";".join(payload["observed"]["failing_query_domains"]),
                        }
                        for metric_name, payload in step1_failure_gate_by_metric.items()
                    ],
                )

            if metadata_sample_oracle_summary is not None:
                _write_rows_csv(
                    run_ctx.reports_dir / "sample_level_oracle_gap_per_sample.csv",
                    fieldnames=[
                        "sample_index",
                        "query_domain",
                        "selected_expert",
                        "oracle_best_expert",
                        "top1_oracle_hit",
                        "selected_nelbo",
                        "oracle_nelbo",
                        "oracle_gap",
                        "oracle_gap_pct",
                    ],
                    rows=list(metadata_sample_oracle_summary.get("per_sample", [])),
                )
                _write_rows_csv(
                    run_ctx.reports_dir / "sample_level_oracle_gap_per_domain.csv",
                    fieldnames=["query_domain", "n_samples", "mean_oracle_gap_pct"],
                    rows=list(metadata_sample_oracle_summary.get("per_domain", [])),
                )

            if sample_level_oracle_benchmark_by_proxy:
                benchmark_rows = []
                for payload in sample_level_oracle_benchmark_by_proxy.values():
                    benchmark_rows.append(
                        {
                            "proxy": str(payload["proxy"]),
                            "n_samples": int(payload["n_samples"]),
                            "top1_oracle_hit": float(payload["top1"]),
                            "routed_nelbo_mean": float(payload["routed_nelbo_mean"]),
                            "oracle_routed_nelbo_mean": float(payload["oracle_routed_nelbo_mean"]),
                            "mean_oracle_gap": float(payload["mean_oracle_gap"]),
                            "mean_oracle_gap_pct": float(payload["mean_oracle_gap_pct"]),
                        }
                    )

                _write_rows_csv(
                    run_ctx.reports_dir / "sample_level_oracle_benchmark_summary.csv",
                    fieldnames=[
                        "proxy",
                        "n_samples",
                        "top1_oracle_hit",
                        "routed_nelbo_mean",
                        "oracle_routed_nelbo_mean",
                        "mean_oracle_gap",
                        "mean_oracle_gap_pct",
                    ],
                    rows=benchmark_rows,
                )
                with (run_ctx.reports_dir / "sample_level_oracle_benchmark_summary.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "routing_granularity": str(latent_cfg["routing_granularity"]),
                            "oracle_definition": "per_sample_min_nelbo",
                            "summary": benchmark_rows,
                        },
                        f,
                        indent=2,
                    )
                _write_rows_csv(
                    run_ctx.reports_dir / "sample_level_oracle_benchmark_per_sample.csv",
                    fieldnames=[
                        "proxy",
                        "sample_index",
                        "query_domain",
                        "selected_expert",
                        "oracle_best_expert",
                        "top1_oracle_hit",
                        "selected_nelbo",
                        "oracle_nelbo",
                        "oracle_gap",
                        "oracle_gap_pct",
                    ],
                    rows=sample_level_oracle_benchmark_per_sample_rows,
                )
                _write_rows_csv(
                    run_ctx.reports_dir / "sample_level_oracle_benchmark_per_domain.csv",
                    fieldnames=["proxy", "query_domain", "n_samples", "mean_oracle_gap_pct"],
                    rows=sample_level_oracle_benchmark_per_domain_rows,
                )

        learned_eligibility_rows = _build_learned_comparison_eligibility_rows(
            cfg=cfg,
            domain_order=domain_order,
            strict_context_match=bool(latent_cfg["learned_comparison"]["strict_context_match"]),
        )
        _write_rows_csv(
            run_ctx.reports_dir / "learned_comparison_eligibility.csv",
            fieldnames=[
                "source_file",
                "run_id",
                "method",
                "heldout_query_domain",
                "status",
                "dataset_match",
                "backbone_match",
                "seed_match",
                "variant_match",
                "domain_set_match",
                "split_semantics_match",
                "eligible",
                "reason",
            ],
            rows=learned_eligibility_rows,
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
                f.write("\n## Proxy Quality vs Oracle (Domain-Level Aggregate)\n\n")
                f.write("- note: this table is query-domain aggregated and is not used for Step 1 sample-level failure gates.\n")
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

                if step1_failure_gate_by_metric:
                    f.write("\n## Step 1 Failure Gates\n\n")
                    thresholds = latent_cfg["step1_failure_gates"]
                    f.write(
                        "- thresholds: "
                        f"overall_top1_lt={float(thresholds['overall_top1_lt']):.2f}, "
                        f"mean_oracle_gap_pct_gt={float(thresholds['mean_oracle_gap_pct_gt']):.2f}, "
                        f"per_domain_top1_lt={float(thresholds['per_domain_top1_lt']):.2f}, "
                        f"min_failing_query_domains={int(thresholds['min_failing_query_domains'])}\n"
                    )
                    if metadata_sample_oracle_summary is not None:
                        f.write(
                            "- oracle_gap_source=sample_level_nelbo, "
                            f"metadata_top1_oracle_hit={float(metadata_sample_oracle_summary['top1']):.4f}, "
                            f"metadata_mean_oracle_gap={float(metadata_sample_oracle_summary['mean_oracle_gap']):.6f}, "
                            f"metadata_mean_oracle_gap_pct={float(metadata_sample_oracle_summary['mean_oracle_gap_pct']):.2f}, "
                            "artifacts=(sample_level_oracle_gap_per_sample.csv, sample_level_oracle_gap_per_domain.csv)\n"
                        )
                    for metric_name in latent_cfg["metrics"]:
                        if metric_name not in step1_failure_gate_by_metric:
                            continue
                        d = step1_failure_gate_by_metric[metric_name]
                        o = d["observed"]
                        c = d["checks"]
                        f.write(
                            f"- {metric_name}: failure_confirmed={bool(d['failure_confirmed'])}, "
                            f"overall_top1={float(o['overall_top1']):.4f}, "
                            f"mean_oracle_gap_pct={float(o['mean_oracle_gap_pct']):.2f}, "
                            f"failing_query_domains={int(o['n_failing_query_domains'])}, "
                            f"checks=({c['overall_top1_lt']}, {c['mean_oracle_gap_pct_gt']}, {c['min_failing_query_domains']})\n"
                        )

                if sample_level_oracle_benchmark_by_proxy:
                    f.write("\n## Step 2 Oracle Benchmark (Sample-Level NELBO)\n\n")
                    f.write("| proxy | top1_oracle_hit | routed_nelbo_mean | oracle_routed_nelbo_mean | mean_oracle_gap | mean_oracle_gap_pct |\n")
                    f.write("|---|---:|---:|---:|---:|---:|\n")
                    for proxy_name in ["metadata"] + [m for m in latent_cfg["metrics"] if m in sample_level_oracle_benchmark_by_proxy] + ["oracle"]:
                        if proxy_name not in sample_level_oracle_benchmark_by_proxy:
                            continue
                        p = sample_level_oracle_benchmark_by_proxy[proxy_name]
                        f.write(
                            f"| {proxy_name} | {float(p['top1']):.4f} | {float(p['routed_nelbo_mean']):.6f} | {float(p['oracle_routed_nelbo_mean']):.6f} | {float(p['mean_oracle_gap']):.6f} | {float(p['mean_oracle_gap_pct']):.2f} |\n"
                        )

                if coverage_status is not None:
                    f.write("\n## Backbone Coverage\n\n")
                    f.write(
                        f"- backbone={coverage_status['backbone_type']}, status={coverage_status['status']}, "
                        f"all_loqdo_query_folds_present={coverage_status['all_loqdo_query_folds_present']}, "
                        f"all_expected_query_domains_present={coverage_status['all_expected_query_domains_present']}, "
                        f"has_partial_metric_rows={coverage_status['has_partial_metric_rows']}\n"
                    )

                eligible_count = sum(int(r.get("eligible", 0)) for r in learned_eligibility_rows)
                total_count = len(learned_eligibility_rows)
                f.write("\n## Learned Comparison Eligibility\n\n")
                f.write(
                    f"- strict_context_match={bool(latent_cfg['learned_comparison']['strict_context_match'])}, "
                    f"eligible_rows={eligible_count}/{total_count}\n"
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
                "learned_comparison_eligibility_artifact": "learned_comparison_eligibility.csv",
                "acceptance_decision_artifact": "acceptance_decision_summary.json",
                "step1_failure_gate_json_artifact": "step1_failure_gate_summary.json",
                "step1_failure_gate_csv_artifact": "step1_failure_gate_summary.csv",
                "sample_oracle_gap_per_sample_artifact": "sample_level_oracle_gap_per_sample.csv",
                "sample_oracle_gap_per_domain_artifact": "sample_level_oracle_gap_per_domain.csv",
                "sample_oracle_benchmark_summary_csv_artifact": "sample_level_oracle_benchmark_summary.csv",
                "sample_oracle_benchmark_summary_json_artifact": "sample_level_oracle_benchmark_summary.json",
                "sample_oracle_benchmark_per_sample_csv_artifact": "sample_level_oracle_benchmark_per_sample.csv",
                "sample_oracle_benchmark_per_domain_csv_artifact": "sample_level_oracle_benchmark_per_domain.csv",
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
