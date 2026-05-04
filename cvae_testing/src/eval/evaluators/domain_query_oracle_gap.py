from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from src.eval.evaluators.support_set_calibration import (
    SupportSetRunMeta,
    make_expert_bank,
    read_embedding_cache,
    score_expert_nelbo_matrix,
)
from src.routing.strategies import ordinal_magnification_similarity


EPS = 1e-12


DEFAULT_INTERPRETATION_THRESHOLDS: Dict[str, float] = {
    "low_normalized_gap": 0.10,
    "high_normalized_gap": 0.25,
    "high_normalized_entropy": 0.67,
    "high_switch_rate": 0.50,
    "high_margin_low_margin_share": 0.25,
    "metadata_close_to_fixed_oracle": 0.10,
}


def _as_domain(value: object) -> int:
    return int(str(value).strip().lower().replace("x", ""))


def _stable_argmin(values: Sequence[float], expert_domains: Sequence[int]) -> int:
    return int(min(range(len(values)), key=lambda i: (float(values[i]), int(expert_domains[i]))))


def _stable_argmax(values: Sequence[float], expert_domains: Sequence[int]) -> int:
    return int(max(range(len(values)), key=lambda i: (float(values[i]), -int(expert_domains[i]))))


def _stable_argmax_nelbo(values: Sequence[float], expert_domains: Sequence[int]) -> int:
    return int(max(range(len(values)), key=lambda i: (float(values[i]), -int(expert_domains[i]))))


def _json_float_mapping(expert_domains: Sequence[int], values: Sequence[float]) -> str:
    return json.dumps(
        {str(int(e)): float(v) for e, v in zip(expert_domains, values)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_int_mapping(expert_domains: Sequence[int], values: Sequence[int]) -> str:
    return json.dumps(
        {str(int(e)): int(v) for e, v in zip(expert_domains, values)},
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_gap(selected_nelbo: float, oracle_nelbo: float, worst_nelbo: float, eps: float) -> float:
    denom = max(float(worst_nelbo) - float(oracle_nelbo), float(eps))
    return float(max(float(selected_nelbo) - float(oracle_nelbo), 0.0) / denom)


def _rank_lower_is_better(values: Sequence[float], selected_idx: int, expert_domains: Sequence[int]) -> int:
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), int(expert_domains[i])))
    for rank, idx in enumerate(order, start=1):
        if int(idx) == int(selected_idx):
            return int(rank)
    return int(len(values))


def _per_sample_best_indices(target_scores: np.ndarray, candidate_experts: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [_stable_argmin(row.tolist(), candidate_experts) for row in target_scores],
        dtype=np.int64,
    )


def _per_sample_margins(target_scores: np.ndarray, candidate_experts: Sequence[int]) -> np.ndarray:
    margins: List[float] = []
    for row in target_scores:
        if row.size < 2:
            margins.append(0.0)
            continue
        order = sorted(range(row.size), key=lambda i: (float(row[i]), int(candidate_experts[i])))
        margins.append(float(row[order[1]] - row[order[0]]))
    return np.asarray(margins, dtype=np.float64)


def _low_margin_share(
    margins: np.ndarray,
    target_scores: np.ndarray,
    *,
    low_margin_abs_threshold: float,
    low_margin_rel_threshold: float,
) -> float:
    if margins.size == 0:
        return 0.0
    ranges = np.max(target_scores, axis=1) - np.min(target_scores, axis=1)
    thresholds = np.maximum(float(low_margin_abs_threshold), float(low_margin_rel_threshold) * ranges)
    return float(np.mean(margins <= thresholds))


def _selection_distribution(
    best_indices: np.ndarray,
    candidate_experts: Sequence[int],
) -> Tuple[List[int], float, float, float, float]:
    counts = [int(np.sum(best_indices == i)) for i in range(len(candidate_experts))]
    total = int(sum(counts))
    if total <= 0:
        return counts, 0.0, 0.0, 0.0, 0.0
    probs = np.asarray([c / float(total) for c in counts if c > 0], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(probs))) if probs.size else 0.0
    entropy_norm = float(entropy / math.log(len(candidate_experts))) if len(candidate_experts) > 1 else 0.0
    modal_share = float(max(counts) / float(total))
    switch_rate = float(1.0 - modal_share)
    return counts, entropy, entropy_norm, modal_share, switch_rate


def _domain_centroids(embeddings: np.ndarray, sample_domains: np.ndarray) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for domain in sorted(set(int(v) for v in sample_domains.tolist())):
        idxs = np.where(sample_domains == int(domain))[0]
        if idxs.size:
            centroids[int(domain)] = embeddings[idxs].mean(axis=0)
    return centroids


def _baseline_gap_fields(
    *,
    prefix: str,
    selected_idx: int,
    candidate_experts: Sequence[int],
    candidate_mean: np.ndarray,
    fixed_domain_oracle_nelbo: float,
    per_query_oracle_nelbo: float,
    worst_fixed_expert_nelbo: float,
    eps: float,
    adoption_eligible: int,
    diagnostic_only: int,
) -> Dict[str, object]:
    selected_nelbo = float(candidate_mean[int(selected_idx)])
    gap_to_per_query = float(selected_nelbo - float(per_query_oracle_nelbo))
    gap_to_fixed = float(selected_nelbo - float(fixed_domain_oracle_nelbo))
    fixed_spread = max(float(worst_fixed_expert_nelbo) - float(per_query_oracle_nelbo), float(eps))
    return {
        f"{prefix}_expert": int(candidate_experts[int(selected_idx)]),
        f"{prefix}_nelbo": selected_nelbo,
        f"{prefix}_gap_to_per_query_oracle": gap_to_per_query,
        f"{prefix}_normalized_gap": _normalized_gap(
            selected_nelbo,
            float(per_query_oracle_nelbo),
            float(worst_fixed_expert_nelbo),
            eps,
        ),
        f"{prefix}_gap_to_fixed_oracle": gap_to_fixed,
        f"{prefix}_normalized_gap_to_fixed_oracle": float(max(gap_to_fixed, 0.0) / fixed_spread),
        f"{prefix}_adoption_eligible": int(adoption_eligible),
        f"{prefix}_diagnostic_only": int(diagnostic_only),
    }


def _point_metrics(
    *,
    target_scores: np.ndarray,
    candidate_experts: Sequence[int],
    eps: float,
    low_margin_abs_threshold: float,
    low_margin_rel_threshold: float,
) -> Dict[str, Any]:
    candidate_mean = np.mean(target_scores, axis=0)
    fixed_idx = _stable_argmin(candidate_mean.tolist(), candidate_experts)
    worst_idx = _stable_argmax_nelbo(candidate_mean.tolist(), candidate_experts)
    best_indices = _per_sample_best_indices(target_scores, candidate_experts)
    per_query_best = target_scores[np.arange(target_scores.shape[0]), best_indices]
    margins = _per_sample_margins(target_scores, candidate_experts)
    fixed_ranks = np.asarray(
        [_rank_lower_is_better(row.tolist(), fixed_idx, candidate_experts) for row in target_scores],
        dtype=np.float64,
    )
    counts, entropy, entropy_norm, modal_share, switch_rate = _selection_distribution(best_indices, candidate_experts)

    fixed_nelbo = float(candidate_mean[int(fixed_idx)])
    per_query_nelbo = float(np.mean(per_query_best)) if per_query_best.size else 0.0
    worst_nelbo = float(candidate_mean[int(worst_idx)])
    gap = float(fixed_nelbo - per_query_nelbo)

    return {
        "candidate_mean": candidate_mean,
        "fixed_idx": int(fixed_idx),
        "worst_idx": int(worst_idx),
        "best_indices": best_indices,
        "per_query_best": per_query_best,
        "margins": margins,
        "fixed_ranks": fixed_ranks,
        "counts": counts,
        "fixed_domain_oracle_nelbo": fixed_nelbo,
        "per_query_oracle_nelbo": per_query_nelbo,
        "worst_fixed_expert_nelbo": worst_nelbo,
        "fixed_to_query_oracle_gap": gap,
        "normalized_fixed_to_query_oracle_gap": _normalized_gap(fixed_nelbo, per_query_nelbo, worst_nelbo, eps),
        "per_query_expert_entropy": entropy,
        "per_query_expert_entropy_normalized": entropy_norm,
        "per_query_oracle_modal_share": modal_share,
        "per_query_oracle_switch_rate": switch_rate,
        "per_query_oracle_margin_mean": float(np.mean(margins)) if margins.size else 0.0,
        "per_query_oracle_margin_std": float(np.std(margins)) if margins.size else 0.0,
        "per_query_oracle_margin_median": float(np.median(margins)) if margins.size else 0.0,
        "low_margin_share": _low_margin_share(
            margins,
            target_scores,
            low_margin_abs_threshold=low_margin_abs_threshold,
            low_margin_rel_threshold=low_margin_rel_threshold,
        ),
        "fixed_domain_oracle_sample_rank_mean": float(np.mean(fixed_ranks)) if fixed_ranks.size else 0.0,
        "fixed_domain_oracle_sample_rank_std": float(np.std(fixed_ranks)) if fixed_ranks.size else 0.0,
    }


def _bootstrap_fold_cis(
    *,
    target_scores: np.ndarray,
    candidate_experts: Sequence[int],
    eps: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
    low_margin_abs_threshold: float,
    low_margin_rel_threshold: float,
) -> Dict[str, float]:
    point = _point_metrics(
        target_scores=target_scores,
        candidate_experts=candidate_experts,
        eps=eps,
        low_margin_abs_threshold=low_margin_abs_threshold,
        low_margin_rel_threshold=low_margin_rel_threshold,
    )
    ci_metrics = [
        "fixed_to_query_oracle_gap",
        "normalized_fixed_to_query_oracle_gap",
        "per_query_oracle_switch_rate",
    ]
    if int(bootstrap_reps) <= 0 or target_scores.shape[0] == 0:
        return {
            f"{metric}_ci_{suffix}": float(point[metric])
            for metric in ci_metrics
            for suffix in ("low", "high")
        }

    rng = np.random.default_rng(int(bootstrap_seed))
    samples: Dict[str, List[float]] = {metric: [] for metric in ci_metrics}
    for _ in range(int(bootstrap_reps)):
        idxs = rng.integers(0, target_scores.shape[0], size=target_scores.shape[0])
        boot = _point_metrics(
            target_scores=target_scores[idxs, :],
            candidate_experts=candidate_experts,
            eps=eps,
            low_margin_abs_threshold=low_margin_abs_threshold,
            low_margin_rel_threshold=low_margin_rel_threshold,
        )
        for metric in ci_metrics:
            samples[metric].append(float(boot[metric]))

    out: Dict[str, float] = {}
    for metric, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{metric}_ci_low"] = float(np.percentile(arr, 2.5))
        out[f"{metric}_ci_high"] = float(np.percentile(arr, 97.5))
    return out


def evaluate_domain_query_oracle_gap_from_arrays(
    *,
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
    nelbo_matrix: np.ndarray,
    expert_domains: Sequence[int],
    run_meta: SupportSetRunMeta,
    eps: float = EPS,
    bootstrap_reps: int = 1000,
    bootstrap_seed: int = 1337,
    low_margin_abs_threshold: float = 1e-8,
    low_margin_rel_threshold: float = 0.05,
    metadata_tau: float = 100.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(metadata) != int(embeddings.shape[0]):
        raise ValueError("Embedding and metadata lengths do not match")
    if nelbo_matrix.shape != (int(embeddings.shape[0]), len(expert_domains)):
        raise ValueError("NELBO matrix shape must be n_samples x n_experts")

    expert_domains_int = [int(d) for d in expert_domains]
    sample_domains = np.asarray([_as_domain(m["magnification"]) for m in metadata], dtype=np.int64)
    embedding_arr = np.asarray(embeddings, dtype=np.float64)
    centroids = _domain_centroids(embedding_arr, sample_domains)

    fold_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    for target_domain in sorted(set(int(v) for v in sample_domains.tolist())):
        target_indices = [int(i) for i, d in enumerate(sample_domains.tolist()) if int(d) == int(target_domain)]
        candidate_col_idxs = [i for i, e in enumerate(expert_domains_int) if int(e) != int(target_domain)]
        candidate_experts = [expert_domains_int[i] for i in candidate_col_idxs]
        if not target_indices or not candidate_experts:
            continue

        target_scores = nelbo_matrix[np.asarray(target_indices, dtype=np.int64)[:, None], candidate_col_idxs]
        point = _point_metrics(
            target_scores=target_scores,
            candidate_experts=candidate_experts,
            eps=float(eps),
            low_margin_abs_threshold=float(low_margin_abs_threshold),
            low_margin_rel_threshold=float(low_margin_rel_threshold),
        )
        fold_seed = int(bootstrap_seed) + int(run_meta.seed) * 1009 + int(target_domain) * 9173
        ci = _bootstrap_fold_cis(
            target_scores=target_scores,
            candidate_experts=candidate_experts,
            eps=float(eps),
            bootstrap_reps=int(bootstrap_reps),
            bootstrap_seed=fold_seed,
            low_margin_abs_threshold=float(low_margin_abs_threshold),
            low_margin_rel_threshold=float(low_margin_rel_threshold),
        )

        fixed_idx = int(point["fixed_idx"])
        worst_idx = int(point["worst_idx"])
        candidate_mean = np.asarray(point["candidate_mean"], dtype=np.float64)
        per_query_nelbo = float(point["per_query_oracle_nelbo"])
        worst_nelbo = float(point["worst_fixed_expert_nelbo"])
        fixed_nelbo = float(point["fixed_domain_oracle_nelbo"])
        fixed_spread = max(worst_nelbo - per_query_nelbo, float(eps))
        fold_id = f"{run_meta.run_id}:{int(target_domain)}"

        metadata_scores = [
            ordinal_magnification_similarity(int(target_domain), int(e), float(metadata_tau))
            for e in candidate_experts
        ]
        metadata_idx = _stable_argmax(metadata_scores, candidate_experts)

        target_centroid = centroids.get(int(target_domain))
        embedding_scores: List[float] = []
        for expert_domain in candidate_experts:
            expert_centroid = centroids.get(int(expert_domain))
            if target_centroid is None or expert_centroid is None:
                embedding_scores.append(float("-inf"))
            else:
                embedding_scores.append(-float(np.linalg.norm(target_centroid - expert_centroid, ord=2)))
        target_centroid_idx = _stable_argmax(embedding_scores, candidate_experts)

        row: Dict[str, Any] = {
            "dataset_name": run_meta.dataset_name,
            "seed": int(run_meta.seed),
            "backbone_type": run_meta.backbone_type,
            "run_id": run_meta.run_id,
            "variant": run_meta.variant,
            "run_dir": run_meta.run_dir,
            "fold_id": fold_id,
            "target_domain": int(target_domain),
            "candidate_experts": "|".join(str(int(e)) for e in candidate_experts),
            "excluded_experts": str(int(target_domain)),
            "target_expert_excluded": 1,
            "n_target_samples": int(len(target_indices)),
            "n_candidate_experts": int(len(candidate_experts)),
            "fixed_domain_oracle_expert": int(candidate_experts[fixed_idx]),
            "fixed_domain_oracle_nelbo": fixed_nelbo,
            "per_query_oracle_nelbo": per_query_nelbo,
            "worst_fixed_expert": int(candidate_experts[worst_idx]),
            "worst_fixed_expert_nelbo": worst_nelbo,
            "fixed_to_query_oracle_gap": float(point["fixed_to_query_oracle_gap"]),
            "normalized_fixed_to_query_oracle_gap": float(point["normalized_fixed_to_query_oracle_gap"]),
            "per_query_expert_entropy": float(point["per_query_expert_entropy"]),
            "per_query_expert_entropy_normalized": float(point["per_query_expert_entropy_normalized"]),
            "per_query_oracle_switch_rate": float(point["per_query_oracle_switch_rate"]),
            "per_query_oracle_modal_share": float(point["per_query_oracle_modal_share"]),
            "per_query_selected_expert_counts_json": _json_int_mapping(candidate_experts, point["counts"]),
            "per_query_oracle_margin_mean": float(point["per_query_oracle_margin_mean"]),
            "per_query_oracle_margin_std": float(point["per_query_oracle_margin_std"]),
            "per_query_oracle_margin_median": float(point["per_query_oracle_margin_median"]),
            "low_margin_share": float(point["low_margin_share"]),
            "fixed_domain_oracle_sample_rank_mean": float(point["fixed_domain_oracle_sample_rank_mean"]),
            "fixed_domain_oracle_sample_rank_std": float(point["fixed_domain_oracle_sample_rank_std"]),
            "fixed_to_query_gap_invariant_ok": int(float(point["fixed_to_query_oracle_gap"]) >= -1e-8),
            "baseline_candidate_policy": "exclude_target_domain",
            "oracle_definition": "fixed_domain_vs_per_query_min_nelbo",
            "bootstrap_reps": int(bootstrap_reps),
            "bootstrap_seed": int(fold_seed),
            "low_margin_abs_threshold": float(low_margin_abs_threshold),
            "low_margin_rel_threshold": float(low_margin_rel_threshold),
            **ci,
        }
        row.update(
            _baseline_gap_fields(
                prefix="metadata_ordinal_excluded",
                selected_idx=metadata_idx,
                candidate_experts=candidate_experts,
                candidate_mean=candidate_mean,
                fixed_domain_oracle_nelbo=fixed_nelbo,
                per_query_oracle_nelbo=per_query_nelbo,
                worst_fixed_expert_nelbo=worst_nelbo,
                eps=float(eps),
                adoption_eligible=1,
                diagnostic_only=0,
            )
        )
        row.update(
            _baseline_gap_fields(
                prefix="target_centroid_embedding_excluded",
                selected_idx=target_centroid_idx,
                candidate_experts=candidate_experts,
                candidate_mean=candidate_mean,
                fixed_domain_oracle_nelbo=fixed_nelbo,
                per_query_oracle_nelbo=per_query_nelbo,
                worst_fixed_expert_nelbo=worst_nelbo,
                eps=float(eps),
                adoption_eligible=0,
                diagnostic_only=1,
            )
        )
        fold_rows.append(row)

        best_indices = np.asarray(point["best_indices"], dtype=np.int64)
        margins = np.asarray(point["margins"], dtype=np.float64)
        fixed_ranks = np.asarray(point["fixed_ranks"], dtype=np.float64)
        for local_idx, sample_idx in enumerate(target_indices):
            score_row = target_scores[int(local_idx), :]
            best_idx = int(best_indices[int(local_idx)])
            fixed_sample_nelbo = float(score_row[fixed_idx])
            per_query_sample_nelbo = float(score_row[best_idx])
            meta = metadata[int(sample_idx)]
            sample_row: Dict[str, Any] = {
                "dataset_name": run_meta.dataset_name,
                "seed": int(run_meta.seed),
                "backbone_type": run_meta.backbone_type,
                "run_id": run_meta.run_id,
                "variant": run_meta.variant,
                "run_dir": run_meta.run_dir,
                "fold_id": fold_id,
                "target_domain": int(target_domain),
                "sample_index": int(sample_idx),
                "fixed_domain_oracle_expert": int(candidate_experts[fixed_idx]),
                "fixed_domain_oracle_sample_nelbo": fixed_sample_nelbo,
                "fixed_domain_oracle_sample_rank": int(fixed_ranks[int(local_idx)]),
                "per_query_oracle_expert": int(candidate_experts[best_idx]),
                "per_query_oracle_nelbo": per_query_sample_nelbo,
                "per_query_oracle_margin": float(margins[int(local_idx)]),
                "fixed_to_query_sample_gap": float(fixed_sample_nelbo - per_query_sample_nelbo),
                "nelbo_by_expert_json": _json_float_mapping(candidate_experts, score_row.tolist()),
            }
            if "sample_id" in meta:
                sample_row["sample_id"] = str(meta["sample_id"])
            if "image_path" in meta:
                sample_row["image_path"] = str(meta["image_path"])
            sample_rows.append(sample_row)

    return fold_rows, sample_rows


def evaluate_domain_query_oracle_gap_for_run(
    *,
    test_cache: Path,
    variant_checkpoint: Path | None,
    expert_manifest: Path | None,
    hidden_dim: int,
    latent_dim: int,
    metadata_constraint_cfg: Mapping[str, object] | None,
    run_meta: SupportSetRunMeta,
    batch_size: int = 2048,
    eps: float = EPS,
    bootstrap_reps: int = 1000,
    bootstrap_seed: int = 1337,
    low_margin_abs_threshold: float = 1e-8,
    low_margin_rel_threshold: float = 0.05,
    metadata_tau: float = 100.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    x_cpu, metadata = read_embedding_cache(Path(test_cache))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bank = make_expert_bank(
        variant_checkpoint=variant_checkpoint,
        expert_manifest=expert_manifest,
        input_dim=int(x_cpu.shape[1]),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        metadata_constraint_cfg=metadata_constraint_cfg or {},
        device=device,
    )
    expert_domains = sorted(int(d) for d in bank.domains)
    nelbo = score_expert_nelbo_matrix(
        bank=bank,
        embeddings=x_cpu,
        expert_domains=expert_domains,
        batch_size=int(batch_size),
        device=device,
    )
    return evaluate_domain_query_oracle_gap_from_arrays(
        embeddings=x_cpu.detach().cpu().numpy().astype(np.float64, copy=False),
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=run_meta,
        eps=float(eps),
        bootstrap_reps=int(bootstrap_reps),
        bootstrap_seed=int(bootstrap_seed),
        low_margin_abs_threshold=float(low_margin_abs_threshold),
        low_margin_rel_threshold=float(low_margin_rel_threshold),
        metadata_tau=float(metadata_tau),
    )


def _to_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _bootstrap_mean_ci(values: np.ndarray, *, bootstrap_reps: int, bootstrap_seed: int) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if int(bootstrap_reps) <= 0:
        mu = float(np.mean(values))
        return mu, mu
    rng = np.random.default_rng(int(bootstrap_seed))
    means = []
    for _ in range(int(bootstrap_reps)):
        idxs = rng.integers(0, values.size, size=values.size)
        means.append(float(np.mean(values[idxs])))
    arr = np.asarray(means, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _interpretation_for_row(row: Mapping[str, object], thresholds: Mapping[str, float]) -> Tuple[str, str]:
    norm_gap = _to_float(row, "normalized_fixed_to_query_oracle_gap_mean")
    entropy = _to_float(row, "per_query_expert_entropy_normalized_mean")
    switch = _to_float(row, "per_query_oracle_switch_rate_mean")
    low_margin = _to_float(row, "low_margin_share_mean")
    metadata_gap_to_fixed = _to_float(row, "metadata_ordinal_excluded_normalized_gap_to_fixed_oracle_mean", 1.0)

    low_gap = norm_gap <= float(thresholds["low_normalized_gap"])
    high_gap = norm_gap >= float(thresholds["high_normalized_gap"])
    high_entropy = entropy >= float(thresholds["high_normalized_entropy"])
    high_switch = switch >= float(thresholds["high_switch_rate"])
    high_margin = low_margin <= float(thresholds["high_margin_low_margin_share"])
    metadata_close = metadata_gap_to_fixed <= float(thresholds["metadata_close_to_fixed_oracle"])

    if low_gap and not high_entropy and not high_switch:
        return (
            "low_gap_low_entropy_low_switch",
            "One fixed expert is effectively optimal per target domain.",
        )
    if low_gap and high_switch:
        return (
            "low_gap_high_switch",
            "Expert choices vary, but utility differences are small.",
        )
    if high_gap and metadata_close:
        return (
            "high_gap_metadata_close_to_fixed_oracle",
            "Metadata captures most fixed-expert signal; learned methods must beat metadata stably.",
        )
    if high_gap and high_entropy and high_switch and high_margin:
        return (
            "high_gap_high_entropy_high_margin",
            "Strong within-domain heterogeneity; per-query compatibility or aggregation is justified.",
        )
    if high_gap and not high_entropy:
        return (
            "high_gap_low_entropy",
            "Variation is concentrated; inspect class, subtype, patient, or sample-quality effects.",
        )
    return (
        "intermediate_or_mixed",
        "Pattern is mixed; inspect per-domain fold rows and margins before changing routing granularity.",
    )


def aggregate_domain_query_oracle_gap_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_reps: int = 1000,
    bootstrap_seed: int = 1337,
    interpretation_thresholds: Mapping[str, float] | None = None,
) -> List[Dict[str, Any]]:
    thresholds = dict(DEFAULT_INTERPRETATION_THRESHOLDS)
    thresholds.update(dict(interpretation_thresholds or {}))
    metrics = [
        "fixed_domain_oracle_nelbo",
        "per_query_oracle_nelbo",
        "worst_fixed_expert_nelbo",
        "fixed_to_query_oracle_gap",
        "normalized_fixed_to_query_oracle_gap",
        "per_query_expert_entropy",
        "per_query_expert_entropy_normalized",
        "per_query_oracle_switch_rate",
        "per_query_oracle_modal_share",
        "per_query_oracle_margin_mean",
        "per_query_oracle_margin_std",
        "per_query_oracle_margin_median",
        "low_margin_share",
        "fixed_domain_oracle_sample_rank_mean",
        "fixed_domain_oracle_sample_rank_std",
        "metadata_ordinal_excluded_normalized_gap",
        "metadata_ordinal_excluded_normalized_gap_to_fixed_oracle",
        "target_centroid_embedding_excluded_normalized_gap",
        "target_centroid_embedding_excluded_normalized_gap_to_fixed_oracle",
    ]
    ci_metrics = [
        "fixed_to_query_oracle_gap",
        "normalized_fixed_to_query_oracle_gap",
        "per_query_expert_entropy_normalized",
        "per_query_oracle_switch_rate",
        "per_query_oracle_margin_mean",
        "low_margin_share",
        "fixed_domain_oracle_sample_rank_mean",
    ]

    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row.get("dataset_name", "")), str(row.get("backbone_type", "")), str(row.get("variant", "")))
        groups.setdefault(key, []).append(row)
        all_key = (str(row.get("dataset_name", "")), "all", str(row.get("variant", "")))
        groups.setdefault(all_key, []).append(row)

    out: List[Dict[str, Any]] = []
    for group_idx, (key, vals) in enumerate(sorted(groups.items())):
        dataset_name, backbone_type, variant = key
        row_out: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "backbone_type": backbone_type,
            "variant": variant,
            "n_folds": int(len(vals)),
            "n_runs": int(len(set(str(v.get("run_id", "")) for v in vals))),
            "n_target_samples": int(sum(int(float(v.get("n_target_samples", 0) or 0)) for v in vals)),
            "thresholds_are_descriptive_heuristics": 1,
        }
        for metric in metrics:
            arr = np.asarray([_to_float(v, metric) for v in vals], dtype=np.float64)
            row_out[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else 0.0
            row_out[f"{metric}_std"] = float(np.std(arr)) if arr.size else 0.0
        for metric in ci_metrics:
            arr = np.asarray([_to_float(v, metric) for v in vals], dtype=np.float64)
            lo, hi = _bootstrap_mean_ci(
                arr,
                bootstrap_reps=int(bootstrap_reps),
                bootstrap_seed=int(bootstrap_seed) + group_idx * 7919,
            )
            row_out[f"{metric}_ci_low"] = lo
            row_out[f"{metric}_ci_high"] = hi

        label, consequence = _interpretation_for_row(row_out, thresholds)
        row_out["interpretation_pattern"] = label
        row_out["interpretation_consequence"] = consequence
        out.append(row_out)

    return out
