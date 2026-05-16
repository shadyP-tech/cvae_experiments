from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import spearman_corr
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load


@dataclass
class DomainGaussianStats:
    mean: np.ndarray
    covariance: np.ndarray
    n_samples: int
    used_diagonal_covariance: bool


def load_embeddings_with_domains(
    cache_paths: Dict[str, Path],
    splits: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    vectors: List[np.ndarray] = []
    domains: List[np.ndarray] = []
    metadata_all: List[Dict[str, Any]] = []

    for split in splits:
        if split not in cache_paths:
            raise ValueError(f"Unknown embedding split '{split}'. Available: {sorted(cache_paths)}")
        payload = safe_torch_load(cache_paths[split], map_location="cpu")
        x = payload["embeddings"].detach().cpu().numpy()
        m = payload["metadata"]
        d = np.array([int(item["magnification"]) for item in m], dtype=np.int64)

        vectors.append(x)
        domains.append(d)
        metadata_all.extend(m)

    if not vectors:
        raise ValueError("No embedding splits were selected for latent compatibility analysis.")

    return np.concatenate(vectors, axis=0), np.concatenate(domains, axis=0), metadata_all


def _covariance_matrix(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / float(max(x.shape[0] - 1, 1))


def compute_domain_gaussian_stats(
    embeddings: np.ndarray,
    domains: np.ndarray,
    covariance_regularization_lambda: float,
    min_samples_per_domain: int,
) -> Tuple[List[int], Dict[int, DomainGaussianStats], List[str]]:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array [N, D], got shape={embeddings.shape}")

    dim = int(embeddings.shape[1])
    unique_domains = sorted(set(int(d) for d in domains.tolist()))
    stats: Dict[int, DomainGaussianStats] = {}
    warnings: List[str] = []

    for domain in unique_domains:
        idxs = np.where(domains == domain)[0]
        x_d = embeddings[idxs].astype(np.float64, copy=False)
        n_samples = int(x_d.shape[0])
        if n_samples == 0:
            continue

        mu = x_d.mean(axis=0)
        use_diagonal = False

        if n_samples < int(min_samples_per_domain):
            warnings.append(
                f"Domain {domain} has only {n_samples} samples (< min_samples_per_domain={min_samples_per_domain})."
            )

        if n_samples < dim + 5:
            use_diagonal = True
            warnings.append(
                f"Domain {domain}: n={n_samples} < latent_dim+5={dim + 5}; using diagonal covariance fallback."
            )
            var = x_d.var(axis=0, ddof=0)
            cov = np.diag(var)
        else:
            cov = _covariance_matrix(x_d)

        cov = 0.5 * (cov + cov.T)
        cov = cov + (float(covariance_regularization_lambda) * np.eye(dim, dtype=np.float64))

        stats[domain] = DomainGaussianStats(
            mean=mu,
            covariance=cov,
            n_samples=n_samples,
            used_diagonal_covariance=use_diagonal,
        )

    if not stats:
        raise RuntimeError("No domains found for Gaussian summary computation.")

    return unique_domains, stats, warnings


def _matrix_sqrt_spd(mat: np.ndarray, eigenvalue_floor: float) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eigenvalue_floor))
    sqrt_vals = np.sqrt(vals)
    return (vecs * sqrt_vals) @ vecs.T


def _matrix_inv_spd(mat: np.ndarray, eigenvalue_floor: float) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eigenvalue_floor))
    inv_vals = 1.0 / vals
    return (vecs * inv_vals) @ vecs.T


def _logdet_spd(mat: np.ndarray, eigenvalue_floor: float) -> float:
    sym = 0.5 * (mat + mat.T)
    vals, _ = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eigenvalue_floor))
    return float(np.sum(np.log(vals)))


def _gaussian_kl(
    mu_p: np.ndarray,
    cov_p: np.ndarray,
    mu_q: np.ndarray,
    cov_q: np.ndarray,
    eigenvalue_floor: float,
) -> float:
    dim = mu_p.shape[0]
    inv_cov_q = _matrix_inv_spd(cov_q, eigenvalue_floor=eigenvalue_floor)
    delta = (mu_q - mu_p).reshape(-1, 1)
    trace_term = float(np.trace(inv_cov_q @ cov_p))
    quad_term = float((delta.T @ inv_cov_q @ delta).item())
    logdet_term = _logdet_spd(cov_q, eigenvalue_floor=eigenvalue_floor) - _logdet_spd(
        cov_p, eigenvalue_floor=eigenvalue_floor
    )
    kl = 0.5 * (trace_term + quad_term - dim + logdet_term)
    return float(max(kl, 0.0))


def compute_distance_matrices(
    domain_order: List[int],
    stats: Dict[int, DomainGaussianStats],
    eigenvalue_floor: float,
) -> Dict[str, np.ndarray]:
    n = len(domain_order)
    centroid = np.zeros((n, n), dtype=np.float64)
    wasserstein = np.zeros((n, n), dtype=np.float64)
    kl_sym = np.zeros((n, n), dtype=np.float64)

    for i, di in enumerate(domain_order):
        for j, dj in enumerate(domain_order):
            si = stats[di]
            sj = stats[dj]
            delta = si.mean - sj.mean
            centroid[i, j] = float(np.linalg.norm(delta, ord=2))

            sqrt_cov_i = _matrix_sqrt_spd(si.covariance, eigenvalue_floor=eigenvalue_floor)
            inner = sqrt_cov_i @ sj.covariance @ sqrt_cov_i
            sqrt_inner = _matrix_sqrt_spd(inner, eigenvalue_floor=eigenvalue_floor)
            w2_sq = float(
                np.dot(delta, delta)
                + np.trace(si.covariance + sj.covariance - (2.0 * sqrt_inner))
            )
            wasserstein[i, j] = math.sqrt(max(w2_sq, 0.0))

            kl_ij = _gaussian_kl(
                si.mean,
                si.covariance,
                sj.mean,
                sj.covariance,
                eigenvalue_floor=eigenvalue_floor,
            )
            kl_ji = _gaussian_kl(
                sj.mean,
                sj.covariance,
                si.mean,
                si.covariance,
                eigenvalue_floor=eigenvalue_floor,
            )
            kl_sym[i, j] = 0.5 * (kl_ij + kl_ji)

    # Ensure symmetric KL matrix after numerical drift.
    kl_sym = 0.5 * (kl_sym + kl_sym.T)

    return {
        "centroid": centroid,
        "wasserstein": wasserstein,
        "gaussian_kl": kl_sym,
    }


def distance_to_similarity(distance: np.ndarray, scale_floor: float) -> Tuple[np.ndarray, float]:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape={distance.shape}")

    n = distance.shape[0]
    offdiag = [float(distance[i, j]) for i in range(n) for j in range(n) if i != j]
    median_scale = float(np.median(offdiag)) if offdiag else 1.0
    scale = max(median_scale, float(scale_floor))

    sim = np.exp(-distance / scale)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64), scale


def verify_similarity_matrix(
    matrix: np.ndarray,
    atol: float,
    rtol: float,
    diag_opt_tol: float,
    symmetric_expected: bool,
) -> Dict[str, Any]:
    n = matrix.shape[0]
    finite_ok = bool(np.isfinite(matrix).all())

    symmetry_violations: List[Dict[str, Any]] = []
    if symmetric_expected:
        for i in range(n):
            for j in range(i + 1, n):
                a = float(matrix[i, j])
                b = float(matrix[j, i])
                tol = float(atol + rtol * max(abs(a), abs(b)))
                if abs(a - b) > tol:
                    symmetry_violations.append(
                        {
                            "i": i,
                            "j": j,
                            "a": a,
                            "b": b,
                            "allowed_abs_diff": tol,
                            "actual_abs_diff": abs(a - b),
                        }
                    )

    diagonal_violations: List[Dict[str, Any]] = []
    for i in range(n):
        diag = float(matrix[i, i])
        for j in range(n):
            if i == j:
                continue
            if diag < float(matrix[i, j]) - float(diag_opt_tol):
                diagonal_violations.append(
                    {
                        "i": i,
                        "j": j,
                        "diag": diag,
                        "offdiag": float(matrix[i, j]),
                        "diag_opt_tol": float(diag_opt_tol),
                    }
                )

    return {
        "finite_ok": finite_ok,
        "symmetric_expected": bool(symmetric_expected),
        "symmetry_ok": len(symmetry_violations) == 0,
        "symmetry_violations": symmetry_violations,
        "diagonal_optimality_ok": len(diagonal_violations) == 0,
        "diagonal_violations": diagonal_violations,
    }


def evaluate_routing_alignment(
    domain_order: List[int],
    similarity_matrix: np.ndarray,
    strategy: str,
    tau: float,
    similarity_lookup_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    n = len(domain_order)
    top1_hits = 0
    ranks: List[float] = []
    rows: List[Dict[str, Any]] = []

    for i, query_domain in enumerate(domain_order):
        latent_scores = [float(similarity_matrix[i, j]) for j in range(n)]
        latent_best_idx = int(np.argmax(np.array(latent_scores, dtype=np.float64)))

        metadata_scores = [
            compute_similarity(
                {"magnification": int(query_domain)},
                {"magnification": int(expert_domain)},
                strategy=str(strategy),
                tau=float(tau),
                similarity_matrix=similarity_lookup_matrix,
            )
            for expert_domain in domain_order
        ]
        metadata_best_idx = int(np.argmax(np.array(metadata_scores, dtype=np.float64)))

        # Deterministic ranking: highest score first, ties broken by smallest domain index.
        ordering = sorted(range(n), key=lambda idx: (-latent_scores[idx], idx))
        rank_by_idx = {idx: r + 1 for r, idx in enumerate(ordering)}
        metadata_rank = float(rank_by_idx[metadata_best_idx])

        hit = 1 if metadata_best_idx == latent_best_idx else 0
        top1_hits += hit
        ranks.append(metadata_rank)

        rows.append(
            {
                "query_domain": f"{query_domain}x",
                "metadata_selected_expert": f"{domain_order[metadata_best_idx]}x",
                "latent_best_expert": f"{domain_order[latent_best_idx]}x",
                "top1_match": bool(hit),
                "metadata_selected_rank_in_latent": metadata_rank,
            }
        )

    top1 = float(top1_hits / n) if n > 0 else 0.0
    mean_rank = float(sum(ranks) / len(ranks)) if ranks else 0.0

    return {
        "top1_agreement": top1,
        "mean_rank": mean_rank,
        "per_domain": rows,
    }


def _precompute_domain_cholesky_factors(
    domain_order: List[int],
    stats: Dict[int, DomainGaussianStats],
    eigenvalue_floor: float,
) -> Dict[int, np.ndarray]:
    factors: Dict[int, np.ndarray] = {}
    for domain in domain_order:
        cov = 0.5 * (stats[int(domain)].covariance + stats[int(domain)].covariance.T)
        cov = cov + (float(eigenvalue_floor) * np.eye(cov.shape[0], dtype=np.float64))
        try:
            factors[int(domain)] = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, float(eigenvalue_floor))
            cov_stable = (vecs * vals) @ vecs.T
            factors[int(domain)] = np.linalg.cholesky(cov_stable)
    return factors


def _domain_gaussian_sample_distance_from_cholesky(
    sample: np.ndarray,
    mean: np.ndarray,
    cholesky_factor: np.ndarray,
) -> float:
    # Mahalanobis distance via solve(L, delta), where Cov = L L^T.
    delta = sample - mean
    solved = np.linalg.solve(cholesky_factor, delta)
    md2 = float(np.dot(solved, solved))
    return math.sqrt(max(md2, 0.0))


def evaluate_sample_level_routing_alignment(
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    domain_order: List[int],
    stats: Dict[int, DomainGaussianStats],
    strategy: str,
    tau: float,
    eigenvalue_floor: float,
    similarity_lookup_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    max_samples: Optional[int] = None,
    timing_every: int = 0,
) -> Dict[str, Any]:
    n_domains = len(domain_order)
    top1_hits = 0
    ranks: List[float] = []
    rows: List[Dict[str, Any]] = []
    per_domain_counts: Dict[int, int] = {int(d): 0 for d in domain_order}
    per_domain_hits: Dict[int, int] = {int(d): 0 for d in domain_order}
    n_total_samples = int(embeddings.shape[0])
    n_evaluated_samples = n_total_samples if max_samples is None else min(int(max_samples), n_total_samples)

    cholesky_by_domain = _precompute_domain_cholesky_factors(
        domain_order=domain_order,
        stats=stats,
        eigenvalue_floor=float(eigenvalue_floor),
    )
    metadata_scores_by_query: Dict[int, List[float]] = {}
    for query_domain in set(int(d) for d in sample_domains[:n_evaluated_samples].tolist()):
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

    started = time.perf_counter()

    for idx in range(n_evaluated_samples):
        query_domain = int(sample_domains[idx])
        x = embeddings[idx].astype(np.float64, copy=False)

        latent_distances = [
            _domain_gaussian_sample_distance_from_cholesky(
                sample=x,
                mean=stats[int(expert_domain)].mean,
                cholesky_factor=cholesky_by_domain[int(expert_domain)],
            )
            for expert_domain in domain_order
        ]
        latent_scores = [float(math.exp(-d)) for d in latent_distances]
        latent_best_idx = int(np.argmax(np.asarray(latent_scores, dtype=np.float64)))

        metadata_scores = metadata_scores_by_query[query_domain]
        metadata_best_idx = int(np.argmax(np.asarray(metadata_scores, dtype=np.float64)))

        ordering = sorted(range(n_domains), key=lambda i: (-latent_scores[i], i))
        rank_by_idx = {i: r + 1 for r, i in enumerate(ordering)}
        metadata_rank = float(rank_by_idx[metadata_best_idx])

        hit = int(metadata_best_idx == latent_best_idx)
        top1_hits += hit
        ranks.append(metadata_rank)
        if query_domain in per_domain_counts:
            per_domain_counts[query_domain] += 1
            per_domain_hits[query_domain] += hit

        rows.append(
            {
                "sample_index": int(idx),
                "query_domain": f"{query_domain}x",
                "metadata_selected_expert": f"{domain_order[metadata_best_idx]}x",
                "latent_best_expert": f"{domain_order[latent_best_idx]}x",
                "top1_match": bool(hit),
                "metadata_selected_rank_in_latent": metadata_rank,
            }
        )

        if timing_every > 0 and (((idx + 1) % int(timing_every)) == 0 or (idx + 1) == n_evaluated_samples):
            elapsed = time.perf_counter() - started
            print(
                f"sample-level routing progress: {idx + 1}/{n_evaluated_samples} "
                f"samples in {elapsed:.2f}s ({elapsed / max(idx + 1, 1):.4f}s/sample)"
            )

    n_samples = int(n_evaluated_samples)
    total_elapsed = time.perf_counter() - started
    per_domain_summary = [
        {
            "query_domain": f"{int(d)}x",
            "n_samples": int(per_domain_counts[int(d)]),
            "top1_agreement": float(per_domain_hits[int(d)] / max(per_domain_counts[int(d)], 1)),
        }
        for d in domain_order
    ]
    return {
        "granularity": "sample",
        "n_total_samples": int(n_total_samples),
        "n_samples": n_samples,
        "top1_agreement": float(top1_hits / max(n_samples, 1)),
        "mean_rank": float(sum(ranks) / len(ranks)) if ranks else 0.0,
        "runtime_seconds": float(total_elapsed),
        "per_sample": rows,
        "per_domain_summary": per_domain_summary,
    }


def matrix_to_domain_dict(domain_order: List[int], matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, di in enumerate(domain_order):
        row = {}
        for j, dj in enumerate(domain_order):
            row[f"{dj}x"] = float(matrix[i, j])
        out[f"{di}x"] = row
    return out


def compute_metric_utility_correlation(
    similarity_matrix: np.ndarray,
    utility_matrix: np.ndarray,
) -> float:
    if similarity_matrix.shape != utility_matrix.shape:
        raise ValueError(
            f"Similarity and utility matrices must share the same shape, got {similarity_matrix.shape} vs {utility_matrix.shape}"
        )
    sim_vals = similarity_matrix.reshape(-1).tolist()
    util_vals = utility_matrix.reshape(-1).tolist()
    return float(spearman_corr(sim_vals, util_vals))


def _off_diagonal_points(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    domain_order: List[int],
) -> Tuple[List[float], List[float], List[int]]:
    if x_matrix.shape != y_matrix.shape:
        raise ValueError(
            f"Matrices must share shape, got {x_matrix.shape} vs {y_matrix.shape}"
        )
    n = x_matrix.shape[0]
    xs: List[float] = []
    ys: List[float] = []
    query_domains: List[int] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xs.append(float(x_matrix[i, j]))
            ys.append(float(y_matrix[i, j]))
            query_domains.append(int(domain_order[i]))
    return xs, ys, query_domains


def compute_distance_utility_correlation(
    distance_matrix: np.ndarray,
    utility_matrix: np.ndarray,
    off_diagonal_only: bool = True,
) -> float:
    if distance_matrix.shape != utility_matrix.shape:
        raise ValueError(
            f"Distance and utility matrices must share the same shape, got {distance_matrix.shape} vs {utility_matrix.shape}"
        )

    if off_diagonal_only:
        n = distance_matrix.shape[0]
        dist_vals = [float(distance_matrix[i, j]) for i in range(n) for j in range(n) if i != j]
        util_vals = [float(utility_matrix[i, j]) for i in range(n) for j in range(n) if i != j]
    else:
        dist_vals = distance_matrix.reshape(-1).tolist()
        util_vals = utility_matrix.reshape(-1).tolist()

    return float(spearman_corr(dist_vals, util_vals))


def validate_query_expert_matrix(
    matrix: np.ndarray,
    domain_order: List[int],
    matrix_name: str,
) -> None:
    expected_shape = (len(domain_order), len(domain_order))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{matrix_name} must have shape {expected_shape} with orientation query_domain x expert_domain; got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{matrix_name} contains non-finite values")


def build_metadata_score_matrix(
    domain_order: List[int],
    strategy: str,
    tau: float,
    similarity_lookup_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    n = len(domain_order)
    out = np.zeros((n, n), dtype=np.float64)
    for i, query_domain in enumerate(domain_order):
        for j, expert_domain in enumerate(domain_order):
            out[i, j] = float(
                compute_similarity(
                    {"magnification": int(query_domain)},
                    {"magnification": int(expert_domain)},
                    strategy=str(strategy),
                    tau=float(tau),
                    similarity_matrix=similarity_lookup_matrix,
                )
            )
    return out


def _rank_desc(scores: List[float]) -> Dict[int, int]:
    ordering = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
    return {idx: rank + 1 for rank, idx in enumerate(ordering)}


def compute_proxy_oracle_alignment(
    proxy_name: str,
    score_matrix: np.ndarray,
    oracle_utility_matrix: np.ndarray,
    domain_order: List[int],
) -> Dict[str, Any]:
    validate_query_expert_matrix(score_matrix, domain_order, f"{proxy_name}.score_matrix")
    validate_query_expert_matrix(oracle_utility_matrix, domain_order, "oracle_utility_matrix")

    rows: List[Dict[str, Any]] = []
    spearmans: List[float] = []
    top1_hits = 0
    gaps: List[float] = []
    rel_gaps_pct: List[float] = []

    for i, query_domain in enumerate(domain_order):
        score_row = [float(v) for v in score_matrix[i, :].tolist()]
        utility_row = [float(v) for v in oracle_utility_matrix[i, :].tolist()]

        selected_idx = int(np.argmax(np.asarray(score_row, dtype=np.float64)))
        oracle_best_idx = int(np.argmax(np.asarray(utility_row, dtype=np.float64)))
        top1_hit = int(selected_idx == oracle_best_idx)

        selected_utility = float(utility_row[selected_idx])
        oracle_best_utility = float(utility_row[oracle_best_idx])
        gap = float(oracle_best_utility - selected_utility)
        denom = max(abs(oracle_best_utility), 1e-12)
        rel_gap_pct = float((gap / denom) * 100.0)

        score_rank = _rank_desc(score_row)
        rank_of_oracle = int(score_rank[oracle_best_idx])
        rho = float(spearman_corr(score_row, utility_row))

        spearmans.append(rho)
        top1_hits += top1_hit
        gaps.append(gap)
        rel_gaps_pct.append(rel_gap_pct)

        rows.append(
            {
                "proxy": str(proxy_name),
                "query_domain": f"{query_domain}x",
                "selected_expert": f"{domain_order[selected_idx]}x",
                "oracle_best_expert": f"{domain_order[oracle_best_idx]}x",
                "spearman_with_oracle": rho,
                "top1_oracle_hit": float(top1_hit),
                "rank_of_oracle_best_in_proxy": float(rank_of_oracle),
                "selected_utility": selected_utility,
                "oracle_best_utility": oracle_best_utility,
                "oracle_gap": gap,
                "oracle_gap_pct": rel_gap_pct,
            }
        )

    n = len(domain_order)
    return {
        "proxy": str(proxy_name),
        "n_queries": n,
        "mean_spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "std_spearman": float(np.std(spearmans)) if spearmans else 0.0,
        "top1": float(top1_hits / n) if n > 0 else 0.0,
        "mean_oracle_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_oracle_gap_pct": float(np.mean(rel_gaps_pct)) if rel_gaps_pct else 0.0,
        "per_query": rows,
    }


def build_metric_payload(
    metric_name: str,
    raw_distance_matrix: np.ndarray,
    score_matrix: np.ndarray,
    scale: float,
    domain_order: List[int],
) -> Dict[str, Any]:
    validate_query_expert_matrix(raw_distance_matrix, domain_order, f"{metric_name}.raw_score_matrix")
    validate_query_expert_matrix(score_matrix, domain_order, f"{metric_name}.score_matrix")
    return {
        "metric": str(metric_name),
        "raw_score_matrix": matrix_to_domain_dict(domain_order, raw_distance_matrix),
        "score_matrix": matrix_to_domain_dict(domain_order, score_matrix),
        "score_direction": "higher_is_better",
        "transform": {
            "name": "exp_neg",
            "parameters": {
                "scale": float(scale),
            },
        },
        "provenance": {
            "raw_represents": "distance",
            "score_represents": "compatibility",
            "orientation": "query_domain_x_expert_domain",
        },
    }


def maybe_project_latent_2d(
    embeddings: np.ndarray,
    seed: int,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n = embeddings.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int64), {"method": "none"}

    rng = np.random.default_rng(int(seed))
    if n > int(max_points):
        idxs = np.sort(rng.choice(n, size=int(max_points), replace=False))
    else:
        idxs = np.arange(n)

    x = embeddings[idxs].astype(np.float64, copy=False)

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            random_state=int(seed),
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
        )
        coords = reducer.fit_transform(x)
        info = {
            "method": "umap",
            "random_state": int(seed),
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_points": int(x.shape[0]),
        }
        return coords.astype(np.float64), idxs.astype(np.int64), info
    except Exception:
        x_centered = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        basis = vt[:2].T
        coords = x_centered @ basis
        info = {
            "method": "pca_fallback",
            "random_state": int(seed),
            "n_points": int(x.shape[0]),
        }
        return coords.astype(np.float64), idxs.astype(np.int64), info


def plot_matrix_heatmap(
    matrix: np.ndarray,
    domain_order: List[int],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{d}x" for d in domain_order]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Expert domain")
    plt.ylabel("Query domain")
    plt.title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_latent_map(
    coords: np.ndarray,
    sample_domains: np.ndarray,
    domain_order: List[int],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for domain in domain_order:
        idxs = np.where(sample_domains == domain)[0]
        if idxs.size == 0:
            continue
        pts = coords[idxs]
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, label=f"{domain}x")

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distance_vs_utility(
    distance_matrix: np.ndarray,
    utility_matrix: np.ndarray,
    domain_order: List[int],
    out_path: Path,
    title: str,
    add_regression: bool = True,
    color_by_query: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dists, utils, query_domains = _off_diagonal_points(distance_matrix, utility_matrix, domain_order)
    corr = compute_distance_utility_correlation(
        distance_matrix=distance_matrix,
        utility_matrix=utility_matrix,
        off_diagonal_only=True,
    )

    plt.figure(figsize=(7, 5.5))
    if color_by_query:
        for domain in domain_order:
            idxs = [k for k, q in enumerate(query_domains) if q == domain]
            if not idxs:
                continue
            x_vals = [dists[k] for k in idxs]
            y_vals = [utils[k] for k in idxs]
            plt.scatter(x_vals, y_vals, alpha=0.7, s=30, label=f"{domain}x")
        plt.legend(loc="best", markerscale=1.5)
    else:
        plt.scatter(dists, utils, alpha=0.7, s=30)

    if add_regression and len(dists) >= 2:
        try:
            slope, intercept = np.polyfit(np.asarray(dists, dtype=np.float64), np.asarray(utils, dtype=np.float64), 1)
            x_line = np.linspace(min(dists), max(dists), num=100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, "k--", linewidth=1.5, alpha=0.8)
        except np.linalg.LinAlgError:
            pass

    plt.xlabel("Latent domain distance")
    plt.ylabel("Expert utility (-NELBO)")
    plt.title(f"{title} (Spearman={corr:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_composite_figure(
    coords: np.ndarray,
    sample_domains: np.ndarray,
    domain_order: List[int],
    compatibility_matrix: np.ndarray,
    utility_matrix: Optional[np.ndarray],
    distance_matrix: Optional[np.ndarray],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{d}x" for d in domain_order]

    n_cols = 4 if (utility_matrix is not None and distance_matrix is not None) else (3 if utility_matrix is not None else 2)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))

    ax0 = axes[0]
    for domain in domain_order:
        idxs = np.where(sample_domains == domain)[0]
        if idxs.size == 0:
            continue
        pts = coords[idxs]
        ax0.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, label=f"{domain}x")
    ax0.set_title("A. Latent 2D Map")
    ax0.set_xlabel("Component 1")
    ax0.set_ylabel("Component 2")
    ax0.legend(loc="best", markerscale=2)

    ax1 = axes[1]
    im1 = ax1.imshow(compatibility_matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(range(len(labels)), labels)
    ax1.set_yticks(range(len(labels)), labels)
    ax1.set_title("B. Compatibility Heatmap")
    ax1.set_xlabel("Expert domain")
    ax1.set_ylabel("Query domain")

    if utility_matrix is not None:
        ax2 = axes[2]
        im2 = ax2.imshow(utility_matrix, cmap="magma", aspect="auto")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_xticks(range(len(labels)), labels)
        ax2.set_yticks(range(len(labels)), labels)
        ax2.set_title("C. Expert Utility Heatmap")
        ax2.set_xlabel("Expert domain")
        ax2.set_ylabel("Query domain")

    if utility_matrix is not None and distance_matrix is not None:
        dists, utils, query_domains = _off_diagonal_points(distance_matrix, utility_matrix, domain_order)
        corr = compute_distance_utility_correlation(
            distance_matrix=distance_matrix,
            utility_matrix=utility_matrix,
            off_diagonal_only=True,
        )
        ax3 = axes[3]
        for domain in domain_order:
            idxs = [k for k, q in enumerate(query_domains) if q == domain]
            if not idxs:
                continue
            x_vals = [dists[k] for k in idxs]
            y_vals = [utils[k] for k in idxs]
            ax3.scatter(x_vals, y_vals, alpha=0.7, s=18, label=f"{domain}x")

        if len(dists) >= 2:
            try:
                slope, intercept = np.polyfit(np.asarray(dists, dtype=np.float64), np.asarray(utils, dtype=np.float64), 1)
                x_line = np.linspace(min(dists), max(dists), num=100)
                y_line = slope * x_line + intercept
                ax3.plot(x_line, y_line, "k--", linewidth=1.2, alpha=0.8)
            except np.linalg.LinAlgError:
                pass

        ax3.set_title(f"D. Distance vs Utility (Spearman={corr:.3f})")
        ax3.set_xlabel("Latent distance")
        ax3.set_ylabel("Utility (-NELBO)")
        ax3.grid(alpha=0.3)
        ax3.legend(loc="best", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
