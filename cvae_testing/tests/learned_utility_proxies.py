from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.eval.evaluators.latent_compatibility import (
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    distance_to_similarity,
)
from src.routing.strategies import compute_similarity


_ALLOWED_NORM_POLICIES = {"per_query_zscore", "per_query_minmax"}
_DEFAULT_ALPHA_GRID = [i / 10.0 for i in range(11)]


def _build_random_rank_floor_proxy(sample_domains: np.ndarray, n_experts: int, seed: int) -> np.ndarray:
    out = np.zeros((int(sample_domains.shape[0]), int(n_experts)), dtype=np.float64)
    if int(n_experts) <= 0:
        return out

    rank_by_query: Dict[int, np.ndarray] = {}
    for i in range(out.shape[0]):
        q = int(sample_domains[i])
        if q not in rank_by_query:
            rng = np.random.default_rng(int(seed) + (1009 * int(q)))
            perm = np.asarray(rng.permutation(int(n_experts)), dtype=np.int64)
            rank = np.zeros((int(n_experts),), dtype=np.float64)
            for pos, idx in enumerate(perm.tolist()):
                rank[int(idx)] = float(pos)
            rank_by_query[q] = rank
        out[i, :] = rank_by_query[q]
    return out


def _build_random_score_floor_proxy(n_samples: int, n_experts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n_samples), int(n_experts)), dtype=np.float64)


def _permute_expert_labels_proxy(proxy: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    perm = np.asarray(rng.permutation(proxy.shape[1]), dtype=np.int64)
    return proxy[:, perm]


def _metadata_scores(sample_domains: np.ndarray, expert_domains: Sequence[int], strategy: str, tau: float) -> np.ndarray:
    n = int(sample_domains.shape[0])
    e = len(expert_domains)
    out = np.zeros((n, e), dtype=np.float64)
    cache: Dict[int, np.ndarray] = {}
    for i in range(n):
        q = int(sample_domains[i])
        if q not in cache:
            cache[q] = np.asarray(
                [
                    float(
                        compute_similarity(
                            {"magnification": q},
                            {"magnification": int(ed)},
                            strategy=strategy,
                            tau=float(tau),
                            similarity_matrix=None,
                        )
                    )
                    for ed in expert_domains
                ],
                dtype=np.float64,
            )
        out[i, :] = cache[q]
    return out


def _latent_wasserstein_scores(embeddings: np.ndarray, sample_domains: np.ndarray, expert_domains: Sequence[int]) -> np.ndarray:
    domain_order = sorted(set(int(d) for d in sample_domains.tolist()))
    domain_order, stats, _warnings = compute_domain_gaussian_stats(
        embeddings=embeddings,
        domains=sample_domains,
        covariance_regularization_lambda=1e-4,
        min_samples_per_domain=5,
    )
    distances = compute_distance_matrices(domain_order=domain_order, stats=stats, eigenvalue_floor=1e-10)
    similarity, _scale = distance_to_similarity(distances["wasserstein"], scale_floor=1e-8)

    d_to_row = {int(d): i for i, d in enumerate(domain_order)}
    q_rows = np.asarray([d_to_row[int(q)] for q in sample_domains.tolist()], dtype=np.int64)

    e_rows = np.asarray([d_to_row.get(int(ed), -1) for ed in expert_domains], dtype=np.int64)
    valid = e_rows >= 0

    out = np.full((int(sample_domains.shape[0]), len(expert_domains)), float("-inf"), dtype=np.float64)
    if np.any(valid):
        out[:, valid] = similarity[q_rows[:, None], e_rows[valid][None, :]]
    return out


def _normalize_scores_per_query(scores: np.ndarray, policy: str) -> np.ndarray:
    policy_norm = str(policy).strip().lower()
    if policy_norm not in _ALLOWED_NORM_POLICIES:
        raise ValueError(
            f"Unknown normalization policy: {policy}. Allowed: {sorted(_ALLOWED_NORM_POLICIES)}"
        )

    out = np.zeros_like(scores, dtype=np.float64)
    for i in range(scores.shape[0]):
        row = scores[i, :].astype(np.float64, copy=False)
        if not np.isfinite(row).all():
            raise ValueError("Normalization received non-finite proxy row values")

        if policy_norm == "per_query_zscore":
            mu = float(np.mean(row))
            sigma = float(np.std(row))
            # Explicit zero-variance policy: emit zeros for this row.
            if sigma < 1e-12:
                out[i, :] = 0.0
            else:
                out[i, :] = (row - mu) / sigma
        else:
            lo = float(np.min(row))
            hi = float(np.max(row))
            span = hi - lo
            # Explicit zero-variance policy: emit zeros for this row.
            if span < 1e-12:
                out[i, :] = 0.0
            else:
                out[i, :] = (row - lo) / span

    return out


def _proxy_diagnostic_rows(scores: np.ndarray, sample_domains: np.ndarray, method: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(scores.shape[0]):
        row = scores[i, :]
        rows.append(
            {
                "method": str(method),
                "sample_index": int(i),
                "query_domain": int(sample_domains[i]),
                "row_min": float(np.min(row)),
                "row_max": float(np.max(row)),
                "row_mean": float(np.mean(row)),
                "row_std": float(np.std(row)),
            }
        )
    return rows
