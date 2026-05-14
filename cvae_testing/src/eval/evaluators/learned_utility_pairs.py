from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet, ProtocolError


def _build_pair_features(
    *,
    sample_embeddings: np.ndarray,
    sample_domains: np.ndarray,
    sample_indices: np.ndarray,
    expert_domains: Sequence[int],
    expert_id_domains: Sequence[int] | None = None,
    include_metadata_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e = len(expert_domains)
    expert_id_order = [int(d) for d in (expert_id_domains or expert_domains)]
    expert_id_to_col = {int(d): idx for idx, d in enumerate(expert_id_order)}
    x_sel = sample_embeddings[sample_indices]
    q_sel = sample_domains[sample_indices]
    n = int(x_sel.shape[0])

    sample_rep = np.repeat(x_sel, repeats=e, axis=0)
    expert_oh = np.zeros((n * e, len(expert_id_order)), dtype=np.float64)
    for idx, domain in enumerate(np.tile(np.asarray([int(d) for d in expert_domains], dtype=np.int64), reps=n).tolist()):
        if int(domain) not in expert_id_to_col:
            raise ProtocolError(f"Expert domain {domain} is missing from expert_id_domains")
        expert_oh[int(idx), expert_id_to_col[int(domain)]] = 1.0
    features = [sample_rep, expert_oh]

    query_rep = np.repeat(q_sel.astype(np.float64), repeats=e)
    expert_rep = np.tile(np.asarray([int(d) for d in expert_domains], dtype=np.float64), reps=n)

    if include_metadata_features:
        span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
        abs_diff = np.abs(query_rep - expert_rep) / span
        exact = (query_rep == expert_rep).astype(np.float64)
        features.append(np.stack([abs_diff, exact], axis=1))

    x = np.concatenate(features, axis=1)
    return x, query_rep.astype(np.int64), expert_rep.astype(np.int64), np.repeat(sample_indices.astype(np.int64), repeats=e)


def _build_fold_training_pair_features(
    *,
    sample_embeddings: np.ndarray,
    sample_domains: np.ndarray,
    train_indices: np.ndarray,
    expert_domains: Sequence[int],
    outer_heldout_domain: int,
    include_metadata_features: bool,
    extra_excluded_domains: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    expert_domain_set = {int(d) for d in expert_domains}
    if int(outer_heldout_domain) not in expert_domain_set:
        raise ProtocolError(f"Outer heldout domain {outer_heldout_domain} has no matching expert checkpoint")

    for query_domain in sorted(set(int(sample_domains[int(i)]) for i in train_indices.tolist())):
        if int(query_domain) not in expert_domain_set:
            raise ProtocolError(f"Training query domain {query_domain} has no matching expert checkpoint")
        domain_indices = train_indices[sample_domains[train_indices] == int(query_domain)]
        if domain_indices.size == 0:
            continue
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(query_domain), *[int(v) for v in (extra_excluded_domains or ())]],
        )
        if not fold.candidate_expert_domains:
            raise ProtocolError(
                "learned_pair_policy left zero training candidates for "
                f"outer_heldout_domain={outer_heldout_domain}, query_domain={query_domain}"
            )
        x, q, e, s = _build_pair_features(
            sample_embeddings=sample_embeddings,
            sample_domains=sample_domains,
            sample_indices=domain_indices,
            expert_domains=fold.candidate_expert_domains,
            expert_id_domains=expert_domains,
            include_metadata_features=include_metadata_features,
        )
        x_parts.append(x)
        q_parts.append(q)
        e_parts.append(e)
        s_parts.append(s)

    if not x_parts:
        raise ProtocolError(f"No learned training samples remain for heldout_domain={outer_heldout_domain}")

    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(q_parts, axis=0),
        np.concatenate(e_parts, axis=0),
        np.concatenate(s_parts, axis=0),
    )


def _zscore_features(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def _normalize_targets_per_query(y: np.ndarray, query_domains: np.ndarray) -> np.ndarray:
    y_norm = np.zeros_like(y, dtype=np.float64)
    for q in sorted(set(int(v) for v in query_domains.tolist())):
        idx = np.where(query_domains == int(q))[0]
        vals = y[idx]
        mu = float(vals.mean())
        sigma = float(vals.std())
        if sigma < 1e-8:
            sigma = 1.0
        y_norm[idx] = (vals - mu) / sigma
    return y_norm


def _build_pairwise_training_pairs(
    *,
    y_train: np.ndarray,
    q_train: np.ndarray,
    s_train: np.ndarray,
    experts_per_sample: int,
    near_tie_delta: float,
    hard_pair_fraction: float,
    random_pair_fraction: float,
    max_pairs_per_sample: int,
    max_pairs_per_domain: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    pairs: List[Tuple[int, int]] = []
    diagnostics: List[Dict[str, Any]] = []
    domain_pair_counts: Dict[int, int] = {}

    frac_sum = max(float(hard_pair_fraction) + float(random_pair_fraction), 1e-12)
    hard_ratio = float(hard_pair_fraction) / frac_sum

    for sample_index in sorted(set(int(v) for v in s_train.tolist())):
        idxs = np.where(s_train == int(sample_index))[0]
        if int(idxs.size) != int(experts_per_sample):
            continue

        query_domain = int(q_train[idxs[0]])
        candidates: List[Tuple[int, int, float]] = []
        for i in range(int(idxs.size)):
            for j in range(i + 1, int(idxs.size)):
                ii = int(idxs[i])
                jj = int(idxs[j])
                yi = float(y_train[ii])
                yj = float(y_train[jj])
                diff = abs(yi - yj)
                if diff <= float(near_tie_delta):
                    continue
                if yi < yj:
                    candidates.append((ii, jj, diff))
                else:
                    candidates.append((jj, ii, diff))

        if not candidates:
            diagnostics.append(
                {
                    "sample_index": int(sample_index),
                    "query_domain": int(query_domain),
                    "n_candidates": 0,
                    "n_selected": 0,
                    "n_hard_selected": 0,
                    "n_random_selected": 0,
                    "dropped_by_domain_cap": 0,
                }
            )
            continue

        candidates_sorted = sorted(candidates, key=lambda x: float(x[2]))
        target = min(int(max_pairs_per_sample), len(candidates_sorted))
        n_hard = min(int(round(target * hard_ratio)), target)
        hard_selected = candidates_sorted[:n_hard]

        remaining = candidates_sorted[n_hard:]
        n_random_target = min(target - n_hard, len(remaining))
        random_selected: List[Tuple[int, int, float]] = []
        if n_random_target > 0:
            rand_idxs = rng.choice(len(remaining), size=n_random_target, replace=False)
            random_selected = [remaining[int(k)] for k in np.asarray(rand_idxs).tolist()]

        selected = hard_selected + random_selected
        dropped_by_cap = 0
        added = 0
        for better_idx, worse_idx, _diff in selected:
            cur = int(domain_pair_counts.get(query_domain, 0))
            if cur >= int(max_pairs_per_domain):
                dropped_by_cap += 1
                continue
            pairs.append((int(better_idx), int(worse_idx)))
            domain_pair_counts[query_domain] = cur + 1
            added += 1

        diagnostics.append(
            {
                "sample_index": int(sample_index),
                "query_domain": int(query_domain),
                "n_candidates": int(len(candidates_sorted)),
                "n_selected": int(added),
                "n_hard_selected": int(min(len(hard_selected), added)),
                "n_random_selected": int(max(added - min(len(hard_selected), added), 0)),
                "dropped_by_domain_cap": int(dropped_by_cap),
            }
        )

    return pairs, diagnostics
