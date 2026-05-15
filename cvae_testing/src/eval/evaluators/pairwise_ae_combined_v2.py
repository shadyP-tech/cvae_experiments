from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_models import _PairwiseRanker
from src.eval.evaluators.learned_utility_pairs import (
    _build_pair_features,
    _build_pairwise_training_pairs,
    _zscore_features,
)
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import _pairwise_auc_single, _selection_metrics


BASELINE_METHOD = "pairwise_ranker_ae_combined"
PRIMARY_METHOD = "pairwise_ranker_ae_combined_inner_selected_v2"
RANK_MARGIN_UNWEIGHTED = "pairwise_ranker_ae_combined_rank_margin_unweighted_v2"
RAW_AE_WEIGHTED = "pairwise_ranker_ae_combined_raw_ae_weighted_v2"
RANK_MARGIN_WEIGHTED = "pairwise_ranker_ae_combined_rank_margin_weighted_v2"
V2_CANDIDATE_METHODS = (
    BASELINE_METHOD,
    RANK_MARGIN_UNWEIGHTED,
    RAW_AE_WEIGHTED,
    RANK_MARGIN_WEIGHTED,
)
_SIMPLER_METHOD_ORDER = {method: idx for idx, method in enumerate(V2_CANDIDATE_METHODS)}


@dataclass(frozen=True)
class PairwiseAECombinedV2FoldOutputs:
    sample_rows: List[Dict[str, Any]]
    pair_rows: List[Dict[str, Any]]
    training_pair_rows: List[Dict[str, Any]]
    feature_diagnostic_rows: List[Dict[str, Any]]
    inner_selection_rows: List[Dict[str, Any]]
    decision_rows: List[Dict[str, Any]]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: Sequence[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _stable_argmin(row: np.ndarray) -> int:
    order = np.lexsort((np.arange(row.shape[0], dtype=np.int64), row))
    return int(order[0])


def _metadata_selected_expert(query_domain: int, candidate_domains: Sequence[int]) -> int:
    candidates = [int(v) for v in candidate_domains]
    exact = [d for d in candidates if int(d) == int(query_domain)]
    return int(exact[0]) if exact else int(sorted(candidates)[0])


def _metadata_features(q: np.ndarray, e: np.ndarray, sample_domains: np.ndarray) -> np.ndarray:
    span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
    abs_diff = np.abs(q.astype(np.float64) - e.astype(np.float64)) / span
    exact = (q == e).astype(np.float64)
    return np.stack([abs_diff, exact], axis=1)


def _ae_rank_margin_features(
    *,
    ae_zscore_matrix: np.ndarray,
    sample_indices: np.ndarray,
    candidate_domains: Sequence[int],
    expert_domains: Sequence[int],
) -> Tuple[np.ndarray, List[str]]:
    domain_to_col = {int(domain): int(i) for i, domain in enumerate(expert_domains)}
    candidate_cols = [domain_to_col[int(domain)] for domain in candidate_domains]
    ae = np.asarray(ae_zscore_matrix[np.asarray(sample_indices, dtype=np.int64)][:, candidate_cols], dtype=np.float64)
    n_rows, n_candidates = ae.shape
    ranks = np.zeros_like(ae, dtype=np.float64)
    best_scores = np.zeros((n_rows,), dtype=np.float64)
    second_scores = np.zeros((n_rows,), dtype=np.float64)
    med_scores = np.median(ae, axis=1)
    best_mask = np.zeros_like(ae, dtype=np.float64)
    second_mask = np.zeros_like(ae, dtype=np.float64)
    margins = np.zeros((n_rows,), dtype=np.float64)
    tie = np.arange(n_candidates, dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie, ae[i, :]))
        rank_row = np.empty((n_candidates,), dtype=np.float64)
        rank_row[order] = np.arange(1, n_candidates + 1, dtype=np.float64)
        ranks[i, :] = rank_row
        best_idx = int(order[0])
        second_idx = int(order[1]) if n_candidates > 1 else best_idx
        best_scores[i] = float(ae[i, best_idx])
        second_scores[i] = float(ae[i, second_idx])
        margins[i] = float(second_scores[i] - best_scores[i]) if n_candidates > 1 else float("inf")
        best_mask[i, best_idx] = 1.0
        if n_candidates > 1:
            second_mask[i, second_idx] = 1.0

    norm_den = max(float(n_candidates - 1), 1.0)
    candidate_z = ae.reshape(-1)
    rank_flat = ranks.reshape(-1)
    best_rep = np.repeat(best_scores, n_candidates)
    second_rep = np.repeat(second_scores, n_candidates)
    median_rep = np.repeat(med_scores, n_candidates)
    margin_rep = np.repeat(margins, n_candidates)
    best_flat = best_mask.reshape(-1)
    second_flat = second_mask.reshape(-1)
    not_best_flat = 1.0 - best_flat
    features = np.stack(
        [
            candidate_z,
            rank_flat,
            (rank_flat - 1.0) / norm_den,
            best_flat,
            second_flat,
            candidate_z - best_rep,
            candidate_z - second_rep,
            candidate_z - median_rep,
            best_flat * margin_rep,
            second_flat * margin_rep,
            not_best_flat * margin_rep,
        ],
        axis=1,
    ).astype(np.float64, copy=False)
    names = [
        "candidate_ae_zscore",
        "candidate_ae_rank",
        "candidate_normalized_ae_rank",
        "is_ae_best_candidate",
        "is_ae_second_candidate",
        "candidate_minus_best_ae_zscore",
        "candidate_minus_second_ae_zscore",
        "candidate_minus_median_ae_zscore",
        "ae_margin_if_best",
        "ae_margin_if_second",
        "ae_margin_if_not_best",
    ]
    return features, names


def _build_pairwise_features_for_candidates(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    sample_indices: np.ndarray,
    candidate_domains: Sequence[int],
    expert_domains: Sequence[int],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    x, q, e, s = _build_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=sample_indices,
        expert_domains=candidate_domains,
        expert_id_domains=expert_domains,
        include_metadata_features=True,
    )
    expert_oh = x[:, embedding_feature_dim : embedding_feature_dim + expert_feature_dim]
    meta = _metadata_features(q, e, sample_domains)
    latent = np.concatenate([x[:, :embedding_feature_dim], expert_oh], axis=1)
    combined = np.concatenate([latent, meta], axis=1)
    ae_rank, ae_names = _ae_rank_margin_features(
        ae_zscore_matrix=ae_zscore_matrix,
        sample_indices=sample_indices,
        candidate_domains=candidate_domains,
        expert_domains=expert_domains,
    )
    if str(feature_mode) == "raw_ae_combined":
        features = np.concatenate([ae_rank[:, :1], combined], axis=1)
        names = ["candidate_ae_zscore"] + [f"combined_{i}" for i in range(combined.shape[1])]
    elif str(feature_mode) == "rank_margin_combined":
        features = np.concatenate([ae_rank, combined], axis=1)
        names = ae_names + [f"combined_{i}" for i in range(combined.shape[1])]
    else:
        raise ProtocolError(f"Unknown pairwise AE-combined v2 feature_mode={feature_mode}")
    return features, q, e, s, names


def _build_training_features(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    train_indices: np.ndarray,
    expert_domains: Sequence[int],
    outer_heldout_domain: int,
    globally_excluded_domains: Sequence[int],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[str]]:
    x_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    feature_names: List[str] = []
    candidate_counts: set[int] = set()
    excluded_global = {int(outer_heldout_domain), *[int(v) for v in globally_excluded_domains]}
    for query_domain in sorted(set(int(sample_domains[int(i)]) for i in train_indices.tolist())):
        if int(query_domain) in excluded_global:
            continue
        domain_indices = train_indices[sample_domains[train_indices] == int(query_domain)]
        excluded = sorted(excluded_global | {int(query_domain)})
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=excluded,
        )
        if len(fold.candidate_expert_domains) < 2:
            continue
        x, q, e, s, names = _build_pairwise_features_for_candidates(
            embeddings=embeddings,
            sample_domains=sample_domains,
            sample_indices=domain_indices,
            candidate_domains=fold.candidate_expert_domains,
            expert_domains=expert_domains,
            embedding_feature_dim=embedding_feature_dim,
            expert_feature_dim=expert_feature_dim,
            ae_zscore_matrix=ae_zscore_matrix,
            feature_mode=feature_mode,
        )
        x_parts.append(x)
        q_parts.append(q)
        e_parts.append(e)
        s_parts.append(s)
        feature_names = names
        candidate_counts.add(int(len(fold.candidate_expert_domains)))
    if not x_parts or len(candidate_counts) != 1:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            0,
            feature_names,
        )
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(q_parts, axis=0),
        np.concatenate(e_parts, axis=0),
        np.concatenate(s_parts, axis=0),
        int(next(iter(candidate_counts))),
        feature_names,
    )


def _pair_weight(
    *,
    better_nelbo: float,
    worse_nelbo: float,
    source_inner_median_abs_nelbo: float,
    cfg: Mapping[str, Any],
) -> float:
    scale = max(float(source_inner_median_abs_nelbo), 1.0e-12)
    delta_scaled = abs(float(worse_nelbo) - float(better_nelbo)) / scale
    alpha = float(cfg.get("pair_weight_alpha", 4.0))
    delta_clip = float(cfg.get("pair_weight_delta_clip", 0.50))
    w_min = float(cfg.get("pair_weight_min", 1.0))
    w_max = float(cfg.get("pair_weight_max", 3.0))
    return float(np.clip(1.0 + alpha * min(delta_scaled, delta_clip), w_min, w_max))


def _build_utility_weighted_training_pairs(
    *,
    y_train: np.ndarray,
    q_train: np.ndarray,
    s_train: np.ndarray,
    experts_per_sample: int,
    near_tie_delta: float,
    hard_pair_fraction: float,
    utility_pair_fraction: float,
    random_pair_fraction: float,
    max_pairs_per_sample: int,
    max_pairs_per_domain: int,
    seed: int,
    cfg: Mapping[str, Any],
) -> Tuple[List[Tuple[int, int]], np.ndarray, List[Dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    pairs: List[Tuple[int, int]] = []
    weights: List[float] = []
    diagnostics: List[Dict[str, Any]] = []
    domain_pair_counts: Dict[int, int] = {}
    frac_sum = max(float(hard_pair_fraction) + float(utility_pair_fraction) + float(random_pair_fraction), 1.0e-12)
    hard_ratio = float(hard_pair_fraction) / frac_sum
    utility_ratio = float(utility_pair_fraction) / frac_sum
    source_scale = float(np.median(np.abs(np.asarray(y_train, dtype=np.float64)))) if y_train.size else 1.0
    if not np.isfinite(source_scale) or source_scale <= 1.0e-12:
        source_scale = 1.0

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
                candidates.append((ii, jj, diff) if yi < yj else (jj, ii, diff))
        if not candidates:
            diagnostics.append(
                {
                    "sample_index": int(sample_index),
                    "query_domain": int(query_domain),
                    "n_candidates": 0,
                    "n_selected": 0,
                    "n_hard_selected": 0,
                    "n_utility_selected": 0,
                    "n_random_selected": 0,
                    "mean_pair_weight": float("nan"),
                    "dropped_by_domain_cap": 0,
                    "source_inner_median_abs_nelbo": float(source_scale),
                }
            )
            continue
        target = min(int(max_pairs_per_sample), len(candidates))
        n_hard = min(int(round(target * hard_ratio)), target)
        n_utility = min(int(round(target * utility_ratio)), target - n_hard)
        hard_sorted = sorted(candidates, key=lambda x: float(x[2]))
        utility_sorted = sorted(candidates, key=lambda x: float(x[2]), reverse=True)
        selected: List[Tuple[int, int, float, str]] = []
        used: set[Tuple[int, int]] = set()
        for better, worse, diff in hard_sorted:
            if len([v for v in selected if v[3] == "hard"]) >= n_hard:
                break
            key = (int(better), int(worse))
            if key not in used:
                selected.append((int(better), int(worse), float(diff), "hard"))
                used.add(key)
        for better, worse, diff in utility_sorted:
            if len([v for v in selected if v[3] == "utility"]) >= n_utility:
                break
            key = (int(better), int(worse))
            if key not in used:
                selected.append((int(better), int(worse), float(diff), "utility"))
                used.add(key)
        remaining = [(b, w, d) for b, w, d in candidates if (int(b), int(w)) not in used]
        n_random = min(target - len(selected), len(remaining))
        if n_random > 0:
            for k in rng.choice(len(remaining), size=n_random, replace=False).tolist():
                b, w, d = remaining[int(k)]
                selected.append((int(b), int(w), float(d), "random"))

        dropped = 0
        added_weights: List[float] = []
        added_by_source = {"hard": 0, "utility": 0, "random": 0}
        for better, worse, _diff, source in selected:
            cur = int(domain_pair_counts.get(query_domain, 0))
            if cur >= int(max_pairs_per_domain):
                dropped += 1
                continue
            weight = _pair_weight(
                better_nelbo=float(y_train[int(better)]),
                worse_nelbo=float(y_train[int(worse)]),
                source_inner_median_abs_nelbo=source_scale,
                cfg=cfg,
            )
            pairs.append((int(better), int(worse)))
            weights.append(float(weight))
            added_weights.append(float(weight))
            added_by_source[str(source)] += 1
            domain_pair_counts[query_domain] = cur + 1

        diagnostics.append(
            {
                "sample_index": int(sample_index),
                "query_domain": int(query_domain),
                "n_candidates": int(len(candidates)),
                "n_selected": int(sum(added_by_source.values())),
                "n_hard_selected": int(added_by_source["hard"]),
                "n_utility_selected": int(added_by_source["utility"]),
                "n_random_selected": int(added_by_source["random"]),
                "mean_pair_weight": _finite_mean(added_weights),
                "dropped_by_domain_cap": int(dropped),
                "source_inner_median_abs_nelbo": float(source_scale),
            }
        )
    return pairs, np.asarray(weights, dtype=np.float64), diagnostics


def _train_predict_variant(
    *,
    method: str,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    outer_heldout_domain: int,
    globally_excluded_domains: Sequence[int],
    eval_candidate_domains: Sequence[int],
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    v2_cfg = dict((pairwise_cfg.get("utility_weighted_v2", {}) or {}))
    feature_mode = "rank_margin_combined" if method in {RANK_MARGIN_UNWEIGHTED, RANK_MARGIN_WEIGHTED} else "raw_ae_combined"
    weighted = method in {RAW_AE_WEIGHTED, RANK_MARGIN_WEIGHTED}
    x_train, q_train, e_train, s_train, experts_per_sample, feature_names = _build_training_features(
        embeddings=embeddings,
        sample_domains=sample_domains,
        train_indices=np.asarray(train_idx, dtype=np.int64),
        expert_domains=expert_domains,
        outer_heldout_domain=int(outer_heldout_domain),
        globally_excluded_domains=globally_excluded_domains,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        feature_mode=feature_mode,
    )
    if x_train.size == 0 or int(experts_per_sample) < 2:
        raise ProtocolError(f"No v2 pairwise training rows for method={method}")
    x_eval, _q_eval, _e_eval, _s_eval, _eval_names = _build_pairwise_features_for_candidates(
        embeddings=embeddings,
        sample_domains=sample_domains,
        sample_indices=np.asarray(eval_idx, dtype=np.int64),
        candidate_domains=eval_candidate_domains,
        expert_domains=expert_domains,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        feature_mode=feature_mode,
    )
    x_train_z, x_eval_z = _zscore_features(x_train, x_eval)
    y_train = true_nelbo[s_train, [int(domain_to_idx[int(ed)]) for ed in e_train]]
    if str(method) == BASELINE_METHOD:
        pairs, pair_diags = _build_pairwise_training_pairs(
            y_train=y_train,
            q_train=q_train,
            s_train=s_train,
            experts_per_sample=int(experts_per_sample),
            near_tie_delta=float(pairwise_cfg.get("near_tie_delta", 0.0)),
            hard_pair_fraction=float(pairwise_cfg.get("hard_pair_fraction", 0.5)),
            random_pair_fraction=float(pairwise_cfg.get("random_pair_fraction", 0.5)),
            max_pairs_per_sample=int(pairwise_cfg.get("max_pairs_per_sample", 12)),
            max_pairs_per_domain=int(pairwise_cfg.get("max_pairs_per_domain", 5000)),
            seed=int(seed) + int(outer_heldout_domain),
        )
        weights = np.ones((len(pairs),), dtype=np.float64)
        for row in pair_diags:
            row["n_utility_selected"] = 0
            row["mean_pair_weight"] = 1.0 if int(row.get("n_selected", 0)) > 0 else float("nan")
            row["source_inner_median_abs_nelbo"] = float(
                np.median(np.abs(np.asarray(y_train, dtype=np.float64)))
            )
    else:
        pairs, weights, pair_diags = _build_utility_weighted_training_pairs(
            y_train=y_train,
            q_train=q_train,
            s_train=s_train,
            experts_per_sample=int(experts_per_sample),
            near_tie_delta=float(pairwise_cfg.get("near_tie_delta", 0.0)),
            hard_pair_fraction=float(v2_cfg.get("hard_pair_fraction", 0.40)),
            utility_pair_fraction=float(v2_cfg.get("utility_pair_fraction", 0.40)),
            random_pair_fraction=float(v2_cfg.get("random_pair_fraction", 0.20)),
            max_pairs_per_sample=int(pairwise_cfg.get("max_pairs_per_sample", 12)),
            max_pairs_per_domain=int(pairwise_cfg.get("max_pairs_per_domain", 5000)),
            seed=int(seed) + int(outer_heldout_domain),
            cfg=v2_cfg,
        )
    if not pairs:
        raise ProtocolError(f"No v2 pairwise training pairs for method={method}")
    feature_diag_rows = _feature_diagnostics(method=method, x_train=x_train_z, pairs=pairs, feature_names=feature_names)
    ranker = _PairwiseRanker(
        seed=int(seed),
        hidden_dim=int(pairwise_cfg.get("hidden_dim", 128)),
        epochs=int(pairwise_cfg.get("epochs", 40)),
        lr=float(pairwise_cfg.get("lr", 1e-3)),
        batch_size=int(pairwise_cfg.get("batch_size", 2048)),
        margin=float(pairwise_cfg.get("margin", 1.0)),
        device=str(pairwise_cfg.get("device", "auto")),
    )
    ranker.fit(x_train_z, pairs, pair_weights=weights if weighted else None)
    pred = ranker.predict(x_eval_z).reshape(int(eval_idx.shape[0]), int(len(eval_candidate_domains)))
    diag_rows: List[Dict[str, Any]] = []
    mean_by_query: Dict[int, List[float]] = {}
    for row in pair_diags:
        query = int(row["query_domain"])
        if np.isfinite(float(row.get("mean_pair_weight", float("nan")))):
            mean_by_query.setdefault(query, []).append(float(row["mean_pair_weight"]))
        diag = {
            "method": str(method),
            "outer_heldout_domain": int(outer_heldout_domain),
            "weighted_loss": int(weighted),
            "feature_mode": str(feature_mode),
            **row,
        }
        diag_rows.append(diag)
    for query, vals in mean_by_query.items():
        diag_rows.append(
            {
                "method": str(method),
                "outer_heldout_domain": int(outer_heldout_domain),
                "query_domain": int(query),
                "diagnostic": "mean_pair_weight_by_query_domain",
                "mean_pair_weight_by_query_domain": _finite_mean(vals),
            }
        )
    expert_weight_map: Dict[int, List[float]] = {}
    for pair_idx, (better, worse) in enumerate(pairs):
        weight = float(weights[int(pair_idx)]) if int(pair_idx) < int(weights.shape[0]) else 1.0
        expert_weight_map.setdefault(int(e_train[int(better)]), []).append(weight)
        expert_weight_map.setdefault(int(e_train[int(worse)]), []).append(weight)
    for expert, vals in sorted(expert_weight_map.items()):
        diag_rows.append(
            {
                "method": str(method),
                "outer_heldout_domain": int(outer_heldout_domain),
                "candidate_expert": int(expert),
                "diagnostic": "mean_pair_weight_by_candidate_expert",
                "mean_pair_weight_by_candidate_expert": _finite_mean(vals),
            }
        )
    return pred, weights, diag_rows, feature_diag_rows, feature_names


def _feature_diagnostics(
    *,
    method: str,
    x_train: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    feature_names: Sequence[str],
) -> List[Dict[str, Any]]:
    if not pairs or x_train.size == 0:
        return []
    pair_arr = np.asarray(pairs, dtype=np.int64)
    diffs = x_train[pair_arr[:, 0], :] - x_train[pair_arr[:, 1], :]
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        nonzero = np.abs(diffs[:, idx]) > 1.0e-12
        rows.append(
            {
                "method": str(method),
                "feature_name": str(name),
                "feature_nonzero_rate_after_pairwise_difference": float(np.mean(nonzero)),
            }
        )
    rows.append(
        {
            "method": str(method),
            "feature_name": "__all__",
            "feature_nonzero_rate_after_pairwise_difference": float(np.mean(np.abs(diffs) > 1.0e-12)),
        }
    )
    return rows


def _metrics_from_sample_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {
        "top1_oracle_hit": _finite_mean([float(r.get("top1_oracle_hit", float("nan"))) for r in rows], 0.0),
        "spearman": _finite_mean([float(r.get("spearman", float("nan"))) for r in rows], float("nan")),
        "mean_oracle_gap_pct": _finite_mean([float(r.get("oracle_gap_pct", float("nan"))) for r in rows], 0.0),
    }


def _evaluate_variant_on_indices(
    *,
    method: str,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    outer_heldout_domain: int,
    globally_excluded_domains: Sequence[int],
    eval_fold: FoldCandidateSet,
    global_eval: np.ndarray,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    tie_policy: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    pred, _weights, _diag_rows, _feature_rows, _names = _train_predict_variant(
        method=method,
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        train_idx=train_idx,
        eval_idx=eval_idx,
        outer_heldout_domain=int(outer_heldout_domain),
        globally_excluded_domains=globally_excluded_domains,
        eval_candidate_domains=eval_fold.candidate_expert_domains,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        pairwise_cfg=pairwise_cfg,
        seed=int(seed),
    )
    true_eval = eval_fold.slice_nelbo(true_nelbo, eval_idx)
    return _selection_metrics(
        method=method,
        query_domains=sample_domains[eval_idx],
        expert_domains=eval_fold.candidate_expert_domains,
        score_matrix=pred,
        true_nelbo_matrix=true_eval,
        fold=eval_fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )


def _source_inner_selection(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    outer_fold: FoldCandidateSet,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    ae_zscore_matrix: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    tie_policy: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    inner_rows: List[Dict[str, Any]] = []
    per_method_by_inner: Dict[str, List[Tuple[int, Dict[str, float]]]] = {method: [] for method in V2_CANDIDATE_METHODS}
    for inner_domain in source_domains:
        inner_eval_idx = np.asarray(
            [int(i) for i in np.asarray(train_idx, dtype=np.int64).tolist() if int(sample_domains[int(i)]) == int(inner_domain)],
            dtype=np.int64,
        )
        inner_train_idx = np.asarray(
            [int(i) for i in np.asarray(train_idx, dtype=np.int64).tolist() if int(sample_domains[int(i)]) != int(inner_domain)],
            dtype=np.int64,
        )
        if inner_eval_idx.size == 0 or inner_train_idx.size == 0:
            continue
        inner_fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_fold.heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(inner_domain)],
        )
        if int(outer_fold.heldout_domain) in set(inner_fold.candidate_expert_domains):
            raise ProtocolError("Inner v2 candidate pool includes outer target expert")
        if int(inner_domain) in set(inner_fold.candidate_expert_domains):
            raise ProtocolError("Inner v2 candidate pool includes query-self expert")
        if len(inner_fold.candidate_expert_domains) < 2:
            continue
        global_eval = true_nelbo[inner_eval_idx]
        method_rows: Dict[str, Dict[str, float]] = {}
        for method in V2_CANDIDATE_METHODS:
            metrics, rows = _evaluate_variant_on_indices(
                method=method,
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                train_idx=inner_train_idx,
                eval_idx=inner_eval_idx,
                outer_heldout_domain=int(outer_fold.heldout_domain),
                globally_excluded_domains=[int(inner_domain)],
                eval_fold=inner_fold,
                global_eval=global_eval,
                embedding_feature_dim=int(embedding_feature_dim),
                expert_feature_dim=int(expert_feature_dim),
                ae_zscore_matrix=ae_zscore_matrix,
                pairwise_cfg=pairwise_cfg,
                seed=int(seed) + int(inner_domain) * 17,
                tie_policy=tie_policy,
            )
            _ = rows
            method_rows[method] = metrics
            per_method_by_inner[method].append((int(inner_domain), metrics))
        baseline = method_rows.get(BASELINE_METHOD)
        if not baseline:
            continue
        for method in V2_CANDIDATE_METHODS:
            candidate = method_rows.get(method)
            if not candidate:
                continue
            inner_rows.append(
                {
                    "seed": int(seed),
                    "outer_heldout_center": int(outer_fold.heldout_domain),
                    "inner_heldout_source_center": int(inner_domain),
                    "baseline_method": BASELINE_METHOD,
                    "candidate_method": str(method),
                    "source_inner_macro_gap_baseline": float(baseline["mean_oracle_gap_pct"]),
                    "source_inner_macro_gap_candidate": float(candidate["mean_oracle_gap_pct"]),
                    "source_inner_macro_top1_baseline": float(baseline["top1_oracle_hit"]),
                    "source_inner_macro_top1_candidate": float(candidate["top1_oracle_hit"]),
                    "source_inner_macro_spearman_baseline": float(baseline["spearman"]),
                    "source_inner_macro_spearman_candidate": float(candidate["spearman"]),
                    "inner_gap_delta": float(baseline["mean_oracle_gap_pct"] - candidate["mean_oracle_gap_pct"]),
                    "inner_top1_delta": float(candidate["top1_oracle_hit"] - baseline["top1_oracle_hit"]),
                    "inner_spearman_delta": float(candidate["spearman"] - baseline["spearman"]),
                    "heldout_target_nelbo_used_for_selection": 0,
                }
            )

    baseline_units = per_method_by_inner.get(BASELINE_METHOD, [])
    if not baseline_units:
        selected = BASELINE_METHOD
    else:
        scored: List[Dict[str, Any]] = []
        baseline_by_inner = {inner: metrics for inner, metrics in baseline_units}
        for method in V2_CANDIDATE_METHODS:
            units = per_method_by_inner.get(method, [])
            if not units:
                continue
            gap_deltas = []
            top1_deltas = []
            spearman_deltas = []
            degradations = []
            for inner, metrics in units:
                base = baseline_by_inner.get(inner)
                if not base:
                    continue
                gap_delta = float(base["mean_oracle_gap_pct"] - metrics["mean_oracle_gap_pct"])
                gap_deltas.append(gap_delta)
                top1_deltas.append(float(metrics["top1_oracle_hit"] - base["top1_oracle_hit"]))
                spearman_deltas.append(float(metrics["spearman"] - base["spearman"]))
                degradations.append(float(metrics["mean_oracle_gap_pct"] - base["mean_oracle_gap_pct"]))
            macro_gap = _finite_mean(gap_deltas, 0.0)
            macro_top1 = _finite_mean(top1_deltas, 0.0)
            macro_spearman = _finite_mean(spearman_deltas, 0.0)
            worst_degradation = max(degradations) if degradations else float("inf")
            passed = bool(
                macro_gap >= 0.0
                and macro_top1 >= -0.02
                and macro_spearman >= -0.03
                and worst_degradation <= 1.0
            )
            scored.append(
                {
                    "method": str(method),
                    "macro_gap": float(macro_gap),
                    "macro_top1": float(macro_top1),
                    "macro_spearman": float(macro_spearman),
                    "worst_degradation": float(worst_degradation),
                    "passed": int(passed),
                }
            )
        passing = [row for row in scored if int(row["passed"]) == 1]
        if not passing:
            selected = BASELINE_METHOD
        else:
            selected = str(
                sorted(
                    passing,
                    key=lambda row: (
                        float(row["macro_gap"]),
                        float(row["macro_top1"]),
                        float(row["macro_spearman"]),
                        -float(_SIMPLER_METHOD_ORDER.get(str(row["method"]), 10**6)),
                    ),
                    reverse=True,
                )[0]["method"]
            )

    selected_score_by_method = {row["candidate_method"]: [] for row in inner_rows}
    for row in inner_rows:
        selected_score_by_method.setdefault(str(row["candidate_method"]), []).append(row)
    for method, rows in selected_score_by_method.items():
        base_rows = [r for r in rows if str(r["candidate_method"]) == str(method)]
        if not base_rows:
            continue
        gap_deltas = [float(r["inner_gap_delta"]) for r in base_rows]
        top1_deltas = [float(r["inner_top1_delta"]) for r in base_rows]
        spearman_deltas = [float(r["inner_spearman_delta"]) for r in base_rows]
        worst = max(float(r["source_inner_macro_gap_candidate"]) - float(r["source_inner_macro_gap_baseline"]) for r in base_rows)
        passed = bool(
            _finite_mean(gap_deltas, 0.0) >= 0.0
            and _finite_mean(top1_deltas, 0.0) >= -0.02
            and _finite_mean(spearman_deltas, 0.0) >= -0.03
            and worst <= 1.0
        )
        for row in base_rows:
            row["selected_method"] = str(selected)
            row["selection_reason"] = "selected_by_source_inner_policy" if str(method) == str(selected) else "not_selected"
            row["inner_worst_center_gap_degradation"] = float(worst)
            row["candidate_passed_no_harm_gate"] = int(passed)
            row["fallback_to_baseline"] = int(str(selected) == BASELINE_METHOD)
            row["selected_by_inner_policy"] = int(str(method) == str(selected))
    return selected, inner_rows


def run_pairwise_ae_combined_v2_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    global_eval: np.ndarray,
    pairwise_cfg: Mapping[str, Any],
    seed: int,
    embedding_feature_dim: int,
    expert_feature_dim: int,
    tie_policy: str,
    ae_zscore_matrix: np.ndarray | None,
) -> PairwiseAECombinedV2FoldOutputs:
    v2_cfg = dict((pairwise_cfg.get("utility_weighted_v2", {}) or {}))
    if not bool(pairwise_cfg.get("run_utility_weighted_v2", False)) or not bool(v2_cfg.get("enabled", False)):
        return PairwiseAECombinedV2FoldOutputs([], [], [], [], [], [])
    if ae_zscore_matrix is None:
        raise ProtocolError("pairwise AE-combined v2 requires autoencoder_proxy AE z-score matrix")

    selected_method, inner_rows = _source_inner_selection(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        train_idx=train_idx,
        outer_fold=fold,
        embedding_feature_dim=int(embedding_feature_dim),
        expert_feature_dim=int(expert_feature_dim),
        ae_zscore_matrix=ae_zscore_matrix,
        pairwise_cfg=pairwise_cfg,
        seed=int(seed),
        tie_policy=tie_policy,
    )

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    method_predictions: Dict[str, np.ndarray] = {}
    method_sample_rows: Dict[str, List[Dict[str, Any]]] = {}
    true_eval = fold.slice_nelbo(true_nelbo, test_idx)
    for method in V2_CANDIDATE_METHODS:
        pred, weights, diag_rows, feat_rows, _names = _train_predict_variant(
            method=method,
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            domain_to_idx=domain_to_idx,
            train_idx=train_idx,
            eval_idx=test_idx,
            outer_heldout_domain=int(fold.heldout_domain),
            globally_excluded_domains=[],
            eval_candidate_domains=fold.candidate_expert_domains,
            embedding_feature_dim=int(embedding_feature_dim),
            expert_feature_dim=int(expert_feature_dim),
            ae_zscore_matrix=ae_zscore_matrix,
            pairwise_cfg=pairwise_cfg,
            seed=int(seed),
        )
        method_predictions[method] = pred
        metrics, rows = _selection_metrics(
            method=method,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=pred,
            true_nelbo_matrix=true_eval,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
        )
        _ = metrics
        for row in rows:
            row["sample_index"] = int(test_idx[int(row["sample_index"])])
            row["source_inner_selected_method"] = str(selected_method)
        method_sample_rows[method] = rows
        if method != BASELINE_METHOD:
            sample_rows.extend(rows)
        feature_rows.extend([{**row, "outer_heldout_center": int(fold.heldout_domain)} for row in feat_rows])
        for row in diag_rows:
            training_rows.append({**row, "seed": int(seed)})
        pred_flat = pred.reshape(-1)
        s_rep = np.repeat(test_idx.astype(np.int64), repeats=len(fold.candidate_expert_domains))
        q_rep = np.repeat(sample_domains[test_idx].astype(np.int64), repeats=len(fold.candidate_expert_domains))
        e_rep = np.tile(np.asarray(fold.candidate_expert_domains, dtype=np.int64), reps=int(test_idx.shape[0]))
        true_flat = true_eval.reshape(-1)
        row_protocol = _method_protocol(method)
        for k in range(pred_flat.shape[0]):
            pair_rows.append(
                {
                    **_protocol_row_fields(fold=fold, method_protocol=row_protocol, method=method),
                    "method": method,
                    "sample_index": int(s_rep[k]),
                    "query_domain": int(q_rep[k]),
                    "expert_domain": int(e_rep[k]),
                    "predicted_score": float(pred_flat[k]),
                    "true_nelbo": float(true_flat[k]),
                    "source_inner_selected_method": str(selected_method),
                }
            )

    selected_pred = method_predictions.get(str(selected_method), method_predictions[BASELINE_METHOD])
    selected_metrics, selected_rows = _selection_metrics(
        method=PRIMARY_METHOD,
        query_domains=sample_domains[test_idx],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=selected_pred,
        true_nelbo_matrix=true_eval,
        fold=fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )
    _ = selected_metrics
    baseline_rows = method_sample_rows[BASELINE_METHOD]
    baseline_by_sample = {int(r["sample_index"]): r for r in baseline_rows}
    ae_scores = ae_zscore_matrix[np.asarray(test_idx, dtype=np.int64)][:, list(fold.candidate_col_indices)]
    decision_rows: List[Dict[str, Any]] = []
    for local_i, row in enumerate(selected_rows):
        sample_index = int(test_idx[int(row["sample_index"])])
        row["sample_index"] = sample_index
        row["source_inner_selected_method"] = str(selected_method)
        row["fallback_to_baseline"] = int(str(selected_method) == BASELINE_METHOD)
        sample_rows.append(row)
        base = baseline_by_sample.get(sample_index, {})
        ae_order = np.lexsort((np.arange(ae_scores.shape[1], dtype=np.int64), ae_scores[int(local_i), :]))
        ae_best = int(fold.candidate_expert_domains[int(ae_order[0])])
        ae_margin = (
            float(ae_scores[int(local_i), int(ae_order[1])] - ae_scores[int(local_i), int(ae_order[0])])
            if int(ae_scores.shape[1]) > 1
            else float("inf")
        )
        metadata_selected = _metadata_selected_expert(
            query_domain=int(sample_domains[sample_index]),
            candidate_domains=fold.candidate_expert_domains,
        )
        decision_rows.append(
            {
                "seed": int(seed),
                "outer_heldout_center": int(fold.heldout_domain),
                "sample_index": int(sample_index),
                "selected_method": str(selected_method),
                "selected_expert": int(row["selected_expert"]),
                "oracle_expert": int(row["oracle_expert"]),
                "selected_nelbo": float(row["selected_nelbo"]),
                "oracle_nelbo": float(row["oracle_nelbo"]),
                "oracle_gap_pct": float(row["oracle_gap_pct"]),
                "baseline_selected_expert": int(base.get("selected_expert", -1)),
                "baseline_oracle_gap_pct": float(base.get("oracle_gap_pct", float("nan"))),
                "delta_gap_vs_baseline": float(base.get("oracle_gap_pct", float("nan"))) - float(row["oracle_gap_pct"]),
                "ae_best_expert": int(ae_best),
                "ae_best_vs_second_margin": float(ae_margin),
                "metadata_selected_expert": int(metadata_selected),
                "top1_oracle_hit": int(row["selected_expert"] == row["oracle_expert"]),
            }
        )
    return PairwiseAECombinedV2FoldOutputs(
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        training_pair_rows=training_rows,
        feature_diagnostic_rows=feature_rows,
        inner_selection_rows=inner_rows,
        decision_rows=decision_rows,
    )


def _pairwise_auc_from_prediction_rows(rows: Sequence[Mapping[str, Any]], *, mode: str) -> float:
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("method", "")), int(row.get("sample_index", -1)))
        grouped.setdefault(key, []).append(row)

    pair_records: List[Tuple[float, float]] = []
    sample_aucs: List[float] = []
    for (_method, _sample), sample_rows in grouped.items():
        if len(sample_rows) < 2:
            continue
        scores = np.asarray([float(r.get("predicted_score", float("nan"))) for r in sample_rows], dtype=np.float64)
        true = np.asarray([float(r.get("true_nelbo", float("nan"))) for r in sample_rows], dtype=np.float64)
        if mode == "all":
            sample_aucs.append(float(_pairwise_auc_single(scores, true)))
            continue
        for i in range(len(sample_rows)):
            for j in range(i + 1, len(sample_rows)):
                diff = abs(float(true[i]) - float(true[j]))
                if diff < 1.0e-12:
                    continue
                pred_diff = abs(float(scores[i]) - float(scores[j]))
                if pred_diff < 1.0e-12:
                    correct = 0.5
                else:
                    correct = float((true[i] < true[j]) == (scores[i] < scores[j]))
                pair_records.append((float(diff), float(correct)))
    if mode == "all":
        return _finite_mean(sample_aucs, float("nan"))
    if not pair_records:
        return float("nan")
    diffs = np.asarray([v[0] for v in pair_records], dtype=np.float64)
    if mode == "near_boundary":
        threshold = float(np.quantile(diffs, 0.25))
        selected = [float(score) for diff, score in pair_records if float(diff) <= threshold]
    elif mode == "high_utility_gap":
        threshold = float(np.quantile(diffs, 0.75))
        selected = [float(score) for diff, score in pair_records if float(diff) >= threshold]
    else:
        selected = []
    return _finite_mean(selected, float("nan"))


def _nested_counts(rows: Sequence[Mapping[str, Any]], *, key_field: str, value_field: str = "selected_method") -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = str(row.get(key_field, ""))
        value = str(row.get(value_field, ""))
        if not key or not value:
            continue
        out.setdefault(key, {})
        out[key][value] = int(out[key].get(value, 0)) + 1
    return out


def _top1_when_low_ae_margin(rows: Sequence[Mapping[str, Any]]) -> float:
    vals = [
        (float(r.get("ae_best_vs_second_margin", float("nan"))), float(r.get("top1_oracle_hit", float("nan"))))
        for r in rows
        if np.isfinite(float(r.get("ae_best_vs_second_margin", float("nan"))))
    ]
    if not vals:
        return float("nan")
    threshold = float(np.quantile(np.asarray([v[0] for v in vals], dtype=np.float64), 0.25))
    return _finite_mean([hit for margin, hit in vals if float(margin) <= threshold], float("nan"))


def write_pairwise_ae_combined_v2_artifacts(
    *,
    reports_dir: Path,
    training_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    inner_selection_rows: Sequence[Mapping[str, Any]],
    pair_prediction_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not (training_rows or feature_rows or inner_selection_rows or pair_prediction_rows or decision_rows):
        return {}
    _write_csv(reports_dir / "pairwise_ae_combined_v2_training_pairs.csv", training_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_feature_diagnostics.csv", feature_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_inner_selection_table.csv", inner_selection_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_pair_predictions.csv", pair_prediction_rows)
    _write_csv(reports_dir / "pairwise_ae_combined_v2_decision_table.csv", decision_rows)

    selected_methods = [str(r.get("selected_method", "")) for r in decision_rows]
    n = max(len(selected_methods), 1)
    fallback_count = sum(1 for method in selected_methods if method == BASELINE_METHOD)
    selected_counts = {method: selected_methods.count(method) for method in sorted(set(selected_methods)) if method}
    gap_deltas = [float(r.get("delta_gap_vs_baseline", float("nan"))) for r in decision_rows]
    summary = {
        "method": PRIMARY_METHOD,
        "selected_method_count_total": selected_counts,
        "selected_method_count_by_seed": _nested_counts(decision_rows, key_field="seed"),
        "selected_method_count_by_outer_center": _nested_counts(decision_rows, key_field="outer_heldout_center"),
        "fallback_to_baseline_rate": float(fallback_count / n),
        "v2_adoption_rate": float((n - fallback_count) / n),
        "mean_delta_gap_vs_baseline": _finite_mean(gap_deltas, 0.0),
        "pairwise_auc_all": _pairwise_auc_from_prediction_rows(pair_prediction_rows, mode="all"),
        "pairwise_auc_near_boundary": _pairwise_auc_from_prediction_rows(pair_prediction_rows, mode="near_boundary"),
        "pairwise_auc_high_utility_gap": _pairwise_auc_from_prediction_rows(pair_prediction_rows, mode="high_utility_gap"),
        "top1_when_margin_small": _top1_when_low_ae_margin(decision_rows),
        "top1_when_ae_confidence_low": _top1_when_low_ae_margin(decision_rows),
        "feature_nonzero_rate_after_pairwise_difference": _finite_mean(
            [
                float(r.get("feature_nonzero_rate_after_pairwise_difference", float("nan")))
                for r in feature_rows
                if str(r.get("feature_name", "")) == "__all__"
            ],
            float("nan"),
        ),
        "heldout_target_nelbo_used_for_selection": 0,
    }
    _write_csv(reports_dir / "pairwise_ae_combined_v2_decision_summary.csv", [summary])
    (reports_dir / "pairwise_ae_combined_v2_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return {
        "pairwise_ae_combined_v2_training_pairs": "pairwise_ae_combined_v2_training_pairs.csv",
        "pairwise_ae_combined_v2_feature_diagnostics": "pairwise_ae_combined_v2_feature_diagnostics.csv",
        "pairwise_ae_combined_v2_inner_selection_table": "pairwise_ae_combined_v2_inner_selection_table.csv",
        "pairwise_ae_combined_v2_pair_predictions": "pairwise_ae_combined_v2_pair_predictions.csv",
        "pairwise_ae_combined_v2_decision_table": "pairwise_ae_combined_v2_decision_table.csv",
        "pairwise_ae_combined_v2_decision_summary": "pairwise_ae_combined_v2_decision_summary.json",
    }
