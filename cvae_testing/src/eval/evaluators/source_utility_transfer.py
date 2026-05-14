from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import SourceUtilityTransferConfig
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _aggregate_metrics_from_sample_rows,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.eval.evaluators.support_response_routing import LinearPairwiseRidge
from src.eval.metrics import spearman_corr


DIRECT_METHOD = "source_utility_transfer_metadata_only_v1"
SAFE_METHOD = "source_utility_transfer_metadata_safe_override_v1"
RANDOM_CONTROL_METHOD = "random_metadata_override_matched_coverage"
SHUFFLED_CONTROL_METHOD = "source_utility_transfer_shuffled_profiles"

FEATURE_NAMES = (
    "metadata_distance",
    "source_profile_mean_z",
    "source_profile_mean_rank",
)
MARGIN_EPS = 1.0e-8


@dataclass(frozen=True)
class SourceUtilityTransferOutputs:
    sample_rows: List[Dict[str, Any]]
    pair_rows: List[Dict[str, Any]]
    feature_audit_rows: List[Dict[str, Any]]
    threshold_audit_rows: List[Dict[str, Any]]
    override_rows: List[Dict[str, Any]]
    domain_breakdown_rows: List[Dict[str, Any]]
    clustered_metric_rows: List[Dict[str, Any]]
    random_control_rows: List[Dict[str, Any]]
    negative_control_rows: List[Dict[str, Any]]
    artifacts: Dict[str, str]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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
                fieldnames.append(key_s)
                seen.add(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stable_argmin(scores: Sequence[float], experts: Sequence[int]) -> int:
    order = sorted(range(len(experts)), key=lambda i: (float(scores[i]), int(experts[i])))
    return int(order[0])


def _normalized_margin(scores: Sequence[float], experts: Sequence[int]) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    order = sorted(range(len(experts)), key=lambda i: (float(arr[i]), int(experts[i])))
    raw_margin = float(arr[int(order[1])] - arr[int(order[0])])
    return float(raw_margin / (float(np.std(arr)) + MARGIN_EPS))


def _domain_indices(sample_domains: np.ndarray, domain: int) -> np.ndarray:
    return np.where(sample_domains == int(domain))[0].astype(np.int64, copy=False)


def _mean_nelbo_for_domain(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    domain_to_idx: Mapping[int, int],
    query_domain: int,
    expert_domain: int,
) -> float:
    idx = _domain_indices(sample_domains, int(query_domain))
    if idx.size == 0:
        raise ProtocolError(f"No samples found for query_domain={query_domain}")
    return float(np.mean(true_nelbo[idx, int(domain_to_idx[int(expert_domain)])]))


def _domain_true_vector(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    domain_to_idx: Mapping[int, int],
    query_domain: int,
    candidate_experts: Sequence[int],
) -> np.ndarray:
    return np.asarray(
        [
            _mean_nelbo_for_domain(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                domain_to_idx=domain_to_idx,
                query_domain=int(query_domain),
                expert_domain=int(expert),
            )
            for expert in candidate_experts
        ],
        dtype=np.float64,
    )


def _global_true_vector(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    query_domain: int,
) -> np.ndarray:
    idx = _domain_indices(sample_domains, int(query_domain))
    if idx.size == 0:
        raise ProtocolError(f"No samples found for query_domain={query_domain}")
    return np.mean(true_nelbo[idx], axis=0, dtype=np.float64).reshape(1, -1)


def _metadata_distance_from_similarity(
    *,
    metadata_similarity: np.ndarray,
    sample_domains: np.ndarray,
    domain_to_idx: Mapping[int, int],
    query_domain: int,
    expert_domain: int,
) -> float:
    idx = _domain_indices(sample_domains, int(query_domain))
    if idx.size == 0:
        return 1.0
    sim = float(np.mean(metadata_similarity[idx, int(domain_to_idx[int(expert_domain)])]))
    return float(1.0 - sim)


def _metadata_scores_for_candidates(
    *,
    metadata_similarity: np.ndarray,
    sample_domains: np.ndarray,
    domain_to_idx: Mapping[int, int],
    query_domain: int,
    candidate_experts: Sequence[int],
) -> np.ndarray:
    idx = _domain_indices(sample_domains, int(query_domain))
    if idx.size == 0:
        raise ProtocolError(f"No samples found for query_domain={query_domain}")
    return np.asarray(
        [-float(np.mean(metadata_similarity[idx, int(domain_to_idx[int(expert)])])) for expert in candidate_experts],
        dtype=np.float64,
    )


def _profile_values(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    domain_to_idx: Mapping[int, int],
    profile_domains: Sequence[int],
    comparison_experts: Sequence[int],
) -> Dict[Tuple[int, int], float]:
    values: Dict[Tuple[int, int], float] = {}
    for domain in profile_domains:
        for expert in comparison_experts:
            values[(int(domain), int(expert))] = _mean_nelbo_for_domain(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                domain_to_idx=domain_to_idx,
                query_domain=int(domain),
                expert_domain=int(expert),
            )
    return values


def _profile_stats_by_expert(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    domain_to_idx: Mapping[int, int],
    outer_target_domain: int,
    pseudo_query_domain: int | None,
    expert_domains: Sequence[int],
    candidate_experts: Sequence[int],
) -> Tuple[Dict[int, Dict[str, float]], List[int]]:
    profile_domains = [
        int(d)
        for d in sorted(set(int(v) for v in sample_domains.tolist()))
        if int(d) != int(outer_target_domain)
        and (pseudo_query_domain is None or int(d) != int(pseudo_query_domain))
    ]
    if not profile_domains:
        return {
            int(expert): {
                "source_profile_mean_z": 0.0,
                "source_profile_mean_rank": 0.0,
                "profile_domain_count": 0.0,
            }
            for expert in candidate_experts
        }, []

    comparison_experts = [
        int(e)
        for e in expert_domains
        if int(e) != int(outer_target_domain)
        and (pseudo_query_domain is None or int(e) != int(pseudo_query_domain))
    ]
    values = _profile_values(
        true_nelbo=true_nelbo,
        sample_domains=sample_domains,
        domain_to_idx=domain_to_idx,
        profile_domains=profile_domains,
        comparison_experts=comparison_experts,
    )
    all_vals = np.asarray(list(values.values()), dtype=np.float64)
    mu = float(np.mean(all_vals)) if all_vals.size else 0.0
    sigma = float(np.std(all_vals)) if all_vals.size else 1.0
    if sigma < 1.0e-8:
        sigma = 1.0

    stats: Dict[int, Dict[str, float]] = {}
    for expert in candidate_experts:
        z_vals: List[float] = []
        rank_vals: List[float] = []
        for domain in profile_domains:
            expert_value = float(values[(int(domain), int(expert))])
            z_vals.append((expert_value - mu) / sigma)
            domain_scores = [(int(e), float(values[(int(domain), int(e))])) for e in comparison_experts]
            ordered = sorted(domain_scores, key=lambda item: (float(item[1]), int(item[0])))
            ranks = {int(e): rank + 1 for rank, (e, _score) in enumerate(ordered)}
            rank_vals.append(float(ranks[int(expert)]))
        stats[int(expert)] = {
            "source_profile_mean_z": float(np.mean(z_vals)) if z_vals else 0.0,
            "source_profile_mean_rank": float(np.mean(rank_vals)) if rank_vals else 0.0,
            "profile_domain_count": float(len(profile_domains)),
        }
    return stats, profile_domains


def _feature_rows_for_query(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    metadata_similarity: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    outer_target_domain: int,
    query_domain: int,
    candidate_experts: Sequence[int],
    split_role: str,
    minibag_id: str,
    profile_shuffle_seed: int | None = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    profile_stats, profile_domains = _profile_stats_by_expert(
        true_nelbo=true_nelbo,
        sample_domains=sample_domains,
        domain_to_idx=domain_to_idx,
        outer_target_domain=int(outer_target_domain),
        pseudo_query_domain=None if split_role == "target" else int(query_domain),
        expert_domains=expert_domains,
        candidate_experts=candidate_experts,
    )
    stats_by_expert = {int(k): dict(v) for k, v in profile_stats.items()}
    if profile_shuffle_seed is not None and candidate_experts:
        rng = np.random.default_rng(int(profile_shuffle_seed))
        experts = [int(e) for e in candidate_experts]
        shuffled = list(rng.permutation(experts))
        stats_by_expert = {
            int(expert): dict(profile_stats[int(shuffled[pos])])
            for pos, expert in enumerate(experts)
        }

    feature_rows: List[List[float]] = []
    audit_rows: List[Dict[str, Any]] = []
    for expert in candidate_experts:
        metadata_distance = _metadata_distance_from_similarity(
            metadata_similarity=metadata_similarity,
            sample_domains=sample_domains,
            domain_to_idx=domain_to_idx,
            query_domain=int(query_domain),
            expert_domain=int(expert),
        )
        stats = stats_by_expert[int(expert)]
        feature_vec = [
            float(metadata_distance),
            float(stats["source_profile_mean_z"]),
            float(stats["source_profile_mean_rank"]),
        ]
        feature_rows.append(feature_vec)
        audit_rows.append(
            {
                "outer_target_domain": int(outer_target_domain),
                "query_domain": int(query_domain),
                "candidate_expert": int(expert),
                "split_role": str(split_role),
                "minibag_id": str(minibag_id),
                "feature_names": "|".join(FEATURE_NAMES),
                "metadata_distance": float(metadata_distance),
                "source_profile_mean_z": float(stats["source_profile_mean_z"]),
                "source_profile_mean_rank": float(stats["source_profile_mean_rank"]),
                "profile_domain_count": int(stats["profile_domain_count"]),
                "profile_excluded_domains": "|".join(
                    str(int(v))
                    for v in sorted(
                        {int(outer_target_domain)}
                        | ({int(query_domain)} if str(split_role) != "target" else set())
                    )
                ),
                "profile_domains": "|".join(str(int(v)) for v in profile_domains),
                "feature_source_scope": (
                    "target_metadata|source_utility_profile"
                    if str(split_role) == "target"
                    else f"{str(split_role)}|source_utility_profile"
                ),
                "strict_source_only": 1,
                "diagnostic_only": 0,
                "forbidden_target_eval": 0,
                "profile_shuffled": int(profile_shuffle_seed is not None),
            }
        )
    return np.asarray(feature_rows, dtype=np.float64), audit_rows


def _make_minibags(
    *,
    sample_indices: np.ndarray,
    query_domain: int,
    outer_target_domain: int,
    size: int,
    minibags_per_domain: int,
    seeds: Sequence[int],
) -> List[Tuple[str, np.ndarray]]:
    if sample_indices.size == 0:
        return []
    out: List[Tuple[str, np.ndarray]] = []
    replace = bool(sample_indices.size < int(size))
    for seed in seeds:
        rng = np.random.default_rng(int(seed) + 1009 * int(query_domain) + 9173 * int(outer_target_domain))
        for bag_idx in range(int(minibags_per_domain)):
            chosen = rng.choice(sample_indices, size=int(size), replace=replace)
            minibag_id = f"outer{int(outer_target_domain)}_q{int(query_domain)}_seed{int(seed)}_bag{int(bag_idx)}"
            out.append((minibag_id, np.asarray(chosen, dtype=np.int64)))
    return out


def _build_training_matrix(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    metadata_similarity: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    outer_target_domain: int,
    training_query_domains: Sequence[int],
    cfg: SourceUtilityTransferConfig,
    profile_shuffle_seed: int | None = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    x_rows: List[np.ndarray] = []
    pairs: List[Tuple[int, int]] = []
    pair_rows: List[Dict[str, Any]] = []
    feature_audit: List[Dict[str, Any]] = []

    for query_domain in sorted(int(v) for v in training_query_domains):
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_target_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(query_domain)],
        )
        candidate_experts = list(fold.candidate_expert_domains)
        query_indices = _domain_indices(sample_domains, int(query_domain))
        minibags = _make_minibags(
            sample_indices=query_indices,
            query_domain=int(query_domain),
            outer_target_domain=int(outer_target_domain),
            size=int(cfg.minibag_size),
            minibags_per_domain=int(cfg.minibags_per_domain),
            seeds=cfg.minibag_seeds,
        )
        for minibag_id, minibag_indices in minibags:
            features, audit_rows = _feature_rows_for_query(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                metadata_similarity=metadata_similarity,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                outer_target_domain=int(outer_target_domain),
                query_domain=int(query_domain),
                candidate_experts=candidate_experts,
                split_role="source_train",
                minibag_id=minibag_id,
                profile_shuffle_seed=profile_shuffle_seed,
            )
            start_idx = len(x_rows)
            for row in features:
                x_rows.append(np.asarray(row, dtype=np.float64))
            feature_audit.extend(audit_rows)

            utilities = np.asarray(
                [
                    float(np.mean(true_nelbo[minibag_indices, int(domain_to_idx[int(expert)])]))
                    for expert in candidate_experts
                ],
                dtype=np.float64,
            )
            for i in range(len(candidate_experts)):
                for j in range(i + 1, len(candidate_experts)):
                    if abs(float(utilities[i] - utilities[j])) < 1.0e-12:
                        continue
                    if float(utilities[i]) < float(utilities[j]):
                        better_local, worse_local = i, j
                    else:
                        better_local, worse_local = j, i
                    better_idx = int(start_idx + better_local)
                    worse_idx = int(start_idx + worse_local)
                    pairs.append((better_idx, worse_idx))
                    pair_rows.append(
                        {
                            "outer_target_domain": int(outer_target_domain),
                            "query_domain": int(query_domain),
                            "minibag_id": str(minibag_id),
                            "better_expert": int(candidate_experts[better_local]),
                            "worse_expert": int(candidate_experts[worse_local]),
                            "better_mean_nelbo": float(utilities[better_local]),
                            "worse_mean_nelbo": float(utilities[worse_local]),
                            "label_source": "minibag_mean_nelbo",
                            "query_unit": "minibag",
                            "profile_shuffled": int(profile_shuffle_seed is not None),
                        }
                    )

    if not x_rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), [], pair_rows, feature_audit
    return np.vstack(x_rows).astype(np.float64, copy=False), pairs, pair_rows, feature_audit


def _fit_ranker(x_train: np.ndarray, pairs: Sequence[Tuple[int, int]], ridge_l2: float) -> LinearPairwiseRidge:
    ranker = LinearPairwiseRidge(ridge_l2=float(ridge_l2))
    ranker.fit(x_train, pairs)
    return ranker


def _score_row_for_domain(
    *,
    method: str,
    query_domain: int,
    fold: FoldCandidateSet,
    candidate_experts: Sequence[int],
    scores: np.ndarray,
    true_vec: np.ndarray,
    global_true_vec: np.ndarray,
    expert_domains: Sequence[int],
    selected_idx_override: int | None = None,
    ranking_scores: np.ndarray | None = None,
    sample_index: int = 0,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    _metrics, rows = _selection_metrics(
        method=str(method),
        query_domains=np.asarray([int(query_domain)], dtype=np.int64),
        expert_domains=candidate_experts,
        score_matrix=np.asarray(scores, dtype=np.float64).reshape(1, -1),
        true_nelbo_matrix=np.asarray(true_vec, dtype=np.float64).reshape(1, -1),
        fold=fold,
        global_true_nelbo_matrix=np.asarray(global_true_vec, dtype=np.float64).reshape(1, -1),
        global_expert_domains=expert_domains,
        tie_policy="stable_expert_index",
        selected_idx_override=(
            np.asarray([int(selected_idx_override)], dtype=np.int64)
            if selected_idx_override is not None
            else None
        ),
        ranking_score_matrix=(
            np.asarray(ranking_scores, dtype=np.float64).reshape(1, -1)
            if ranking_scores is not None
            else None
        ),
    )
    row = dict(rows[0])
    row["sample_index"] = int(sample_index)
    if extra_fields:
        row.update(dict(extra_fields))
    return row


def _validation_rows_for_threshold(
    *,
    threshold: float,
    validation_predictions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pred in validation_predictions:
        candidate_experts = list(pred["candidate_experts"])
        learned_scores = np.asarray(pred["learned_scores"], dtype=np.float64)
        true_vec = np.asarray(pred["true_vec"], dtype=np.float64)
        metadata_scores = np.asarray(pred["metadata_scores"], dtype=np.float64)
        fallback_idx = int(pred["fallback_idx"])
        proposal_idx = _stable_argmin(learned_scores, candidate_experts)
        margin = _normalized_margin(learned_scores, candidate_experts)
        accepted = int(proposal_idx != fallback_idx and margin >= float(threshold))
        selected_idx = int(proposal_idx if accepted else fallback_idx)
        oracle_idx = _stable_argmin(true_vec, candidate_experts)
        selected_nelbo = float(true_vec[selected_idx])
        oracle_nelbo = float(true_vec[oracle_idx])
        fallback_nelbo = float(true_vec[fallback_idx])
        gap_pct = float(((selected_nelbo - oracle_nelbo) / max(abs(oracle_nelbo), 1.0e-12)) * 100.0)
        fallback_gap_pct = float(((fallback_nelbo - oracle_nelbo) / max(abs(oracle_nelbo), 1.0e-12)) * 100.0)
        spearman = float(spearman_corr((-learned_scores).tolist(), (-true_vec).tolist()))
        fallback_spearman = float(spearman_corr((-metadata_scores).tolist(), (-true_vec).tolist()))
        rows.append(
            {
                "outer_target_domain": int(pred["outer_target_domain"]),
                "inner_validation_domain": int(pred["query_domain"]),
                "threshold": float(threshold),
                "selected_idx": int(selected_idx),
                "fallback_idx": int(fallback_idx),
                "proposal_idx": int(proposal_idx),
                "accepted_override": int(accepted),
                "true_harmful_override": int(bool(accepted) and selected_nelbo > fallback_nelbo + 1.0e-12),
                "true_improving_override": int(bool(accepted) and selected_nelbo < fallback_nelbo - 1.0e-12),
                "top1_oracle_hit": int(selected_idx == oracle_idx),
                "fallback_top1_oracle_hit": int(fallback_idx == oracle_idx),
                "spearman": float(spearman),
                "fallback_spearman": float(fallback_spearman),
                "oracle_gap_pct": float(gap_pct),
                "fallback_oracle_gap_pct": float(fallback_gap_pct),
                "normalized_margin": float(margin),
            }
        )
    return rows


def _select_threshold(
    *,
    outer_target_domain: int,
    validation_predictions: Sequence[Dict[str, Any]],
    thresholds: Sequence[float],
) -> Tuple[float, List[Dict[str, Any]]]:
    audit_rows: List[Dict[str, Any]] = []
    best_key: Tuple[float, float, float, float, float] | None = None
    selected_threshold = float("inf")
    for threshold in thresholds:
        rows = _validation_rows_for_threshold(
            threshold=float(threshold),
            validation_predictions=validation_predictions,
        )
        if not rows:
            continue
        gap = float(np.mean([float(r["oracle_gap_pct"]) for r in rows]))
        fallback_gap = float(np.mean([float(r["fallback_oracle_gap_pct"]) for r in rows]))
        top1 = float(np.mean([float(r["top1_oracle_hit"]) for r in rows]))
        fallback_top1 = float(np.mean([float(r["fallback_top1_oracle_hit"]) for r in rows]))
        spearman = float(np.mean([float(r["spearman"]) for r in rows]))
        fallback_spearman = float(np.mean([float(r["fallback_spearman"]) for r in rows]))
        accepted = [r for r in rows if int(r["accepted_override"]) == 1]
        coverage = float(len(accepted) / max(len(rows), 1))
        harmful = float(
            sum(int(r["true_harmful_override"]) for r in accepted) / max(len(accepted), 1)
        ) if accepted else 0.0
        constraints_pass = bool(
            harmful <= 0.05
            and (fallback_top1 - top1) <= 0.02
            and (fallback_spearman - spearman) <= 0.05
        )
        audit_rows.append(
            {
                "outer_target_domain": int(outer_target_domain),
                "threshold": float(threshold),
                "threshold_label": "inf" if np.isinf(float(threshold)) else str(float(threshold)),
                "selection_source": "nested_source_inner_domain_aggregate",
                "n_inner_validation_domains": int(len(rows)),
                "mean_oracle_gap_pct": float(gap),
                "fallback_mean_oracle_gap_pct": float(fallback_gap),
                "gap_reduction_vs_fallback": float(fallback_gap - gap),
                "top1_oracle_hit": float(top1),
                "fallback_top1_oracle_hit": float(fallback_top1),
                "spearman": float(spearman),
                "fallback_spearman": float(fallback_spearman),
                "coverage": float(coverage),
                "harmful_override_rate": float(harmful),
                "constraints_pass": int(constraints_pass),
                "selected_threshold": 0,
            }
        )
        if not constraints_pass:
            continue
        key = (gap, -top1, -spearman, -coverage, float(threshold))
        if best_key is None or key < best_key:
            best_key = key
            selected_threshold = float(threshold)

    if best_key is None:
        selected_threshold = float("inf")
    for row in audit_rows:
        row["selected_threshold"] = int(float(row["threshold"]) == float(selected_threshold))
        row["selected_tau"] = "inf" if np.isinf(float(selected_threshold)) else str(float(selected_threshold))
    return selected_threshold, audit_rows


def _source_transfer_extra(
    *,
    cfg: SourceUtilityTransferConfig,
    threshold: float,
    normalized_margin: float,
    accepted_override: int,
    fallback_expert: int,
    proposal_expert: int,
    selected_expert: int,
    fallback_nelbo: float,
    selected_nelbo: float,
    feature_set: str,
    diagnostic_only: int,
    strict_source_only: int = 1,
) -> Dict[str, Any]:
    return {
        "policy_name": "source_only_utility_transfer_v1",
        "query_unit": str(cfg.query_unit),
        "strict_source_only": int(strict_source_only),
        "diagnostic_only": int(diagnostic_only),
        "feature_set": str(feature_set),
        "feature_source_scope": "target_metadata|source_utility_profile",
        "threshold_selection_policy": "nested_source_inner_domain_aggregate",
        "selection_source": "source_inner_only",
        "selected_tau": "inf" if np.isinf(float(threshold)) else str(float(threshold)),
        "normalized_margin": float(normalized_margin),
        "accepted_override": int(accepted_override),
        "override_candidate": int(int(proposal_expert) != int(fallback_expert)),
        "fallback_method": str(cfg.fallback_method),
        "fallback_expert": int(fallback_expert),
        "proposal_expert": int(proposal_expert),
        "selected_expert": int(selected_expert),
        "true_harmful_override": int(bool(accepted_override) and float(selected_nelbo) > float(fallback_nelbo) + 1.0e-12),
        "true_improving_override": int(bool(accepted_override) and float(selected_nelbo) < float(fallback_nelbo) - 1.0e-12),
        "n_minibags": int(cfg.minibags_per_domain * len(cfg.minibag_seeds)),
        "n_effective_domains": 1,
        "score_direction": "predicted_score_is_predicted_mean_nelbo_lower_is_better",
    }


def _clustered_metrics_from_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    metrics_by_method = _aggregate_metrics_from_sample_rows(rows) if rows else {}
    for method, metrics in sorted(metrics_by_method.items()):
        method_rows = [r for r in rows if str(r.get("method", "")) == str(method)]
        accepted = [r for r in method_rows if int(float(r.get("accepted_override", 0) or 0)) == 1]
        out.append(
            {
                "method": str(method),
                "coverage": float(len(accepted) / max(len(method_rows), 1)),
                "top1_oracle_hit": float(metrics.get("top1_oracle_hit", 0.0)),
                "spearman": float(metrics.get("spearman", 0.0)),
                "mean_oracle_gap_pct": float(metrics.get("mean_oracle_gap_pct", 0.0)),
                "harmful_override_rate": float(
                    sum(int(float(r.get("true_harmful_override", 0) or 0)) for r in accepted)
                    / max(len(accepted), 1)
                )
                if accepted
                else 0.0,
                "conditional_win_rate": float(
                    sum(int(float(r.get("true_improving_override", 0) or 0)) for r in accepted)
                    / max(len(accepted), 1)
                )
                if accepted
                else 0.0,
                "n_minibags": int(max([int(float(r.get("n_minibags", 0) or 0)) for r in method_rows] or [0])),
                "n_effective_domains": int(len(set(int(r["query_domain"]) for r in method_rows))),
                "strict_source_only": int(max(int(float(r.get("strict_source_only", 0) or 0)) for r in method_rows)),
                "diagnostic_only": int(max(int(float(r.get("diagnostic_only", 0) or 0)) for r in method_rows)),
            }
        )
    return out


def evaluate_source_utility_transfer(
    *,
    true_nelbo: np.ndarray,
    sample_domains: np.ndarray,
    metadata_similarity: np.ndarray,
    expert_domains: Sequence[int],
    domain_to_idx: Mapping[int, int],
    seed: int,
    cfg: SourceUtilityTransferConfig,
    reports_dir: Path,
) -> SourceUtilityTransferOutputs:
    if not bool(cfg.enabled):
        return SourceUtilityTransferOutputs([], [], [], [], [], [], [], [], [], {})
    if tuple(cfg.variants) != ("metadata_only",):
        raise ValueError("source_utility_transfer v1 supports only variants=['metadata_only']")
    if str(cfg.query_unit) != "minibag":
        raise ValueError("source_utility_transfer v1 supports only query_unit='minibag'")
    if str(cfg.ranker) != "linear_pairwise_ridge":
        raise ValueError("source_utility_transfer v1 supports only linear_pairwise_ridge")

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    feature_audit_rows: List[Dict[str, Any]] = []
    threshold_audit_rows: List[Dict[str, Any]] = []
    override_rows: List[Dict[str, Any]] = []
    random_control_rows: List[Dict[str, Any]] = []
    negative_control_rows: List[Dict[str, Any]] = []
    baseline_cluster_rows: List[Dict[str, Any]] = []
    source_domains = sorted(set(int(v) for v in sample_domains.tolist()))
    sample_index = 0

    for outer_target in source_domains:
        target_fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_target),
            expert_domains=expert_domains,
        )
        target_candidates = list(target_fold.candidate_expert_domains)
        source_query_domains = [int(d) for d in source_domains if int(d) != int(outer_target)]

        validation_predictions: List[Dict[str, Any]] = []
        for inner_val in source_query_domains:
            train_domains = [int(d) for d in source_query_domains if int(d) != int(inner_val)]
            x_inner, pairs_inner, pair_inner, feature_inner = _build_training_matrix(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                metadata_similarity=metadata_similarity,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                outer_target_domain=int(outer_target),
                training_query_domains=train_domains,
                cfg=cfg,
            )
            pair_rows.extend(pair_inner)
            feature_audit_rows.extend(feature_inner)
            if x_inner.shape[0] == 0:
                continue
            ranker_inner = _fit_ranker(x_inner, pairs_inner, ridge_l2=float(cfg.ridge_l2))
            inner_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_target),
                expert_domains=expert_domains,
                excluded_domains=[int(inner_val)],
            )
            inner_candidates = list(inner_fold.candidate_expert_domains)
            x_val, val_audit = _feature_rows_for_query(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                metadata_similarity=metadata_similarity,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                outer_target_domain=int(outer_target),
                query_domain=int(inner_val),
                candidate_experts=inner_candidates,
                split_role="source_val",
                minibag_id=f"outer{outer_target}_inner{inner_val}_domain",
            )
            feature_audit_rows.extend(val_audit)
            learned_scores = ranker_inner.predict(x_val)
            metadata_scores = _metadata_scores_for_candidates(
                metadata_similarity=metadata_similarity,
                sample_domains=sample_domains,
                domain_to_idx=domain_to_idx,
                query_domain=int(inner_val),
                candidate_experts=inner_candidates,
            )
            validation_predictions.append(
                {
                    "outer_target_domain": int(outer_target),
                    "query_domain": int(inner_val),
                    "candidate_experts": inner_candidates,
                    "learned_scores": learned_scores,
                    "metadata_scores": metadata_scores,
                    "fallback_idx": _stable_argmin(metadata_scores, inner_candidates),
                    "true_vec": _domain_true_vector(
                        true_nelbo=true_nelbo,
                        sample_domains=sample_domains,
                        domain_to_idx=domain_to_idx,
                        query_domain=int(inner_val),
                        candidate_experts=inner_candidates,
                    ),
                }
            )

        selected_threshold, threshold_rows = _select_threshold(
            outer_target_domain=int(outer_target),
            validation_predictions=validation_predictions,
            thresholds=cfg.normalized_margin_thresholds,
        )
        threshold_audit_rows.extend(threshold_rows)

        x_train, pairs, pair_train, feature_train = _build_training_matrix(
            true_nelbo=true_nelbo,
            sample_domains=sample_domains,
            metadata_similarity=metadata_similarity,
            expert_domains=expert_domains,
            domain_to_idx=domain_to_idx,
            outer_target_domain=int(outer_target),
            training_query_domains=source_query_domains,
            cfg=cfg,
        )
        pair_rows.extend(pair_train)
        feature_audit_rows.extend(feature_train)
        ranker = _fit_ranker(x_train, pairs, ridge_l2=float(cfg.ridge_l2))
        x_target, target_feature_audit = _feature_rows_for_query(
            true_nelbo=true_nelbo,
            sample_domains=sample_domains,
            metadata_similarity=metadata_similarity,
            expert_domains=expert_domains,
            domain_to_idx=domain_to_idx,
            outer_target_domain=int(outer_target),
            query_domain=int(outer_target),
            candidate_experts=target_candidates,
            split_role="target",
            minibag_id=f"outer{outer_target}_target_domain",
        )
        feature_audit_rows.extend(target_feature_audit)
        learned_scores = ranker.predict(x_target)
        metadata_scores = _metadata_scores_for_candidates(
            metadata_similarity=metadata_similarity,
            sample_domains=sample_domains,
            domain_to_idx=domain_to_idx,
            query_domain=int(outer_target),
            candidate_experts=target_candidates,
        )
        true_vec = _domain_true_vector(
            true_nelbo=true_nelbo,
            sample_domains=sample_domains,
            domain_to_idx=domain_to_idx,
            query_domain=int(outer_target),
            candidate_experts=target_candidates,
        )
        global_true_vec = _global_true_vector(
            true_nelbo=true_nelbo,
            sample_domains=sample_domains,
            query_domain=int(outer_target),
        )
        fallback_idx = _stable_argmin(metadata_scores, target_candidates)
        proposal_idx = _stable_argmin(learned_scores, target_candidates)
        margin = _normalized_margin(learned_scores, target_candidates)
        accepted = int(proposal_idx != fallback_idx and margin >= float(selected_threshold))
        safe_idx = int(proposal_idx if accepted else fallback_idx)
        direct_row = _score_row_for_domain(
            method=DIRECT_METHOD,
            query_domain=int(outer_target),
            fold=target_fold,
            candidate_experts=target_candidates,
            scores=learned_scores,
            true_vec=true_vec,
            global_true_vec=global_true_vec,
            expert_domains=expert_domains,
            sample_index=sample_index,
            extra_fields=_source_transfer_extra(
                cfg=cfg,
                threshold=selected_threshold,
                normalized_margin=margin,
                accepted_override=int(proposal_idx != fallback_idx),
                fallback_expert=int(target_candidates[fallback_idx]),
                proposal_expert=int(target_candidates[proposal_idx]),
                selected_expert=int(target_candidates[proposal_idx]),
                fallback_nelbo=float(true_vec[fallback_idx]),
                selected_nelbo=float(true_vec[proposal_idx]),
                feature_set="metadata_profile",
                diagnostic_only=1,
            ),
        )
        sample_rows.append(direct_row)
        sample_index += 1
        safe_row = _score_row_for_domain(
            method=SAFE_METHOD,
            query_domain=int(outer_target),
            fold=target_fold,
            candidate_experts=target_candidates,
            scores=learned_scores,
            true_vec=true_vec,
            global_true_vec=global_true_vec,
            expert_domains=expert_domains,
            selected_idx_override=safe_idx,
            ranking_scores=learned_scores,
            sample_index=sample_index,
            extra_fields=_source_transfer_extra(
                cfg=cfg,
                threshold=selected_threshold,
                normalized_margin=margin,
                accepted_override=accepted,
                fallback_expert=int(target_candidates[fallback_idx]),
                proposal_expert=int(target_candidates[proposal_idx]),
                selected_expert=int(target_candidates[safe_idx]),
                fallback_nelbo=float(true_vec[fallback_idx]),
                selected_nelbo=float(true_vec[safe_idx]),
                feature_set="metadata_profile",
                diagnostic_only=0,
            ),
        )
        sample_rows.append(safe_row)
        override_rows.append(safe_row)
        sample_index += 1

        baseline_row = _score_row_for_domain(
            method="metadata_routing",
            query_domain=int(outer_target),
            fold=target_fold,
            candidate_experts=target_candidates,
            scores=metadata_scores,
            true_vec=true_vec,
            global_true_vec=global_true_vec,
            expert_domains=expert_domains,
            sample_index=sample_index,
            extra_fields={
                "strict_source_only": 1,
                "diagnostic_only": 0,
                "query_unit": "domain",
                "feature_source_scope": "target_metadata",
                "n_minibags": 0,
                "n_effective_domains": 1,
            },
        )
        baseline_cluster_rows.append(baseline_row)
        sample_index += 1

        for control_seed in cfg.random_control_seeds:
            rng = np.random.default_rng(int(control_seed) + 7919 * int(seed) + 101 * int(outer_target))
            if accepted:
                non_fallback = [i for i in range(len(target_candidates)) if int(i) != int(fallback_idx)]
                random_idx = int(rng.choice(non_fallback)) if non_fallback else int(fallback_idx)
            else:
                random_idx = int(fallback_idx)
            random_scores = rng.random(len(target_candidates), dtype=np.float64)
            random_row = _score_row_for_domain(
                method=RANDOM_CONTROL_METHOD,
                query_domain=int(outer_target),
                fold=target_fold,
                candidate_experts=target_candidates,
                scores=random_scores,
                true_vec=true_vec,
                global_true_vec=global_true_vec,
                expert_domains=expert_domains,
                selected_idx_override=random_idx,
                ranking_scores=random_scores,
                sample_index=sample_index,
                extra_fields={
                    **_source_transfer_extra(
                        cfg=cfg,
                        threshold=selected_threshold,
                        normalized_margin=float(margin),
                        accepted_override=int(accepted),
                        fallback_expert=int(target_candidates[fallback_idx]),
                        proposal_expert=int(target_candidates[random_idx]),
                        selected_expert=int(target_candidates[random_idx]),
                        fallback_nelbo=float(true_vec[fallback_idx]),
                        selected_nelbo=float(true_vec[random_idx]),
                        feature_set="matched_random_override",
                        diagnostic_only=0,
                    ),
                    "control_seed": int(control_seed),
                    "method_role": "control",
                    "adoption_eligible": 0,
                },
            )
            sample_rows.append(random_row)
            random_control_rows.append(random_row)
            sample_index += 1

        if bool(cfg.enable_shuffled_profile_control):
            shuffle_seed = int(seed) + 100003 * int(outer_target)
            x_shuf_train, shuf_pairs, shuf_pair_rows, shuf_feature_rows = _build_training_matrix(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                metadata_similarity=metadata_similarity,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                outer_target_domain=int(outer_target),
                training_query_domains=source_query_domains,
                cfg=cfg,
                profile_shuffle_seed=shuffle_seed,
            )
            pair_rows.extend(shuf_pair_rows)
            feature_audit_rows.extend(shuf_feature_rows)
            shuf_ranker = _fit_ranker(x_shuf_train, shuf_pairs, ridge_l2=float(cfg.ridge_l2))
            x_shuf_target, shuf_target_audit = _feature_rows_for_query(
                true_nelbo=true_nelbo,
                sample_domains=sample_domains,
                metadata_similarity=metadata_similarity,
                expert_domains=expert_domains,
                domain_to_idx=domain_to_idx,
                outer_target_domain=int(outer_target),
                query_domain=int(outer_target),
                candidate_experts=target_candidates,
                split_role="target",
                minibag_id=f"outer{outer_target}_target_domain_shuffled",
                profile_shuffle_seed=shuffle_seed,
            )
            feature_audit_rows.extend(shuf_target_audit)
            shuffled_scores = shuf_ranker.predict(x_shuf_target)
            shuffled_idx = _stable_argmin(shuffled_scores, target_candidates)
            shuffled_row = _score_row_for_domain(
                method=SHUFFLED_CONTROL_METHOD,
                query_domain=int(outer_target),
                fold=target_fold,
                candidate_experts=target_candidates,
                scores=shuffled_scores,
                true_vec=true_vec,
                global_true_vec=global_true_vec,
                expert_domains=expert_domains,
                selected_idx_override=shuffled_idx,
                ranking_scores=shuffled_scores,
                sample_index=sample_index,
                extra_fields={
                    **_source_transfer_extra(
                        cfg=cfg,
                        threshold=selected_threshold,
                        normalized_margin=_normalized_margin(shuffled_scores, target_candidates),
                        accepted_override=int(shuffled_idx != fallback_idx),
                        fallback_expert=int(target_candidates[fallback_idx]),
                        proposal_expert=int(target_candidates[shuffled_idx]),
                        selected_expert=int(target_candidates[shuffled_idx]),
                        fallback_nelbo=float(true_vec[fallback_idx]),
                        selected_nelbo=float(true_vec[shuffled_idx]),
                        feature_set="shuffled_profiles",
                        diagnostic_only=0,
                    ),
                    "method_role": "control",
                    "adoption_eligible": 0,
                },
            )
            sample_rows.append(shuffled_row)
            negative_control_rows.append(shuffled_row)
            sample_index += 1

    source_rows_for_cluster = list(baseline_cluster_rows) + list(sample_rows)
    domain_breakdown_rows = []
    if source_rows_for_cluster:
        from src.eval.evaluators.learned_utility_protocol import _domain_breakdown_rows

        domain_breakdown_rows = _domain_breakdown_rows(source_rows_for_cluster)
    clustered_metric_rows = _clustered_metrics_from_rows(source_rows_for_cluster)

    artifacts = {
        "source_utility_transfer_sample_selections": "source_utility_transfer_sample_selections.csv",
        "source_utility_transfer_pair_predictions": "source_utility_transfer_pair_predictions.csv",
        "source_utility_transfer_feature_audit": "source_utility_transfer_feature_audit.csv",
        "source_utility_transfer_threshold_audit": "source_utility_transfer_threshold_audit.csv",
        "source_utility_transfer_override_diagnostics": "source_utility_transfer_override_diagnostics.csv",
        "source_utility_transfer_domain_breakdown": "source_utility_transfer_domain_breakdown.csv",
        "source_utility_transfer_clustered_metrics": "source_utility_transfer_clustered_metrics.csv",
        "source_utility_transfer_random_matched_control": "source_utility_transfer_random_matched_control.csv",
        "source_utility_transfer_negative_controls": "source_utility_transfer_negative_controls.csv",
    }
    _write_csv(reports_dir / artifacts["source_utility_transfer_sample_selections"], source_rows_for_cluster)
    _write_csv(reports_dir / artifacts["source_utility_transfer_pair_predictions"], pair_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_feature_audit"], feature_audit_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_threshold_audit"], threshold_audit_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_override_diagnostics"], override_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_domain_breakdown"], domain_breakdown_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_clustered_metrics"], clustered_metric_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_random_matched_control"], random_control_rows)
    _write_csv(reports_dir / artifacts["source_utility_transfer_negative_controls"], negative_control_rows)

    return SourceUtilityTransferOutputs(
        sample_rows=sample_rows,
        pair_rows=pair_rows,
        feature_audit_rows=feature_audit_rows,
        threshold_audit_rows=threshold_audit_rows,
        override_rows=override_rows,
        domain_breakdown_rows=domain_breakdown_rows,
        clustered_metric_rows=clustered_metric_rows,
        random_control_rows=random_control_rows,
        negative_control_rows=negative_control_rows,
        artifacts=artifacts,
    )
