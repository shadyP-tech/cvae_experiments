from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import (
    _pairwise_auc_single,
    _selected_rank_in_true_matrix,
    _stable_argmin_indices,
)
from src.eval.metrics import spearman_corr


HIGH_REGRET_GAP_PCT_THRESHOLD = 2.0


@dataclass(frozen=True)
class TournamentPolicySelection:
    base_method: str
    threshold: float
    topk: int
    selected_by_inner_validation: bool
    diagnostic_only_reason: str = ""
    source_inner_rows: int = 0
    source_inner_gap_pct: float = float("nan")
    source_inner_high_regret_rate: float = float("nan")
    source_inner_oracle_in_route_set: float = float("nan")
    source_inner_top1: float = float("nan")
    source_inner_sparse_mix_rate: float = float("nan")


def tournament_win_scores(score_matrix: np.ndarray, *, temperature: float) -> np.ndarray:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be 2D")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")
    n, k = scores.shape
    if k <= 0:
        raise ValueError("score_matrix must have at least one candidate column")
    if k == 1:
        return np.ones((n, 1), dtype=np.float64)

    diff = (scores[:, None, :] - scores[:, :, None]) / float(temperature)
    # diff[i, a, b] = score_b - score_a. Lower score_a means better predicted NELBO.
    wins = 1.0 / (1.0 + np.exp(-diff))
    mask = ~np.eye(k, dtype=bool)
    return wins[:, mask].reshape(n, k, k - 1).mean(axis=2)


def tournament_order_and_margin(
    score_matrix: np.ndarray,
    *,
    expert_domains: Sequence[int],
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = tournament_win_scores(score_matrix, temperature=float(temperature))
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    if win.shape[1] != experts.shape[0]:
        raise ProtocolError("Tournament score width does not match expert_domains")

    orders = np.zeros_like(win, dtype=np.int64)
    margins = np.zeros((win.shape[0],), dtype=np.float64)
    for i in range(win.shape[0]):
        # Higher win score is better; ties break by smaller expert-domain label.
        order = np.lexsort((experts, -win[i, :]))
        orders[i, :] = order
        margins[i] = float(win[i, order[0]] - win[i, order[1]]) if win.shape[1] > 1 else float("inf")
    return win, orders, margins


def _route_strings(experts: Sequence[int], weights: Sequence[float]) -> Tuple[str, str]:
    return "|".join(str(int(v)) for v in experts), "|".join(f"{float(w):.12g}" for w in weights)


def _route_rows(
    *,
    method: str,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    route_indices_by_row: Sequence[Sequence[int]],
    tournament_orders: np.ndarray,
    tournament_margins: np.ndarray,
    policy_name: str,
    base_method: str,
    threshold: float,
    topk: int,
    temperature: float,
    temperature_policy: str,
    sparse_mix_active: Sequence[int],
    selected_by_inner_validation: bool,
    threshold_selection_policy: str,
    diagnostic_only_reason: str = "",
    source_inner_summary: TournamentPolicySelection | None = None,
) -> List[Dict[str, Any]]:
    if score_matrix.shape != true_nelbo_matrix.shape:
        raise ProtocolError("Tournament score and true NELBO matrix shape mismatch")
    if score_matrix.shape[0] != len(route_indices_by_row):
        raise ProtocolError("Tournament route count does not match score rows")

    method_protocol = _method_protocol(method)
    oracle_idx = _stable_argmin_indices(true_nelbo_matrix)
    global_oracle_idx = _stable_argmin_indices(global_true_nelbo_matrix)
    selected_idx = tournament_orders[:, 0].astype(np.int64, copy=False)
    selected_rank = _selected_rank_in_true_matrix(selected_idx, true_nelbo_matrix)
    rank_metrics_valid = int(score_matrix.shape[1]) >= 2

    rows: List[Dict[str, Any]] = []
    expert_domains_int = [int(v) for v in expert_domains]
    global_expert_domains_int = [int(v) for v in global_expert_domains]
    source = source_inner_summary
    for i in range(score_matrix.shape[0]):
        route_idx = [int(v) for v in route_indices_by_row[i]]
        if not route_idx:
            raise ProtocolError("Tournament route set is empty")
        route_experts = [expert_domains_int[j] for j in route_idx]
        route_weights = [1.0 / float(len(route_idx)) for _ in route_idx]

        top1_idx = int(selected_idx[i])
        top1_nelbo = float(true_nelbo_matrix[i, top1_idx])
        route_nelbos = true_nelbo_matrix[i, route_idx].astype(np.float64, copy=False)
        routed_nelbo = float(np.mean(route_nelbos))
        oracle_nelbo = float(true_nelbo_matrix[i, int(oracle_idx[i])])
        gap = float(routed_nelbo - oracle_nelbo)
        gap_pct = float((gap / max(abs(oracle_nelbo), 1e-12)) * 100.0)
        fallback_delta = float(routed_nelbo - top1_nelbo) if int(sparse_mix_active[i]) else float("nan")

        pair_auc = (
            float(_pairwise_auc_single(score_matrix[i, :], true_nelbo_matrix[i, :]))
            if rank_metrics_valid
            else float("nan")
        )
        rho = (
            float(spearman_corr((-score_matrix[i, :]).tolist(), (-true_nelbo_matrix[i, :]).tolist()))
            if rank_metrics_valid
            else float("nan")
        )

        protocol_fields = _protocol_row_fields(fold=fold, method_protocol=method_protocol, method=method)
        if diagnostic_only_reason:
            protocol_fields.update(
                {
                    "method_role": "diagnostic",
                    "adoption_eligible": 0,
                    "diagnostic_only": 1,
                    "routing_uses_eval_nelbo": int(method == "oracle_confidence_set_diagnostic"),
                }
            )

        row = {
            **protocol_fields,
            "sample_index": int(i),
            "query_domain": int(query_domains[i]),
            "selected_expert": int(expert_domains_int[top1_idx]),
            "candidate_oracle_expert": int(expert_domains_int[int(oracle_idx[i])]),
            "candidate_oracle_nelbo": oracle_nelbo,
            "global_oracle_expert": int(global_expert_domains_int[int(global_oracle_idx[i])]),
            "global_oracle_nelbo": float(global_true_nelbo_matrix[i, int(global_oracle_idx[i])]),
            "global_oracle_excluded_by_policy": int(
                not fold.contains(int(global_expert_domains_int[int(global_oracle_idx[i])]))
            ),
            "oracle_expert": int(expert_domains_int[int(oracle_idx[i])]),
            "selected_nelbo": routed_nelbo,
            "oracle_nelbo": oracle_nelbo,
            "oracle_gap": gap,
            "oracle_gap_pct": gap_pct,
            "top1_oracle_hit": int(top1_idx == int(oracle_idx[i])),
            "selected_rank": float(selected_rank[i]),
            "pairwise_auc": pair_auc,
            "spearman": rho,
            "rank_metrics_valid": int(rank_metrics_valid),
            "policy_name": str(policy_name),
            "base_method": str(base_method),
            "selected_tau": float(threshold),
            "sparse_mix_topk": int(topk),
            "score_temperature": float(temperature),
            "temperature_policy": str(temperature_policy),
            "threshold_selection_policy": str(threshold_selection_policy),
            "selected_by_inner_validation": int(bool(selected_by_inner_validation)),
            "route_experts": _route_strings(route_experts, route_weights)[0],
            "route_weights": _route_strings(route_experts, route_weights)[1],
            "route_size": int(len(route_idx)),
            "route_mode": "sparse_mix_uniform" if int(sparse_mix_active[i]) else "hard_top1",
            "tournament_margin": float(tournament_margins[i]),
            "sparse_mix_active": int(sparse_mix_active[i]),
            "oracle_in_route_set": int(int(oracle_idx[i]) in set(route_idx)),
            "mean_nelbo_spread_in_route_set": float(np.max(route_nelbos) - np.min(route_nelbos)),
            "route_set_regret": gap,
            "top1_nelbo": top1_nelbo,
            "fallback_delta": fallback_delta,
            "fallback_help": int(fallback_delta < 0.0) if np.isfinite(fallback_delta) else float("nan"),
            "fallback_harm": int(fallback_delta > 0.0) if np.isfinite(fallback_delta) else float("nan"),
            "high_regret_selection": int(gap_pct > HIGH_REGRET_GAP_PCT_THRESHOLD),
            "bottom_half_selection": int(float(selected_rank[i]) > float((score_matrix.shape[1] + 1) / 2.0)),
            "catastrophic_mistake": int(gap_pct > 10.0),
            "diagnostic_only_reason": str(diagnostic_only_reason),
        }
        if source is not None:
            row.update(
                {
                    "source_inner_rows": int(source.source_inner_rows),
                    "source_inner_gap_pct": float(source.source_inner_gap_pct),
                    "source_inner_high_regret_rate": float(source.source_inner_high_regret_rate),
                    "source_inner_oracle_in_route_set": float(source.source_inner_oracle_in_route_set),
                    "source_inner_top1": float(source.source_inner_top1),
                    "source_inner_sparse_mix_rate": float(source.source_inner_sparse_mix_rate),
                }
            )
        rows.append(row)
    return rows


def tournament_route_rows(
    *,
    method: str,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    policy_name: str,
    base_method: str,
    threshold: float,
    topk: int,
    temperature: float,
    temperature_policy: str,
    selected_by_inner_validation: bool,
    threshold_selection_policy: str,
    diagnostic_only_reason: str = "",
    source_inner_summary: TournamentPolicySelection | None = None,
) -> List[Dict[str, Any]]:
    _win, orders, margins = tournament_order_and_margin(
        score_matrix,
        expert_domains=expert_domains,
        temperature=float(temperature),
    )
    k_eff = min(max(int(topk), 1), int(score_matrix.shape[1]))
    route_indices: List[List[int]] = []
    active: List[int] = []
    for i in range(score_matrix.shape[0]):
        use_sparse = bool(float(margins[i]) < float(threshold) and k_eff > 1)
        active.append(int(use_sparse))
        route_indices.append([int(v) for v in orders[i, : (k_eff if use_sparse else 1)].tolist()])

    return _route_rows(
        method=method,
        fold=fold,
        query_domains=query_domains,
        expert_domains=expert_domains,
        score_matrix=score_matrix,
        true_nelbo_matrix=true_nelbo_matrix,
        global_true_nelbo_matrix=global_true_nelbo_matrix,
        global_expert_domains=global_expert_domains,
        route_indices_by_row=route_indices,
        tournament_orders=orders,
        tournament_margins=margins,
        policy_name=policy_name,
        base_method=base_method,
        threshold=float(threshold),
        topk=k_eff,
        temperature=float(temperature),
        temperature_policy=temperature_policy,
        sparse_mix_active=active,
        selected_by_inner_validation=selected_by_inner_validation,
        threshold_selection_policy=threshold_selection_policy,
        diagnostic_only_reason=diagnostic_only_reason,
        source_inner_summary=source_inner_summary,
    )


def oracle_confidence_set_rows(
    *,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    policy_name: str,
    base_method: str,
    topk: int,
    temperature: float,
    temperature_policy: str,
) -> List[Dict[str, Any]]:
    _win, orders, margins = tournament_order_and_margin(
        score_matrix,
        expert_domains=expert_domains,
        temperature=float(temperature),
    )
    k_eff = min(max(int(topk), 1), int(score_matrix.shape[1]))
    route_indices: List[List[int]] = []
    active: List[int] = []
    for i in range(score_matrix.shape[0]):
        hard_idx = int(orders[i, 0])
        sparse_idx = [int(v) for v in orders[i, :k_eff].tolist()]
        hard_nelbo = float(true_nelbo_matrix[i, hard_idx])
        sparse_nelbo = float(np.mean(true_nelbo_matrix[i, sparse_idx]))
        use_sparse = bool(k_eff > 1 and sparse_nelbo < hard_nelbo)
        active.append(int(use_sparse))
        route_indices.append(sparse_idx if use_sparse else [hard_idx])

    return _route_rows(
        method="oracle_confidence_set_diagnostic",
        fold=fold,
        query_domains=query_domains,
        expert_domains=expert_domains,
        score_matrix=score_matrix,
        true_nelbo_matrix=true_nelbo_matrix,
        global_true_nelbo_matrix=global_true_nelbo_matrix,
        global_expert_domains=global_expert_domains,
        route_indices_by_row=route_indices,
        tournament_orders=orders,
        tournament_margins=margins,
        policy_name=policy_name,
        base_method=base_method,
        threshold=float("nan"),
        topk=k_eff,
        temperature=float(temperature),
        temperature_policy=temperature_policy,
        sparse_mix_active=active,
        selected_by_inner_validation=False,
        threshold_selection_policy="heldout_true_nelbo_oracle_gate_diagnostic",
        diagnostic_only_reason="oracle_confidence_set_diagnostic",
    )


def summarize_tournament_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "top1_oracle_hit": float("nan"),
            "mean_oracle_gap_pct": float("nan"),
            "high_regret_selection_rate": float("nan"),
            "oracle_in_route_set": float("nan"),
            "sparse_mix_active": float("nan"),
        }
    return {
        "n_rows": float(len(rows)),
        "top1_oracle_hit": float(np.mean([float(r["top1_oracle_hit"]) for r in rows])),
        "mean_oracle_gap_pct": float(np.mean([float(r["oracle_gap_pct"]) for r in rows])),
        "high_regret_selection_rate": float(np.mean([float(r.get("high_regret_selection", 0.0)) for r in rows])),
        "oracle_in_route_set": float(np.mean([float(r.get("oracle_in_route_set", 0.0)) for r in rows])),
        "sparse_mix_active": float(np.mean([float(r.get("sparse_mix_active", 0.0)) for r in rows])),
    }
