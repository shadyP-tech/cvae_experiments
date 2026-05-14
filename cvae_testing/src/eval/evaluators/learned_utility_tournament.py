from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import FallbackBenefitGateConfig
from src.eval.evaluators.learned_utility_models import _LinearRegressor
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


@dataclass(frozen=True)
class DeltaGateModel:
    feature_set: str
    feature_names: Tuple[str, ...]
    impute_values: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    regressor: _LinearRegressor


@dataclass(frozen=True)
class DeltaGatePolicySelection:
    base_method: str
    feature_set: str
    threshold: float
    topk: int
    selected_by_inner_validation: bool
    selection_status: str
    diagnostic_only_reason: str = ""
    source_inner_rows: int = 0
    source_inner_validation_domains: int = 0
    source_inner_active_rows: int = 0
    source_inner_active_domains: int = 0
    source_inner_gap_pct: float = float("nan")
    source_inner_paired_gap_reduction_vs_hard: float = float("nan")
    source_inner_high_regret_rate: float = float("nan")
    source_inner_paired_high_regret_reduction_vs_hard: float = float("nan")
    source_inner_activation_rate: float = float("nan")
    source_inner_help_rate_active_only: float = float("nan")
    source_inner_harm_rate_active_only: float = float("nan")
    source_inner_help_rate_all_rows: float = float("nan")
    source_inner_harm_rate_all_rows: float = float("nan")
    source_inner_mean_delta_pct_when_active: float = float("nan")
    source_inner_median_delta_pct_when_active: float = float("nan")
    source_inner_spearman_pred_vs_true_delta: float = float("nan")
    source_inner_auc_help_vs_harm: float = float("nan")
    model: DeltaGateModel | None = None


_LATENT_ONLY_FEATURES: Tuple[str, ...] = (
    "tournament_margin",
    "win_top1",
    "win_top2",
    "score_gap_top1_top2",
    "score_gap_top1_top3",
    "score_range_all_candidates",
    "score_std_all_candidates",
    "topk_score_spread",
    "topk_win_spread",
)
_COMBINED_DIAGNOSTIC_FEATURES: Tuple[str, ...] = (
    *_LATENT_ONLY_FEATURES,
    "latent_combined_top1_agreement",
    "latent_combined_topk_jaccard",
)
_DIAGNOSTIC_REASON_PRIORITY: Tuple[str, ...] = (
    "insufficient_validation_domains",
    "insufficient_active_rows",
    "insufficient_active_domains",
    "activation_rate_too_high",
    "harm_rate_too_high",
    "help_minus_harm_too_low",
    "insufficient_gap_reduction",
)


def delta_gate_feature_names(feature_set: str) -> Tuple[str, ...]:
    name = str(feature_set)
    if name == "tournament_uncertainty_latent_only_v1":
        return _LATENT_ONLY_FEATURES
    if name == "tournament_uncertainty_combined_diagnostic_v1":
        return _COMBINED_DIAGNOSTIC_FEATURES
    raise ValueError(f"Unknown fallback benefit gate feature_set={feature_set!r}")


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


def _safe_finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_finite_median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _binary_auc(score: Sequence[float], label: Sequence[int]) -> float:
    pairs = [
        (float(s), int(y))
        for s, y in zip(score, label)
        if np.isfinite(float(s)) and int(y) in {0, 1}
    ]
    positives = [s for s, y in pairs if y == 1]
    negatives = [s for s, y in pairs if y == 0]
    if not positives or not negatives:
        return float("nan")
    total = 0.0
    correct = 0.0
    for p in positives:
        for n in negatives:
            total += 1.0
            if p > n:
                correct += 1.0
            elif abs(p - n) < 1e-12:
                correct += 0.5
    return float(correct / total)


def _impute_standardize_train_only(x_train: np.ndarray, x_apply: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(x_train, dtype=np.float64)
    if train.ndim != 2:
        raise ValueError("x_train must be 2D")
    impute = np.zeros((train.shape[1],), dtype=np.float64)
    filled_train = train.copy()
    for j in range(train.shape[1]):
        col = train[:, j]
        finite = col[np.isfinite(col)]
        impute[j] = float(np.median(finite)) if finite.size else 0.0
        filled_train[~np.isfinite(filled_train[:, j]), j] = impute[j]
    mean = filled_train.mean(axis=0)
    scale = filled_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_z = (filled_train - mean) / scale
    apply_z = None
    if x_apply is not None:
        apply_arr = np.asarray(x_apply, dtype=np.float64).copy()
        for j in range(apply_arr.shape[1]):
            apply_arr[~np.isfinite(apply_arr[:, j]), j] = impute[j]
        apply_z = (apply_arr - mean) / scale
    return train_z, apply_z, impute, mean, scale


def _apply_delta_gate_model(model: DeltaGateModel, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64).copy()
    if x.ndim != 2 or x.shape[1] != int(model.impute_values.shape[0]):
        raise ProtocolError("Delta gate feature width does not match fitted model")
    for j in range(x.shape[1]):
        x[~np.isfinite(x[:, j]), j] = float(model.impute_values[j])
    x_z = (x - model.feature_mean) / model.feature_scale
    return model.regressor.predict(x_z).astype(np.float64, copy=False)


def _fit_delta_gate_model(
    *,
    features: np.ndarray,
    target_delta_pct_clipped: np.ndarray,
    feature_set: str,
    feature_names: Sequence[str],
    ridge_l2: float,
) -> DeltaGateModel:
    x_z, _unused, impute, mean, scale = _impute_standardize_train_only(np.asarray(features, dtype=np.float64))
    reg = _LinearRegressor(l2=float(ridge_l2))
    reg.fit(x_z, np.asarray(target_delta_pct_clipped, dtype=np.float64))
    return DeltaGateModel(
        feature_set=str(feature_set),
        feature_names=tuple(str(v) for v in feature_names),
        impute_values=impute,
        feature_mean=mean,
        feature_scale=scale,
        regressor=reg,
    )


def delta_gate_feature_matrix(
    *,
    score_matrix: np.ndarray,
    expert_domains: Sequence[int],
    temperature: float,
    topk: int,
    feature_set: str,
    latent_score_matrix: np.ndarray | None = None,
    combined_score_matrix: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    win, orders, margins = tournament_order_and_margin(
        scores,
        expert_domains=expert_domains,
        temperature=float(temperature),
    )
    k = int(scores.shape[1])
    k_eff = min(max(int(topk), 1), k)
    feature_names = delta_gate_feature_names(feature_set)
    out = np.zeros((scores.shape[0], len(feature_names)), dtype=np.float64)

    latent_orders = None
    combined_orders = None
    if "latent_combined_top1_agreement" in feature_names or "latent_combined_topk_jaccard" in feature_names:
        if latent_score_matrix is None or combined_score_matrix is None:
            raise ProtocolError("Combined diagnostic delta-gate features require latent and combined score matrices")
        _lw, latent_orders, _lm = tournament_order_and_margin(
            np.asarray(latent_score_matrix, dtype=np.float64),
            expert_domains=expert_domains,
            temperature=float(temperature),
        )
        _cw, combined_orders, _cm = tournament_order_and_margin(
            np.asarray(combined_score_matrix, dtype=np.float64),
            expert_domains=expert_domains,
            temperature=float(temperature),
        )

    for i in range(scores.shape[0]):
        order = orders[i, :]
        ordered_scores = scores[i, order]
        ordered_wins = win[i, order]
        values = {
            "tournament_margin": float(margins[i]),
            "win_top1": float(ordered_wins[0]),
            "win_top2": float(ordered_wins[1]) if k >= 2 else float("nan"),
            "score_gap_top1_top2": float(ordered_scores[1] - ordered_scores[0]) if k >= 2 else float("nan"),
            "score_gap_top1_top3": float(ordered_scores[2] - ordered_scores[0]) if k >= 3 else float("nan"),
            "score_range_all_candidates": float(np.max(scores[i, :]) - np.min(scores[i, :])),
            "score_std_all_candidates": float(np.std(scores[i, :])),
            "topk_score_spread": float(np.max(ordered_scores[:k_eff]) - np.min(ordered_scores[:k_eff])),
            "topk_win_spread": float(np.max(ordered_wins[:k_eff]) - np.min(ordered_wins[:k_eff])),
        }
        if latent_orders is not None and combined_orders is not None:
            latent_topk = set(int(v) for v in latent_orders[i, :k_eff].tolist())
            combined_topk = set(int(v) for v in combined_orders[i, :k_eff].tolist())
            union = latent_topk | combined_topk
            values["latent_combined_top1_agreement"] = float(
                int(int(latent_orders[i, 0]) == int(combined_orders[i, 0]))
            )
            values["latent_combined_topk_jaccard"] = float(
                len(latent_topk & combined_topk) / len(union) if union else 0.0
            )
        for j, name in enumerate(feature_names):
            out[i, j] = float(values[name])
    return out, win, orders, margins, feature_names


def fallback_delta_pct_arrays(
    *,
    true_nelbo_matrix: np.ndarray,
    tournament_orders: np.ndarray,
    topk: int,
    clip_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    orders = np.asarray(tournament_orders, dtype=np.int64)
    k_eff = min(max(int(topk), 1), int(true.shape[1]))
    top1_idx = orders[:, 0]
    topk_idx = orders[:, :k_eff]
    top1_nelbo = true[np.arange(true.shape[0]), top1_idx]
    topk_nelbo = np.asarray([float(np.mean(true[i, topk_idx[i, :]])) for i in range(true.shape[0])], dtype=np.float64)
    raw = 100.0 * (topk_nelbo - top1_nelbo) / np.maximum(np.abs(top1_nelbo), 1e-12)
    clipped = np.clip(raw, float(clip_bounds[0]), float(clip_bounds[1]))
    return raw.astype(np.float64, copy=False), clipped.astype(np.float64, copy=False), top1_nelbo, topk_nelbo


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
    common_extra_fields: Dict[str, Any] | None = None,
    row_extra_fields: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    if score_matrix.shape != true_nelbo_matrix.shape:
        raise ProtocolError("Tournament score and true NELBO matrix shape mismatch")
    if score_matrix.shape[0] != len(route_indices_by_row):
        raise ProtocolError("Tournament route count does not match score rows")
    if row_extra_fields is not None and len(row_extra_fields) != score_matrix.shape[0]:
        raise ProtocolError("Tournament row extra field count does not match score rows")

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
        if common_extra_fields:
            row.update(common_extra_fields)
        if row_extra_fields is not None:
            row.update(row_extra_fields[i])
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
    common_extra_fields: Dict[str, Any] | None = None,
    row_extra_fields: Sequence[Dict[str, Any]] | None = None,
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
        common_extra_fields=common_extra_fields,
        row_extra_fields=row_extra_fields,
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


def build_delta_gate_calibration_rows(
    *,
    validation_domain: int,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    feature_set: str,
    base_method: str,
    topk: int,
    temperature: float,
    gate_cfg: FallbackBenefitGateConfig,
    latent_score_matrix: np.ndarray | None = None,
    combined_score_matrix: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    features, _win, orders, _margins, feature_names = delta_gate_feature_matrix(
        score_matrix=score_matrix,
        expert_domains=expert_domains,
        temperature=float(temperature),
        topk=int(topk),
        feature_set=str(feature_set),
        latent_score_matrix=latent_score_matrix,
        combined_score_matrix=combined_score_matrix,
    )
    raw_delta, clipped_delta, _top1_nelbo, topk_nelbo = fallback_delta_pct_arrays(
        true_nelbo_matrix=true_nelbo_matrix,
        tournament_orders=orders,
        topk=int(topk),
        clip_bounds=gate_cfg.target_clip_delta_pct,
    )
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    oracle_idx = _stable_argmin_indices(true)
    oracle_nelbo = true[np.arange(true.shape[0]), oracle_idx]
    hard_idx = orders[:, 0]
    hard_nelbo = true[np.arange(true.shape[0]), hard_idx]
    hard_gap_pct = 100.0 * (hard_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)
    topk_gap_pct = 100.0 * (topk_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)
    topk_eff = min(max(int(topk), 1), int(true.shape[1]))

    rows: List[Dict[str, Any]] = []
    for i in range(true.shape[0]):
        topk_set = set(int(v) for v in orders[i, :topk_eff].tolist())
        rows.append(
            {
                "validation_domain": int(validation_domain),
                "query_domain": int(query_domains[i]),
                "base_method": str(base_method),
                "feature_set": str(feature_set),
                "feature_names": feature_names,
                "features": features[i, :].astype(np.float64, copy=True),
                "fallback_delta_pct_raw": float(raw_delta[i]),
                "fallback_delta_pct_clipped_for_training": float(clipped_delta[i]),
                "hard_oracle_gap_pct": float(hard_gap_pct[i]),
                "topk_oracle_gap_pct": float(topk_gap_pct[i]),
                "hard_high_regret_selection": int(float(hard_gap_pct[i]) > HIGH_REGRET_GAP_PCT_THRESHOLD),
                "topk_high_regret_selection": int(float(topk_gap_pct[i]) > HIGH_REGRET_GAP_PCT_THRESHOLD),
                "hard_top1_oracle_hit": int(int(hard_idx[i]) == int(oracle_idx[i])),
                "topk_oracle_in_route_set": int(int(oracle_idx[i]) in topk_set),
            }
        )
    return rows


def _delta_rows_to_arrays(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    x = np.stack([np.asarray(r["features"], dtype=np.float64) for r in rows], axis=0)
    y = np.asarray([float(r["fallback_delta_pct_clipped_for_training"]) for r in rows], dtype=np.float64)
    return x, y


def _fit_delta_gate_model_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    feature_set: str,
    ridge_l2: float,
) -> DeltaGateModel:
    if not rows:
        raise ProtocolError("Cannot fit delta gate on zero source-inner rows")
    x, y = _delta_rows_to_arrays(rows)
    feature_names = tuple(str(v) for v in rows[0]["feature_names"])
    return _fit_delta_gate_model(
        features=x,
        target_delta_pct_clipped=y,
        feature_set=str(feature_set),
        feature_names=feature_names,
        ridge_l2=float(ridge_l2),
    )


def _summarize_delta_gate_evaluations(evaluated_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not evaluated_rows:
        return {
            "n_rows": 0.0,
            "n_validation_domains": 0.0,
            "active_rows": 0.0,
            "active_domains": 0.0,
            "mean_oracle_gap_pct": float("nan"),
            "paired_gap_reduction_vs_hard": float("nan"),
            "high_regret_selection_rate": float("nan"),
            "paired_high_regret_reduction_vs_hard": float("nan"),
            "activation_rate": float("nan"),
            "fallback_help_rate_active_only": float("nan"),
            "fallback_harm_rate_active_only": float("nan"),
            "fallback_help_rate_all_rows": float("nan"),
            "fallback_harm_rate_all_rows": float("nan"),
            "mean_delta_pct_when_active": float("nan"),
            "median_delta_pct_when_active": float("nan"),
            "spearman_pred_vs_true_delta": float("nan"),
            "auc_help_vs_harm": float("nan"),
        }

    active_rows = [r for r in evaluated_rows if int(r["active"]) == 1]
    pred = [float(r["predicted_fallback_delta_pct"]) for r in evaluated_rows]
    true_delta = [float(r["fallback_delta_pct_raw"]) for r in evaluated_rows]
    help_labels = [int(float(r["fallback_delta_pct_raw"]) < 0.0) for r in evaluated_rows]
    spearman = (
        float(spearman_corr(pred, true_delta))
        if len({float(v) for v in pred}) > 1 and len({float(v) for v in true_delta}) > 1
        else float("nan")
    )
    return {
        "n_rows": float(len(evaluated_rows)),
        "n_validation_domains": float(len(set(int(r["validation_domain"]) for r in evaluated_rows))),
        "active_rows": float(len(active_rows)),
        "active_domains": float(len(set(int(r["validation_domain"]) for r in active_rows))),
        "mean_oracle_gap_pct": _safe_finite_mean([float(r["gate_oracle_gap_pct"]) for r in evaluated_rows]),
        "paired_gap_reduction_vs_hard": _safe_finite_mean(
            [float(r["hard_oracle_gap_pct"]) - float(r["gate_oracle_gap_pct"]) for r in evaluated_rows]
        ),
        "high_regret_selection_rate": _safe_finite_mean(
            [float(r["gate_high_regret_selection"]) for r in evaluated_rows]
        ),
        "paired_high_regret_reduction_vs_hard": _safe_finite_mean(
            [
                float(r["hard_high_regret_selection"]) - float(r["gate_high_regret_selection"])
                for r in evaluated_rows
            ]
        ),
        "activation_rate": _safe_finite_mean([float(r["active"]) for r in evaluated_rows]),
        "fallback_help_rate_active_only": _safe_finite_mean(
            [float(r["fallback_help"]) for r in active_rows]
        ),
        "fallback_harm_rate_active_only": _safe_finite_mean(
            [float(r["fallback_harm"]) for r in active_rows]
        ),
        "fallback_help_rate_all_rows": float(
            np.mean(
                [
                    1.0 if int(r["active"]) == 1 and float(r["fallback_delta_pct_raw"]) < 0.0 else 0.0
                    for r in evaluated_rows
                ]
            )
        ),
        "fallback_harm_rate_all_rows": float(
            np.mean(
                [
                    1.0 if int(r["active"]) == 1 and float(r["fallback_delta_pct_raw"]) > 0.0 else 0.0
                    for r in evaluated_rows
                ]
            )
        ),
        "mean_delta_pct_when_active": _safe_finite_mean(
            [float(r["fallback_delta_pct_raw"]) for r in active_rows]
        ),
        "median_delta_pct_when_active": _safe_finite_median(
            [float(r["fallback_delta_pct_raw"]) for r in active_rows]
        ),
        "spearman_pred_vs_true_delta": spearman,
        "auc_help_vs_harm": _binary_auc([-float(v) for v in pred], help_labels),
    }


def _guard_reason(summary: Dict[str, float], gate_cfg: FallbackBenefitGateConfig) -> str:
    checks = {
        "insufficient_validation_domains": float(summary.get("n_validation_domains", 0.0))
        < float(gate_cfg.min_source_inner_validation_domains),
        "insufficient_active_rows": float(summary.get("active_rows", 0.0))
        < float(gate_cfg.min_source_inner_active_rows),
        "insufficient_active_domains": float(summary.get("active_domains", 0.0))
        < float(gate_cfg.min_source_inner_active_domains),
        "activation_rate_too_high": float(summary.get("activation_rate", float("inf")))
        > float(gate_cfg.max_sparse_mix_activation_rate),
        "harm_rate_too_high": float(summary.get("fallback_harm_rate_active_only", float("inf")))
        > float(gate_cfg.max_fallback_harm_rate_active_only),
        "help_minus_harm_too_low": (
            float(summary.get("fallback_help_rate_active_only", 0.0))
            - float(summary.get("fallback_harm_rate_active_only", 0.0))
        )
        < float(gate_cfg.min_fallback_help_minus_harm_active_only),
        "insufficient_gap_reduction": float(summary.get("paired_gap_reduction_vs_hard", -float("inf")))
        < float(gate_cfg.min_source_inner_gap_reduction_pct),
    }
    for reason in _DIAGNOSTIC_REASON_PRIORITY:
        if bool(checks[reason]):
            return reason
    return ""


def _evaluate_delta_gate_candidate(
    *,
    rows: Sequence[Dict[str, Any]],
    threshold: float,
    gate_cfg: FallbackBenefitGateConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    domains = sorted(set(int(r["validation_domain"]) for r in rows))
    evaluated: List[Dict[str, Any]] = []
    for domain in domains:
        train_rows = [r for r in rows if int(r["validation_domain"]) != int(domain)]
        val_rows = [r for r in rows if int(r["validation_domain"]) == int(domain)]
        if not train_rows or not val_rows:
            continue
        model = _fit_delta_gate_model_from_rows(
            train_rows,
            feature_set=str(rows[0]["feature_set"]),
            ridge_l2=float(gate_cfg.ridge_l2),
        )
        x_val, _y_val = _delta_rows_to_arrays(val_rows)
        pred = _apply_delta_gate_model(model, x_val)
        for row, pred_delta in zip(val_rows, pred.tolist()):
            active = int(float(pred_delta) <= float(threshold))
            gate_gap = float(row["topk_oracle_gap_pct"]) if active else float(row["hard_oracle_gap_pct"])
            gate_high = int(row["topk_high_regret_selection"]) if active else int(row["hard_high_regret_selection"])
            raw_delta = float(row["fallback_delta_pct_raw"])
            evaluated.append(
                {
                    **row,
                    "predicted_fallback_delta_pct": float(pred_delta),
                    "active": int(active),
                    "gate_oracle_gap_pct": float(gate_gap),
                    "gate_high_regret_selection": int(gate_high),
                    "fallback_help": int(raw_delta < 0.0) if active else float("nan"),
                    "fallback_harm": int(raw_delta > 0.0) if active else float("nan"),
                }
            )
    return evaluated, _summarize_delta_gate_evaluations(evaluated)


def select_delta_gate_policy(
    *,
    rows_by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]],
    gate_cfg: FallbackBenefitGateConfig,
) -> DeltaGatePolicySelection | None:
    def finite_or(value: float, default: float) -> float:
        return float(value) if np.isfinite(float(value)) else float(default)

    candidates: List[
        Tuple[Tuple[float, float, float, float, float, float], Tuple[str, str, int, float], Dict[str, float], str]
    ] = []
    for (base_method, feature_set, topk), rows in rows_by_key.items():
        if not rows:
            continue
        for threshold in gate_cfg.predicted_delta_pct_thresholds:
            _eval_rows, summary = _evaluate_delta_gate_candidate(
                rows=rows,
                threshold=float(threshold),
                gate_cfg=gate_cfg,
            )
            reason = _guard_reason(summary, gate_cfg)
            score = (
                -finite_or(summary.get("mean_oracle_gap_pct", float("inf")), float("inf")),
                finite_or(summary.get("paired_gap_reduction_vs_hard", -float("inf")), -float("inf")),
                -finite_or(summary.get("high_regret_selection_rate", float("inf")), float("inf")),
                -finite_or(summary.get("fallback_harm_rate_active_only", float("inf")), float("inf")),
                finite_or(summary.get("fallback_help_rate_active_only", -float("inf")), -float("inf")),
                -finite_or(summary.get("activation_rate", float("inf")), float("inf")),
            )
            candidates.append((score, (str(base_method), str(feature_set), int(topk), float(threshold)), summary, reason))
    if not candidates:
        return None

    ordered_candidates = sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    )
    selected_candidate = next((item for item in ordered_candidates if not item[3]), ordered_candidates[0])
    _score, (base_method, feature_set, topk, threshold), summary, reason = selected_candidate
    rows = rows_by_key[(base_method, feature_set, topk)]
    status = "selected"
    model: DeltaGateModel | None = None
    if reason:
        status = "insufficient_evidence_noop" if str(reason).startswith("insufficient_") else "failed_guards_noop"
    else:
        model = _fit_delta_gate_model_from_rows(
            rows,
            feature_set=str(feature_set),
            ridge_l2=float(gate_cfg.ridge_l2),
        )

    return DeltaGatePolicySelection(
        base_method=str(base_method),
        feature_set=str(feature_set),
        threshold=float(threshold),
        topk=int(topk),
        selected_by_inner_validation=True,
        selection_status=str(status),
        diagnostic_only_reason=str(reason),
        source_inner_rows=int(summary.get("n_rows", 0.0)),
        source_inner_validation_domains=int(summary.get("n_validation_domains", 0.0)),
        source_inner_active_rows=int(summary.get("active_rows", 0.0)),
        source_inner_active_domains=int(summary.get("active_domains", 0.0)),
        source_inner_gap_pct=float(summary.get("mean_oracle_gap_pct", float("nan"))),
        source_inner_paired_gap_reduction_vs_hard=float(
            summary.get("paired_gap_reduction_vs_hard", float("nan"))
        ),
        source_inner_high_regret_rate=float(summary.get("high_regret_selection_rate", float("nan"))),
        source_inner_paired_high_regret_reduction_vs_hard=float(
            summary.get("paired_high_regret_reduction_vs_hard", float("nan"))
        ),
        source_inner_activation_rate=float(summary.get("activation_rate", float("nan"))),
        source_inner_help_rate_active_only=float(summary.get("fallback_help_rate_active_only", float("nan"))),
        source_inner_harm_rate_active_only=float(summary.get("fallback_harm_rate_active_only", float("nan"))),
        source_inner_help_rate_all_rows=float(summary.get("fallback_help_rate_all_rows", float("nan"))),
        source_inner_harm_rate_all_rows=float(summary.get("fallback_harm_rate_all_rows", float("nan"))),
        source_inner_mean_delta_pct_when_active=float(summary.get("mean_delta_pct_when_active", float("nan"))),
        source_inner_median_delta_pct_when_active=float(summary.get("median_delta_pct_when_active", float("nan"))),
        source_inner_spearman_pred_vs_true_delta=float(summary.get("spearman_pred_vs_true_delta", float("nan"))),
        source_inner_auc_help_vs_harm=float(summary.get("auc_help_vs_harm", float("nan"))),
        model=model,
    )


def delta_gate_route_rows(
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
    selection: DeltaGatePolicySelection,
    temperature: float,
    temperature_policy: str,
    gate_cfg: FallbackBenefitGateConfig,
    diagnostic_only_reason: str = "",
    latent_score_matrix: np.ndarray | None = None,
    combined_score_matrix: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    features, _win, orders, margins, _feature_names = delta_gate_feature_matrix(
        score_matrix=score_matrix,
        expert_domains=expert_domains,
        temperature=float(temperature),
        topk=int(selection.topk),
        feature_set=str(selection.feature_set),
        latent_score_matrix=latent_score_matrix,
        combined_score_matrix=combined_score_matrix,
    )
    raw_delta, clipped_delta, _top1_nelbo, _topk_nelbo = fallback_delta_pct_arrays(
        true_nelbo_matrix=true_nelbo_matrix,
        tournament_orders=orders,
        topk=int(selection.topk),
        clip_bounds=gate_cfg.target_clip_delta_pct,
    )
    if selection.model is not None and selection.selection_status == "selected":
        pred_delta = _apply_delta_gate_model(selection.model, features)
        active = (pred_delta <= float(selection.threshold)).astype(np.int64)
    else:
        pred_delta = np.full((score_matrix.shape[0],), float("nan"), dtype=np.float64)
        active = np.zeros((score_matrix.shape[0],), dtype=np.int64)

    k_eff = min(max(int(selection.topk), 1), int(score_matrix.shape[1]))
    route_indices: List[List[int]] = []
    for i in range(score_matrix.shape[0]):
        route_indices.append([int(v) for v in orders[i, : (k_eff if int(active[i]) else 1)].tolist()])

    hard_route_rows = _route_rows(
        method=method,
        fold=fold,
        query_domains=query_domains,
        expert_domains=expert_domains,
        score_matrix=score_matrix,
        true_nelbo_matrix=true_nelbo_matrix,
        global_true_nelbo_matrix=global_true_nelbo_matrix,
        global_expert_domains=global_expert_domains,
        route_indices_by_row=[[int(v)] for v in orders[:, 0].tolist()],
        tournament_orders=orders,
        tournament_margins=margins,
        policy_name=policy_name,
        base_method=selection.base_method,
        threshold=float(selection.threshold),
        topk=1,
        temperature=float(temperature),
        temperature_policy=temperature_policy,
        sparse_mix_active=[0 for _ in range(score_matrix.shape[0])],
        selected_by_inner_validation=selection.selected_by_inner_validation,
        threshold_selection_policy=gate_cfg.calibration_policy,
    )
    hard_gap_pct = np.asarray([float(r["oracle_gap_pct"]) for r in hard_route_rows], dtype=np.float64)
    hard_high_regret = np.asarray([int(r["high_regret_selection"]) for r in hard_route_rows], dtype=np.int64)

    row_extras: List[Dict[str, Any]] = []
    for i in range(score_matrix.shape[0]):
        row_extras.append(
            {
                "predicted_fallback_delta_pct": float(pred_delta[i]),
                "fallback_delta_pct_raw": float(raw_delta[i]),
                "fallback_delta_pct_clipped_for_training": float(clipped_delta[i]),
                "delta_gate_selection_status": str(selection.selection_status),
                "delta_gate_active": int(active[i]),
                "delta_gate_threshold": float(selection.threshold),
                "delta_gate_feature_set": str(selection.feature_set),
                "delta_gate_source_inner_gap_pct": float(selection.source_inner_gap_pct),
                "delta_gate_source_inner_paired_gap_reduction_vs_hard": float(
                    selection.source_inner_paired_gap_reduction_vs_hard
                ),
                "delta_gate_source_inner_activation_rate": float(selection.source_inner_activation_rate),
                "delta_gate_source_inner_harm_rate_active_only": float(
                    selection.source_inner_harm_rate_active_only
                ),
                "delta_gate_source_inner_help_rate_active_only": float(
                    selection.source_inner_help_rate_active_only
                ),
                "delta_gate_spearman_pred_vs_true_delta_source_inner": float(
                    selection.source_inner_spearman_pred_vs_true_delta
                ),
                "delta_gate_auc_help_vs_harm_source_inner": float(selection.source_inner_auc_help_vs_harm),
                "delta_gate_diagnostic_only_reason": str(selection.diagnostic_only_reason),
                "hard_oracle_gap_pct": float(hard_gap_pct[i]),
                "hard_high_regret_selection": int(hard_high_regret[i]),
            }
        )

    reason = str(diagnostic_only_reason or selection.diagnostic_only_reason)
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
        base_method=selection.base_method,
        threshold=float(selection.threshold),
        topk=int(selection.topk),
        temperature=float(temperature),
        temperature_policy=temperature_policy,
        sparse_mix_active=active.tolist(),
        selected_by_inner_validation=selection.selected_by_inner_validation,
        threshold_selection_policy=gate_cfg.calibration_policy,
        diagnostic_only_reason=reason,
        row_extra_fields=row_extras,
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
