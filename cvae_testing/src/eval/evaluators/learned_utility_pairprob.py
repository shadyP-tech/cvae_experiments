from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import PairprobTournamentConfig
from src.eval.evaluators.learned_utility_models import _LogisticRidgePairprob
from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet, ProtocolError
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.eval.evaluators.learned_utility_pairs import _zscore_features


@dataclass(frozen=True)
class PairprobTrainingData:
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    query_domains: np.ndarray
    total_pairs: int
    dropped_near_tie: int
    kept_by_domain: Dict[int, int]


@dataclass(frozen=True)
class PairprobModelBundle:
    feature_set: str
    ridge_l2: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    model: _LogisticRidgePairprob


@dataclass(frozen=True)
class PairprobPolicySelection:
    method: str
    feature_set: str
    ridge_l2: float
    selected_by_inner_validation: bool
    diagnostic_only_reason: str = ""
    source_inner_validation_domains: int = 0
    source_inner_rows: int = 0
    source_inner_mean_oracle_gap_pct: float = float("nan")
    source_inner_worst_domain_oracle_gap_pct: float = float("nan")
    source_inner_relative_catastrophic_rate: float = float("nan")
    source_inner_absolute_high_regret_rate: float = float("nan")
    source_inner_top1: float = float("nan")
    source_inner_spearman: float = float("nan")
    source_inner_std_oracle_gap_pct: float = float("nan")
    source_inner_std_top1: float = float("nan")
    source_inner_max_minus_min_oracle_gap_pct: float = float("nan")
    pairwise_near_tie_drop_rate: float = float("nan")
    pairwise_train_pairs_after_filter: int = 0
    pairwise_validation_pairs_after_filter: int = 0
    pairwise_train_domains_after_filter: int = 0


def pairprob_feature_names(feature_set: str, *, embedding_dim: int, expert_feature_dim: int, metadata_dim: int) -> Tuple[str, ...]:
    name = str(feature_set)
    if name not in {"pairprob_latent_only_v1", "pairprob_combined_diagnostic_v1"}:
        raise ValueError(f"Unknown pairprob feature_set={feature_set!r}")

    names: List[str] = []
    names.extend(f"query_embedding_{i}" for i in range(int(embedding_dim)))
    names.extend(f"expert_a_identity_{i}" for i in range(int(expert_feature_dim)))
    names.extend(f"expert_b_identity_{i}" for i in range(int(expert_feature_dim)))
    names.extend(f"query_by_expert_a_{i}_{j}" for i in range(int(embedding_dim)) for j in range(int(expert_feature_dim)))
    names.extend(f"query_by_expert_b_{i}_{j}" for i in range(int(embedding_dim)) for j in range(int(expert_feature_dim)))
    names.extend(f"expert_identity_signed_diff_{i}" for i in range(int(expert_feature_dim)))
    names.extend(f"expert_identity_abs_diff_{i}" for i in range(int(expert_feature_dim)))
    if name == "pairprob_combined_diagnostic_v1":
        names.extend(f"expert_a_metadata_{i}" for i in range(int(metadata_dim)))
        names.extend(f"expert_b_metadata_{i}" for i in range(int(metadata_dim)))
        names.extend(f"metadata_signed_diff_{i}" for i in range(int(metadata_dim)))
        names.extend(f"metadata_abs_diff_{i}" for i in range(int(metadata_dim)))
    return tuple(names)


def _pair_feature(
    row_a: np.ndarray,
    row_b: np.ndarray,
    *,
    embedding_dim: int,
    expert_feature_dim: int,
    feature_set: str,
) -> np.ndarray:
    name = str(feature_set)
    query = np.asarray(row_a[:embedding_dim], dtype=np.float64)
    expert_a = np.asarray(row_a[embedding_dim : embedding_dim + expert_feature_dim], dtype=np.float64)
    expert_b = np.asarray(row_b[embedding_dim : embedding_dim + expert_feature_dim], dtype=np.float64)
    interaction_a = (query[:, None] * expert_a[None, :]).reshape(-1)
    interaction_b = (query[:, None] * expert_b[None, :]).reshape(-1)
    parts = [
        query,
        expert_a,
        expert_b,
        interaction_a,
        interaction_b,
        expert_a - expert_b,
        np.abs(expert_a - expert_b),
    ]
    if name == "pairprob_combined_diagnostic_v1":
        meta_a = np.asarray(row_a[embedding_dim + expert_feature_dim :], dtype=np.float64)
        meta_b = np.asarray(row_b[embedding_dim + expert_feature_dim :], dtype=np.float64)
        parts.extend([meta_a, meta_b, meta_a - meta_b, np.abs(meta_a - meta_b)])
    elif name != "pairprob_latent_only_v1":
        raise ValueError(f"Unknown pairprob feature_set={feature_set!r}")
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def build_pairprob_training_data(
    *,
    x_rows: np.ndarray,
    q_rows: np.ndarray,
    e_rows: np.ndarray,
    s_rows: np.ndarray,
    y_rows: np.ndarray,
    embedding_dim: int,
    expert_feature_dim: int,
    feature_set: str,
    near_tie_delta_pct: float,
    margin_weight_scale_pct: float,
    margin_weight_clip: Tuple[float, float],
) -> PairprobTrainingData:
    features: List[np.ndarray] = []
    labels: List[float] = []
    weights: List[float] = []
    query_domains: List[int] = []
    total_pairs = 0
    dropped = 0
    kept_by_domain: Dict[int, int] = {}

    for sample_index in sorted(set(int(v) for v in np.asarray(s_rows, dtype=np.int64).tolist())):
        idxs = np.where(np.asarray(s_rows, dtype=np.int64) == int(sample_index))[0]
        if idxs.size < 2:
            continue
        ordered = sorted([int(idx) for idx in idxs.tolist()], key=lambda idx: int(e_rows[idx]))
        query_domain = int(q_rows[ordered[0]])
        for pos_a in range(len(ordered)):
            for pos_b in range(pos_a + 1, len(ordered)):
                ia = int(ordered[pos_a])
                ib = int(ordered[pos_b])
                domain_a = int(e_rows[ia])
                domain_b = int(e_rows[ib])
                if domain_a >= domain_b:
                    raise ProtocolError("Pair-prob canonical expert order must be ascending by domain")
                ya = float(y_rows[ia])
                yb = float(y_rows[ib])
                denom = max(abs(min(ya, yb)), 1e-12)
                delta_pct = 100.0 * abs(ya - yb) / denom
                total_pairs += 1
                if delta_pct < float(near_tie_delta_pct):
                    dropped += 1
                    continue
                features.append(
                    _pair_feature(
                        np.asarray(x_rows[ia], dtype=np.float64),
                        np.asarray(x_rows[ib], dtype=np.float64),
                        embedding_dim=int(embedding_dim),
                        expert_feature_dim=int(expert_feature_dim),
                        feature_set=str(feature_set),
                    )
                )
                labels.append(1.0 if ya < yb else 0.0)
                low, high = float(margin_weight_clip[0]), float(margin_weight_clip[1])
                weights.append(float(np.clip(delta_pct / float(margin_weight_scale_pct), low, high)))
                query_domains.append(int(query_domain))
                kept_by_domain[query_domain] = int(kept_by_domain.get(query_domain, 0)) + 1

    x = np.vstack(features).astype(np.float64, copy=False) if features else np.zeros((0, 0), dtype=np.float64)
    return PairprobTrainingData(
        x=x,
        y=np.asarray(labels, dtype=np.float64),
        weight=np.asarray(weights, dtype=np.float64),
        query_domains=np.asarray(query_domains, dtype=np.int64),
        total_pairs=int(total_pairs),
        dropped_near_tie=int(dropped),
        kept_by_domain=kept_by_domain,
    )


def pairprob_evidence_reason(
    *,
    train_data: PairprobTrainingData,
    validation_data: PairprobTrainingData | None,
    validation_domains: int,
    cfg: PairprobTournamentConfig,
) -> str:
    if int(validation_domains) < int(cfg.min_source_inner_validation_domains):
        return "insufficient_pairwise_evidence"
    if int(train_data.x.shape[0]) < int(cfg.min_pairwise_train_pairs):
        return "insufficient_pairwise_evidence"
    if validation_data is not None and int(validation_data.x.shape[0]) < int(cfg.min_pairwise_validation_pairs):
        return "insufficient_pairwise_evidence"
    if len(train_data.kept_by_domain) < 1:
        return "insufficient_pairwise_evidence"
    if any(int(v) < int(cfg.min_non_tie_pairs_per_inner_domain) for v in train_data.kept_by_domain.values()):
        return "insufficient_pairwise_evidence"
    return ""


def fit_pairprob_model(
    *,
    train_data: PairprobTrainingData,
    feature_set: str,
    ridge_l2: float,
    device: str,
) -> PairprobModelBundle:
    if train_data.x.shape[0] <= 0:
        raise ProtocolError("Cannot fit pair-prob model without training pairs")
    x_z, _x_unused = _zscore_features(train_data.x, train_data.x)
    mean = train_data.x.mean(axis=0)
    scale = train_data.x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    clf = _LogisticRidgePairprob(l2=float(ridge_l2), device=str(device))
    clf.fit(x_z, train_data.y, train_data.weight)
    return PairprobModelBundle(
        feature_set=str(feature_set),
        ridge_l2=float(ridge_l2),
        feature_mean=mean.astype(np.float64, copy=False),
        feature_scale=scale.astype(np.float64, copy=False),
        model=clf,
    )


def _apply_pairprob_model(bundle: PairprobModelBundle, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    z = (arr - bundle.feature_mean) / bundle.feature_scale
    return bundle.model.predict_proba(z)


def pairprob_probability_matrix(
    *,
    bundle: PairprobModelBundle,
    x_rows: np.ndarray,
    expert_domains: Sequence[int],
    embedding_dim: int,
    expert_feature_dim: int,
) -> np.ndarray:
    expert_domains_int = [int(v) for v in expert_domains]
    k = len(expert_domains_int)
    if k <= 0:
        raise ProtocolError("Pair-prob routing requires at least one candidate expert")
    if x_rows.shape[0] % k != 0:
        raise ProtocolError("Pair-prob feature rows are not divisible by candidate expert count")
    n = int(x_rows.shape[0] // k)
    probs = np.full((n, k, k), 0.5, dtype=np.float64)
    if k == 1:
        return probs

    feature_rows: List[np.ndarray] = []
    pair_refs: List[Tuple[int, int, int]] = []
    for row_idx in range(n):
        base = row_idx * k
        for a in range(k):
            for b in range(a + 1, k):
                if expert_domains_int[a] >= expert_domains_int[b]:
                    raise ProtocolError("Pair-prob candidate experts must be sorted ascending by domain")
                feature_rows.append(
                    _pair_feature(
                        np.asarray(x_rows[base + a], dtype=np.float64),
                        np.asarray(x_rows[base + b], dtype=np.float64),
                        embedding_dim=int(embedding_dim),
                        expert_feature_dim=int(expert_feature_dim),
                        feature_set=str(bundle.feature_set),
                    )
                )
                pair_refs.append((row_idx, a, b))
    pred = _apply_pairprob_model(bundle, np.vstack(feature_rows))
    for p, (row_idx, a, b) in zip(pred.tolist(), pair_refs):
        prob = float(np.clip(p, 0.0, 1.0))
        probs[row_idx, a, b] = prob
        probs[row_idx, b, a] = 1.0 - prob
    return probs


def pairprob_win_scores(prob_matrix: np.ndarray) -> np.ndarray:
    probs = np.asarray(prob_matrix, dtype=np.float64)
    if probs.ndim != 3 or probs.shape[1] != probs.shape[2]:
        raise ValueError("prob_matrix must have shape (n, k, k)")
    n, k, _ = probs.shape
    if k == 1:
        return np.ones((n, 1), dtype=np.float64)
    mask = ~np.eye(k, dtype=bool)
    return probs[:, mask].reshape(n, k, k - 1).mean(axis=2)


def pairprob_order_and_margin(
    prob_matrix: np.ndarray,
    *,
    expert_domains: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    win = pairprob_win_scores(prob_matrix)
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    if win.shape[1] != experts.shape[0]:
        raise ProtocolError("Pair-prob win width does not match expert_domains")
    orders = np.zeros_like(win, dtype=np.int64)
    margins = np.zeros((win.shape[0],), dtype=np.float64)
    for i in range(win.shape[0]):
        order = np.lexsort((experts, -win[i, :]))
        orders[i, :] = order
        margins[i] = float(win[i, order[0]] - win[i, order[1]]) if win.shape[1] > 1 else float("inf")
    return win, orders, margins


def _pairprob_cycle_rate_for_row(prob: np.ndarray) -> float:
    k = int(prob.shape[0])
    if k < 3:
        return float("nan")
    total = 0.0
    cycles = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            for c in range(b + 1, k):
                ab = float(prob[a, b]) > 0.5
                bc = float(prob[b, c]) > 0.5
                ca = float(prob[c, a]) > 0.5
                ba = float(prob[b, a]) > 0.5
                cb = float(prob[c, b]) > 0.5
                ac = float(prob[a, c]) > 0.5
                total += 1.0
                if (ab and bc and ca) or (ba and cb and ac):
                    cycles += 1.0
    return float(cycles / total) if total > 0.0 else float("nan")


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


def _pair_diagnostics_for_row(prob: np.ndarray, true_nelbo: np.ndarray) -> Dict[str, float]:
    k = int(prob.shape[0])
    pair_probs: List[float] = []
    labels: List[int] = []
    confidences: List[float] = []
    for a in range(k):
        for b in range(a + 1, k):
            p = float(prob[a, b])
            y = 1 if float(true_nelbo[a]) < float(true_nelbo[b]) else 0
            pair_probs.append(p)
            labels.append(y)
            confidences.append(abs(p - 0.5) * 2.0)
    if not pair_probs:
        return {
            "pairwise_cycle_rate": float("nan"),
            "mean_pairwise_confidence": float("nan"),
            "pairwise_calibration_brier": float("nan"),
            "pairwise_auc_helpful_preferences": float("nan"),
        }
    brier = float(np.mean([(p - y) ** 2 for p, y in zip(pair_probs, labels)]))
    return {
        "pairwise_cycle_rate": _pairprob_cycle_rate_for_row(prob),
        "mean_pairwise_confidence": float(np.mean(confidences)),
        "pairwise_calibration_brier": brier,
        "pairwise_auc_helpful_preferences": _binary_auc(pair_probs, labels),
    }


def pairprob_route_rows(
    *,
    method: str,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    prob_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    policy_name: str,
    selection: PairprobPolicySelection,
    hard_oracle_gap_pct: np.ndarray | None,
    diagnostic_only_reason: str = "",
    absolute_high_regret_gap_pct: float = 5.0,
    catastrophic_regression_vs_hard_gap_pct: float = 5.0,
) -> List[Dict[str, Any]]:
    win, orders, margins = pairprob_order_and_margin(prob_matrix, expert_domains=expert_domains)
    ranking_score = -win
    selected_idx = orders[:, 0].astype(np.int64, copy=False)
    _metrics, rows = _selection_metrics(
        method=method,
        query_domains=query_domains,
        expert_domains=expert_domains,
        score_matrix=ranking_score,
        true_nelbo_matrix=true_nelbo_matrix,
        fold=fold,
        global_true_nelbo_matrix=global_true_nelbo_matrix,
        global_expert_domains=global_expert_domains,
        selected_idx_override=selected_idx,
        ranking_score_matrix=ranking_score,
    )
    reason = str(diagnostic_only_reason or selection.diagnostic_only_reason)
    hard_gap = (
        np.asarray(hard_oracle_gap_pct, dtype=np.float64)
        if hard_oracle_gap_pct is not None
        else np.full((len(rows),), float("nan"), dtype=np.float64)
    )
    if hard_gap.shape[0] != len(rows):
        hard_gap = np.full((len(rows),), float("nan"), dtype=np.float64)

    for i, row in enumerate(rows):
        selected_col = int(selected_idx[i])
        selected_expert = int(expert_domains[selected_col])
        pair_diag = _pair_diagnostics_for_row(prob_matrix[i, :, :], true_nelbo_matrix[i, :])
        row.update(
            {
                "policy_name": str(policy_name),
                "base_method": str(selection.method),
                "feature_set": str(selection.feature_set),
                "selected_tau": float(selection.ridge_l2),
                "selected_by_inner_validation": int(bool(selection.selected_by_inner_validation)),
                "threshold_selection_policy": "source_inner_group_robust_worst_gap_then_catastrophic_then_mean_gap_v1",
                "route_experts": str(selected_expert),
                "route_weights": "1",
                "route_size": 1,
                "route_mode": "pairprob_hard_top1",
                "pairprob_predictor": "logistic_ridge_pairprob",
                "pairprob_probability_calibration": "none_v1",
                "pairprob_ridge_l2": float(selection.ridge_l2),
                "pairprob_feature_set": str(selection.feature_set),
                "pairprob_selection_policy": "source_inner_group_robust_worst_gap_then_catastrophic_then_mean_gap_v1",
                "pairprob_win_top1": float(win[i, selected_col]),
                "top1_win_margin": float(margins[i]),
                "tournament_margin": float(margins[i]),
                "absolute_high_regret_gap_gt_5": int(
                    float(row["oracle_gap_pct"]) > float(absolute_high_regret_gap_pct)
                ),
                "relative_catastrophic_regression_vs_hard_gt_5": int(
                    np.isfinite(float(hard_gap[i]))
                    and float(row["oracle_gap_pct"]) - float(hard_gap[i])
                    > float(catastrophic_regression_vs_hard_gap_pct)
                ),
                "hard_oracle_gap_pct": float(hard_gap[i]),
                "worst_inner_domain_oracle_gap_pct": float(selection.source_inner_worst_domain_oracle_gap_pct),
                "relative_catastrophic_regression_vs_hard_gt_5_rate": float(
                    selection.source_inner_relative_catastrophic_rate
                ),
                "absolute_high_regret_rate_gap_gt_5": float(selection.source_inner_absolute_high_regret_rate),
                "std_oracle_gap_pct_across_inner_domains": float(selection.source_inner_std_oracle_gap_pct),
                "std_top1_across_inner_domains": float(selection.source_inner_std_top1),
                "max_minus_min_oracle_gap_pct_across_inner_domains": float(
                    selection.source_inner_max_minus_min_oracle_gap_pct
                ),
                "pairwise_near_tie_drop_rate": float(selection.pairwise_near_tie_drop_rate),
                "pairwise_train_pairs_after_filter": int(selection.pairwise_train_pairs_after_filter),
                "pairwise_validation_pairs_after_filter": int(selection.pairwise_validation_pairs_after_filter),
                "pairwise_train_domains_after_filter": int(selection.pairwise_train_domains_after_filter),
                "diagnostic_only_reason": str(reason),
                **pair_diag,
            }
        )
        if reason:
            row.update(
                {
                    "method_role": "diagnostic",
                    "adoption_eligible": 0,
                    "diagnostic_only": 1,
                }
            )
    return rows


def summarize_pairprob_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "mean_oracle_gap_pct": float("nan"),
            "worst_inner_domain_oracle_gap_pct": float("nan"),
            "relative_catastrophic_regression_vs_hard_gt_5_rate": float("nan"),
            "absolute_high_regret_rate_gap_gt_5": float("nan"),
            "top1_oracle_hit": float("nan"),
            "spearman": float("nan"),
            "std_oracle_gap_pct_across_inner_domains": float("nan"),
            "std_top1_across_inner_domains": float("nan"),
            "max_minus_min_oracle_gap_pct_across_inner_domains": float("nan"),
        }
    by_domain: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(int(row["query_domain"]), []).append(row)

    domain_gap = []
    domain_top1 = []
    for domain_rows in by_domain.values():
        domain_gap.append(float(np.mean([float(r["oracle_gap_pct"]) for r in domain_rows])))
        domain_top1.append(float(np.mean([float(r["top1_oracle_hit"]) for r in domain_rows])))

    spearman_vals = [float(r["spearman"]) for r in rows if np.isfinite(float(r["spearman"]))]
    return {
        "n_rows": float(len(rows)),
        "validation_domains": float(len(by_domain)),
        "mean_oracle_gap_pct": float(np.mean([float(r["oracle_gap_pct"]) for r in rows])),
        "worst_inner_domain_oracle_gap_pct": float(max(domain_gap)) if domain_gap else float("nan"),
        "relative_catastrophic_regression_vs_hard_gt_5_rate": float(
            np.mean([float(r.get("relative_catastrophic_regression_vs_hard_gt_5", 0.0)) for r in rows])
        ),
        "absolute_high_regret_rate_gap_gt_5": float(
            np.mean([float(r.get("absolute_high_regret_gap_gt_5", 0.0)) for r in rows])
        ),
        "top1_oracle_hit": float(np.mean([float(r["top1_oracle_hit"]) for r in rows])),
        "spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "std_oracle_gap_pct_across_inner_domains": float(np.std(domain_gap)) if domain_gap else float("nan"),
        "std_top1_across_inner_domains": float(np.std(domain_top1)) if domain_top1 else float("nan"),
        "max_minus_min_oracle_gap_pct_across_inner_domains": (
            float(max(domain_gap) - min(domain_gap)) if domain_gap else float("nan")
        ),
    }


def select_pairprob_policy(
    *,
    rows_by_key: Dict[Tuple[str, str, float], List[Dict[str, Any]]],
    method: str,
    cfg: PairprobTournamentConfig,
    selection_mode: str,
    evidence_by_key: Dict[Tuple[str, str, float], Dict[str, float]],
) -> PairprobPolicySelection | None:
    candidates: List[Tuple[Tuple[float, ...], Tuple[str, str, float], Dict[str, float], str]] = []
    for key, rows in rows_by_key.items():
        candidate_method, feature_set, l2 = key
        if str(candidate_method) != str(method):
            continue
        summary = summarize_pairprob_rows(rows)
        if int(summary.get("n_rows", 0.0)) <= 0:
            continue
        evidence = evidence_by_key.get(key, {})
        reason = ""
        if int(summary.get("validation_domains", 0.0)) < int(cfg.min_source_inner_validation_domains):
            reason = "insufficient_pairwise_evidence"
        if selection_mode == "group_robust":
            score = (
                -float(summary["worst_inner_domain_oracle_gap_pct"]),
                -float(summary["relative_catastrophic_regression_vs_hard_gt_5_rate"]),
                -float(summary["mean_oracle_gap_pct"]),
                float(summary["top1_oracle_hit"]),
                -float(summary["std_oracle_gap_pct_across_inner_domains"]),
                float(summary["spearman"]) if np.isfinite(float(summary["spearman"])) else -1e9,
                -float(l2),
            )
        else:
            score = (
                -float(summary["mean_oracle_gap_pct"]),
                -float(summary["relative_catastrophic_regression_vs_hard_gt_5_rate"]),
                float(summary["top1_oracle_hit"]),
                float(summary["spearman"]) if np.isfinite(float(summary["spearman"])) else -1e9,
                -float(l2),
            )
        candidates.append((score, key, summary, reason or str(evidence.get("diagnostic_only_reason", ""))))
    if not candidates:
        return None
    _score, (candidate_method, feature_set, l2), summary, reason = sorted(
        candidates,
        key=lambda item: item[0],
        reverse=True,
    )[0]
    evidence = evidence_by_key.get((candidate_method, feature_set, l2), {})
    return PairprobPolicySelection(
        method=str(candidate_method),
        feature_set=str(feature_set),
        ridge_l2=float(l2),
        selected_by_inner_validation=True,
        diagnostic_only_reason=str(reason),
        source_inner_validation_domains=int(summary.get("validation_domains", 0.0)),
        source_inner_rows=int(summary.get("n_rows", 0.0)),
        source_inner_mean_oracle_gap_pct=float(summary["mean_oracle_gap_pct"]),
        source_inner_worst_domain_oracle_gap_pct=float(summary["worst_inner_domain_oracle_gap_pct"]),
        source_inner_relative_catastrophic_rate=float(
            summary["relative_catastrophic_regression_vs_hard_gt_5_rate"]
        ),
        source_inner_absolute_high_regret_rate=float(summary["absolute_high_regret_rate_gap_gt_5"]),
        source_inner_top1=float(summary["top1_oracle_hit"]),
        source_inner_spearman=float(summary["spearman"]),
        source_inner_std_oracle_gap_pct=float(summary["std_oracle_gap_pct_across_inner_domains"]),
        source_inner_std_top1=float(summary["std_top1_across_inner_domains"]),
        source_inner_max_minus_min_oracle_gap_pct=float(
            summary["max_minus_min_oracle_gap_pct_across_inner_domains"]
        ),
        pairwise_near_tie_drop_rate=float(evidence.get("pairwise_near_tie_drop_rate", float("nan"))),
        pairwise_train_pairs_after_filter=int(evidence.get("pairwise_train_pairs_after_filter", 0.0)),
        pairwise_validation_pairs_after_filter=int(evidence.get("pairwise_validation_pairs_after_filter", 0.0)),
        pairwise_train_domains_after_filter=int(evidence.get("pairwise_train_domains_after_filter", 0.0)),
    )
