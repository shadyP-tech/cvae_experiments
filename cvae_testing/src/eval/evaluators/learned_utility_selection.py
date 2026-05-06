from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    ProtocolError,
    _MIN_CANDIDATES_FOR_RANK_METRICS,
    _assert_method_eligibility,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.metrics import spearman_corr


def _stable_argmin_indices(matrix: np.ndarray) -> np.ndarray:
    n_rows, n_cols = matrix.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    out = np.zeros((n_rows,), dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, matrix[i, :]))
        out[i] = int(order[0])
    return out


def _selected_rank_in_true_matrix(selected_idx: np.ndarray, true_nelbo_matrix: np.ndarray) -> np.ndarray:
    n_rows, n_cols = true_nelbo_matrix.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    out = np.zeros((n_rows,), dtype=np.float64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, true_nelbo_matrix[i, :]))
        ranks = np.empty((n_cols,), dtype=np.int64)
        ranks[order] = np.arange(1, n_cols + 1, dtype=np.int64)
        out[i] = float(ranks[int(selected_idx[i])])
    return out


def _pairwise_auc_single(score_row: np.ndarray, true_row: np.ndarray) -> float:
    n = int(score_row.shape[0])
    if n <= 1:
        return float("nan")
    total = 0.0
    correct = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti = float(true_row[i])
            tj = float(true_row[j])
            if abs(ti - tj) < 1e-12:
                continue

            si = float(score_row[i])
            sj = float(score_row[j])
            total += 1.0
            if abs(si - sj) < 1e-12:
                correct += 0.5
                continue

            true_better_i = ti < tj
            pred_better_i = si < sj
            if true_better_i == pred_better_i:
                correct += 1.0

    return float(correct / total) if total > 0.0 else float("nan")


def _pairwise_auc_matrix(score_matrix: np.ndarray, true_nelbo_matrix: np.ndarray) -> float:
    vals = [
        _pairwise_auc_single(score_matrix[i, :], true_nelbo_matrix[i, :])
        for i in range(score_matrix.shape[0])
    ]
    return float(np.mean(vals)) if vals else 0.5


def _selection_metrics(
    *,
    method: str,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    score_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    fold: FoldCandidateSet,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    tie_policy: str = "stable_expert_index",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if str(tie_policy).strip().lower() != "stable_expert_index":
        raise ValueError("Only tie_policy='stable_expert_index' is currently supported")
    if score_matrix.shape != true_nelbo_matrix.shape:
        raise ProtocolError(
            f"score_matrix and true_nelbo_matrix shape mismatch for {method}: "
            f"{score_matrix.shape} vs {true_nelbo_matrix.shape}"
        )
    if score_matrix.shape[1] != len(expert_domains):
        raise ProtocolError(
            f"Score matrix candidate width does not match expert domains for {method}: "
            f"{score_matrix.shape[1]} vs {len(expert_domains)}"
        )

    method_protocol = _method_protocol(method)
    _assert_method_eligibility(method, method_protocol)

    selected_idx = _stable_argmin_indices(score_matrix)
    oracle_idx = _stable_argmin_indices(true_nelbo_matrix)
    global_oracle_idx = _stable_argmin_indices(global_true_nelbo_matrix)

    selected_nelbo = true_nelbo_matrix[np.arange(true_nelbo_matrix.shape[0]), selected_idx]
    oracle_nelbo = true_nelbo_matrix[np.arange(true_nelbo_matrix.shape[0]), oracle_idx]
    global_oracle_nelbo = global_true_nelbo_matrix[np.arange(global_true_nelbo_matrix.shape[0]), global_oracle_idx]
    selected_rank = _selected_rank_in_true_matrix(selected_idx, true_nelbo_matrix)

    top1 = float(np.mean(selected_idx == oracle_idx)) if selected_idx.size else 0.0
    mean_rank = float(np.mean(selected_rank)) if selected_rank.size else 0.0
    gap = selected_nelbo - oracle_nelbo
    denom = np.maximum(np.abs(oracle_nelbo), 1e-12)
    gap_pct = (gap / denom) * 100.0

    rho_vals: List[float] = []
    pair_auc_vals: List[float] = []
    rank_metrics_valid = int(score_matrix.shape[1]) >= _MIN_CANDIDATES_FOR_RANK_METRICS
    for i in range(score_matrix.shape[0]):
        if rank_metrics_valid:
            pair_auc_vals.append(
                float(_pairwise_auc_single(score_matrix[i, :], true_nelbo_matrix[i, :]))
            )
            rho_vals.append(
                float(
                    spearman_corr(
                        (-score_matrix[i, :]).tolist(),
                        (-true_nelbo_matrix[i, :]).tolist(),
                    )
                )
            )
        else:
            pair_auc_vals.append(float("nan"))
            rho_vals.append(float("nan"))

    rows: List[Dict[str, Any]] = []
    for i in range(score_matrix.shape[0]):
        selected_expert = int(expert_domains[int(selected_idx[i])])
        candidate_oracle_expert = int(expert_domains[int(oracle_idx[i])])
        global_oracle_expert = int(global_expert_domains[int(global_oracle_idx[i])])
        if not fold.contains(selected_expert):
            raise ProtocolError(f"Selected expert {selected_expert} is outside candidate pool for {method}")
        if not fold.contains(candidate_oracle_expert):
            raise ProtocolError(
                f"Candidate oracle expert {candidate_oracle_expert} is outside candidate pool for {method}"
            )
        rows.append(
            {
                **_protocol_row_fields(fold=fold, method_protocol=method_protocol, method=method),
                "sample_index": int(i),
                "query_domain": int(query_domains[i]),
                "selected_expert": selected_expert,
                "candidate_oracle_expert": candidate_oracle_expert,
                "candidate_oracle_nelbo": float(oracle_nelbo[i]),
                "global_oracle_expert": global_oracle_expert,
                "global_oracle_nelbo": float(global_oracle_nelbo[i]),
                "global_oracle_excluded_by_policy": int(not fold.contains(global_oracle_expert)),
                "oracle_expert": candidate_oracle_expert,
                "selected_nelbo": float(selected_nelbo[i]),
                "oracle_nelbo": float(oracle_nelbo[i]),
                "oracle_gap": float(gap[i]),
                "oracle_gap_pct": float(gap_pct[i]),
                "top1_oracle_hit": int(selected_idx[i] == oracle_idx[i]),
                "selected_rank": float(selected_rank[i]),
                "pairwise_auc": float(pair_auc_vals[i]),
                "spearman": float(rho_vals[i]),
                "rank_metrics_valid": int(rank_metrics_valid),
            }
        )

    valid_pair_auc = [v for v in pair_auc_vals if np.isfinite(v)]
    valid_spearman = [v for v in rho_vals if np.isfinite(v)]
    metrics = {
        "top1_oracle_hit": float(top1),
        "mean_rank": float(mean_rank),
        "mean_oracle_gap": float(np.mean(gap)) if gap.size else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gap_pct)) if gap_pct.size else 0.0,
        "pairwise_auc": float(np.mean(valid_pair_auc)) if valid_pair_auc else float("nan"),
        "spearman": float(np.mean(valid_spearman)) if valid_spearman else float("nan"),
        "selected_nelbo": float(np.mean(selected_nelbo)) if selected_nelbo.size else 0.0,
        "oracle_nelbo": float(np.mean(oracle_nelbo)) if oracle_nelbo.size else 0.0,
        "candidate_oracle_nelbo": float(np.mean(oracle_nelbo)) if oracle_nelbo.size else 0.0,
        "global_oracle_nelbo": float(np.mean(global_oracle_nelbo)) if global_oracle_nelbo.size else 0.0,
        "n_valid_spearman_samples": float(len(valid_spearman)),
        "n_valid_auc_samples": float(len(valid_pair_auc)),
    }
    return metrics, rows
