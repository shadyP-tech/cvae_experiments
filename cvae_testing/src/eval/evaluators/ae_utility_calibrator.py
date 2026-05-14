from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import AEUtilityCalibratorConfig
from src.eval.evaluators.learned_utility_models import _LinearRegressor, _PairwiseRanker
from src.eval.evaluators.learned_utility_pairs import _zscore_features
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    MethodProtocol,
    ProtocolError,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics, _stable_argmin_indices
from src.eval.evaluators.support_free_ae import (
    AutoencoderScoreMatrices,
    _metadata_selected_local_indices,
    _oracle_ranks_for_matrix,
    _threshold_label,
    _write_csv,
)
from src.eval.metrics import spearman_corr


PRIMARY_METHOD = "ae_utility_calibrated_safe_override_v1"
HYBRID_METADATA_METHOD = "ae_metadata_utility_calibrated_safe_override_v1"
HYBRID_COMBINED_METHOD = "ae_combined_utility_calibrated_safe_override_v1"
PAIRWISE_DIAG_METHOD = "ae_utility_pairwise_ranker_diagnostic_v1"
ORACLE_HEADROOM_METHOD = "oracle_safe_override_over_ae_argmin"


@dataclass(frozen=True)
class AEUtilityCalibratorFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    raw_rows: List[Dict[str, Any]]
    source_inner_validation_rows: List[Dict[str, Any]]
    policy_audit_rows: List[Dict[str, Any]]
    override_diagnostic_rows: List[Dict[str, Any]]
    oracle_headroom_rows: List[Dict[str, Any]]
    selected_feature_rows: List[Dict[str, Any]]
    override_precision_rows: List[Dict[str, Any]]
    anchor_rank_rows: List[Dict[str, Any]]


@dataclass(frozen=True)
class _FeatureRows:
    x: np.ndarray
    y_delta: np.ndarray
    y_nelbo: np.ndarray
    sample_indices: np.ndarray
    query_domains: np.ndarray
    expert_domains: np.ndarray
    sample_positions: np.ndarray
    candidate_local_indices: np.ndarray
    anchor_local_indices: np.ndarray
    candidate_col_indices: np.ndarray
    anchor_col_indices: np.ndarray
    ae_z: np.ndarray
    anchor_ae_z: np.ndarray
    ae_rank: np.ndarray
    ae_margin: np.ndarray


@dataclass(frozen=True)
class _SelectedConfig:
    method: str
    feature_set: str
    delta_threshold: float
    margin_threshold: float
    selected_by_source_inner: int


def _safe_div(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return num / np.maximum(np.abs(denom), 1e-12)


def _finite_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _quality_by_domain(ae_scores: AutoencoderScoreMatrices) -> Dict[int, Dict[str, Any]]:
    return {int(row.get("source_domain")): dict(row) for row in ae_scores.quality_rows}


def _rank_order(values: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    primary = values if lower_is_better else -values
    order = np.lexsort((np.arange(values.shape[0], dtype=np.int64), primary))
    ranks = np.empty((values.shape[0],), dtype=np.int64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.int64)
    return ranks


def _metadata_features(
    *,
    sample_index: int,
    query_domain: int,
    candidate_domain: int,
    candidate_col: int,
    candidate_cols: Sequence[int],
    metadata_similarity: np.ndarray,
    sample_domains: np.ndarray,
) -> List[float]:
    sims = metadata_similarity[int(sample_index), list(candidate_cols)]
    ranks = _rank_order(sims, lower_is_better=False)
    local = list(candidate_cols).index(int(candidate_col))
    sorted_sims = np.sort(sims)
    margin = float(sorted_sims[-1] - sorted_sims[-2]) if sims.shape[0] > 1 else 0.0
    span = max(float(np.max(sample_domains) - np.min(sample_domains)), 1.0)
    return [
        float(metadata_similarity[int(sample_index), int(candidate_col)]),
        float(ranks[int(local)]),
        float(margin),
        float(abs(float(query_domain) - float(candidate_domain)) / span),
        float(int(query_domain) == int(candidate_domain)),
    ]


def _feature_vector(
    *,
    feature_set: str,
    sample_index: int,
    query_domain: int,
    candidate_domain: int,
    candidate_col: int,
    anchor_domain: int,
    anchor_col: int,
    candidate_cols: Sequence[int],
    candidate_local: int,
    ae_z: np.ndarray,
    ae_raw: np.ndarray,
    ae_ranks: np.ndarray,
    ae_margin: float,
    embeddings: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    sample_domains: np.ndarray,
    quality: Mapping[int, Mapping[str, Any]],
) -> List[float]:
    candidate_z = float(ae_z[int(candidate_local)])
    anchor_local = list(candidate_cols).index(int(anchor_col))
    anchor_z = float(ae_z[int(anchor_local)])
    candidate_raw = float(ae_raw[int(candidate_local)])
    anchor_raw = float(ae_raw[int(anchor_local)])
    n_candidates = max(float(len(candidate_cols)), 1.0)
    values: List[float] = [
        candidate_z,
        anchor_z,
        candidate_z - anchor_z,
        float(ae_ranks[int(candidate_local)]),
        float(ae_ranks[int(candidate_local)]) / n_candidates,
        float(ae_margin),
        candidate_raw,
        anchor_raw,
    ]
    if str(feature_set) in {"ae_quality", "ae_metadata", "ae_combined"}:
        q_candidate = quality.get(int(candidate_domain), {})
        q_anchor = quality.get(int(anchor_domain), {})
        values.extend(
            [
                float(q_candidate.get("source_val_reconstruction_mse_by_domain", 0.0)),
                float(q_candidate.get("source_val_reconstruction_std_by_domain", 1.0)),
                float(q_candidate.get("ae_source_val_count", q_candidate.get("val_size", 0))),
                float(q_candidate.get("ae_z_sigma_floor", 0.0)),
                float(q_candidate.get("ae_z_sigma_floor_applied", 0)),
                float(q_candidate.get("ae_val_loss", 0.0)),
                float(q_anchor.get("source_val_reconstruction_mse_by_domain", 0.0)),
                float(q_anchor.get("source_val_reconstruction_std_by_domain", 1.0)),
            ]
        )
    if str(feature_set) in {"ae_metadata", "ae_combined"}:
        values.extend(
            _metadata_features(
                sample_index=int(sample_index),
                query_domain=int(query_domain),
                candidate_domain=int(candidate_domain),
                candidate_col=int(candidate_col),
                candidate_cols=candidate_cols,
                metadata_similarity=metadata_similarity,
                sample_domains=sample_domains,
            )
        )
    if str(feature_set) == "ae_combined":
        values.extend(float(v) for v in embeddings[int(sample_index), :].tolist())
        one_hot = [0.0] * len(expert_domains)
        domain_to_pos = {int(domain): i for i, domain in enumerate(expert_domains)}
        one_hot[domain_to_pos[int(candidate_domain)]] = 1.0
        values.extend(one_hot)
    return [0.0 if not np.isfinite(float(v)) else float(v) for v in values]


def _build_feature_rows(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    sample_indices: np.ndarray,
    fold_for_sample: Callable[[int], FoldCandidateSet],
    metadata_similarity: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    feature_set: str,
    exclude_anchor: bool,
) -> _FeatureRows:
    quality = _quality_by_domain(ae_scores)
    domain_to_col = {int(domain): int(idx) for idx, domain in enumerate(expert_domains)}
    x_rows: List[List[float]] = []
    y_delta: List[float] = []
    y_nelbo: List[float] = []
    row_sample_indices: List[int] = []
    row_query_domains: List[int] = []
    row_expert_domains: List[int] = []
    row_sample_positions: List[int] = []
    row_candidate_local: List[int] = []
    row_anchor_local: List[int] = []
    row_candidate_col: List[int] = []
    row_anchor_col: List[int] = []
    row_ae_z: List[float] = []
    row_anchor_ae_z: List[float] = []
    row_ae_rank: List[int] = []
    row_ae_margin: List[float] = []

    for sample_position, sample_index_raw in enumerate(np.asarray(sample_indices, dtype=np.int64).tolist()):
        sample_index = int(sample_index_raw)
        fold = fold_for_sample(sample_index)
        candidate_cols = list(fold.candidate_col_indices)
        candidate_domains = [int(d) for d in fold.candidate_expert_domains]
        ae_z = ae_scores.zscore_matrix[sample_index, candidate_cols]
        ae_raw = ae_scores.raw_mse_matrix[sample_index, candidate_cols]
        if not np.isfinite(ae_z).all() or not np.isfinite(ae_raw).all():
            raise ProtocolError("AE utility calibrator features require finite AE scores")
        order = np.lexsort((np.arange(len(candidate_cols), dtype=np.int64), ae_z))
        anchor_local = int(order[0])
        anchor_col = int(candidate_cols[anchor_local])
        anchor_domain = int(expert_domains[anchor_col])
        ranks = _rank_order(ae_z, lower_is_better=True)
        ae_margin = (
            float(ae_z[int(order[1])] - ae_z[int(order[0])])
            if len(candidate_cols) > 1
            else float("inf")
        )
        anchor_nelbo = float(true_nelbo[sample_index, anchor_col])
        for candidate_local, candidate_domain in enumerate(candidate_domains):
            candidate_col = int(domain_to_col[int(candidate_domain)])
            if bool(exclude_anchor) and int(candidate_local) == int(anchor_local):
                continue
            candidate_nelbo = float(true_nelbo[sample_index, candidate_col])
            x_rows.append(
                _feature_vector(
                    feature_set=feature_set,
                    sample_index=sample_index,
                    query_domain=int(sample_domains[sample_index]),
                    candidate_domain=int(candidate_domain),
                    candidate_col=candidate_col,
                    anchor_domain=anchor_domain,
                    anchor_col=anchor_col,
                    candidate_cols=candidate_cols,
                    candidate_local=int(candidate_local),
                    ae_z=ae_z,
                    ae_raw=ae_raw,
                    ae_ranks=ranks,
                    ae_margin=ae_margin,
                    embeddings=embeddings,
                    expert_domains=expert_domains,
                    metadata_similarity=metadata_similarity,
                    sample_domains=sample_domains,
                    quality=quality,
                )
            )
            y_delta.append(float((anchor_nelbo - candidate_nelbo) / max(abs(anchor_nelbo), 1e-12)))
            y_nelbo.append(candidate_nelbo)
            row_sample_indices.append(sample_index)
            row_query_domains.append(int(sample_domains[sample_index]))
            row_expert_domains.append(int(candidate_domain))
            row_sample_positions.append(int(sample_position))
            row_candidate_local.append(int(candidate_local))
            row_anchor_local.append(int(anchor_local))
            row_candidate_col.append(int(candidate_col))
            row_anchor_col.append(int(anchor_col))
            row_ae_z.append(float(ae_z[int(candidate_local)]))
            row_anchor_ae_z.append(float(ae_z[int(anchor_local)]))
            row_ae_rank.append(int(ranks[int(candidate_local)]))
            row_ae_margin.append(float(ae_margin))

    if not x_rows:
        return _FeatureRows(
            x=np.zeros((0, 0), dtype=np.float64),
            y_delta=np.asarray([], dtype=np.float64),
            y_nelbo=np.asarray([], dtype=np.float64),
            sample_indices=np.asarray([], dtype=np.int64),
            query_domains=np.asarray([], dtype=np.int64),
            expert_domains=np.asarray([], dtype=np.int64),
            sample_positions=np.asarray([], dtype=np.int64),
            candidate_local_indices=np.asarray([], dtype=np.int64),
            anchor_local_indices=np.asarray([], dtype=np.int64),
            candidate_col_indices=np.asarray([], dtype=np.int64),
            anchor_col_indices=np.asarray([], dtype=np.int64),
            ae_z=np.asarray([], dtype=np.float64),
            anchor_ae_z=np.asarray([], dtype=np.float64),
            ae_rank=np.asarray([], dtype=np.int64),
            ae_margin=np.asarray([], dtype=np.float64),
        )

    return _FeatureRows(
        x=np.asarray(x_rows, dtype=np.float64),
        y_delta=np.asarray(y_delta, dtype=np.float64),
        y_nelbo=np.asarray(y_nelbo, dtype=np.float64),
        sample_indices=np.asarray(row_sample_indices, dtype=np.int64),
        query_domains=np.asarray(row_query_domains, dtype=np.int64),
        expert_domains=np.asarray(row_expert_domains, dtype=np.int64),
        sample_positions=np.asarray(row_sample_positions, dtype=np.int64),
        candidate_local_indices=np.asarray(row_candidate_local, dtype=np.int64),
        anchor_local_indices=np.asarray(row_anchor_local, dtype=np.int64),
        candidate_col_indices=np.asarray(row_candidate_col, dtype=np.int64),
        anchor_col_indices=np.asarray(row_anchor_col, dtype=np.int64),
        ae_z=np.asarray(row_ae_z, dtype=np.float64),
        anchor_ae_z=np.asarray(row_anchor_ae_z, dtype=np.float64),
        ae_rank=np.asarray(row_ae_rank, dtype=np.int64),
        ae_margin=np.asarray(row_ae_margin, dtype=np.float64),
    )


def _fit_predict_delta(
    *,
    train_rows: _FeatureRows,
    eval_rows: _FeatureRows,
    ridge_l2: float,
) -> np.ndarray:
    if train_rows.x.size == 0 or eval_rows.x.size == 0:
        return np.asarray([], dtype=np.float64)
    x_train_z, x_eval_z = _zscore_features(train_rows.x, eval_rows.x)
    model = _LinearRegressor(l2=float(ridge_l2))
    model.fit(x_train_z, train_rows.y_delta)
    return model.predict(x_eval_z).astype(np.float64, copy=False)


def _prediction_matrix_from_rows(
    *,
    rows: _FeatureRows,
    predicted_delta: np.ndarray,
    n_samples: int,
    n_candidates: int,
) -> np.ndarray:
    pred = np.zeros((int(n_samples), int(n_candidates)), dtype=np.float64)
    if predicted_delta.size:
        for k, value in enumerate(predicted_delta.tolist()):
            pred[int(rows.sample_positions[k]), int(rows.candidate_local_indices[k])] = float(value)
    return pred


def _anchor_indices_from_rows(*, rows: _FeatureRows, n_samples: int) -> np.ndarray:
    anchors = np.zeros((int(n_samples),), dtype=np.int64)
    for sample_pos, anchor in zip(rows.sample_positions.tolist(), rows.anchor_local_indices.tolist()):
        anchors[int(sample_pos)] = int(anchor)
    return anchors


def _apply_safe_override_policy(
    *,
    pred_delta_matrix: np.ndarray,
    anchor_idx: np.ndarray,
    delta_threshold: float,
    margin_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols = pred_delta_matrix.shape
    selected = np.asarray(anchor_idx, dtype=np.int64).copy()
    best_override = np.full((n_rows,), -1, dtype=np.int64)
    best_delta = np.full((n_rows,), float("-inf"), dtype=np.float64)
    margins = np.full((n_rows,), float("inf"), dtype=np.float64)
    accepted = np.zeros((n_rows,), dtype=bool)
    for i in range(n_rows):
        candidates = [j for j in range(n_cols) if int(j) != int(anchor_idx[i])]
        if not candidates:
            continue
        order = sorted(candidates, key=lambda j: (-float(pred_delta_matrix[i, j]), int(j)))
        best = int(order[0])
        second_delta = float(pred_delta_matrix[i, int(order[1])]) if len(order) > 1 else float("-inf")
        best_override[i] = best
        best_delta[i] = float(pred_delta_matrix[i, best])
        margins[i] = (
            float(best_delta[i] - second_delta)
            if np.isfinite(second_delta)
            else float("inf")
        )
        if float(best_delta[i]) >= float(delta_threshold) and float(margins[i]) >= float(margin_threshold):
            selected[i] = best
            accepted[i] = True
    return selected, best_override, best_delta, margins


def _true_delta_matrix(true_eval: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    rows = np.arange(true_eval.shape[0])
    anchor_nelbo = true_eval[rows, np.asarray(anchor_idx, dtype=np.int64)]
    return _safe_div(anchor_nelbo.reshape(-1, 1) - true_eval, anchor_nelbo.reshape(-1, 1))


def _spearman_delta_metrics(
    *,
    pred_delta_matrix: np.ndarray,
    true_delta_matrix: np.ndarray,
    anchor_idx: np.ndarray,
) -> Tuple[float, float]:
    non_anchor_vals: List[float] = []
    with_anchor_vals: List[float] = []
    for i in range(pred_delta_matrix.shape[0]):
        non_anchor = [j for j in range(pred_delta_matrix.shape[1]) if int(j) != int(anchor_idx[i])]
        if len(non_anchor) >= 2:
            non_anchor_vals.append(
                float(
                    spearman_corr(
                        pred_delta_matrix[i, non_anchor].tolist(),
                        true_delta_matrix[i, non_anchor].tolist(),
                    )
                )
            )
        if pred_delta_matrix.shape[1] >= 2:
            with_anchor_vals.append(
                float(
                    spearman_corr(
                        pred_delta_matrix[i, :].tolist(),
                        true_delta_matrix[i, :].tolist(),
                    )
                )
            )
    return _finite_mean(non_anchor_vals, default=float("nan")), _finite_mean(with_anchor_vals, default=float("nan"))


def _policy_summary(
    *,
    selected_idx: np.ndarray,
    anchor_idx: np.ndarray,
    pred_delta_matrix: np.ndarray,
    true_eval: np.ndarray,
    metadata_idx: np.ndarray,
    ae_zscore_eval: np.ndarray,
) -> Dict[str, float]:
    rows = np.arange(true_eval.shape[0])
    oracle_idx = _stable_argmin_indices(true_eval)
    selected_nelbo = true_eval[rows, selected_idx]
    anchor_nelbo = true_eval[rows, anchor_idx]
    metadata_nelbo = true_eval[rows, metadata_idx]
    oracle_nelbo = true_eval[rows, oracle_idx]
    true_delta = _true_delta_matrix(true_eval, anchor_idx)
    pred_rho_non_anchor, pred_rho_with_anchor = _spearman_delta_metrics(
        pred_delta_matrix=pred_delta_matrix,
        true_delta_matrix=true_delta,
        anchor_idx=anchor_idx,
    )
    ae_delta_pred = -np.asarray(ae_zscore_eval, dtype=np.float64)
    ae_rho_non_anchor, ae_rho_with_anchor = _spearman_delta_metrics(
        pred_delta_matrix=ae_delta_pred,
        true_delta_matrix=true_delta,
        anchor_idx=anchor_idx,
    )
    gap = selected_nelbo - oracle_nelbo
    gap_pct = (gap / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    anchor_gap = anchor_nelbo - oracle_nelbo
    anchor_gap_pct = (anchor_gap / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    metadata_gap = metadata_nelbo - oracle_nelbo
    metadata_gap_pct = (metadata_gap / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    active = np.asarray(selected_idx, dtype=np.int64) != np.asarray(anchor_idx, dtype=np.int64)
    active_count = int(np.sum(active))
    improving_active = active & (selected_nelbo < anchor_nelbo)
    harmful_active = active & (selected_nelbo > anchor_nelbo)
    selected_precision = (
        float(np.sum(improving_active) / active_count)
        if active_count > 0
        else float("nan")
    )
    oracle_improvable = oracle_nelbo < anchor_nelbo
    oracle_improvable_count = int(np.sum(oracle_improvable))
    return {
        "top1_oracle_hit": float(np.mean(selected_idx == oracle_idx)) if selected_idx.size else 0.0,
        "mean_oracle_gap_pct": float(np.mean(gap_pct)) if gap_pct.size else 0.0,
        "mean_oracle_gap": float(np.mean(gap)) if gap.size else 0.0,
        "raw_predicted_delta_spearman_non_anchor": float(pred_rho_non_anchor),
        "raw_predicted_delta_spearman_with_anchor": float(pred_rho_with_anchor),
        "ae_delta_spearman_non_anchor": float(ae_rho_non_anchor),
        "ae_delta_spearman_with_anchor": float(ae_rho_with_anchor),
        "ae_argmin_top1_oracle_hit": float(np.mean(anchor_idx == oracle_idx)) if anchor_idx.size else 0.0,
        "ae_argmin_mean_oracle_gap_pct": float(np.mean(anchor_gap_pct)) if anchor_gap_pct.size else 0.0,
        "ae_argmin_mean_oracle_gap": float(np.mean(anchor_gap)) if anchor_gap.size else 0.0,
        "metadata_top1_oracle_hit": float(np.mean(metadata_idx == oracle_idx)) if metadata_idx.size else 0.0,
        "metadata_mean_oracle_gap_pct": float(np.mean(metadata_gap_pct)) if metadata_gap_pct.size else 0.0,
        "active_override_rate": float(np.mean(active)) if active.size else 0.0,
        "fallback_to_ae_argmin_rate": float(1.0 - np.mean(active)) if active.size else 1.0,
        "net_gain_vs_ae_argmin": float(np.mean(anchor_nelbo - selected_nelbo)) if selected_nelbo.size else 0.0,
        "net_gain_vs_metadata": float(np.mean(metadata_nelbo - selected_nelbo)) if selected_nelbo.size else 0.0,
        "harmful_vs_ae_argmin_rate": float(np.mean(harmful_active)) if active.size else 0.0,
        "improving_vs_ae_argmin_rate": float(np.mean(improving_active)) if active.size else 0.0,
        "harmful_vs_metadata_rate": float(np.mean(selected_nelbo > metadata_nelbo)) if selected_nelbo.size else 0.0,
        "improving_vs_metadata_rate": float(np.mean(selected_nelbo < metadata_nelbo)) if selected_nelbo.size else 0.0,
        "selected_override_precision": selected_precision,
        "oracle_headroom_vs_ae_argmin": float(np.mean(anchor_nelbo - oracle_nelbo)) if anchor_nelbo.size else 0.0,
        "oracle_improvable_query_rate": float(np.mean(oracle_improvable)) if oracle_improvable.size else 0.0,
        "override_opportunity_rate": float(np.mean(oracle_improvable)) if oracle_improvable.size else 0.0,
        "override_capture_rate": (
            float(np.sum(improving_active) / oracle_improvable_count)
            if oracle_improvable_count > 0
            else float("nan")
        ),
        "ae_argmin_already_oracle_rate": float(np.mean(anchor_idx == oracle_idx)) if anchor_idx.size else 0.0,
    }


def _passes_risk_gates(summary: Mapping[str, float], cfg: AEUtilityCalibratorConfig) -> bool:
    top1_drop = float(summary["ae_argmin_top1_oracle_hit"]) - float(summary["top1_oracle_hit"])
    spearman_drop = float(summary["ae_delta_spearman_non_anchor"]) - float(
        summary["raw_predicted_delta_spearman_non_anchor"]
    )
    gap_degradation = float(summary["mean_oracle_gap_pct"]) - float(summary["ae_argmin_mean_oracle_gap_pct"])
    return bool(
        top1_drop <= float(cfg.max_top1_drop_vs_ae_argmin_abs)
        and spearman_drop <= float(cfg.max_spearman_drop_vs_ae_argmin_abs)
        and gap_degradation <= float(cfg.max_gap_pct_degradation_vs_ae_argmin)
        and float(summary["harmful_vs_ae_argmin_rate"]) <= float(summary["improving_vs_ae_argmin_rate"])
    )


def _train_predict_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    train_fold_for_sample: Callable[[int], FoldCandidateSet],
    eval_fold_for_sample: Callable[[int], FoldCandidateSet],
    metadata_similarity: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    feature_set: str,
    ridge_l2: float,
    n_eval_candidates: int,
) -> Tuple[_FeatureRows, np.ndarray, np.ndarray, np.ndarray]:
    train_rows = _build_feature_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        sample_indices=train_idx,
        fold_for_sample=train_fold_for_sample,
        metadata_similarity=metadata_similarity,
        ae_scores=ae_scores,
        feature_set=feature_set,
        exclude_anchor=True,
    )
    eval_rows = _build_feature_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        sample_indices=eval_idx,
        fold_for_sample=eval_fold_for_sample,
        metadata_similarity=metadata_similarity,
        ae_scores=ae_scores,
        feature_set=feature_set,
        exclude_anchor=True,
    )
    pred = _fit_predict_delta(train_rows=train_rows, eval_rows=eval_rows, ridge_l2=float(ridge_l2))
    pred_matrix = _prediction_matrix_from_rows(
        rows=eval_rows,
        predicted_delta=pred,
        n_samples=int(eval_idx.shape[0]),
        n_candidates=int(n_eval_candidates),
    )
    anchor_idx = _anchor_indices_from_rows(rows=eval_rows, n_samples=int(eval_idx.shape[0]))
    return eval_rows, pred, pred_matrix, anchor_idx


def _select_config_for_method(
    *,
    method: str,
    feature_sets: Sequence[str],
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    outer_fold: FoldCandidateSet,
    metadata_similarity: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    cfg: AEUtilityCalibratorConfig,
) -> Tuple[_SelectedConfig, List[Dict[str, Any]]]:
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64)))
    validation_rows: List[Dict[str, Any]] = []
    config_summaries: List[Dict[str, Any]] = []
    deltas = tuple(dict.fromkeys(float(v) for v in cfg.delta_thresholds))
    if not any(not np.isfinite(v) for v in deltas):
        deltas = tuple(list(deltas) + [float("inf")])
    margins = tuple(dict.fromkeys(float(v) for v in cfg.margin_thresholds))

    for feature_set in feature_sets:
        threshold_domain_summaries: Dict[Tuple[float, float], List[Dict[str, float]]] = {}
        for pseudo_domain in source_domains:
            val_idx = np.asarray(
                [i for i in train_idx.tolist() if int(sample_domains[int(i)]) == int(pseudo_domain)],
                dtype=np.int64,
            )
            inner_train_idx = np.asarray(
                [i for i in train_idx.tolist() if int(sample_domains[int(i)]) != int(pseudo_domain)],
                dtype=np.int64,
            )
            if val_idx.size == 0 or inner_train_idx.size == 0:
                continue
            inner_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_fold.heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(pseudo_domain)],
            )
            if len(inner_fold.candidate_expert_domains) < 2:
                continue
            train_fold_for_sample = lambda sample_index, h=int(outer_fold.heldout_domain), p=int(pseudo_domain): FoldCandidateSet.for_heldout_domain(
                heldout_domain=h,
                expert_domains=expert_domains,
                excluded_domains=[int(sample_domains[int(sample_index)]), p],
            )
            eval_fold_for_sample = lambda _sample_index, f=inner_fold: f
            eval_rows, _pred, pred_matrix, anchor_idx = _train_predict_for_fold(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                train_idx=inner_train_idx,
                eval_idx=val_idx,
                train_fold_for_sample=train_fold_for_sample,
                eval_fold_for_sample=eval_fold_for_sample,
                metadata_similarity=metadata_similarity,
                ae_scores=ae_scores,
                feature_set=str(feature_set),
                ridge_l2=float(cfg.ridge_l2),
                n_eval_candidates=len(inner_fold.candidate_expert_domains),
            )
            if eval_rows.x.size == 0:
                continue
            true_val = inner_fold.slice_nelbo(true_nelbo, val_idx)
            metadata_val = metadata_similarity[val_idx][:, list(inner_fold.candidate_col_indices)]
            metadata_idx = _metadata_selected_local_indices(metadata_val)
            ae_val = ae_scores.zscore_matrix[val_idx][:, list(inner_fold.candidate_col_indices)]
            for delta_threshold in deltas:
                for margin_threshold in margins:
                    selected_idx, _best, _best_delta, _override_margin = _apply_safe_override_policy(
                        pred_delta_matrix=pred_matrix,
                        anchor_idx=anchor_idx,
                        delta_threshold=float(delta_threshold),
                        margin_threshold=float(margin_threshold),
                    )
                    summary = _policy_summary(
                        selected_idx=selected_idx,
                        anchor_idx=anchor_idx,
                        pred_delta_matrix=pred_matrix,
                        true_eval=true_val,
                        metadata_idx=metadata_idx,
                        ae_zscore_eval=ae_val,
                    )
                    threshold_domain_summaries.setdefault((float(delta_threshold), float(margin_threshold)), []).append(summary)
                    validation_rows.append(
                        {
                            "method": str(method),
                            "feature_set": str(feature_set),
                            "model_type": "ridge_delta",
                            "fold_query_domain": int(outer_fold.heldout_domain),
                            "source_inner_pseudo_query_domain": int(pseudo_domain),
                            "delta_threshold": _threshold_label(float(delta_threshold)),
                            "margin_threshold": _threshold_label(float(margin_threshold)),
                            "threshold_selection_policy": "source_inner_ae_argmin_noninferiority_then_gap",
                            "n_validation_samples": int(val_idx.shape[0]),
                            "candidate_experts": inner_fold.label(),
                            "excluded_target_ae": 1,
                            "excluded_target_cvae": 1,
                            "excluded_pseudo_query_ae": 1,
                            "excluded_pseudo_query_cvae": 1,
                            "heldout_target_nelbo_used_for_selection": 0,
                            **{f"macro_{k}": float(v) for k, v in summary.items()},
                        }
                    )

        for (delta_threshold, margin_threshold), summaries in threshold_domain_summaries.items():
            if not summaries:
                continue
            keys = set().union(*(row.keys() for row in summaries))
            macro = {k: _finite_mean([float(row.get(k, float("nan"))) for row in summaries], default=float("nan")) for k in keys}
            passes = _passes_risk_gates(macro, cfg)
            config_summaries.append(
                {
                    "method": str(method),
                    "feature_set": str(feature_set),
                    "delta_threshold": float(delta_threshold),
                    "margin_threshold": float(margin_threshold),
                    "passes_source_inner_risk_gates": bool(passes),
                    **macro,
                }
            )

    passing = [row for row in config_summaries if bool(row.get("passes_source_inner_risk_gates", False))]
    if not passing:
        selected = _SelectedConfig(str(method), str(feature_sets[0]), float("inf"), 0.0, 0)
    else:
        selected_row = sorted(
            passing,
            key=lambda row: (
                float(row["ae_argmin_mean_oracle_gap_pct"]) - float(row["mean_oracle_gap_pct"]),
                float(row["top1_oracle_hit"]),
                float(row["raw_predicted_delta_spearman_non_anchor"]),
                -float(row["harmful_vs_ae_argmin_rate"]),
                float(row["delta_threshold"]),
                float(row["margin_threshold"]),
            ),
            reverse=True,
        )[0]
        selected = _SelectedConfig(
            str(method),
            str(selected_row["feature_set"]),
            float(selected_row["delta_threshold"]),
            float(selected_row["margin_threshold"]),
            1,
        )
    for row in validation_rows:
        row["selected_feature_set"] = selected.feature_set
        row["selected_delta_threshold"] = _threshold_label(selected.delta_threshold)
        row["selected_margin_threshold"] = _threshold_label(selected.margin_threshold)
        row["selected_by_source_inner_validation"] = int(
            row["feature_set"] == selected.feature_set
            and row["delta_threshold"] == _threshold_label(selected.delta_threshold)
            and row["margin_threshold"] == _threshold_label(selected.margin_threshold)
        )
    return selected, validation_rows


def _method_feature_sets(cfg: AEUtilityCalibratorConfig) -> List[Tuple[str, Tuple[str, ...], str]]:
    methods = [(PRIMARY_METHOD, tuple(cfg.feature_sets_primary), "primary_metadata_free")]
    if "ae_metadata" in set(cfg.feature_sets_diagnostic):
        methods.append((HYBRID_METADATA_METHOD, ("ae_metadata",), "hybrid_metadata"))
    if "ae_combined" in set(cfg.feature_sets_diagnostic):
        methods.append((HYBRID_COMBINED_METHOD, ("ae_combined",), "hybrid_combined"))
    return methods


def _oracle_headroom_rows(
    *,
    fold: FoldCandidateSet,
    test_idx: np.ndarray,
    sample_domains: np.ndarray,
    true_eval: np.ndarray,
    anchor_idx: np.ndarray,
    selected_idx: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    oracle_idx = _stable_argmin_indices(true_eval)
    oracle_ranks = _oracle_ranks_for_matrix(true_eval)
    rows = np.arange(true_eval.shape[0])
    anchor_nelbo = true_eval[rows, anchor_idx]
    oracle_nelbo = true_eval[rows, oracle_idx]
    selected_nelbo = true_eval[rows, selected_idx]
    headroom_rows: List[Dict[str, Any]] = []
    anchor_rank_rows: List[Dict[str, Any]] = []
    fields = _protocol_row_fields(
        fold=fold,
        method_protocol=MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_eval_nelbo=1,
        ),
        method=ORACLE_HEADROOM_METHOD,
    )
    for local, sample_index in enumerate(np.asarray(test_idx, dtype=np.int64).tolist()):
        row = {
            **fields,
            "method": ORACLE_HEADROOM_METHOD,
            "sample_index": int(sample_index),
            "query_domain": int(sample_domains[int(sample_index)]),
            "ae_anchor_expert": int(fold.candidate_expert_domains[int(anchor_idx[local])]),
            "selected_expert": int(fold.candidate_expert_domains[int(selected_idx[local])]),
            "oracle_best_expert": int(fold.candidate_expert_domains[int(oracle_idx[local])]),
            "oracle_headroom_vs_ae_argmin": float(anchor_nelbo[local] - oracle_nelbo[local]),
            "oracle_improvable": int(oracle_nelbo[local] < anchor_nelbo[local]),
            "ae_argmin_already_oracle": int(int(anchor_idx[local]) == int(oracle_idx[local])),
            "selected_improves_over_ae_argmin": int(selected_nelbo[local] < anchor_nelbo[local]),
            "ae_anchor_oracle_rank": int(oracle_ranks[local, int(anchor_idx[local])]),
            "selected_oracle_rank": int(oracle_ranks[local, int(selected_idx[local])]),
        }
        headroom_rows.append(row)
        anchor_rank_rows.append(
            {
                "method": PRIMARY_METHOD,
                "fold_query_domain": int(fold.heldout_domain),
                "sample_index": int(sample_index),
                "query_domain": int(sample_domains[int(sample_index)]),
                "ae_anchor_expert": int(fold.candidate_expert_domains[int(anchor_idx[local])]),
                "selected_expert": int(fold.candidate_expert_domains[int(selected_idx[local])]),
                "oracle_best_expert": int(fold.candidate_expert_domains[int(oracle_idx[local])]),
                "ae_anchor_oracle_rank": int(oracle_ranks[local, int(anchor_idx[local])]),
                "selected_oracle_rank": int(oracle_ranks[local, int(selected_idx[local])]),
            }
        )
    return headroom_rows, anchor_rank_rows


def _run_pairwise_diagnostic(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    global_eval: np.ndarray,
    metadata_similarity: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    cfg: AEUtilityCalibratorConfig,
    seed: int,
    tie_policy: str,
) -> List[Dict[str, Any]]:
    if "pairwise_ranker" not in set(cfg.diagnostic_model_types):
        return []
    train_fold_for_sample = lambda sample_index: FoldCandidateSet.for_heldout_domain(
        heldout_domain=int(fold.heldout_domain),
        expert_domains=expert_domains,
        excluded_domains=[int(sample_domains[int(sample_index)])],
    )
    eval_fold_for_sample = lambda _sample_index: fold
    train_rows = _build_feature_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        sample_indices=train_idx,
        fold_for_sample=train_fold_for_sample,
        metadata_similarity=metadata_similarity,
        ae_scores=ae_scores,
        feature_set="ae_quality",
        exclude_anchor=False,
    )
    eval_rows = _build_feature_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        sample_indices=test_idx,
        fold_for_sample=eval_fold_for_sample,
        metadata_similarity=metadata_similarity,
        ae_scores=ae_scores,
        feature_set="ae_quality",
        exclude_anchor=False,
    )
    if train_rows.x.size == 0 or eval_rows.x.size == 0:
        return []
    pairs: List[Tuple[int, int]] = []
    for sample_index in sorted(set(int(v) for v in train_rows.sample_indices.tolist())):
        idxs = np.where(train_rows.sample_indices == int(sample_index))[0]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia = int(idxs[a])
                ib = int(idxs[b])
                if abs(float(train_rows.y_nelbo[ia]) - float(train_rows.y_nelbo[ib])) < 1e-12:
                    continue
                if float(train_rows.y_nelbo[ia]) < float(train_rows.y_nelbo[ib]):
                    pairs.append((ia, ib))
                else:
                    pairs.append((ib, ia))
    if not pairs:
        return []
    x_train_z, x_eval_z = _zscore_features(train_rows.x, eval_rows.x)
    ranker = _PairwiseRanker(seed=int(seed), epochs=20, hidden_dim=64, batch_size=2048)
    ranker.fit(x_train_z, pairs)
    pred = ranker.predict(x_eval_z)
    pred_matrix = np.zeros((int(test_idx.shape[0]), len(fold.candidate_expert_domains)), dtype=np.float64)
    for k, value in enumerate(pred.tolist()):
        pred_matrix[int(eval_rows.sample_positions[k]), int(eval_rows.candidate_local_indices[k])] = float(value)
    true_eval = fold.slice_nelbo(true_nelbo, test_idx)
    _metrics, rows = _selection_metrics(
        method=PAIRWISE_DIAG_METHOD,
        query_domains=sample_domains[test_idx],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=pred_matrix,
        true_nelbo_matrix=true_eval,
        fold=fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )
    for row in rows:
        row["sample_index"] = int(test_idx[int(row["sample_index"])])
        row["model_type"] = "pairwise_ranker"
        row["diagnostic_only_reason"] = "pairwise scores are not calibrated predicted_delta_u_ae_pct"
    return rows


def run_ae_utility_calibrator_methods_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    true_nelbo: np.ndarray,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    metadata_similarity: np.ndarray,
    metadata_similarity_eval: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    cfg: AEUtilityCalibratorConfig,
    seed: int,
    tie_policy: str,
) -> AEUtilityCalibratorFoldOutputs:
    if not bool(cfg.enabled):
        return AEUtilityCalibratorFoldOutputs([], [], [], [], [], [], [], [], [])
    if str(cfg.primary_method) != PRIMARY_METHOD:
        raise ProtocolError("AE utility calibrator primary_method must be ae_utility_calibrated_safe_override_v1")
    if str(cfg.primary_model_type) != "ridge_delta" or set(cfg.model_types) != {"ridge_delta"}:
        raise ProtocolError("AE utility calibrator v1 primary model_types must be ['ridge_delta']")
    if str(cfg.fallback_policy) != "ae_argmin_zscore":
        raise ProtocolError("AE utility calibrator fallback_policy must be ae_argmin_zscore")

    sample_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    override_diag_rows: List[Dict[str, Any]] = []
    oracle_headroom_rows: List[Dict[str, Any]] = []
    selected_feature_rows: List[Dict[str, Any]] = []
    override_precision_rows: List[Dict[str, Any]] = []
    anchor_rank_rows: List[Dict[str, Any]] = []

    candidate_cols = list(fold.candidate_col_indices)
    ae_zscore_eval = ae_scores.zscore_matrix[np.asarray(test_idx, dtype=np.int64)][:, candidate_cols]
    metadata_idx = _metadata_selected_local_indices(metadata_similarity_eval)
    train_fold_for_sample = lambda sample_index: FoldCandidateSet.for_heldout_domain(
        heldout_domain=int(fold.heldout_domain),
        expert_domains=expert_domains,
        excluded_domains=[int(sample_domains[int(sample_index)])],
    )
    eval_fold_for_sample = lambda _sample_index: fold

    primary_anchor_idx: np.ndarray | None = None
    primary_selected_idx: np.ndarray | None = None

    for method, feature_sets, method_kind in _method_feature_sets(cfg):
        selected_cfg, rows = _select_config_for_method(
            method=method,
            feature_sets=feature_sets,
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            train_idx=train_idx,
            outer_fold=fold,
            metadata_similarity=metadata_similarity,
            ae_scores=ae_scores,
            cfg=cfg,
        )
        validation_rows.extend(rows)
        eval_rows, pred_flat, pred_matrix, anchor_idx = _train_predict_for_fold(
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            train_idx=train_idx,
            eval_idx=test_idx,
            train_fold_for_sample=train_fold_for_sample,
            eval_fold_for_sample=eval_fold_for_sample,
            metadata_similarity=metadata_similarity,
            ae_scores=ae_scores,
            feature_set=selected_cfg.feature_set,
            ridge_l2=float(cfg.ridge_l2),
            n_eval_candidates=len(fold.candidate_expert_domains),
        )
        selected_idx, best_override, best_delta, override_margin = _apply_safe_override_policy(
            pred_delta_matrix=pred_matrix,
            anchor_idx=anchor_idx,
            delta_threshold=float(selected_cfg.delta_threshold),
            margin_threshold=float(selected_cfg.margin_threshold),
        )
        summary = _policy_summary(
            selected_idx=selected_idx,
            anchor_idx=anchor_idx,
            pred_delta_matrix=pred_matrix,
            true_eval=true_eval,
            metadata_idx=metadata_idx,
            ae_zscore_eval=ae_zscore_eval,
        )
        score_matrix = -pred_matrix
        _metrics_unused, rows_for_method = _selection_metrics(
            method=method,
            query_domains=sample_domains[test_idx],
            expert_domains=fold.candidate_expert_domains,
            score_matrix=score_matrix,
            true_nelbo_matrix=true_eval,
            fold=fold,
            global_true_nelbo_matrix=global_eval,
            global_expert_domains=expert_domains,
            tie_policy=tie_policy,
            selected_idx_override=selected_idx,
            ranking_score_matrix=score_matrix,
        )
        rows_idx = np.arange(true_eval.shape[0])
        anchor_nelbo = true_eval[rows_idx, anchor_idx]
        selected_nelbo = true_eval[rows_idx, selected_idx]
        true_delta = _true_delta_matrix(true_eval, anchor_idx)
        for row in rows_for_method:
            local = int(row["sample_index"])
            row_metadata_nelbo = float(true_eval[local, int(metadata_idx[local])])
            row_improves_anchor = int(selected_nelbo[local] < anchor_nelbo[local])
            row_harms_anchor = int(selected_nelbo[local] > anchor_nelbo[local])
            row_active = int(int(selected_idx[local]) != int(anchor_idx[local]))
            row["sample_index"] = int(test_idx[local])
            row.update(
                {
                    "model_type": "ridge_delta",
                    "feature_set": selected_cfg.feature_set,
                    "method_kind": method_kind,
                    "fallback_policy": "ae_argmin_zscore",
                    "ae_anchor_expert": int(fold.candidate_expert_domains[int(anchor_idx[local])]),
                    "override_candidate_expert": (
                        int(fold.candidate_expert_domains[int(best_override[local])])
                        if int(best_override[local]) >= 0
                        else ""
                    ),
                    "override_accepted": int(int(selected_idx[local]) != int(anchor_idx[local])),
                    "predicted_delta_best_override": float(best_delta[local]),
                    "true_delta_best_override": (
                        float(true_delta[local, int(best_override[local])])
                        if int(best_override[local]) >= 0
                        else float("nan")
                    ),
                    "predicted_override_margin": float(override_margin[local]),
                    "selected_delta_threshold": _threshold_label(float(selected_cfg.delta_threshold)),
                    "selected_margin_threshold": _threshold_label(float(selected_cfg.margin_threshold)),
                    "selected_by_source_inner_validation": int(selected_cfg.selected_by_source_inner),
                    "target_support_free": 1,
                    "target_support_used": 0,
                    "target_ae_excluded": 1,
                    "source_inner_self_ae_excluded": 1,
                    "source_inner_self_expert_excluded": 1,
                    "metadata_role": "not_used" if method == PRIMARY_METHOD else "hybrid_auxiliary_feature",
                    "proxy_claim_boundary": "AE reconstruction fit is a proxy for CVAE utility, not compatibility.",
                    "net_gain_vs_ae_argmin": float(anchor_nelbo[local] - selected_nelbo[local]),
                    "net_gain_vs_metadata": float(row_metadata_nelbo - selected_nelbo[local]),
                    "active_override": row_active,
                    "harmful_vs_ae_argmin": row_harms_anchor,
                    "improving_vs_ae_argmin": row_improves_anchor,
                    "selected_override_precision": (
                        float(row_improves_anchor)
                        if row_active
                        else float("nan")
                    ),
                    "raw_predicted_delta_spearman_non_anchor": float(
                        summary["raw_predicted_delta_spearman_non_anchor"]
                    ),
                    "raw_predicted_delta_spearman_with_anchor": float(
                        summary["raw_predicted_delta_spearman_with_anchor"]
                    ),
                }
            )
            sample_rows.append(row)

        for k, value in enumerate(pred_flat.tolist()):
            local = int(eval_rows.sample_positions[k])
            raw_rows.append(
                {
                    "method": method,
                    "fold_query_domain": int(fold.heldout_domain),
                    "sample_index": int(eval_rows.sample_indices[k]),
                    "query_domain": int(eval_rows.query_domains[k]),
                    "feature_set": selected_cfg.feature_set,
                    "candidate_expert": int(eval_rows.expert_domains[k]),
                    "ae_anchor_expert": int(fold.candidate_expert_domains[int(anchor_idx[local])]),
                    "predicted_delta_u_ae_pct": float(value),
                    "true_delta_u_ae_pct": float(eval_rows.y_delta[k]),
                    "ae_zscore": float(eval_rows.ae_z[k]),
                    "anchor_ae_zscore": float(eval_rows.anchor_ae_z[k]),
                    "ae_rank": int(eval_rows.ae_rank[k]),
                    "ae_margin": float(eval_rows.ae_margin[k]),
                    "override_candidate_expert": int(fold.candidate_expert_domains[int(best_override[local])])
                    if int(best_override[local]) >= 0
                    else "",
                    "override_accepted": int(int(selected_idx[local]) != int(anchor_idx[local])),
                    "predicted_override_margin": float(override_margin[local]),
                    "heldout_target_nelbo_used_for_selection": 0,
                }
            )

        policy = {
            "method": method,
            "outer_heldout_domain": int(fold.heldout_domain),
            "fold_query_domain": int(fold.heldout_domain),
            "query_domain": int(fold.heldout_domain),
            "aggregation_unit": "seed_x_heldout_domain_x_query_domain",
            "primary_aggregation": "macro_by_domain",
            "model_type": "ridge_delta",
            "feature_set": selected_cfg.feature_set,
            "method_kind": method_kind,
            "selected_delta_threshold": _threshold_label(float(selected_cfg.delta_threshold)),
            "selected_margin_threshold": _threshold_label(float(selected_cfg.margin_threshold)),
            "source_train_domains": "|".join(
                str(int(v)) for v in sorted(set(int(sample_domains[int(i)]) for i in train_idx.tolist()))
            ),
            "source_inner_pseudo_query_domain": "source_inner_loqdo",
            "excluded_target_ae": 1,
            "excluded_target_cvae": 1,
            "excluded_pseudo_query_ae": 1,
            "excluded_pseudo_query_cvae": 1,
            "ae_stats_domains_used": fold.label(),
            "threshold_selection_domains_used": "source_train_domains_only",
            "model_training_domains_used": "source_train_domains_only",
            "heldout_target_nelbo_used_for_selection": 0,
            **summary,
        }
        policy_rows.append(policy)
        override_precision_rows.append(
            {
                "method": method,
                "fold_query_domain": int(fold.heldout_domain),
                "active_overrides": int(np.sum(selected_idx != anchor_idx)),
                "selected_override_precision": float(summary["selected_override_precision"]),
                "active_override_rate": float(summary["active_override_rate"]),
                "override_capture_rate": float(summary["override_capture_rate"]),
            }
        )
        override_diag_rows.append(
            {
                "method": method,
                "fold_query_domain": int(fold.heldout_domain),
                "active_override_rate": float(summary["active_override_rate"]),
                "fallback_to_ae_argmin_rate": float(summary["fallback_to_ae_argmin_rate"]),
                "net_gain_vs_ae_argmin": float(summary["net_gain_vs_ae_argmin"]),
                "net_gain_vs_metadata": float(summary["net_gain_vs_metadata"]),
                "harmful_vs_ae_argmin_rate": float(summary["harmful_vs_ae_argmin_rate"]),
                "improving_vs_ae_argmin_rate": float(summary["improving_vs_ae_argmin_rate"]),
                "harmful_vs_metadata_rate": float(summary["harmful_vs_metadata_rate"]),
                "improving_vs_metadata_rate": float(summary["improving_vs_metadata_rate"]),
            }
        )
        selected_feature_rows.append(
            {
                "method": method,
                "fold_query_domain": int(fold.heldout_domain),
                "selected_feature_set": selected_cfg.feature_set,
                "selected_delta_threshold": _threshold_label(float(selected_cfg.delta_threshold)),
                "selected_margin_threshold": _threshold_label(float(selected_cfg.margin_threshold)),
                "selected_by_source_inner_validation": int(selected_cfg.selected_by_source_inner),
            }
        )
        if method == PRIMARY_METHOD:
            primary_anchor_idx = anchor_idx
            primary_selected_idx = selected_idx

    if primary_anchor_idx is not None and primary_selected_idx is not None:
        headroom, anchor_rows = _oracle_headroom_rows(
            fold=fold,
            test_idx=test_idx,
            sample_domains=sample_domains,
            true_eval=true_eval,
            anchor_idx=primary_anchor_idx,
            selected_idx=primary_selected_idx,
        )
        oracle_headroom_rows.extend(headroom)
        anchor_rank_rows.extend(anchor_rows)

    sample_rows.extend(
        _run_pairwise_diagnostic(
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            train_idx=train_idx,
            test_idx=test_idx,
            fold=fold,
            global_eval=global_eval,
            metadata_similarity=metadata_similarity,
            ae_scores=ae_scores,
            cfg=cfg,
            seed=int(seed),
            tie_policy=tie_policy,
        )
    )

    return AEUtilityCalibratorFoldOutputs(
        sample_rows=sample_rows,
        raw_rows=raw_rows,
        source_inner_validation_rows=validation_rows,
        policy_audit_rows=policy_rows,
        override_diagnostic_rows=override_diag_rows,
        oracle_headroom_rows=oracle_headroom_rows,
        selected_feature_rows=selected_feature_rows,
        override_precision_rows=override_precision_rows,
        anchor_rank_rows=anchor_rank_rows,
    )


def write_ae_utility_calibrator_artifacts(
    *,
    reports_dir: Path,
    raw_rows: Sequence[Dict[str, Any]],
    source_inner_validation_rows: Sequence[Dict[str, Any]],
    policy_audit_rows: Sequence[Dict[str, Any]],
    override_diagnostic_rows: Sequence[Dict[str, Any]],
    oracle_headroom_rows: Sequence[Dict[str, Any]],
    selected_feature_rows: Sequence[Dict[str, Any]],
    override_precision_rows: Sequence[Dict[str, Any]],
    anchor_rank_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not (
        raw_rows
        or source_inner_validation_rows
        or policy_audit_rows
        or override_diagnostic_rows
        or oracle_headroom_rows
        or selected_feature_rows
        or override_precision_rows
        or anchor_rank_rows
    ):
        return {}
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(reports_dir / "ae_utility_calibrator_raw.csv", raw_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_source_inner_validation.csv", source_inner_validation_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_policy_audit.csv", policy_audit_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_override_diagnostics.csv", override_diagnostic_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_oracle_headroom.csv", oracle_headroom_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_selected_feature_sets.csv", selected_feature_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_override_precision.csv", override_precision_rows)
    _write_csv(reports_dir / "ae_utility_calibrator_anchor_rank_diagnostics.csv", anchor_rank_rows)
    return {
        "ae_utility_calibrator_raw": "ae_utility_calibrator_raw.csv",
        "ae_utility_calibrator_source_inner_validation": "ae_utility_calibrator_source_inner_validation.csv",
        "ae_utility_calibrator_policy_audit": "ae_utility_calibrator_policy_audit.csv",
        "ae_utility_calibrator_override_diagnostics": "ae_utility_calibrator_override_diagnostics.csv",
        "ae_utility_calibrator_oracle_headroom": "ae_utility_calibrator_oracle_headroom.csv",
        "ae_utility_calibrator_selected_feature_sets": "ae_utility_calibrator_selected_feature_sets.csv",
        "ae_utility_calibrator_override_precision": "ae_utility_calibrator_override_precision.csv",
        "ae_utility_calibrator_anchor_rank_diagnostics": "ae_utility_calibrator_anchor_rank_diagnostics.csv",
    }
