from __future__ import annotations

from dataclasses import dataclass
import math
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
PRIMARY_METHOD_V11 = "ae_utility_calibrated_precision_lcb_safe_override_v11"
PRIMARY_METHOD_V2 = "ae_utility_calibrated_consensus_safe_override_v2"
HYBRID_METADATA_METHOD = "ae_metadata_utility_calibrated_safe_override_v1"
HYBRID_COMBINED_METHOD = "ae_combined_utility_calibrated_safe_override_v1"
HYBRID_METADATA_METHOD_V2 = "ae_metadata_utility_calibrated_consensus_safe_override_v2"
HYBRID_COMBINED_METHOD_V2 = "ae_combined_utility_calibrated_consensus_safe_override_v2"
PAIRWISE_DIAG_METHOD = "ae_utility_pairwise_ranker_diagnostic_v1"
ORACLE_HEADROOM_METHOD = "oracle_safe_override_over_ae_argmin"
V2_METHODS = {PRIMARY_METHOD_V2, HYBRID_METADATA_METHOD_V2, HYBRID_COMBINED_METHOD_V2}
V11_METHODS = {PRIMARY_METHOD_V11}
V2_PRIMARY_FEATURE_SETS = {"ae_consensus_core", "ae_consensus_quality"}
V2_DIAGNOSTIC_FEATURE_SETS = {"ae_metadata_consensus", "ae_combined_consensus"}


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
    consensus_threshold: float = 0.0
    selection_status: str = "source_inner_selected"
    fallback_reason: str = ""


@dataclass(frozen=True)
class _ConsensusPredictions:
    mean_matrix: np.ndarray
    std_matrix: np.ndarray
    lower_matrix: np.ndarray
    positive_rate_matrix: np.ndarray
    n_members_matrix: np.ndarray
    n_positive_matrix: np.ndarray
    member_labels: Tuple[str, ...]


def _safe_div(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return num / np.maximum(np.abs(denom), 1e-12)


def _finite_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float(default)


def _quality_by_domain(ae_scores: AutoencoderScoreMatrices) -> Dict[int, Dict[str, Any]]:
    return {int(row.get("source_domain")): dict(row) for row in ae_scores.quality_rows}


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def _uses_quality_features(feature_set: str) -> bool:
    return str(feature_set) in {
        "ae_quality",
        "ae_metadata",
        "ae_combined",
        "ae_consensus_quality",
        "ae_metadata_consensus",
        "ae_combined_consensus",
    }


def _uses_metadata_features(feature_set: str) -> bool:
    return str(feature_set) in {
        "ae_metadata",
        "ae_combined",
        "ae_metadata_consensus",
        "ae_combined_consensus",
    }


def _uses_combined_features(feature_set: str) -> bool:
    return str(feature_set) in {"ae_combined", "ae_combined_consensus"}


def _uses_consensus_features(feature_set: str) -> bool:
    return str(feature_set) in V2_PRIMARY_FEATURE_SETS | V2_DIAGNOSTIC_FEATURE_SETS


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
    if _uses_consensus_features(feature_set):
        candidate_quantile = _normal_cdf(candidate_z)
        anchor_quantile = _normal_cdf(anchor_z)
        values: List[float] = [
            float(ae_ranks[int(candidate_local)]),
            float(ae_ranks[int(candidate_local)]) / n_candidates,
            float(ae_margin),
            candidate_z - anchor_z,
            candidate_quantile,
            anchor_quantile,
            candidate_quantile - anchor_quantile,
        ]
    else:
        values = [
            candidate_z,
            anchor_z,
            candidate_z - anchor_z,
            float(ae_ranks[int(candidate_local)]),
            float(ae_ranks[int(candidate_local)]) / n_candidates,
            float(ae_margin),
            candidate_raw,
            anchor_raw,
        ]
    if _uses_quality_features(feature_set):
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
    if _uses_metadata_features(feature_set):
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
    if _uses_combined_features(feature_set):
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


def _wilson_bounds(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if int(total) <= 0:
        return float("nan"), float("nan")
    n = float(total)
    p = float(successes) / n
    z2 = float(z) ** 2
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    radius = float(z) * math.sqrt(max((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)), 0.0))
    return float(max(0.0, (centre - radius) / denom)), float(min(1.0, (centre + radius) / denom))


def _bootstrap_lcb(values: Sequence[float], *, reps: int, seed: int, quantile: float = 0.025) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    if vals.size == 1 or int(reps) <= 0:
        return float(vals[0])
    rng = np.random.default_rng(int(seed))
    means = np.empty((int(reps),), dtype=np.float64)
    for i in range(int(reps)):
        idx = rng.integers(0, vals.size, size=vals.size)
        means[i] = float(np.mean(vals[idx]))
    return float(np.quantile(means, float(quantile)))


def _override_classification_summary(
    *,
    selected_idx: np.ndarray,
    anchor_idx: np.ndarray,
    true_eval: np.ndarray,
    neutral_gap_pct_band: float,
) -> Dict[str, float]:
    rows = np.arange(true_eval.shape[0])
    oracle_idx = _stable_argmin_indices(true_eval)
    selected_nelbo = true_eval[rows, selected_idx]
    anchor_nelbo = true_eval[rows, anchor_idx]
    oracle_nelbo = true_eval[rows, oracle_idx]
    selected_gap_pct = ((selected_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    anchor_gap_pct = ((anchor_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
    delta_gap_pct = selected_gap_pct - anchor_gap_pct
    active = np.asarray(selected_idx, dtype=np.int64) != np.asarray(anchor_idx, dtype=np.int64)
    active_count = int(np.sum(active))
    band = float(neutral_gap_pct_band)
    improving = active & (delta_gap_pct <= -band)
    harmful = active & (delta_gap_pct >= band)
    neutral = active & ~(improving | harmful)
    improving_count = int(np.sum(improving))
    neutral_count = int(np.sum(neutral))
    harmful_count = int(np.sum(harmful))
    strict_lcb, _strict_ucb = _wilson_bounds(improving_count, active_count)
    safe_lcb, _safe_ucb = _wilson_bounds(improving_count + neutral_count, active_count)
    _harm_lcb, harmful_ucb = _wilson_bounds(harmful_count, active_count)
    gains = anchor_nelbo - selected_nelbo
    improving_gains = gains[improving]
    harmful_losses = selected_nelbo[harmful] - anchor_nelbo[harmful]
    return {
        "active_override_count": active_count,
        "active_override_rate_report": float(np.mean(active)) if active.size else 0.0,
        "strict_improvement_precision": (
            float(improving_count / active_count) if active_count > 0 else float("nan")
        ),
        "strict_improvement_precision_lcb": strict_lcb,
        "safe_override_precision": (
            float((improving_count + neutral_count) / active_count) if active_count > 0 else float("nan")
        ),
        "safe_override_precision_lcb": safe_lcb,
        "harmful_override_rate": float(harmful_count / active_count) if active_count > 0 else float("nan"),
        "harmful_override_rate_ucb": harmful_ucb,
        "improving_override_rate": float(improving_count / active_count) if active_count > 0 else float("nan"),
        "neutral_override_rate": float(neutral_count / active_count) if active_count > 0 else float("nan"),
        "mean_gain_improving_overrides": (
            float(np.mean(improving_gains)) if improving_gains.size else float("nan")
        ),
        "mean_loss_harmful_overrides": (
            float(np.mean(harmful_losses)) if harmful_losses.size else float("nan")
        ),
    }


def _policy_summary(
    *,
    selected_idx: np.ndarray,
    anchor_idx: np.ndarray,
    pred_delta_matrix: np.ndarray,
    true_eval: np.ndarray,
    metadata_idx: np.ndarray,
    ae_zscore_eval: np.ndarray,
    abstention_correct_gap_pct_epsilon: float = 1.0,
    neutral_gap_pct_band: float = 0.25,
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
    oracle_headroom = np.maximum(anchor_nelbo - oracle_nelbo, 0.0)
    captured_headroom = np.maximum(anchor_nelbo - selected_nelbo, 0.0)
    headroom_denom = float(np.sum(oracle_headroom))
    abstained = ~active
    abstained_count = int(np.sum(abstained))
    abstention_correct = abstained & (
        (anchor_idx == oracle_idx) | (anchor_gap_pct <= float(abstention_correct_gap_pct_epsilon))
    )
    abstention_missed_gain_values = anchor_nelbo[abstained] - oracle_nelbo[abstained]
    override_classification = _override_classification_summary(
        selected_idx=selected_idx,
        anchor_idx=anchor_idx,
        true_eval=true_eval,
        neutral_gap_pct_band=float(neutral_gap_pct_band),
    )
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
        "captured_oracle_headroom_rate": (
            float(np.sum(captured_headroom) / headroom_denom)
            if headroom_denom > 0.0
            else float("nan")
        ),
        "ae_argmin_already_oracle_rate": float(np.mean(anchor_idx == oracle_idx)) if anchor_idx.size else 0.0,
        "abstention_rate": float(np.mean(abstained)) if abstained.size else 1.0,
        "abstention_correct_rate": (
            float(np.sum(abstention_correct) / abstained_count)
            if abstained_count > 0
            else float("nan")
        ),
        "abstention_missed_gain": (
            float(np.mean(abstention_missed_gain_values))
            if abstention_missed_gain_values.size
            else float("nan")
        ),
        **override_classification,
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


def _worst_pseudo_domain_gap_degradation(summary_rows: Sequence[Mapping[str, float]]) -> float:
    values = [
        float(row["mean_oracle_gap_pct"]) - float(row["ae_argmin_mean_oracle_gap_pct"])
        for row in summary_rows
        if np.isfinite(float(row.get("mean_oracle_gap_pct", float("nan"))))
        and np.isfinite(float(row.get("ae_argmin_mean_oracle_gap_pct", float("nan"))))
    ]
    return float(max(values)) if values else float("inf")


def _source_inner_gap_reduction_lcb(
    summary_rows: Sequence[Mapping[str, float]],
    cfg: AEUtilityCalibratorConfig,
) -> float:
    reductions = [_gap_reduction(row) for row in summary_rows]
    return _bootstrap_lcb(
        reductions,
        reps=int(cfg.precision_bootstrap_reps),
        seed=int(cfg.precision_bootstrap_seed),
    )


def _precision_lcb_metrics(
    summary_rows: Sequence[Mapping[str, float]],
    cfg: AEUtilityCalibratorConfig,
) -> Dict[str, float]:
    active_count = int(sum(int(float(row.get("active_override_count", 0.0))) for row in summary_rows))
    improving_count = 0
    neutral_count = 0
    harmful_count = 0
    for row in summary_rows:
        row_active = int(float(row.get("active_override_count", 0.0)))
        if row_active <= 0:
            continue
        improving_count += int(round(float(row.get("improving_override_rate", 0.0)) * row_active))
        neutral_count += int(round(float(row.get("neutral_override_rate", 0.0)) * row_active))
        harmful_count += int(round(float(row.get("harmful_override_rate", 0.0)) * row_active))
    strict_lcb, _strict_ucb = _wilson_bounds(improving_count, active_count)
    safe_lcb, _safe_ucb = _wilson_bounds(improving_count + neutral_count, active_count)
    _harm_lcb, harmful_ucb = _wilson_bounds(harmful_count, active_count)
    strict_precision = float(improving_count / active_count) if active_count > 0 else float("nan")
    safe_precision = float((improving_count + neutral_count) / active_count) if active_count > 0 else float("nan")
    harmful_rate = float(harmful_count / active_count) if active_count > 0 else float("nan")
    improving_rate = float(improving_count / active_count) if active_count > 0 else float("nan")
    neutral_rate = float(neutral_count / active_count) if active_count > 0 else float("nan")
    macro_gap_lcb = _source_inner_gap_reduction_lcb(summary_rows, cfg)
    worst_gap_degradation = _worst_pseudo_domain_gap_degradation(summary_rows)
    passes = bool(
        active_count >= int(cfg.min_active_override_count)
        and strict_precision >= float(cfg.min_strict_improvement_precision)
        and strict_lcb >= float(cfg.min_strict_improvement_precision_lcb)
        and _finite_mean([float(row.get("active_override_rate", 0.0)) for row in summary_rows]) >= float(cfg.min_active_override_rate)
        and _finite_mean([float(row.get("net_gain_vs_ae_argmin", 0.0)) for row in summary_rows]) >= float(cfg.min_net_gain_vs_ae_argmin)
        and harmful_rate <= improving_rate
        and worst_gap_degradation <= float(cfg.max_worst_pseudo_domain_gap_degradation_pp)
    )
    return {
        "active_override_count_source_inner": active_count,
        "active_override_rate_source_inner": _finite_mean(
            [float(row.get("active_override_rate", 0.0)) for row in summary_rows]
        ),
        "net_gain_vs_ae_argmin_source_inner": _finite_mean(
            [float(row.get("net_gain_vs_ae_argmin", 0.0)) for row in summary_rows]
        ),
        "strict_improvement_precision_source_inner": strict_precision,
        "strict_improvement_precision_lcb_source_inner": strict_lcb,
        "safe_override_precision_source_inner": safe_precision,
        "safe_override_precision_lcb_source_inner": safe_lcb,
        "harmful_override_rate_ucb_source_inner": harmful_ucb,
        "improving_override_rate_source_inner": improving_rate,
        "neutral_override_rate_source_inner": neutral_rate,
        "harmful_override_rate_source_inner": harmful_rate,
        "source_inner_macro_gap_reduction_lcb": macro_gap_lcb,
        "worst_pseudo_domain_gap_degradation_pp": worst_gap_degradation,
        "passes_precision_lcb_gates": int(passes),
    }


def _gap_reduction(summary: Mapping[str, float]) -> float:
    return float(summary["ae_argmin_mean_oracle_gap_pct"]) - float(summary["mean_oracle_gap_pct"])


def _material_degradation_vs_ae_argmin(summary: Mapping[str, float], cfg: AEUtilityCalibratorConfig) -> bool:
    top1_drop = float(summary["ae_argmin_top1_oracle_hit"]) - float(summary["top1_oracle_hit"])
    spearman_drop = float(summary["ae_delta_spearman_non_anchor"]) - float(
        summary["raw_predicted_delta_spearman_non_anchor"]
    )
    gap_degradation = float(summary["mean_oracle_gap_pct"]) - float(summary["ae_argmin_mean_oracle_gap_pct"])
    return bool(
        top1_drop > float(cfg.max_top1_drop_vs_ae_argmin_abs)
        or spearman_drop > float(cfg.max_spearman_drop_vs_ae_argmin_abs)
        or gap_degradation > float(cfg.max_gap_pct_degradation_vs_ae_argmin)
    )


def _passes_consensus_risk_gates(summary: Mapping[str, float], cfg: AEUtilityCalibratorConfig) -> bool:
    precision = float(summary.get("selected_override_precision", float("nan")))
    precision_ok = (
        float(summary.get("active_override_rate", 0.0)) <= 0.0
        or (np.isfinite(precision) and precision >= 0.50)
    )
    return bool(
        not _material_degradation_vs_ae_argmin(summary, cfg)
        and float(summary["harmful_vs_ae_argmin_rate"]) <= float(summary["improving_vs_ae_argmin_rate"])
        and precision_ok
        and float(summary["net_gain_vs_ae_argmin"]) >= 0.0
    )


def _source_inner_stability(summary_rows: Sequence[Mapping[str, float]], cfg: AEUtilityCalibratorConfig) -> Dict[str, float]:
    if not summary_rows:
        return {
            "source_inner_pseudo_domain_positive_rate": 0.0,
            "max_pseudo_domain_gain_share": 1.0,
            "max_source_inner_fold_gain_share": 1.0,
            "source_inner_material_degradation_count": 1,
            "passes_source_inner_stability_gates": 0,
        }
    gains = np.asarray([max(_gap_reduction(row), 0.0) for row in summary_rows], dtype=np.float64)
    total_gain = float(np.sum(gains))
    max_share = float(np.max(gains) / total_gain) if total_gain > 0.0 and gains.size else 1.0
    positive_rate = float(np.mean(gains > 0.0)) if gains.size else 0.0
    material_count = int(sum(1 for row in summary_rows if _material_degradation_vs_ae_argmin(row, cfg)))
    passes = bool(
        material_count == 0
        and positive_rate >= float(cfg.min_pseudo_domain_positive_rate)
        and max_share <= float(cfg.max_pseudo_domain_gain_share)
        and max_share <= float(cfg.max_source_inner_fold_gain_share)
    )
    return {
        "source_inner_pseudo_domain_positive_rate": positive_rate,
        "max_pseudo_domain_gain_share": max_share,
        "max_source_inner_fold_gain_share": max_share,
        "source_inner_material_degradation_count": material_count,
        "passes_source_inner_stability_gates": int(passes),
    }


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


def _member_train_indices(
    *,
    train_idx: np.ndarray,
    sample_domains: np.ndarray,
    excluded_member_domain: int | None,
) -> np.ndarray:
    if excluded_member_domain is None:
        return np.asarray(train_idx, dtype=np.int64)
    return np.asarray(
        [i for i in np.asarray(train_idx, dtype=np.int64).tolist() if int(sample_domains[int(i)]) != int(excluded_member_domain)],
        dtype=np.int64,
    )


def _train_predict_consensus_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    heldout_domain: int,
    eval_fold_for_sample: Callable[[int], FoldCandidateSet],
    metadata_similarity: np.ndarray,
    ae_scores: AutoencoderScoreMatrices,
    feature_set: str,
    ridge_l2: float,
    n_eval_candidates: int,
    member_excluded_domains: Sequence[int | None],
    uncertainty_multiplier: float,
) -> Tuple[_FeatureRows, _ConsensusPredictions, np.ndarray]:
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
    anchor_idx = _anchor_indices_from_rows(rows=eval_rows, n_samples=int(eval_idx.shape[0]))
    pred_matrices: List[np.ndarray] = []
    member_labels: List[str] = []
    for excluded_member_domain in member_excluded_domains:
        member_train_idx = _member_train_indices(
            train_idx=train_idx,
            sample_domains=sample_domains,
            excluded_member_domain=excluded_member_domain,
        )
        if member_train_idx.size == 0:
            continue
        train_fold_for_sample = (
            lambda sample_index, h=int(heldout_domain), excluded=excluded_member_domain: FoldCandidateSet.for_heldout_domain(
                heldout_domain=h,
                expert_domains=expert_domains,
                excluded_domains=[
                    v
                    for v in [int(sample_domains[int(sample_index)]), excluded]
                    if v is not None
                ],
            )
        )
        train_rows = _build_feature_rows(
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            sample_indices=member_train_idx,
            fold_for_sample=train_fold_for_sample,
            metadata_similarity=metadata_similarity,
            ae_scores=ae_scores,
            feature_set=feature_set,
            exclude_anchor=True,
        )
        pred = _fit_predict_delta(train_rows=train_rows, eval_rows=eval_rows, ridge_l2=float(ridge_l2))
        if train_rows.x.size == 0 or pred.size == 0:
            continue
        pred_matrices.append(
            _prediction_matrix_from_rows(
                rows=eval_rows,
                predicted_delta=pred,
                n_samples=int(eval_idx.shape[0]),
                n_candidates=int(n_eval_candidates),
            )
        )
        member_labels.append("full_source" if excluded_member_domain is None else f"leave_domain_{int(excluded_member_domain)}")

    if len(pred_matrices) < 2:
        shape = (int(eval_idx.shape[0]), int(n_eval_candidates))
        consensus = _ConsensusPredictions(
            mean_matrix=np.zeros(shape, dtype=np.float64),
            std_matrix=np.full(shape, float("nan"), dtype=np.float64),
            lower_matrix=np.full(shape, float("-inf"), dtype=np.float64),
            positive_rate_matrix=np.zeros(shape, dtype=np.float64),
            n_members_matrix=np.zeros(shape, dtype=np.float64),
            n_positive_matrix=np.zeros(shape, dtype=np.float64),
            member_labels=tuple(member_labels),
        )
        return eval_rows, consensus, anchor_idx

    stacked = np.stack(pred_matrices, axis=0).astype(np.float64, copy=False)
    mean_matrix = np.mean(stacked, axis=0)
    std_matrix = np.std(stacked, axis=0)
    lower_matrix = mean_matrix - float(uncertainty_multiplier) * std_matrix
    n_positive = np.sum(stacked > 0.0, axis=0).astype(np.float64, copy=False)
    n_members = np.full(mean_matrix.shape, float(stacked.shape[0]), dtype=np.float64)
    consensus = _ConsensusPredictions(
        mean_matrix=mean_matrix,
        std_matrix=std_matrix,
        lower_matrix=lower_matrix,
        positive_rate_matrix=n_positive / np.maximum(n_members, 1.0),
        n_members_matrix=n_members,
        n_positive_matrix=n_positive,
        member_labels=tuple(member_labels),
    )
    return eval_rows, consensus, anchor_idx


def _apply_consensus_safe_override_policy(
    *,
    consensus: _ConsensusPredictions,
    anchor_idx: np.ndarray,
    delta_threshold: float,
    margin_threshold: float,
    consensus_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols = consensus.lower_matrix.shape
    selected = np.asarray(anchor_idx, dtype=np.int64).copy()
    best_override = np.full((n_rows,), -1, dtype=np.int64)
    second_override = np.full((n_rows,), -1, dtype=np.int64)
    lower_best = np.full((n_rows,), float("-inf"), dtype=np.float64)
    lower_second = np.full((n_rows,), float("-inf"), dtype=np.float64)
    margins = np.full((n_rows,), float("inf"), dtype=np.float64)
    positive_rate_best = np.zeros((n_rows,), dtype=np.float64)
    for i in range(n_rows):
        candidates = [j for j in range(n_cols) if int(j) != int(anchor_idx[i])]
        if not candidates:
            continue
        order = sorted(candidates, key=lambda j: (-float(consensus.lower_matrix[i, j]), int(j)))
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else -1
        best_override[i] = best
        second_override[i] = second
        lower_best[i] = float(consensus.lower_matrix[i, best])
        lower_second[i] = float(consensus.lower_matrix[i, second]) if second >= 0 else float("-inf")
        margins[i] = (
            float(lower_best[i] - lower_second[i])
            if np.isfinite(lower_second[i])
            else float("inf")
        )
        positive_rate_best[i] = float(consensus.positive_rate_matrix[i, best])
        if (
            float(lower_best[i]) >= float(delta_threshold)
            and float(margins[i]) >= float(margin_threshold)
            and float(positive_rate_best[i]) >= float(consensus_threshold)
        ):
            selected[i] = best
    return selected, best_override, second_override, lower_best, lower_second, margins, positive_rate_best


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
                        neutral_gap_pct_band=float(cfg.neutral_override_gap_pct_band),
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
                            "threshold_selection_policy": (
                                "source_inner_precision_lcb_then_gap_lcb"
                                if str(method) == PRIMARY_METHOD_V11
                                else "source_inner_ae_argmin_noninferiority_then_gap"
                            ),
                            "selection_mode": (
                                str(cfg.selection_mode)
                                if str(method) == PRIMARY_METHOD_V11
                                else ""
                            ),
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
            precision_metrics = _precision_lcb_metrics(summaries, cfg)
            config_summaries.append(
                {
                    "method": str(method),
                    "feature_set": str(feature_set),
                    "delta_threshold": float(delta_threshold),
                    "margin_threshold": float(margin_threshold),
                    "passes_source_inner_risk_gates": bool(passes),
                    **precision_metrics,
                    **macro,
                }
            )
            if str(method) == PRIMARY_METHOD_V11:
                validation_rows.append(
                    {
                        "method": str(method),
                        "feature_set": str(feature_set),
                        "model_type": "ridge_delta",
                        "fold_query_domain": int(outer_fold.heldout_domain),
                        "source_inner_pseudo_query_domain": "source_inner_macro",
                        "delta_threshold": _threshold_label(float(delta_threshold)),
                        "margin_threshold": _threshold_label(float(margin_threshold)),
                        "threshold_selection_policy": "source_inner_precision_lcb_then_gap_lcb",
                        "selection_mode": str(cfg.selection_mode),
                        "n_validation_samples": int(
                            sum(int(row.get("active_override_count", 0.0)) for row in summaries)
                        ),
                        "candidate_experts": "source_inner_macro",
                        "excluded_target_ae": 1,
                        "excluded_target_cvae": 1,
                        "excluded_pseudo_query_ae": 1,
                        "excluded_pseudo_query_cvae": 1,
                        "heldout_target_nelbo_used_for_selection": 0,
                        "precision_gate_passed": int(precision_metrics["passes_precision_lcb_gates"]),
                        "strict_improvement_precision_source_inner": float(
                            precision_metrics["strict_improvement_precision_source_inner"]
                        ),
                        "strict_improvement_precision_lcb_source_inner": float(
                            precision_metrics["strict_improvement_precision_lcb_source_inner"]
                        ),
                        "safe_override_precision_source_inner": float(
                            precision_metrics["safe_override_precision_source_inner"]
                        ),
                        "safe_override_precision_lcb_source_inner": float(
                            precision_metrics["safe_override_precision_lcb_source_inner"]
                        ),
                        "harmful_override_rate_ucb_source_inner": float(
                            precision_metrics["harmful_override_rate_ucb_source_inner"]
                        ),
                        "active_override_count_source_inner": int(
                            precision_metrics["active_override_count_source_inner"]
                        ),
                        "active_override_rate_source_inner": float(
                            precision_metrics["active_override_rate_source_inner"]
                        ),
                        "net_gain_vs_ae_argmin_source_inner": float(
                            precision_metrics["net_gain_vs_ae_argmin_source_inner"]
                        ),
                        "source_inner_macro_gap_reduction_lcb": float(
                            precision_metrics["source_inner_macro_gap_reduction_lcb"]
                        ),
                        "worst_pseudo_domain_gap_degradation_pp": float(
                            precision_metrics["worst_pseudo_domain_gap_degradation_pp"]
                        ),
                        "passes_source_inner_risk_gates": int(passes),
                        **{f"macro_{k}": float(v) for k, v in macro.items()},
                    }
                )

    passing = [row for row in config_summaries if bool(row.get("passes_source_inner_risk_gates", False))]
    if not passing:
        selected = _SelectedConfig(
            str(method),
            str(feature_sets[0]),
            float("inf"),
            0.0,
            0,
            selection_status="fallback_to_ae_argmin_no_v1_risk_safe_config"
            if str(method) == PRIMARY_METHOD_V11
            else "fallback_to_ae_argmin_no_source_inner_config",
            fallback_reason="no_source_inner_risk_gate_config",
        )
    else:
        v1_selected_row = sorted(
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
        if str(method) == PRIMARY_METHOD_V11:
            precision_passing = [
                row
                for row in passing
                if int(row.get("passes_precision_lcb_gates", 0)) == 1
            ]
            if precision_passing:
                selected_row = sorted(
                    precision_passing,
                    key=lambda row: (
                        float(row.get("source_inner_macro_gap_reduction_lcb", float("-inf"))),
                        float(row.get("strict_improvement_precision_lcb_source_inner", float("-inf"))),
                        -float(row.get("harmful_override_rate_ucb_source_inner", float("inf"))),
                        float(row.get("active_override_rate", 0.0)),
                        float(row["top1_oracle_hit"]),
                        float(row["raw_predicted_delta_spearman_non_anchor"]),
                        float(row["delta_threshold"]),
                        float(row["margin_threshold"]),
                    ),
                    reverse=True,
                )[0]
                selected_by_source_inner = 1
                selection_status = "precision_lcb_selected"
                fallback_reason = ""
            else:
                selected_row = v1_selected_row
                selected_by_source_inner = 0
                selection_status = "fallback_to_v1_no_precision_lcb_safe_config"
                fallback_reason = "no_precision_lcb_safe_config"
        else:
            selected_row = v1_selected_row
            selected_by_source_inner = 1
            selection_status = "source_inner_selected"
            fallback_reason = ""
        selected = _SelectedConfig(
            str(method),
            str(selected_row["feature_set"]),
            float(selected_row["delta_threshold"]),
            float(selected_row["margin_threshold"]),
            int(selected_by_source_inner),
            selection_status=selection_status,
            fallback_reason=fallback_reason,
        )
    for row in validation_rows:
        row["selected_feature_set"] = selected.feature_set
        row["selected_delta_threshold"] = _threshold_label(selected.delta_threshold)
        row["selected_margin_threshold"] = _threshold_label(selected.margin_threshold)
        row["selection_status"] = selected.selection_status
        row["fallback_reason"] = selected.fallback_reason
        row["selected_by_source_inner_validation"] = int(
            selected.selected_by_source_inner
            and
            row["feature_set"] == selected.feature_set
            and row["delta_threshold"] == _threshold_label(selected.delta_threshold)
            and row["margin_threshold"] == _threshold_label(selected.margin_threshold)
        )
    return selected, validation_rows


def _select_consensus_config_for_method(
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
    consensus_thresholds = tuple(dict.fromkeys(float(v) for v in cfg.consensus_thresholds))

    for feature_set in feature_sets:
        threshold_domain_summaries: Dict[Tuple[float, float, float], List[Dict[str, float]]] = {}
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
            inner_train_domains = sorted(set(int(sample_domains[int(i)]) for i in inner_train_idx.tolist()))
            member_exclusions: List[int | None] = [None] + inner_train_domains
            eval_fold_for_sample = lambda _sample_index, f=inner_fold: f
            eval_rows, consensus, anchor_idx = _train_predict_consensus_for_fold(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                train_idx=inner_train_idx,
                eval_idx=val_idx,
                heldout_domain=int(outer_fold.heldout_domain),
                eval_fold_for_sample=eval_fold_for_sample,
                metadata_similarity=metadata_similarity,
                ae_scores=ae_scores,
                feature_set=str(feature_set),
                ridge_l2=float(cfg.ridge_l2),
                n_eval_candidates=len(inner_fold.candidate_expert_domains),
                member_excluded_domains=member_exclusions,
                uncertainty_multiplier=float(cfg.uncertainty_multiplier),
            )
            if eval_rows.x.size == 0:
                continue
            true_val = inner_fold.slice_nelbo(true_nelbo, val_idx)
            metadata_val = metadata_similarity[val_idx][:, list(inner_fold.candidate_col_indices)]
            metadata_idx = _metadata_selected_local_indices(metadata_val)
            ae_val = ae_scores.zscore_matrix[val_idx][:, list(inner_fold.candidate_col_indices)]
            for delta_threshold in deltas:
                for margin_threshold in margins:
                    for consensus_threshold in consensus_thresholds:
                        selected_idx, _best, _second, _lower_best, _lower_second, _override_margin, _positive_rate = (
                            _apply_consensus_safe_override_policy(
                                consensus=consensus,
                                anchor_idx=anchor_idx,
                                delta_threshold=float(delta_threshold),
                                margin_threshold=float(margin_threshold),
                                consensus_threshold=float(consensus_threshold),
                            )
                        )
                        summary = _policy_summary(
                            selected_idx=selected_idx,
                            anchor_idx=anchor_idx,
                            pred_delta_matrix=consensus.lower_matrix,
                            true_eval=true_val,
                            metadata_idx=metadata_idx,
                            ae_zscore_eval=ae_val,
                            abstention_correct_gap_pct_epsilon=float(cfg.abstention_correct_gap_pct_epsilon),
                            neutral_gap_pct_band=float(cfg.neutral_override_gap_pct_band),
                        )
                        threshold_domain_summaries.setdefault(
                            (float(delta_threshold), float(margin_threshold), float(consensus_threshold)),
                            [],
                        ).append(summary)
                        validation_rows.append(
                            {
                                "method": str(method),
                                "feature_set": str(feature_set),
                                "model_type": "ridge_delta_consensus",
                                "fold_query_domain": int(outer_fold.heldout_domain),
                                "source_inner_pseudo_query_domain": int(pseudo_domain),
                                "delta_threshold": _threshold_label(float(delta_threshold)),
                                "margin_threshold": _threshold_label(float(margin_threshold)),
                                "consensus_threshold": float(consensus_threshold),
                                "threshold_selection_policy": "source_inner_stability_then_ae_argmin_gap",
                                "n_validation_samples": int(val_idx.shape[0]),
                                "n_ensemble_members": int(len(consensus.member_labels)),
                                "ensemble_member_labels": "|".join(consensus.member_labels),
                                "ensemble_training_domains_used": "|".join(str(int(d)) for d in inner_train_domains),
                                "source_inner_validation_domain_excluded_from_ensemble_training": 1,
                                "candidate_experts": inner_fold.label(),
                                "excluded_target_ae": 1,
                                "excluded_target_cvae": 1,
                                "excluded_pseudo_query_ae": 1,
                                "excluded_pseudo_query_cvae": 1,
                                "heldout_target_nelbo_used_for_selection": 0,
                                **{f"macro_{k}": float(v) for k, v in summary.items()},
                            }
                        )

        for (delta_threshold, margin_threshold, consensus_threshold), summaries in threshold_domain_summaries.items():
            if not summaries:
                continue
            keys = set().union(*(row.keys() for row in summaries))
            macro = {k: _finite_mean([float(row.get(k, float("nan"))) for row in summaries], default=float("nan")) for k in keys}
            passes_risk = _passes_consensus_risk_gates(macro, cfg)
            stability = _source_inner_stability(summaries, cfg)
            config_summaries.append(
                {
                    "method": str(method),
                    "feature_set": str(feature_set),
                    "delta_threshold": float(delta_threshold),
                    "margin_threshold": float(margin_threshold),
                    "consensus_threshold": float(consensus_threshold),
                    "passes_source_inner_risk_gates": bool(passes_risk),
                    "passes_source_inner_stability_gates": bool(stability["passes_source_inner_stability_gates"]),
                    **stability,
                    **macro,
                }
            )

    passing = [
        row
        for row in config_summaries
        if bool(row.get("passes_source_inner_risk_gates", False))
        and bool(row.get("passes_source_inner_stability_gates", False))
    ]
    if not passing:
        selected = _SelectedConfig(str(method), str(feature_sets[0]), float("inf"), 0.0, 0, 1.0)
    else:
        selected_row = sorted(
            passing,
            key=lambda row: (
                _gap_reduction(row),
                float(row.get("selected_override_precision", float("nan")))
                if np.isfinite(float(row.get("selected_override_precision", float("nan"))))
                else -1.0,
                float(row.get("captured_oracle_headroom_rate", float("nan")))
                if np.isfinite(float(row.get("captured_oracle_headroom_rate", float("nan"))))
                else -1.0,
                -float(row["harmful_vs_ae_argmin_rate"]),
                float(row["consensus_threshold"]),
                float(row["delta_threshold"]),
            ),
            reverse=True,
        )[0]
        selected = _SelectedConfig(
            str(method),
            str(selected_row["feature_set"]),
            float(selected_row["delta_threshold"]),
            float(selected_row["margin_threshold"]),
            1,
            float(selected_row["consensus_threshold"]),
        )
    for row in validation_rows:
        row["selected_feature_set"] = selected.feature_set
        row["selected_delta_threshold"] = _threshold_label(selected.delta_threshold)
        row["selected_margin_threshold"] = _threshold_label(selected.margin_threshold)
        row["selected_consensus_threshold"] = float(selected.consensus_threshold)
        row["selected_by_source_inner_validation"] = int(
            row["feature_set"] == selected.feature_set
            and row["delta_threshold"] == _threshold_label(selected.delta_threshold)
            and row["margin_threshold"] == _threshold_label(selected.margin_threshold)
            and float(row.get("consensus_threshold", -1.0)) == float(selected.consensus_threshold)
        )
    return selected, validation_rows


def _method_feature_sets(cfg: AEUtilityCalibratorConfig) -> List[Tuple[str, Tuple[str, ...], str]]:
    if str(cfg.primary_method) == PRIMARY_METHOD_V2:
        methods = [(PRIMARY_METHOD_V2, tuple(cfg.feature_sets_primary), "primary_metadata_free")]
        if "ae_metadata_consensus" in set(cfg.feature_sets_diagnostic):
            methods.append((HYBRID_METADATA_METHOD_V2, ("ae_metadata_consensus",), "hybrid_metadata"))
        if "ae_combined_consensus" in set(cfg.feature_sets_diagnostic):
            methods.append((HYBRID_COMBINED_METHOD_V2, ("ae_combined_consensus",), "hybrid_combined"))
        return methods
    if str(cfg.primary_method) == PRIMARY_METHOD_V11:
        return [
            (PRIMARY_METHOD, tuple(cfg.feature_sets_primary), "v1_baseline"),
            (PRIMARY_METHOD_V11, tuple(cfg.feature_sets_primary), "primary_metadata_free_precision_lcb"),
        ]
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
    primary_method: str = PRIMARY_METHOD,
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
                "method": primary_method,
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
    is_v2 = str(cfg.primary_method) == PRIMARY_METHOD_V2
    if str(cfg.primary_method) not in {PRIMARY_METHOD, PRIMARY_METHOD_V11, PRIMARY_METHOD_V2}:
        raise ProtocolError(
            "AE utility calibrator primary_method must be "
            f"{PRIMARY_METHOD}, {PRIMARY_METHOD_V11}, or {PRIMARY_METHOD_V2}"
        )
    if is_v2:
        if str(cfg.primary_model_type) != "ridge_delta_consensus" or set(cfg.model_types) != {"ridge_delta_consensus"}:
            raise ProtocolError("AE utility calibrator v2 primary model_types must be ['ridge_delta_consensus']")
    elif str(cfg.primary_model_type) != "ridge_delta" or set(cfg.model_types) != {"ridge_delta"}:
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
        if method in V2_METHODS:
            selected_cfg, rows = _select_consensus_config_for_method(
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
        else:
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
        selected_source_inner_row = next(
            (
                row
                for row in rows
                if str(row.get("source_inner_pseudo_query_domain")) == "source_inner_macro"
                and str(row.get("feature_set")) == str(selected_cfg.feature_set)
                and str(row.get("delta_threshold")) == _threshold_label(float(selected_cfg.delta_threshold))
                and str(row.get("margin_threshold")) == _threshold_label(float(selected_cfg.margin_threshold))
            ),
            {},
        )
        if method in V2_METHODS:
            source_domains = sorted(set(int(sample_domains[int(i)]) for i in train_idx.tolist()))
            eval_rows, consensus, anchor_idx = _train_predict_consensus_for_fold(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                train_idx=train_idx,
                eval_idx=test_idx,
                heldout_domain=int(fold.heldout_domain),
                eval_fold_for_sample=eval_fold_for_sample,
                metadata_similarity=metadata_similarity,
                ae_scores=ae_scores,
                feature_set=selected_cfg.feature_set,
                ridge_l2=float(cfg.ridge_l2),
                n_eval_candidates=len(fold.candidate_expert_domains),
                member_excluded_domains=[None] + source_domains,
                uncertainty_multiplier=float(cfg.uncertainty_multiplier),
            )
            selected_idx, best_override, second_override, best_delta, second_delta, override_margin, positive_rate_best = (
                _apply_consensus_safe_override_policy(
                    consensus=consensus,
                    anchor_idx=anchor_idx,
                    delta_threshold=float(selected_cfg.delta_threshold),
                    margin_threshold=float(selected_cfg.margin_threshold),
                    consensus_threshold=float(selected_cfg.consensus_threshold),
                )
            )
            pred_matrix = consensus.lower_matrix
            pred_flat = np.asarray(
                [
                    float(consensus.mean_matrix[int(eval_rows.sample_positions[k]), int(eval_rows.candidate_local_indices[k])])
                    for k in range(eval_rows.sample_positions.shape[0])
                ],
                dtype=np.float64,
            )
        else:
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
            second_override = np.full(best_override.shape, -1, dtype=np.int64)
            second_delta = np.full(best_delta.shape, float("-inf"), dtype=np.float64)
            positive_rate_best = np.ones(best_delta.shape, dtype=np.float64)
        summary = _policy_summary(
            selected_idx=selected_idx,
            anchor_idx=anchor_idx,
            pred_delta_matrix=pred_matrix,
            true_eval=true_eval,
            metadata_idx=metadata_idx,
            ae_zscore_eval=ae_zscore_eval,
            abstention_correct_gap_pct_epsilon=float(cfg.abstention_correct_gap_pct_epsilon),
            neutral_gap_pct_band=float(cfg.neutral_override_gap_pct_band),
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
        oracle_idx = _stable_argmin_indices(true_eval)
        oracle_nelbo = true_eval[rows_idx, oracle_idx]
        selected_gap_pct = ((selected_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
        anchor_gap_pct = ((anchor_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0
        override_delta_gap_pct = selected_gap_pct - anchor_gap_pct
        active_override_mask = np.asarray(selected_idx, dtype=np.int64) != np.asarray(anchor_idx, dtype=np.int64)
        band = float(cfg.neutral_override_gap_pct_band)
        override_class = np.full((true_eval.shape[0],), "not_active", dtype=object)
        override_class[active_override_mask & (override_delta_gap_pct <= -band)] = "improving"
        override_class[active_override_mask & (override_delta_gap_pct >= band)] = "harmful"
        override_class[active_override_mask & (np.abs(override_delta_gap_pct) < band)] = "neutral"
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
                    "model_type": "ridge_delta_consensus" if method in V2_METHODS else "ridge_delta",
                    "ensemble_strategy": str(cfg.ensemble_strategy) if method in V2_METHODS else "",
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
                    "mean_predicted_delta_best": (
                        float(consensus.mean_matrix[local, int(best_override[local])])
                        if method in V2_METHODS and int(best_override[local]) >= 0
                        else float(best_delta[local])
                    ),
                    "std_predicted_delta_best": (
                        float(consensus.std_matrix[local, int(best_override[local])])
                        if method in V2_METHODS and int(best_override[local]) >= 0
                        else float("nan")
                    ),
                    "lower_confidence_delta_best": (
                        float(best_delta[local])
                        if method in V2_METHODS
                        else float("nan")
                    ),
                    "positive_consensus_rate_best": (
                        float(positive_rate_best[local])
                        if method in V2_METHODS
                        else float("nan")
                    ),
                    "mean_predicted_delta_second": (
                        float(consensus.mean_matrix[local, int(second_override[local])])
                        if method in V2_METHODS and int(second_override[local]) >= 0
                        else float("nan")
                    ),
                    "lower_confidence_delta_second": (
                        float(second_delta[local])
                        if method in V2_METHODS
                        else float("nan")
                    ),
                    "n_ensemble_members": (
                        int(consensus.n_members_matrix[local, int(best_override[local])])
                        if method in V2_METHODS and int(best_override[local]) >= 0
                        else 1
                    ),
                    "n_positive_members": (
                        int(consensus.n_positive_matrix[local, int(best_override[local])])
                        if method in V2_METHODS and int(best_override[local]) >= 0
                        else int(float(best_delta[local]) > 0.0)
                    ),
                    "true_delta_best_override": (
                        float(true_delta[local, int(best_override[local])])
                        if int(best_override[local]) >= 0
                        else float("nan")
                    ),
                    "predicted_override_margin": float(override_margin[local]),
                    "selected_delta_threshold": _threshold_label(float(selected_cfg.delta_threshold)),
                    "selected_margin_threshold": _threshold_label(float(selected_cfg.margin_threshold)),
                    "selected_consensus_threshold": (
                        float(selected_cfg.consensus_threshold)
                        if method in V2_METHODS
                        else ""
                    ),
                    "selected_by_source_inner_validation": int(selected_cfg.selected_by_source_inner),
                    "target_support_free": 1,
                    "target_support_used": 0,
                    "target_ae_excluded": 1,
                    "target_cvae_excluded": 1,
                    "source_inner_self_ae_excluded": 1,
                    "source_inner_self_expert_excluded": 1,
                    "metadata_role": (
                        "not_used"
                        if method in {PRIMARY_METHOD, PRIMARY_METHOD_V11, PRIMARY_METHOD_V2}
                        else "hybrid_auxiliary_feature"
                    ),
                    "proxy_claim_boundary": "AE reconstruction fit is a proxy for CVAE utility, not compatibility.",
                    "selection_status": selected_cfg.selection_status,
                    "fallback_reason": selected_cfg.fallback_reason,
                    "heldout_precision_report_only": 1 if method == PRIMARY_METHOD_V11 else "",
                    "net_gain_vs_ae_argmin": float(anchor_nelbo[local] - selected_nelbo[local]),
                    "net_gain_vs_metadata": float(row_metadata_nelbo - selected_nelbo[local]),
                    "active_override": row_active,
                    "harmful_vs_ae_argmin": row_harms_anchor,
                    "improving_vs_ae_argmin": row_improves_anchor,
                    "override_delta_gap_pct": float(override_delta_gap_pct[local]),
                    "override_class": str(override_class[local]),
                    "selected_override_precision": (
                        float(row_improves_anchor)
                        if row_active
                        else float("nan")
                    ),
                    "strict_improvement_precision": float(summary["strict_improvement_precision"]),
                    "safe_override_precision": float(summary["safe_override_precision"]),
                    "harmful_override_rate": float(summary["harmful_override_rate"]),
                    "improving_override_rate": float(summary["improving_override_rate"]),
                    "neutral_override_rate": float(summary["neutral_override_rate"]),
                    "raw_predicted_delta_spearman_non_anchor": float(
                        summary["raw_predicted_delta_spearman_non_anchor"]
                    ),
                    "raw_predicted_delta_spearman_with_anchor": float(
                        summary["raw_predicted_delta_spearman_with_anchor"]
                    ),
                    "abstention_rate": float(summary["abstention_rate"]),
                    "abstention_correct_rate": float(summary["abstention_correct_rate"]),
                    "abstention_missed_gain": float(summary["abstention_missed_gain"]),
                    "captured_oracle_headroom_rate": float(summary["captured_oracle_headroom_rate"]),
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
                    "mean_predicted_delta": (
                        float(value) if method in V2_METHODS else ""
                    ),
                    "std_predicted_delta": (
                        float(consensus.std_matrix[local, int(eval_rows.candidate_local_indices[k])])
                        if method in V2_METHODS
                        else ""
                    ),
                    "lower_confidence_delta": (
                        float(consensus.lower_matrix[local, int(eval_rows.candidate_local_indices[k])])
                        if method in V2_METHODS
                        else ""
                    ),
                    "positive_consensus_rate": (
                        float(consensus.positive_rate_matrix[local, int(eval_rows.candidate_local_indices[k])])
                        if method in V2_METHODS
                        else ""
                    ),
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
                    "selection_status": selected_cfg.selection_status,
                    "fallback_reason": selected_cfg.fallback_reason,
                    "heldout_precision_report_only": 1 if method == PRIMARY_METHOD_V11 else "",
                    "selected_consensus_threshold": (
                        float(selected_cfg.consensus_threshold)
                        if method in V2_METHODS
                        else ""
                    ),
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
            "model_type": "ridge_delta_consensus" if method in V2_METHODS else "ridge_delta",
            "feature_set": selected_cfg.feature_set,
            "method_kind": method_kind,
            "selected_delta_threshold": _threshold_label(float(selected_cfg.delta_threshold)),
            "selected_margin_threshold": _threshold_label(float(selected_cfg.margin_threshold)),
            "selected_by_source_inner_validation": int(selected_cfg.selected_by_source_inner),
            "selection_status": selected_cfg.selection_status,
            "fallback_reason": selected_cfg.fallback_reason,
            "selection_mode": str(cfg.selection_mode) if method == PRIMARY_METHOD_V11 else "",
            "selected_consensus_threshold": (
                float(selected_cfg.consensus_threshold)
                if method in V2_METHODS
                else ""
            ),
            "ensemble_strategy": str(cfg.ensemble_strategy) if method in V2_METHODS else "",
            "uncertainty_multiplier": float(cfg.uncertainty_multiplier) if method in V2_METHODS else "",
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
            "target_ae_excluded": 1,
            "target_cvae_excluded": 1,
            "source_inner_self_ae_excluded": 1,
            "source_inner_self_expert_excluded": 1,
            "heldout_precision_report_only": 1 if method == PRIMARY_METHOD_V11 else "",
            "active_override_count_heldout": int(summary["active_override_count"]),
            "active_override_rate_heldout": float(summary["active_override_rate"]),
            "active_override_count_source_inner": selected_source_inner_row.get(
                "active_override_count_source_inner", ""
            ),
            "active_override_rate_source_inner": selected_source_inner_row.get(
                "active_override_rate_source_inner", ""
            ),
            "strict_improvement_precision_source_inner": selected_source_inner_row.get(
                "strict_improvement_precision_source_inner", ""
            ),
            "strict_improvement_precision_lcb_source_inner": selected_source_inner_row.get(
                "strict_improvement_precision_lcb_source_inner", ""
            ),
            "safe_override_precision_source_inner": selected_source_inner_row.get(
                "safe_override_precision_source_inner", ""
            ),
            "safe_override_precision_lcb_source_inner": selected_source_inner_row.get(
                "safe_override_precision_lcb_source_inner", ""
            ),
            "harmful_override_rate_ucb_source_inner": selected_source_inner_row.get(
                "harmful_override_rate_ucb_source_inner", ""
            ),
            "source_inner_macro_gap_reduction_lcb": selected_source_inner_row.get(
                "source_inner_macro_gap_reduction_lcb", ""
            ),
            "worst_pseudo_domain_gap_degradation_pp": selected_source_inner_row.get(
                "worst_pseudo_domain_gap_degradation_pp", ""
            ),
            **summary,
        }
        policy_rows.append(policy)
        override_precision_rows.append(
            {
                "method": method,
                "fold_query_domain": int(fold.heldout_domain),
                "active_overrides": int(np.sum(selected_idx != anchor_idx)),
                "active_override_count_heldout": int(summary["active_override_count"]),
                "active_override_rate_heldout": float(summary["active_override_rate"]),
                "active_override_count_source_inner": selected_source_inner_row.get(
                    "active_override_count_source_inner", ""
                ),
                "active_override_rate_source_inner": selected_source_inner_row.get(
                    "active_override_rate_source_inner", ""
                ),
                "selected_override_precision": float(summary["selected_override_precision"]),
                "strict_improvement_precision": float(summary["strict_improvement_precision"]),
                "strict_improvement_precision_source_inner": selected_source_inner_row.get(
                    "strict_improvement_precision_source_inner", ""
                ),
                "strict_improvement_precision_lcb_source_inner": selected_source_inner_row.get(
                    "strict_improvement_precision_lcb_source_inner", ""
                ),
                "safe_override_precision": float(summary["safe_override_precision"]),
                "safe_override_precision_source_inner": selected_source_inner_row.get(
                    "safe_override_precision_source_inner", ""
                ),
                "safe_override_precision_lcb_source_inner": selected_source_inner_row.get(
                    "safe_override_precision_lcb_source_inner", ""
                ),
                "harmful_override_rate_ucb_source_inner": selected_source_inner_row.get(
                    "harmful_override_rate_ucb_source_inner", ""
                ),
                "active_override_rate": float(summary["active_override_rate"]),
                "override_capture_rate": float(summary["override_capture_rate"]),
                "captured_oracle_headroom_rate": float(summary["captured_oracle_headroom_rate"]),
                "heldout_precision_report_only": 1 if method == PRIMARY_METHOD_V11 else "",
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
                "improving_override_rate": float(summary["improving_override_rate"]),
                "neutral_override_rate": float(summary["neutral_override_rate"]),
                "harmful_override_rate": float(summary["harmful_override_rate"]),
                "mean_gain_improving_overrides": float(summary["mean_gain_improving_overrides"]),
                "mean_loss_harmful_overrides": float(summary["mean_loss_harmful_overrides"]),
                "captured_oracle_headroom_rate": float(summary["captured_oracle_headroom_rate"]),
                "abstention_rate": float(summary["abstention_rate"]),
                "abstention_correct_rate": float(summary["abstention_correct_rate"]),
                "abstention_missed_gain": float(summary["abstention_missed_gain"]),
            }
        )
        selected_feature_rows.append(
            {
                "method": method,
                "fold_query_domain": int(fold.heldout_domain),
                "selected_feature_set": selected_cfg.feature_set,
                "selected_delta_threshold": _threshold_label(float(selected_cfg.delta_threshold)),
                "selected_margin_threshold": _threshold_label(float(selected_cfg.margin_threshold)),
                "selected_consensus_threshold": (
                    float(selected_cfg.consensus_threshold)
                    if method in V2_METHODS
                    else ""
                ),
                "selected_by_source_inner_validation": int(selected_cfg.selected_by_source_inner),
                "selection_status": selected_cfg.selection_status,
                "fallback_reason": selected_cfg.fallback_reason,
                "active_override_count_source_inner": selected_source_inner_row.get(
                    "active_override_count_source_inner", ""
                ),
                "strict_improvement_precision_lcb_source_inner": selected_source_inner_row.get(
                    "strict_improvement_precision_lcb_source_inner", ""
                ),
                "source_inner_macro_gap_reduction_lcb": selected_source_inner_row.get(
                    "source_inner_macro_gap_reduction_lcb", ""
                ),
            }
        )
        if method in {PRIMARY_METHOD, PRIMARY_METHOD_V11, PRIMARY_METHOD_V2}:
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
            primary_method=str(cfg.primary_method),
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
    artifacts = {
        "ae_utility_calibrator_raw": "ae_utility_calibrator_raw.csv",
        "ae_utility_calibrator_source_inner_validation": "ae_utility_calibrator_source_inner_validation.csv",
        "ae_utility_calibrator_policy_audit": "ae_utility_calibrator_policy_audit.csv",
        "ae_utility_calibrator_override_diagnostics": "ae_utility_calibrator_override_diagnostics.csv",
        "ae_utility_calibrator_oracle_headroom": "ae_utility_calibrator_oracle_headroom.csv",
        "ae_utility_calibrator_selected_feature_sets": "ae_utility_calibrator_selected_feature_sets.csv",
        "ae_utility_calibrator_override_precision": "ae_utility_calibrator_override_precision.csv",
        "ae_utility_calibrator_anchor_rank_diagnostics": "ae_utility_calibrator_anchor_rank_diagnostics.csv",
    }
    v11_validation_rows = [row for row in source_inner_validation_rows if str(row.get("method")) == PRIMARY_METHOD_V11]
    v11_policy_rows = [row for row in policy_audit_rows if str(row.get("method")) == PRIMARY_METHOD_V11]
    v11_override_rows = [row for row in override_diagnostic_rows if str(row.get("method")) == PRIMARY_METHOD_V11]
    v11_precision_rows = [row for row in override_precision_rows if str(row.get("method")) == PRIMARY_METHOD_V11]
    v11_selected_rows = [row for row in selected_feature_rows if str(row.get("method")) == PRIMARY_METHOD_V11]
    v11_tradeoff_rows = [
        row
        for row in v11_validation_rows
        if str(row.get("source_inner_pseudo_query_domain")) == "source_inner_macro"
    ]
    if v11_validation_rows or v11_policy_rows:
        _write_csv(
            reports_dir / "ae_utility_calibrator_precision_v11_source_inner_validation.csv",
            v11_validation_rows,
        )
        _write_csv(
            reports_dir / "ae_utility_calibrator_precision_v11_policy_audit.csv",
            v11_policy_rows,
        )
        _write_csv(
            reports_dir / "ae_utility_calibrator_precision_v11_override_diagnostics.csv",
            v11_override_rows,
        )
        _write_csv(
            reports_dir / "ae_utility_calibrator_precision_v11_precision_tradeoff.csv",
            v11_tradeoff_rows or v11_precision_rows,
        )
        _write_csv(
            reports_dir / "ae_utility_calibrator_precision_v11_selection_status.csv",
            v11_selected_rows,
        )
        artifacts.update(
            {
                "ae_utility_calibrator_precision_v11_source_inner_validation": (
                    "ae_utility_calibrator_precision_v11_source_inner_validation.csv"
                ),
                "ae_utility_calibrator_precision_v11_policy_audit": (
                    "ae_utility_calibrator_precision_v11_policy_audit.csv"
                ),
                "ae_utility_calibrator_precision_v11_override_diagnostics": (
                    "ae_utility_calibrator_precision_v11_override_diagnostics.csv"
                ),
                "ae_utility_calibrator_precision_v11_precision_tradeoff": (
                    "ae_utility_calibrator_precision_v11_precision_tradeoff.csv"
                ),
                "ae_utility_calibrator_precision_v11_selection_status": (
                    "ae_utility_calibrator_precision_v11_selection_status.csv"
                ),
            }
        )
    v2_raw_rows = [row for row in raw_rows if str(row.get("method")) in V2_METHODS]
    v2_validation_rows = [row for row in source_inner_validation_rows if str(row.get("method")) in V2_METHODS]
    v2_policy_rows = [row for row in policy_audit_rows if str(row.get("method")) in V2_METHODS]
    v2_override_rows = [row for row in override_diagnostic_rows if str(row.get("method")) in V2_METHODS]
    v2_headroom_rows = [row for row in oracle_headroom_rows if str(row.get("method")) in V2_METHODS | {ORACLE_HEADROOM_METHOD}]
    v2_precision_rows = [row for row in override_precision_rows if str(row.get("method")) in V2_METHODS]
    v2_anchor_rows = [row for row in anchor_rank_rows if str(row.get("method")) in V2_METHODS]
    v2_sample_rows = [row for row in raw_rows if str(row.get("method")) in V2_METHODS]
    if v2_raw_rows or v2_policy_rows:
        _write_csv(reports_dir / "ae_utility_calibrator_v2_raw.csv", v2_raw_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_source_inner_validation.csv", v2_validation_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_policy_audit.csv", v2_policy_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_override_diagnostics.csv", v2_override_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_override_precision.csv", v2_precision_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_consensus_diagnostics.csv", v2_sample_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_coverage_precision_tradeoff.csv", v2_policy_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_oracle_headroom.csv", v2_headroom_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_anchor_rank_diagnostics.csv", v2_anchor_rows)
        _write_csv(reports_dir / "ae_utility_calibrator_v2_abstention_diagnostics.csv", v2_override_rows)
        artifacts.update(
            {
                "ae_utility_calibrator_v2_raw": "ae_utility_calibrator_v2_raw.csv",
                "ae_utility_calibrator_v2_source_inner_validation": "ae_utility_calibrator_v2_source_inner_validation.csv",
                "ae_utility_calibrator_v2_policy_audit": "ae_utility_calibrator_v2_policy_audit.csv",
                "ae_utility_calibrator_v2_override_diagnostics": "ae_utility_calibrator_v2_override_diagnostics.csv",
                "ae_utility_calibrator_v2_override_precision": "ae_utility_calibrator_v2_override_precision.csv",
                "ae_utility_calibrator_v2_consensus_diagnostics": "ae_utility_calibrator_v2_consensus_diagnostics.csv",
                "ae_utility_calibrator_v2_coverage_precision_tradeoff": "ae_utility_calibrator_v2_coverage_precision_tradeoff.csv",
                "ae_utility_calibrator_v2_oracle_headroom": "ae_utility_calibrator_v2_oracle_headroom.csv",
                "ae_utility_calibrator_v2_anchor_rank_diagnostics": "ae_utility_calibrator_v2_anchor_rank_diagnostics.csv",
                "ae_utility_calibrator_v2_abstention_diagnostics": "ae_utility_calibrator_v2_abstention_diagnostics.csv",
            }
        )
    return artifacts
