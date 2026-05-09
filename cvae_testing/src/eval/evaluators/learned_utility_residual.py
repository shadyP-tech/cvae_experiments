from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import ResidualRoutingConfig
from src.eval.evaluators.learned_utility_protocol import (
    FoldCandidateSet,
    MethodProtocol,
    ProtocolError,
    _aggregate_metrics_from_sample_rows,
    _domain_breakdown_rows,
    _method_protocol,
    _protocol_row_fields,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics


_RESIDUAL_EPS = 1e-12
_CATASTROPHIC_TOP1_UPLIFT_MIN = -0.05
_CATASTROPHIC_SPEARMAN_UPLIFT_MIN = -0.05
_CATASTROPHIC_GAP_PCT_REDUCTION_MIN = -2.0
_FEATURE_SETS = {"minimal", "latent", "calibrated"}
_SAFE_OVERRIDE_POLICY_V2 = "metadata_residual_safe_override_v2"
_SAFE_V2_METHODS = {"metadata_residual_thresholded_safe_v2", "metadata_residual_group_robust_safe_v2"}
_RESIDUAL_ADOPTION_METHODS = {
    "metadata_residual_thresholded",
    "metadata_residual_group_robust",
    *_SAFE_V2_METHODS,
}


@dataclass(frozen=True)
class ResidualFoldOutputs:
    sample_rows: List[Dict[str, Any]]
    raw_rows: List[Dict[str, Any]]
    override_rows: List[Dict[str, Any]]
    audit_rows: List[Dict[str, Any]]
    confusion_rows: List[Dict[str, Any]]


@dataclass(frozen=True)
class _ResidualModel:
    w: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_z = (x - self.mu) / self.sigma
        x_aug = np.concatenate([x_z, np.ones((x_z.shape[0], 1), dtype=np.float64)], axis=1)
        return x_aug @ self.w


@dataclass(frozen=True)
class _FeatureContext:
    feature_set: str
    centroid_by_domain: Mapping[int, np.ndarray]
    calibration_by_domain: Mapping[int, Tuple[float, float]]


@dataclass(frozen=True)
class _SelectedResidualConfig:
    method: str
    feature_set: str
    tau: float
    validation_top1_uplift: float
    validation_spearman_uplift: float
    validation_gap_reduction: float
    validation_safety_pass: bool
    threshold_selection_policy: str
    fallback_used: bool = False
    validation_max_harmful_override_rate: float = 0.0
    validation_min_utility_improving_override_rate: float = 0.0
    validation_mean_override_rate: float = 0.0
    validation_domains_failed_gate: str = ""
    validation_worst_domain_top1_uplift: float = 0.0
    validation_worst_domain_gap_pct_reduction: float = 0.0


def _stable_argmax_indices(matrix: np.ndarray) -> np.ndarray:
    n_rows, n_cols = matrix.shape
    tie_break = np.arange(n_cols, dtype=np.int64)
    out = np.zeros((n_rows,), dtype=np.int64)
    for i in range(n_rows):
        order = np.lexsort((tie_break, -matrix[i, :]))
        out[i] = int(order[0])
    return out


def _threshold_label(value: float) -> str:
    return "inf" if math.isinf(float(value)) else f"{float(value):.6g}"


def _is_safe_override_v2(residual_cfg: ResidualRoutingConfig) -> bool:
    return str(residual_cfg.residual_policy_version) == _SAFE_OVERRIDE_POLICY_V2


def _feature_complexity_rank(feature_set: str) -> int:
    order = {"minimal": 0, "latent": 1, "calibrated": 2}
    return int(order.get(str(feature_set).strip().lower(), 99))


def _safe_v2_adoption_feature_sets(residual_cfg: ResidualRoutingConfig) -> Tuple[str, ...]:
    configured = tuple(
        str(v).strip().lower()
        for v in (residual_cfg.adoption_feature_sets or ("minimal", "latent"))
    )
    if bool(residual_cfg.allow_calibrated_adoption):
        return tuple(v for v in configured if v in _FEATURE_SETS)
    return tuple(v for v in configured if v in _FEATURE_SETS and v != "calibrated")


def _safe_v2_diagnostic_feature_sets(residual_cfg: ResidualRoutingConfig) -> Tuple[str, ...]:
    configured = tuple(
        str(v).strip().lower()
        for v in (residual_cfg.diagnostic_feature_sets or ("calibrated",))
    )
    return tuple(v for v in configured if v in _FEATURE_SETS)


def _metadata_selected_local_indices(metadata_similarity_eval: np.ndarray) -> np.ndarray:
    return _stable_argmax_indices(np.asarray(metadata_similarity_eval, dtype=np.float64))


def _fit_weighted_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, l2: float) -> _ResidualModel:
    if x.size == 0 or y.size == 0:
        raise ProtocolError("Residual ridge received zero training rows")
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    x_z = (x - mu) / sigma
    x_aug = np.concatenate([x_z, np.ones((x_z.shape[0], 1), dtype=np.float64)], axis=1)
    w_sqrt = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 0.0)).reshape(-1, 1)
    x_w = x_aug * w_sqrt
    y_w = y.reshape(-1, 1) * w_sqrt
    xtx = x_w.T @ x_w
    xtx += float(l2) * np.eye(xtx.shape[0], dtype=np.float64)
    coef = np.linalg.solve(xtx, x_w.T @ y_w).reshape(-1)
    return _ResidualModel(w=coef, mu=mu, sigma=sigma)


def _domain_balanced_weights(query_domains: np.ndarray, *, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ones((int(query_domains.shape[0]),), dtype=np.float64)
    weights = np.zeros((int(query_domains.shape[0]),), dtype=np.float64)
    unique = sorted(set(int(q) for q in query_domains.tolist()))
    for q in unique:
        idx = np.where(query_domains == int(q))[0]
        if idx.size:
            weights[idx] = 1.0 / float(idx.size)
    if weights.sum() > 0:
        weights *= float(query_domains.shape[0]) / float(weights.sum())
    return weights


def _feature_context(
    *,
    feature_set: str,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    stats_indices: np.ndarray,
) -> _FeatureContext:
    normalized = str(feature_set).strip().lower()
    if normalized not in _FEATURE_SETS:
        raise ProtocolError(f"Unknown residual feature_set={feature_set!r}")

    centroid_by_domain: Dict[int, np.ndarray] = {}
    calibration_by_domain: Dict[int, Tuple[float, float]] = {}
    stats_indices = np.asarray(stats_indices, dtype=np.int64)
    for domain in sorted(set(int(v) for v in sample_domains[stats_indices].tolist())):
        idx = stats_indices[sample_domains[stats_indices] == int(domain)]
        if idx.size:
            centroid_by_domain[int(domain)] = embeddings[idx].mean(axis=0)

    for col, domain in enumerate(expert_domains):
        allowed_idx = stats_indices[sample_domains[stats_indices] != int(domain)]
        vals = -true_nelbo[allowed_idx, int(col)] if allowed_idx.size else np.asarray([], dtype=np.float64)
        if vals.size:
            calibration_by_domain[int(domain)] = (float(np.mean(vals)), float(np.var(vals)))
        else:
            calibration_by_domain[int(domain)] = (0.0, 0.0)

    return _FeatureContext(
        feature_set=normalized,
        centroid_by_domain=centroid_by_domain,
        calibration_by_domain=calibration_by_domain,
    )


def _metadata_distances(
    *,
    metadata_similarity: np.ndarray,
    sample_index: int,
    candidate_col: int,
    meta_col: int,
) -> Tuple[float, float, float]:
    d_candidate = -float(metadata_similarity[int(sample_index), int(candidate_col)])
    d_meta = -float(metadata_similarity[int(sample_index), int(meta_col)])
    return d_candidate, d_meta, float(d_candidate - d_meta)


def _latent_distances(
    *,
    context: _FeatureContext,
    embeddings: np.ndarray,
    sample_index: int,
    candidate_domain: int,
    meta_domain: int,
) -> Tuple[float, float, float]:
    emb = embeddings[int(sample_index)]
    cand_centroid = context.centroid_by_domain.get(int(candidate_domain))
    meta_centroid = context.centroid_by_domain.get(int(meta_domain))
    d_candidate = float(np.linalg.norm(emb - cand_centroid)) if cand_centroid is not None else 0.0
    d_meta = float(np.linalg.norm(emb - meta_centroid)) if meta_centroid is not None else 0.0
    return d_candidate, d_meta, float(d_candidate - d_meta)


def _calibration_features(
    *,
    context: _FeatureContext,
    candidate_domain: int,
    meta_domain: int,
) -> Tuple[float, ...]:
    cand_mean, cand_var = context.calibration_by_domain.get(int(candidate_domain), (0.0, 0.0))
    meta_mean, meta_var = context.calibration_by_domain.get(int(meta_domain), (0.0, 0.0))
    return (
        float(cand_mean),
        float(meta_mean),
        float(cand_mean - meta_mean),
        float(cand_var),
        float(meta_var),
        float(cand_var - meta_var),
    )


def _build_feature_matrix(
    *,
    sample_indices: np.ndarray,
    fold: FoldCandidateSet,
    metadata_similarity: np.ndarray,
    embeddings: np.ndarray,
    context: _FeatureContext,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    candidate_cols = np.asarray(fold.candidate_col_indices, dtype=np.int64)
    candidate_domains = np.asarray(fold.candidate_expert_domains, dtype=np.int64)
    metadata_eval = metadata_similarity[sample_indices][:, candidate_cols]
    meta_local_idx = _metadata_selected_local_indices(metadata_eval)

    features: List[List[float]] = []
    row_meta_local: List[int] = []
    row_sample_local: List[int] = []
    for local_sample_idx, sample_index in enumerate(sample_indices.tolist()):
        meta_local = int(meta_local_idx[local_sample_idx])
        meta_col = int(candidate_cols[meta_local])
        meta_domain = int(candidate_domains[meta_local])
        for local_candidate_idx, (candidate_col, candidate_domain) in enumerate(
            zip(candidate_cols.tolist(), candidate_domains.tolist())
        ):
            d_meta = _metadata_distances(
                metadata_similarity=metadata_similarity,
                sample_index=int(sample_index),
                candidate_col=int(candidate_col),
                meta_col=int(meta_col),
            )
            vals: List[float] = [*d_meta]
            if context.feature_set in {"latent", "calibrated"}:
                vals.extend(
                    _latent_distances(
                        context=context,
                        embeddings=embeddings,
                        sample_index=int(sample_index),
                        candidate_domain=int(candidate_domain),
                        meta_domain=int(meta_domain),
                    )
                )
            if context.feature_set == "calibrated":
                vals.extend(
                    _calibration_features(
                        context=context,
                        candidate_domain=int(candidate_domain),
                        meta_domain=int(meta_domain),
                    )
                )
            features.append(vals)
            row_meta_local.append(meta_local)
            row_sample_local.append(local_sample_idx)
            _ = local_candidate_idx

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(row_meta_local, dtype=np.int64),
        np.asarray(row_sample_local, dtype=np.int64),
    )


def _build_residual_training_rows(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    outer_heldout_domain: int,
    train_indices: np.ndarray,
    context: _FeatureContext,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    expert_col_by_domain = {int(domain): idx for idx, domain in enumerate(expert_domains)}

    for query_domain in sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_indices, dtype=np.int64).tolist())):
        domain_indices = np.asarray(train_indices, dtype=np.int64)[sample_domains[np.asarray(train_indices, dtype=np.int64)] == int(query_domain)]
        if domain_indices.size == 0:
            continue
        fold = FoldCandidateSet.for_heldout_domain(
            heldout_domain=int(outer_heldout_domain),
            expert_domains=expert_domains,
            excluded_domains=[int(query_domain)],
        )
        x, meta_local_per_row, sample_local_per_row = _build_feature_matrix(
            sample_indices=domain_indices,
            fold=fold,
            metadata_similarity=metadata_similarity,
            embeddings=embeddings,
            context=context,
        )
        candidate_cols = np.asarray(fold.candidate_col_indices, dtype=np.int64)
        e_count = len(fold.candidate_expert_domains)
        y = np.zeros((x.shape[0],), dtype=np.float64)
        for row_idx in range(x.shape[0]):
            local_sample = int(sample_local_per_row[row_idx])
            sample_index = int(domain_indices[local_sample])
            local_candidate = int(row_idx % e_count)
            meta_local = int(meta_local_per_row[row_idx])
            candidate_col = int(candidate_cols[local_candidate])
            meta_col = int(candidate_cols[meta_local])
            if candidate_col not in set(expert_col_by_domain.values()):
                raise ProtocolError("Residual training candidate column is outside expert domains")
            nelbo_meta = float(true_nelbo[sample_index, meta_col])
            nelbo_candidate = float(true_nelbo[sample_index, candidate_col])
            y[row_idx] = (nelbo_meta - nelbo_candidate) / max(abs(nelbo_meta), _RESIDUAL_EPS)
        x_parts.append(x)
        y_parts.append(y)
        q_parts.append(np.full((x.shape[0],), int(query_domain), dtype=np.int64))

    if not x_parts:
        raise ProtocolError(f"No residual training rows remain for heldout_domain={outer_heldout_domain}")
    return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(q_parts)


def _fit_residual_model(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    outer_heldout_domain: int,
    train_indices: np.ndarray,
    feature_set: str,
    group_balanced: bool,
    l2: float,
) -> Tuple[_ResidualModel, _FeatureContext]:
    context = _feature_context(
        feature_set=feature_set,
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        stats_indices=train_indices,
    )
    x, y, q = _build_residual_training_rows(
        embeddings=embeddings,
        sample_domains=sample_domains,
        true_nelbo=true_nelbo,
        expert_domains=expert_domains,
        metadata_similarity=metadata_similarity,
        outer_heldout_domain=outer_heldout_domain,
        train_indices=train_indices,
        context=context,
    )
    weights = _domain_balanced_weights(q, enabled=bool(group_balanced))
    return _fit_weighted_ridge(x, y, weights, l2=float(l2)), context


def _metadata_rows(
    *,
    sample_domains: np.ndarray,
    expert_domains: Sequence[int],
    sample_indices: np.ndarray,
    fold: FoldCandidateSet,
    metadata_similarity: np.ndarray,
    true_eval: np.ndarray,
    global_eval: np.ndarray,
    tie_policy: str,
) -> List[Dict[str, Any]]:
    score = -metadata_similarity[np.asarray(sample_indices, dtype=np.int64)][:, list(fold.candidate_col_indices)]
    _metrics, rows = _selection_metrics(
        method="metadata_routing",
        query_domains=sample_domains[np.asarray(sample_indices, dtype=np.int64)],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score,
        true_nelbo_matrix=true_eval,
        fold=fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
    )
    for row in rows:
        row["sample_index"] = int(sample_indices[int(row["sample_index"])])
    return rows


def _selected_from_residual(raw_scores: np.ndarray, meta_local_idx: np.ndarray, *, tau: float) -> np.ndarray:
    if math.isinf(float(tau)):
        return np.asarray(meta_local_idx, dtype=np.int64)
    best = _stable_argmax_indices(raw_scores)
    selected = np.asarray(meta_local_idx, dtype=np.int64).copy()
    for i, best_idx in enumerate(best.tolist()):
        if float(raw_scores[i, int(best_idx)]) > float(tau):
            selected[i] = int(best_idx)
    return selected


def _evaluate_residual_config(
    *,
    method: str,
    residual_cfg: ResidualRoutingConfig,
    sample_domains: np.ndarray,
    embeddings: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    sample_indices: np.ndarray,
    fold: FoldCandidateSet,
    global_eval: np.ndarray,
    model: _ResidualModel,
    context: _FeatureContext,
    tau: float,
    selected_feature_set: str,
    selected_variant: str,
    threshold_selection_policy: str,
    validation_summary: Mapping[str, Any],
    tie_policy: str,
    selected_by_inner_validation: bool,
    adoption_selected_method: str,
    force_diagnostic: bool = False,
    emit_raw_rows: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    x, meta_local_per_row, sample_local_per_row = _build_feature_matrix(
        sample_indices=sample_indices,
        fold=fold,
        metadata_similarity=metadata_similarity,
        embeddings=embeddings,
        context=context,
    )
    e_count = len(fold.candidate_expert_domains)
    n_samples = int(sample_indices.shape[0])
    raw = model.predict(x).reshape(n_samples, e_count)
    meta_local_idx = np.zeros((n_samples,), dtype=np.int64)
    for row_idx, local_sample in enumerate(sample_local_per_row.tolist()):
        meta_local_idx[int(local_sample)] = int(meta_local_per_row[row_idx])
    raw[np.arange(n_samples), meta_local_idx] = 0.0

    selected_idx = _selected_from_residual(raw, meta_local_idx, tau=float(tau))
    true_eval = fold.slice_nelbo(true_nelbo, sample_indices)
    ranking_score = -raw

    _metrics, rows = _selection_metrics(
        method=method,
        query_domains=sample_domains[sample_indices],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=ranking_score,
        true_nelbo_matrix=true_eval,
        fold=fold,
        global_true_nelbo_matrix=global_eval,
        global_expert_domains=expert_domains,
        tie_policy=tie_policy,
        selected_idx_override=selected_idx,
        ranking_score_matrix=ranking_score,
    )

    row_protocol = _method_protocol(method)
    if method in _RESIDUAL_ADOPTION_METHODS and (
        not bool(selected_by_inner_validation) or bool(force_diagnostic)
    ):
        row_protocol = MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    method_fields = _protocol_row_fields(fold=fold, method_protocol=row_protocol, method=method)
    extra = {
        "residual_policy_version": str(residual_cfg.residual_policy_version),
        "threshold_selection_policy": str(threshold_selection_policy),
        "feature_set": str(selected_feature_set),
        "residual_variant": str(selected_variant),
        "selected_tau": _threshold_label(float(tau)),
        "residual_score_direction": "higher_is_better",
        "residual_target_scale": "delta_u_pct",
        "spearman_score_source": "raw_residual_pre_threshold",
        "selected_by_inner_validation": int(bool(selected_by_inner_validation)),
        "adoption_selected_method": str(adoption_selected_method),
        "validation_top1_uplift": float(validation_summary.get("top1_uplift", 0.0)),
        "validation_spearman_uplift": float(validation_summary.get("spearman_uplift", 0.0)),
        "validation_gap_pct_reduction": float(validation_summary.get("gap_pct_reduction", 0.0)),
        "validation_safety_pass": int(bool(validation_summary.get("safety_pass", False))),
        "inner_validation_mean_top1_uplift": float(validation_summary.get("top1_uplift", 0.0)),
        "inner_validation_mean_gap_reduction": float(validation_summary.get("gap_pct_reduction", 0.0)),
        "inner_validation_max_harmful_override_rate": float(
            validation_summary.get("max_harmful_override_rate", 0.0)
        ),
        "inner_validation_min_utility_improving_override_rate": float(
            validation_summary.get("min_utility_improving_override_rate", 0.0)
        ),
        "inner_validation_domains_failed_gate": str(validation_summary.get("domains_failed_gate", "")),
        "inner_validation_worst_domain_top1_uplift": float(
            validation_summary.get("worst_domain_top1_uplift", 0.0)
        ),
        "inner_validation_worst_domain_gap_pct_reduction": float(
            validation_summary.get("worst_domain_gap_pct_reduction", 0.0)
        ),
        "fallback_used": int(bool(validation_summary.get("fallback_used", False))),
        "harmful_override_max": float(residual_cfg.harmful_override_max),
        "allow_calibrated_adoption": int(bool(residual_cfg.allow_calibrated_adoption)),
    }
    for row in rows:
        row["sample_index"] = int(sample_indices[int(row["sample_index"])])
        row.update(method_fields)
        row.update(extra)

    raw_rows: List[Dict[str, Any]] = []
    if emit_raw_rows:
        candidate_cols = np.asarray(fold.candidate_col_indices, dtype=np.int64)
        candidate_domains = np.asarray(fold.candidate_expert_domains, dtype=np.int64)
        for i, sample_index in enumerate(sample_indices.tolist()):
            meta_local = int(meta_local_idx[i])
            meta_col = int(candidate_cols[meta_local])
            nelbo_meta = float(true_nelbo[int(sample_index), meta_col])
            for local_candidate, candidate_domain in enumerate(candidate_domains.tolist()):
                cand_col = int(candidate_cols[int(local_candidate)])
                nelbo_candidate = float(true_nelbo[int(sample_index), cand_col])
                raw_rows.append(
                    {
                        "method": str(method),
                        "sample_index": int(sample_index),
                        "query_domain": int(sample_domains[int(sample_index)]),
                        "expert_domain": int(candidate_domain),
                        "metadata_selected_expert": int(candidate_domains[meta_local]),
                        "selected_expert": int(candidate_domains[int(selected_idx[i])]),
                        "candidate_oracle_expert": int(rows[i]["candidate_oracle_expert"]),
                        "raw_residual_score": float(raw[i, int(local_candidate)]),
                        "true_residual_pct": float(
                            (nelbo_meta - nelbo_candidate) / max(abs(nelbo_meta), _RESIDUAL_EPS)
                        ),
                        "true_nelbo": float(nelbo_candidate),
                        "metadata_selected_nelbo": float(nelbo_meta),
                        **method_fields,
                        **extra,
                    }
                )

    metadata_selected_experts = [int(fold.candidate_expert_domains[int(v)]) for v in meta_local_idx.tolist()]
    selected_experts = [int(fold.candidate_expert_domains[int(v)]) for v in selected_idx.tolist()]
    oracle_experts = [int(r["candidate_oracle_expert"]) for r in rows]
    selected_nelbo = np.asarray([float(r["selected_nelbo"]) for r in rows], dtype=np.float64)
    meta_nelbo = true_eval[np.arange(n_samples), meta_local_idx]
    override_mask = np.asarray(
        [int(s) != int(m) for s, m in zip(selected_experts, metadata_selected_experts)],
        dtype=bool,
    )
    n_override = int(np.sum(override_mask))
    improving = np.asarray(selected_nelbo < (meta_nelbo - 1e-12), dtype=bool)
    harmful = np.asarray(selected_nelbo > (meta_nelbo + 1e-12), dtype=bool)
    oracle_correct = np.asarray(
        [int(s) == int(o) for s, o in zip(selected_experts, oracle_experts)],
        dtype=bool,
    )
    denom = max(n_override, 1)
    override_delta_nelbo = selected_nelbo[override_mask] - meta_nelbo[override_mask]
    harmful_delta_nelbo = override_delta_nelbo[override_delta_nelbo > 1e-12]
    override_diag = {
        "method": str(method),
        "fold_query_domain": int(fold.heldout_domain),
        "n_samples": int(n_samples),
        "n_overrides": int(n_override),
        "override_rate": float(n_override / max(n_samples, 1)),
        "utility_improving_override_rate": float(np.sum(override_mask & improving) / denom),
        "oracle_correct_override_rate": float(np.sum(override_mask & oracle_correct) / denom),
        "harmful_override_rate": float(np.sum(override_mask & harmful) / denom),
        "mean_delta_nelbo_for_overrides": float(np.mean(override_delta_nelbo)) if n_override else 0.0,
        "median_delta_nelbo_for_overrides": float(np.median(override_delta_nelbo)) if n_override else 0.0,
        "max_harmful_delta_nelbo": float(np.max(harmful_delta_nelbo)) if harmful_delta_nelbo.size else 0.0,
        **method_fields,
        **extra,
    }

    confusion_counts: Dict[Tuple[int, int], int] = {}
    for selected, oracle in zip(selected_experts, oracle_experts):
        key = (int(selected), int(oracle))
        confusion_counts[key] = int(confusion_counts.get(key, 0)) + 1
    confusion_rows = [
        {
            "method": str(method),
            "fold_query_domain": int(fold.heldout_domain),
            "selected_expert": int(selected),
            "oracle_expert": int(oracle),
            "count": int(count),
            **method_fields,
            **extra,
        }
        for (selected, oracle), count in sorted(confusion_counts.items())
    ]
    return rows, raw_rows, override_diag, confusion_rows


def _validation_report(
    *,
    candidate_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    method: str,
) -> Dict[str, Any]:
    candidate_metrics = _aggregate_metrics_from_sample_rows(candidate_rows).get(str(method), {})
    baseline_metrics = _aggregate_metrics_from_sample_rows(baseline_rows).get("metadata_routing", {})
    domain_rows = _domain_breakdown_rows(list(candidate_rows) + list(baseline_rows))
    by_key = {(str(r["method"]), int(r["query_domain"])): r for r in domain_rows}
    domains = sorted(q for m, q in by_key if m == "metadata_routing")
    safety_pass = True
    worst_top1 = 0.0
    worst_spearman = 0.0
    worst_gap = 0.0
    for q in domains:
        base = by_key.get(("metadata_routing", q))
        cand = by_key.get((str(method), q))
        if base is None or cand is None:
            safety_pass = False
            continue
        top1_uplift = float(cand["top1_oracle_hit"]) - float(base["top1_oracle_hit"])
        spearman_uplift = float(cand["spearman"]) - float(base["spearman"])
        gap_reduction = float(base["mean_oracle_gap_pct"]) - float(cand["mean_oracle_gap_pct"])
        worst_top1 = min(worst_top1, top1_uplift)
        worst_spearman = min(worst_spearman, spearman_uplift)
        worst_gap = min(worst_gap, gap_reduction)
        if (
            top1_uplift < _CATASTROPHIC_TOP1_UPLIFT_MIN
            or spearman_uplift < _CATASTROPHIC_SPEARMAN_UPLIFT_MIN
            or gap_reduction < _CATASTROPHIC_GAP_PCT_REDUCTION_MIN
        ):
            safety_pass = False

    top1_uplift = float(candidate_metrics.get("top1_oracle_hit", 0.0)) - float(
        baseline_metrics.get("top1_oracle_hit", 0.0)
    )
    spearman_uplift = float(candidate_metrics.get("spearman", 0.0)) - float(
        baseline_metrics.get("spearman", 0.0)
    )
    gap_reduction = float(baseline_metrics.get("mean_oracle_gap_pct", 0.0)) - float(
        candidate_metrics.get("mean_oracle_gap_pct", 0.0)
    )
    return {
        "top1_uplift": float(top1_uplift),
        "spearman_uplift": float(spearman_uplift),
        "gap_pct_reduction": float(gap_reduction),
        "safety_pass": bool(safety_pass),
        "worst_domain_top1_uplift": float(worst_top1),
        "worst_domain_spearman_uplift": float(worst_spearman),
        "worst_domain_gap_pct_reduction": float(worst_gap),
    }


def _safe_v2_validation_report(
    *,
    candidate_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    override_rows: Sequence[Dict[str, Any]],
    method: str,
    residual_cfg: ResidualRoutingConfig,
) -> Dict[str, Any]:
    report = _validation_report(candidate_rows=candidate_rows, baseline_rows=baseline_rows, method=method)
    domain_rows = _domain_breakdown_rows(list(candidate_rows) + list(baseline_rows))
    by_key = {(str(r["method"]), int(r["query_domain"])): r for r in domain_rows}
    override_by_domain = {
        int(r.get("validation_query_domain", r.get("fold_query_domain", 0))): r
        for r in override_rows
    }

    candidate_by_domain: Dict[int, List[Dict[str, Any]]] = {}
    baseline_by_domain: Dict[int, List[Dict[str, Any]]] = {}
    for row in candidate_rows:
        candidate_by_domain.setdefault(int(row["query_domain"]), []).append(row)
    for row in baseline_rows:
        baseline_by_domain.setdefault(int(row["query_domain"]), []).append(row)

    failed: List[str] = []
    harmful_rates: List[float] = []
    improving_rates: List[float] = []
    override_rates: List[float] = []
    worst_top1 = 0.0
    worst_gap = 0.0

    domains = sorted(q for m, q in by_key if m == "metadata_routing")
    for q in domains:
        base = by_key.get(("metadata_routing", q))
        cand = by_key.get((str(method), q))
        diag = override_by_domain.get(int(q), {})
        reasons: List[str] = []
        if base is None or cand is None:
            reasons.append("missing_domain_metrics")
        else:
            top1_uplift = float(cand["top1_oracle_hit"]) - float(base["top1_oracle_hit"])
            gap_reduction = float(base["mean_oracle_gap_pct"]) - float(cand["mean_oracle_gap_pct"])
            worst_top1 = min(worst_top1, top1_uplift)
            worst_gap = min(worst_gap, gap_reduction)
            if top1_uplift < float(residual_cfg.catastrophic_top1_floor):
                reasons.append("catastrophic_top1")
            if gap_reduction < -float(residual_cfg.gap_regression_max):
                reasons.append("gap_regression")

        harmful_rate = float(diag.get("harmful_override_rate", 0.0))
        improving_rate = float(diag.get("utility_improving_override_rate", 0.0))
        override_rate = float(diag.get("override_rate", 0.0))
        harmful_rates.append(harmful_rate)
        improving_rates.append(improving_rate)
        override_rates.append(override_rate)
        if harmful_rate > float(residual_cfg.harmful_override_max):
            reasons.append("harmful_override")
        if override_rate > 0.0:
            if improving_rate <= harmful_rate:
                reasons.append("override_usefulness")
        else:
            base_selected_by_sample = {
                int(r["sample_index"]): int(r["selected_expert"])
                for r in baseline_by_domain.get(int(q), [])
            }
            exact_metadata = all(
                int(r["selected_expert"]) == int(base_selected_by_sample.get(int(r["sample_index"]), -1))
                for r in candidate_by_domain.get(int(q), [])
            )
            if not exact_metadata:
                reasons.append("zero_override_not_metadata")

        if reasons:
            failed.append(f"{int(q)}:{'|'.join(sorted(set(reasons)))}")

    report.update(
        {
            "safety_pass": bool(not failed and domains),
            "catastrophic_regression_breach": int(
                any("catastrophic_top1" in item or "gap_regression" in item for item in failed)
            ),
            "max_harmful_override_rate": float(max(harmful_rates)) if harmful_rates else 0.0,
            "mean_harmful_override_rate": float(np.mean(harmful_rates)) if harmful_rates else 0.0,
            "min_utility_improving_override_rate": float(min(improving_rates)) if improving_rates else 0.0,
            "mean_override_rate": float(np.mean(override_rates)) if override_rates else 0.0,
            "domains_failed_gate": ";".join(failed),
            "worst_domain_top1_uplift": float(worst_top1),
            "worst_domain_gap_pct_reduction": float(worst_gap),
        }
    )
    return report


def _safe_v2_fallback_config(method: str, residual_cfg: ResidualRoutingConfig, policy: str) -> _SelectedResidualConfig:
    feature = "minimal"
    if feature not in _safe_v2_adoption_feature_sets(residual_cfg):
        feature = str((_safe_v2_adoption_feature_sets(residual_cfg) or ("minimal",))[0])
    return _SelectedResidualConfig(
        method=method,
        feature_set=feature,
        tau=float("inf"),
        validation_top1_uplift=0.0,
        validation_spearman_uplift=0.0,
        validation_gap_reduction=0.0,
        validation_safety_pass=True,
        threshold_selection_policy=policy,
        fallback_used=True,
        validation_max_harmful_override_rate=0.0,
        validation_min_utility_improving_override_rate=0.0,
        validation_mean_override_rate=0.0,
        validation_domains_failed_gate="",
        validation_worst_domain_top1_uplift=0.0,
        validation_worst_domain_gap_pct_reduction=0.0,
    )


def _select_inner_config_safe_v2(
    *,
    method: str,
    residual_cfg: ResidualRoutingConfig,
    sample_domains: np.ndarray,
    embeddings: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    outer_heldout_domain: int,
    train_idx: np.ndarray,
    group_balanced: bool,
    tie_policy: str,
) -> _SelectedResidualConfig:
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    finite_thresholds = [float(v) for v in residual_cfg.thresholds if not math.isinf(float(v))]
    feature_sets = _safe_v2_adoption_feature_sets(residual_cfg)
    candidates: List[Tuple[Tuple[float, float, float, float, int, float], str, float, Dict[str, Any]]] = []

    if len(source_domains) < 2 or not finite_thresholds or not feature_sets:
        return _safe_v2_fallback_config(
            method,
            residual_cfg,
            "safe_override_v2_inner_loqdo_insufficient_source_domains_fallback_inf",
        )

    for feature in feature_sets:
        if feature not in _FEATURE_SETS:
            continue
        by_threshold_rows: Dict[float, List[Dict[str, Any]]] = {float(t): [] for t in finite_thresholds}
        by_threshold_override_rows: Dict[float, List[Dict[str, Any]]] = {float(t): [] for t in finite_thresholds}
        baseline_rows: List[Dict[str, Any]] = []
        for validation_domain in source_domains:
            train_idx_arr = np.asarray(train_idx, dtype=np.int64)
            inner_train_idx = train_idx_arr[sample_domains[train_idx_arr] != int(validation_domain)]
            validation_idx = train_idx_arr[sample_domains[train_idx_arr] == int(validation_domain)]
            if inner_train_idx.size == 0 or validation_idx.size == 0:
                continue
            validation_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(validation_domain)],
            )
            model, context = _fit_residual_model(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                metadata_similarity=metadata_similarity,
                outer_heldout_domain=int(outer_heldout_domain),
                train_indices=inner_train_idx,
                feature_set=feature,
                group_balanced=bool(group_balanced),
                l2=float(residual_cfg.ridge_l2),
            )
            true_eval = validation_fold.slice_nelbo(true_nelbo, validation_idx)
            global_eval = true_nelbo[np.asarray(validation_idx, dtype=np.int64)]
            baseline_rows.extend(
                _metadata_rows(
                    sample_domains=sample_domains,
                    expert_domains=expert_domains,
                    sample_indices=validation_idx,
                    fold=validation_fold,
                    metadata_similarity=metadata_similarity,
                    true_eval=true_eval,
                    global_eval=global_eval,
                    tie_policy=tie_policy,
                )
            )
            for tau in finite_thresholds:
                rows, _raw, diag, _conf = _evaluate_residual_config(
                    method=method,
                    residual_cfg=residual_cfg,
                    sample_domains=sample_domains,
                    embeddings=embeddings,
                    true_nelbo=true_nelbo,
                    expert_domains=expert_domains,
                    metadata_similarity=metadata_similarity,
                    sample_indices=validation_idx,
                    fold=validation_fold,
                    global_eval=global_eval,
                    model=model,
                    context=context,
                    tau=float(tau),
                    selected_feature_set=feature,
                    selected_variant=method,
                    threshold_selection_policy="safe_override_v2_inner_leave_query_domain_out",
                    validation_summary={},
                    tie_policy=tie_policy,
                    selected_by_inner_validation=True,
                    adoption_selected_method=str(method),
                    emit_raw_rows=False,
                )
                by_threshold_rows[float(tau)].extend(rows)
                diag = dict(diag)
                diag["validation_query_domain"] = int(validation_domain)
                by_threshold_override_rows[float(tau)].append(diag)

        for tau, rows in by_threshold_rows.items():
            if not rows or not baseline_rows:
                continue
            report = _safe_v2_validation_report(
                candidate_rows=rows,
                baseline_rows=baseline_rows,
                override_rows=by_threshold_override_rows[float(tau)],
                method=method,
                residual_cfg=residual_cfg,
            )
            if not bool(report["safety_pass"]):
                continue
            score = (
                float(report["gap_pct_reduction"]),
                float(report["top1_uplift"]),
                -float(report["mean_harmful_override_rate"]),
                -float(report["mean_override_rate"]),
                -_feature_complexity_rank(feature),
                float(tau),
            )
            candidates.append((score, feature, float(tau), report))

    if not candidates:
        return _safe_v2_fallback_config(
            method,
            residual_cfg,
            "safe_override_v2_inner_leave_query_domain_out_fallback_inf",
        )

    _score, feature, tau, report = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return _SelectedResidualConfig(
        method=method,
        feature_set=str(feature),
        tau=float(tau),
        validation_top1_uplift=float(report["top1_uplift"]),
        validation_spearman_uplift=float(report["spearman_uplift"]),
        validation_gap_reduction=float(report["gap_pct_reduction"]),
        validation_safety_pass=bool(report["safety_pass"]),
        threshold_selection_policy="safe_override_v2_inner_leave_query_domain_out",
        fallback_used=False,
        validation_max_harmful_override_rate=float(report["max_harmful_override_rate"]),
        validation_min_utility_improving_override_rate=float(report["min_utility_improving_override_rate"]),
        validation_mean_override_rate=float(report["mean_override_rate"]),
        validation_domains_failed_gate=str(report["domains_failed_gate"]),
        validation_worst_domain_top1_uplift=float(report["worst_domain_top1_uplift"]),
        validation_worst_domain_gap_pct_reduction=float(report["worst_domain_gap_pct_reduction"]),
    )


def _select_inner_config(
    *,
    method: str,
    residual_cfg: ResidualRoutingConfig,
    sample_domains: np.ndarray,
    embeddings: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    outer_heldout_domain: int,
    train_idx: np.ndarray,
    group_balanced: bool,
    tie_policy: str,
    force_tau_zero: bool = False,
) -> _SelectedResidualConfig:
    if _is_safe_override_v2(residual_cfg) and not bool(force_tau_zero):
        return _select_inner_config_safe_v2(
            method=method,
            residual_cfg=residual_cfg,
            sample_domains=sample_domains,
            embeddings=embeddings,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            metadata_similarity=metadata_similarity,
            outer_heldout_domain=int(outer_heldout_domain),
            train_idx=train_idx,
            group_balanced=bool(group_balanced),
            tie_policy=tie_policy,
        )
    source_domains = sorted(set(int(sample_domains[int(i)]) for i in np.asarray(train_idx, dtype=np.int64).tolist()))
    thresholds = [0.0] if force_tau_zero else list(residual_cfg.thresholds)
    finite_thresholds = [float(v) for v in thresholds if not math.isinf(float(v))]
    candidates: List[Tuple[Tuple[float, float, float], str, float, Dict[str, Any]]] = []

    if len(source_domains) < 2:
        return _SelectedResidualConfig(
            method=method,
            feature_set=str(residual_cfg.feature_sets[0] if residual_cfg.feature_sets else "minimal"),
            tau=float("inf"),
            validation_top1_uplift=0.0,
            validation_spearman_uplift=0.0,
            validation_gap_reduction=0.0,
            validation_safety_pass=False,
            threshold_selection_policy="inner_loqdo_insufficient_source_domains_fallback_inf",
        )

    for feature_set in residual_cfg.feature_sets:
        feature = str(feature_set).strip().lower()
        if feature not in _FEATURE_SETS:
            continue
        by_threshold_rows: Dict[float, List[Dict[str, Any]]] = {float(t): [] for t in finite_thresholds}
        baseline_rows: List[Dict[str, Any]] = []
        for validation_domain in source_domains:
            inner_train_idx = np.asarray(train_idx, dtype=np.int64)[sample_domains[np.asarray(train_idx, dtype=np.int64)] != int(validation_domain)]
            validation_idx = np.asarray(train_idx, dtype=np.int64)[sample_domains[np.asarray(train_idx, dtype=np.int64)] == int(validation_domain)]
            if inner_train_idx.size == 0 or validation_idx.size == 0:
                continue
            validation_fold = FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(outer_heldout_domain),
                expert_domains=expert_domains,
                excluded_domains=[int(validation_domain)],
            )
            model, context = _fit_residual_model(
                embeddings=embeddings,
                sample_domains=sample_domains,
                true_nelbo=true_nelbo,
                expert_domains=expert_domains,
                metadata_similarity=metadata_similarity,
                outer_heldout_domain=int(outer_heldout_domain),
                train_indices=inner_train_idx,
                feature_set=feature,
                group_balanced=bool(group_balanced),
                l2=float(residual_cfg.ridge_l2),
            )
            true_eval = validation_fold.slice_nelbo(true_nelbo, validation_idx)
            global_eval = true_nelbo[np.asarray(validation_idx, dtype=np.int64)]
            baseline_rows.extend(
                _metadata_rows(
                    sample_domains=sample_domains,
                    expert_domains=expert_domains,
                    sample_indices=validation_idx,
                    fold=validation_fold,
                    metadata_similarity=metadata_similarity,
                    true_eval=true_eval,
                    global_eval=global_eval,
                    tie_policy=tie_policy,
                )
            )
            for tau in finite_thresholds:
                rows, _raw, _diag, _conf = _evaluate_residual_config(
                    method=method,
                    residual_cfg=residual_cfg,
                    sample_domains=sample_domains,
                    embeddings=embeddings,
                    true_nelbo=true_nelbo,
                    expert_domains=expert_domains,
                    metadata_similarity=metadata_similarity,
                    sample_indices=validation_idx,
                    fold=validation_fold,
                    global_eval=global_eval,
                    model=model,
                    context=context,
                    tau=float(tau),
                    selected_feature_set=feature,
                    selected_variant=method,
                    threshold_selection_policy="inner_leave_query_domain_out",
                    validation_summary={},
                    tie_policy=tie_policy,
                    selected_by_inner_validation=True,
                    adoption_selected_method=str(method),
                    emit_raw_rows=False,
                )
                by_threshold_rows[float(tau)].extend(rows)

        for tau, rows in by_threshold_rows.items():
            if not rows or not baseline_rows:
                continue
            report = _validation_report(candidate_rows=rows, baseline_rows=baseline_rows, method=method)
            improves = bool(float(report["gap_pct_reduction"]) > 0.0 or float(report["top1_uplift"]) > 0.0)
            if (bool(report["safety_pass"]) and improves) or bool(force_tau_zero):
                score = (
                    float(report["gap_pct_reduction"]),
                    float(report["top1_uplift"]),
                    float(report["spearman_uplift"]),
                )
                candidates.append((score, feature, float(tau), report))

    if not candidates:
        return _SelectedResidualConfig(
            method=method,
            feature_set=str(residual_cfg.feature_sets[0] if residual_cfg.feature_sets else "minimal"),
            tau=float("inf"),
            validation_top1_uplift=0.0,
            validation_spearman_uplift=0.0,
            validation_gap_reduction=0.0,
            validation_safety_pass=False,
            threshold_selection_policy="inner_leave_query_domain_out_fallback_inf",
        )

    candidates = sorted(candidates, key=lambda item: (item[0][0], item[0][1], item[0][2], -item[2]), reverse=True)
    _score, feature, tau, report = candidates[0]
    return _SelectedResidualConfig(
        method=method,
        feature_set=str(feature),
        tau=float(tau),
        validation_top1_uplift=float(report["top1_uplift"]),
        validation_spearman_uplift=float(report["spearman_uplift"]),
        validation_gap_reduction=float(report["gap_pct_reduction"]),
        validation_safety_pass=bool(report["safety_pass"]),
        threshold_selection_policy="inner_leave_query_domain_out",
    )


def _select_adoption_variant(configs: Sequence[_SelectedResidualConfig]) -> str:
    if not configs:
        return ""
    if not any(str(item.method) in _SAFE_V2_METHODS for item in configs):
        ordered_v1 = sorted(
            configs,
            key=lambda item: (
                int(bool(item.validation_safety_pass)),
                float(item.validation_gap_reduction),
                float(item.validation_top1_uplift),
                float(item.validation_spearman_uplift),
                -float(item.tau) if math.isfinite(float(item.tau)) else float("-inf"),
                str(item.method),
            ),
            reverse=True,
        )
        return str(ordered_v1[0].method)
    ordered = sorted(
        configs,
        key=lambda item: (
            int(bool(item.validation_safety_pass)),
            float(item.validation_gap_reduction),
            float(item.validation_top1_uplift),
            float(item.validation_spearman_uplift),
            -float(item.validation_max_harmful_override_rate),
            -float(item.validation_mean_override_rate),
            -int(bool(item.fallback_used)),
            float(item.tau) if math.isfinite(float(item.tau)) else float("inf"),
            str(item.method),
        ),
        reverse=True,
    )
    return str(ordered[0].method)


def _copy_rows_as_inner_selected_method(
    *,
    rows: Sequence[Dict[str, Any]],
    fold: FoldCandidateSet,
    source_method: str,
) -> List[Dict[str, Any]]:
    method = "metadata_residual_inner_selected"
    method_fields = _protocol_row_fields(
        fold=fold,
        method_protocol=_method_protocol(method),
        method=method,
    )
    copied: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row.update(method_fields)
        new_row["source_residual_method"] = str(source_method)
        new_row["residual_variant"] = str(source_method)
        new_row["selected_by_inner_validation"] = 1
        copied.append(new_row)
    return copied


def _copy_diagnostic_rows_as_inner_selected_method(
    *,
    rows: Sequence[Dict[str, Any]],
    source_method: str,
) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["method"] = "metadata_residual_inner_selected"
        candidate_label = str(new_row.get("candidate_experts", "")).strip()
        if candidate_label and "fold_query_domain" in new_row:
            candidate_domains = [int(v) for v in candidate_label.split("|") if str(v).strip()]
            fold_domain = int(new_row["fold_query_domain"])
            method_fields = _protocol_row_fields(
                fold=FoldCandidateSet.for_heldout_domain(
                    heldout_domain=fold_domain,
                    expert_domains=candidate_domains + [fold_domain],
                ),
                method_protocol=_method_protocol("metadata_residual_inner_selected"),
                method="metadata_residual_inner_selected",
            )
            new_row.update(method_fields)
        new_row["source_residual_method"] = str(source_method)
        new_row["residual_variant"] = str(source_method)
        new_row["selected_by_inner_validation"] = 1
        copied.append(new_row)
    return copied


def _alias_unconstrained_reference(
    *,
    residual_cfg: ResidualRoutingConfig,
    learned_sample_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in learned_sample_rows:
        if str(row.get("method", "")) != str(residual_cfg.unconstrained_reference_method):
            continue
        new_row = dict(row)
        protocol = MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
        method_fields = _protocol_row_fields(
            fold=FoldCandidateSet.for_heldout_domain(
                heldout_domain=int(new_row["fold_query_domain"]),
                expert_domains=[int(v) for v in str(new_row["candidate_experts"]).split("|")] + [int(new_row["fold_query_domain"])],
            ),
            method_protocol=protocol,
            method="unconstrained_learned_reference",
        )
        new_row.update(method_fields)
        new_row["method"] = "unconstrained_learned_reference"
        new_row["unconstrained_reference_method"] = str(residual_cfg.unconstrained_reference_method)
        new_row["residual_policy_version"] = str(residual_cfg.residual_policy_version)
        out.append(new_row)
    return out


def _audit_row(
    *,
    method: str,
    fold: FoldCandidateSet,
    selected: _SelectedResidualConfig,
    residual_cfg: ResidualRoutingConfig,
    adoption_selected_method: str,
    override_diag: Mapping[str, Any],
    heldout_report: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_by_inner = int(str(method) == str(adoption_selected_method))
    heldout_harmful = float(override_diag.get("harmful_override_rate", 0.0))
    policy_pass = bool(
        fold.target_expert_excluded
        and not math.isnan(float(selected.validation_gap_reduction))
        and (
            not _is_safe_override_v2(residual_cfg)
            or (
                float(selected.validation_max_harmful_override_rate) <= float(residual_cfg.harmful_override_max)
                and int(heldout_report.get("catastrophic_regression_breach", 0)) == 0
            )
        )
    )
    return {
        "method": str(method),
        "fold_query_domain": int(fold.heldout_domain),
        "target_expert_excluded": int(fold.target_expert_excluded),
        "heldout_query_domain_used_for_training": 0,
        "heldout_query_domain_used_for_threshold_tuning": 0,
        "uses_eval_domain_latent_statistics": 0,
        "uses_raw_expert_or_query_identity": 0,
        "metadata_baseline_unchanged": 1,
        "score_direction_consistent": 1,
        "domain_40_specific_tuning": 0,
        "residual_policy_version": str(residual_cfg.residual_policy_version),
        "feature_set": str(selected.feature_set),
        "selected_tau": _threshold_label(float(selected.tau)),
        "threshold_selection_policy": str(selected.threshold_selection_policy),
        "adoption_selected_method": str(adoption_selected_method),
        "selected_by_inner_validation": int(selected_by_inner),
        "fallback_used": int(bool(selected.fallback_used)),
        "harmful_override_max": float(residual_cfg.harmful_override_max),
        "allow_calibrated_adoption": int(bool(residual_cfg.allow_calibrated_adoption)),
        "inner_validation_mean_top1_uplift": float(selected.validation_top1_uplift),
        "inner_validation_mean_gap_reduction": float(selected.validation_gap_reduction),
        "inner_validation_max_harmful_override_rate": float(selected.validation_max_harmful_override_rate),
        "inner_validation_min_utility_improving_override_rate": float(
            selected.validation_min_utility_improving_override_rate
        ),
        "inner_validation_domains_failed_gate": str(selected.validation_domains_failed_gate),
        "inner_validation_worst_domain_top1_uplift": float(selected.validation_worst_domain_top1_uplift),
        "inner_validation_worst_domain_gap_pct_reduction": float(
            selected.validation_worst_domain_gap_pct_reduction
        ),
        "heldout_top1_uplift": float(heldout_report.get("top1_uplift", 0.0)),
        "heldout_gap_reduction": float(heldout_report.get("gap_pct_reduction", 0.0)),
        "heldout_harmful_override_rate": float(heldout_harmful),
        "catastrophic_regression_breach": int(heldout_report.get("catastrophic_regression_breach", 0)),
        "policy_audit_pass": int(policy_pass),
    }


def run_residual_methods_for_fold(
    *,
    embeddings: np.ndarray,
    sample_domains: np.ndarray,
    true_nelbo: np.ndarray,
    expert_domains: Sequence[int],
    metadata_similarity: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: FoldCandidateSet,
    global_eval: np.ndarray,
    residual_cfg: ResidualRoutingConfig,
    learned_sample_rows: Sequence[Dict[str, Any]],
    tie_policy: str,
) -> ResidualFoldOutputs:
    if not bool(residual_cfg.enabled):
        return ResidualFoldOutputs([], [], [], [], [])
    if "ridge" not in {str(v).strip().lower() for v in residual_cfg.models}:
        raise ProtocolError("Residual routing supports only models containing 'ridge'")

    sample_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    override_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    sample_rows.extend(
        _alias_unconstrained_reference(
            residual_cfg=residual_cfg,
            learned_sample_rows=learned_sample_rows,
        )
    )

    if _is_safe_override_v2(residual_cfg):
        method_specs = [
            ("metadata_residual_argmax", False, True),
            ("metadata_residual_thresholded_safe_v2", False, False),
            ("metadata_residual_group_robust_safe_v2", True, False),
        ]
        adoption_methods = ("metadata_residual_thresholded_safe_v2", "metadata_residual_group_robust_safe_v2")
    else:
        method_specs = [
            ("metadata_residual_argmax", False, True),
            ("metadata_residual_thresholded", False, False),
            ("metadata_residual_group_robust", True, False),
        ]
        adoption_methods = ("metadata_residual_thresholded", "metadata_residual_group_robust")
    selected_by_method: Dict[str, _SelectedResidualConfig] = {}
    for method, group_balanced, force_tau_zero in method_specs:
        selection_cfg = residual_cfg
        if _is_safe_override_v2(residual_cfg) and bool(force_tau_zero):
            diagnostic_feature_sets = _safe_v2_diagnostic_feature_sets(residual_cfg)
            if diagnostic_feature_sets:
                selection_cfg = replace(residual_cfg, feature_sets=diagnostic_feature_sets)
        selected_by_method[str(method)] = _select_inner_config(
            method=method,
            residual_cfg=selection_cfg,
            sample_domains=sample_domains,
            embeddings=embeddings,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            metadata_similarity=metadata_similarity,
            outer_heldout_domain=int(fold.heldout_domain),
            train_idx=train_idx,
            group_balanced=bool(group_balanced),
            tie_policy=tie_policy,
            force_tau_zero=bool(force_tau_zero),
        )

    adoption_selected_method = _select_adoption_variant(
        [
            selected_by_method[m]
            for m in adoption_methods
            if m in selected_by_method
        ]
    )

    test_true_eval = fold.slice_nelbo(true_nelbo, test_idx)
    metadata_eval_rows = _metadata_rows(
        sample_domains=sample_domains,
        expert_domains=expert_domains,
        sample_indices=test_idx,
        fold=fold,
        metadata_similarity=metadata_similarity,
        true_eval=test_true_eval,
        global_eval=global_eval,
        tie_policy=tie_policy,
    )

    for method, group_balanced, _force_tau_zero in method_specs:
        selected = selected_by_method[str(method)]
        is_selected_adoption_candidate = bool(method == adoption_selected_method)
        model, context = _fit_residual_model(
            embeddings=embeddings,
            sample_domains=sample_domains,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            metadata_similarity=metadata_similarity,
            outer_heldout_domain=int(fold.heldout_domain),
            train_indices=np.asarray(train_idx, dtype=np.int64),
            feature_set=str(selected.feature_set),
            group_balanced=bool(group_balanced),
            l2=float(residual_cfg.ridge_l2),
        )
        eval_rows, eval_raw, override_diag, eval_confusion = _evaluate_residual_config(
            method=method,
            residual_cfg=residual_cfg,
            sample_domains=sample_domains,
            embeddings=embeddings,
            true_nelbo=true_nelbo,
            expert_domains=expert_domains,
            metadata_similarity=metadata_similarity,
            sample_indices=test_idx,
            fold=fold,
            global_eval=global_eval,
            model=model,
            context=context,
            tau=float(selected.tau),
            selected_feature_set=str(selected.feature_set),
            selected_variant=str(method),
            threshold_selection_policy=str(selected.threshold_selection_policy),
            validation_summary={
                "top1_uplift": float(selected.validation_top1_uplift),
                "spearman_uplift": float(selected.validation_spearman_uplift),
                "gap_pct_reduction": float(selected.validation_gap_reduction),
                "safety_pass": bool(selected.validation_safety_pass),
                "max_harmful_override_rate": float(selected.validation_max_harmful_override_rate),
                "min_utility_improving_override_rate": float(
                    selected.validation_min_utility_improving_override_rate
                ),
                "domains_failed_gate": str(selected.validation_domains_failed_gate),
                "worst_domain_top1_uplift": float(selected.validation_worst_domain_top1_uplift),
                "worst_domain_gap_pct_reduction": float(selected.validation_worst_domain_gap_pct_reduction),
                "fallback_used": bool(selected.fallback_used),
            },
            tie_policy=tie_policy,
            selected_by_inner_validation=is_selected_adoption_candidate,
            adoption_selected_method=str(adoption_selected_method),
            force_diagnostic=True,
            emit_raw_rows=True,
        )
        sample_rows.extend(eval_rows)
        raw_rows.extend(eval_raw)
        override_rows.append(override_diag)
        confusion_rows.extend(eval_confusion)
        if is_selected_adoption_candidate:
            sample_rows.extend(
                _copy_rows_as_inner_selected_method(
                    rows=eval_rows,
                    fold=fold,
                    source_method=str(method),
                )
            )
            raw_rows.extend(
                _copy_diagnostic_rows_as_inner_selected_method(
                    rows=eval_raw,
                    source_method=str(method),
                )
            )
            override_rows.extend(
                _copy_diagnostic_rows_as_inner_selected_method(
                    rows=[override_diag],
                    source_method=str(method),
                )
            )
            confusion_rows.extend(
                _copy_diagnostic_rows_as_inner_selected_method(
                    rows=eval_confusion,
                    source_method=str(method),
                )
            )
        if _is_safe_override_v2(residual_cfg):
            heldout_report = _safe_v2_validation_report(
                candidate_rows=eval_rows,
                baseline_rows=metadata_eval_rows,
                override_rows=[override_diag],
                method=str(method),
                residual_cfg=residual_cfg,
            )
        else:
            heldout_report = _validation_report(
                candidate_rows=eval_rows,
                baseline_rows=metadata_eval_rows,
                method=str(method),
            )
        audit_rows.append(
            _audit_row(
                method=method,
                fold=fold,
                selected=selected,
                residual_cfg=residual_cfg,
                adoption_selected_method=str(adoption_selected_method),
                override_diag=override_diag,
                heldout_report=heldout_report,
            )
        )

    return ResidualFoldOutputs(
        sample_rows=sample_rows,
        raw_rows=raw_rows,
        override_rows=override_rows,
        audit_rows=audit_rows,
        confusion_rows=confusion_rows,
    )
