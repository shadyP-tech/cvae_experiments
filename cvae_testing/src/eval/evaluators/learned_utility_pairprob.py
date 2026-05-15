from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.evaluators.learned_utility_config import (
    ConformalRegretSetConfig,
    JackknifeLCBTournamentConfig,
    PairprobTournamentConfig,
    Top2MarginRerankerConfig,
)
from src.eval.evaluators.learned_utility_models import _LogisticRidgePairprob
from src.eval.evaluators.learned_utility_protocol import FoldCandidateSet, ProtocolError
from src.eval.evaluators.learned_utility_selection import _selection_metrics
from src.eval.evaluators.learned_utility_pairs import _zscore_features
from src.eval.metrics import spearman_corr


DIRECT_PAIRPROB_DIAGNOSTIC_METHOD = "pairwise_direct_pairprob_tournament_v1"
DIRECT_PAIRPROB_ADOPTION_METHOD = "pairwise_direct_pairprob_adoption_v1"
GROUP_ROBUST_PAIRPROB_METHOD = "pairwise_group_robust_pairprob_tournament_v1"
COMBINED_PAIRPROB_DIAGNOSTIC_METHOD = "pairwise_pairprob_combined_diagnostic_v1"
TOP2_RERANK_METHOD = "pairwise_direct_top2_margin_reranker_v1"
ORACLE_TOP2_RERANK_DIAGNOSTIC_METHOD = "oracle_top2_margin_reranker_diagnostic_v1"
TOP2_RERANK_FEATURE_SET = "top2_rerank_latent_context_v1"
DIRECT_PAIRPROB_SELECTION_POLICY = "source_inner_mean_gap_then_catastrophic_then_top1_v1"
GROUP_ROBUST_PAIRPROB_SELECTION_POLICY = (
    "source_inner_group_robust_worst_gap_then_catastrophic_then_mean_gap_v1"
)
DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY = "pairprob_latent_only_v1"
TOP2_RERANK_GUARD_PRIORITY = (
    "insufficient_source_inner_rerank_rows",
    "insufficient_source_inner_positive_rows",
    "insufficient_source_inner_negative_rows",
    "insufficient_source_inner_active_domains",
    "low_margin_not_high_regret_enriched",
    "activation_rate_too_high",
    "switch_rate_too_high",
    "harm_rate_too_high",
    "insufficient_gap_reduction",
    "unstable_source_inner_selection",
    "weak_reranker_auc_or_calibration",
    "worsens_direct_pairprob",
)
DIRECT_ADOPTION_AUDIT_REASONS = {
    "none",
    "missing_diagnostic_direct_row",
    "route_hash_mismatch",
    "nelbo_metric_mismatch",
    "source_only_audit_failed",
    "target_leakage_audit_failed",
    "invalid_feature_family",
    "duplicate_sign_ci_candidate",
}
_DIRECT_ROUTE_HASH_FIELDS = (
    "selected_expert",
    "route_experts",
    "route_weights",
    "route_size",
    "route_mode",
    "selected_nelbo",
    "oracle_nelbo",
    "oracle_gap",
    "oracle_gap_pct",
    "top1_oracle_hit",
    "selected_rank",
    "spearman",
    "pairwise_auc",
)
_DIRECT_ALIAS_MATCH_FIELDS = (
    "selected_expert",
    "route_experts",
    "route_weights",
    "selected_nelbo",
    "oracle_gap_pct",
    "top1_oracle_hit",
    "spearman",
    "selected_rank",
)


def _pairprob_selection_policy_for_method(method: str) -> str:
    if str(method) in {DIRECT_PAIRPROB_DIAGNOSTIC_METHOD, DIRECT_PAIRPROB_ADOPTION_METHOD}:
        return DIRECT_PAIRPROB_SELECTION_POLICY
    return GROUP_ROBUST_PAIRPROB_SELECTION_POLICY


def _direct_pairprob_route_hash(row: Mapping[str, Any]) -> str:
    payload = "\x1f".join(str(row.get(field, "")) for field in _DIRECT_ROUTE_HASH_FIELDS)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _direct_adoption_audit_reason(
    *,
    diagnostic_hash: str,
    adoption_hash: str,
    metrics_match: bool,
    source_only_audit_pass: bool,
    target_leakage_audit_pass: bool,
    feature_family: str,
    duplicate_sign_ci_candidate: bool = False,
) -> str:
    if not str(diagnostic_hash):
        return "missing_diagnostic_direct_row"
    if str(diagnostic_hash) != str(adoption_hash):
        return "route_hash_mismatch"
    if not bool(metrics_match):
        return "nelbo_metric_mismatch"
    if str(feature_family) != DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY:
        return "invalid_feature_family"
    if not bool(source_only_audit_pass):
        return "source_only_audit_failed"
    if not bool(target_leakage_audit_pass):
        return "target_leakage_audit_failed"
    if bool(duplicate_sign_ci_candidate):
        return "duplicate_sign_ci_candidate"
    return "none"


def clone_direct_pairprob_adoption_rows(
    direct_rows: Sequence[Mapping[str, Any]],
    *,
    adoption_method: str = DIRECT_PAIRPROB_ADOPTION_METHOD,
) -> List[Dict[str, Any]]:
    cloned: List[Dict[str, Any]] = []
    for direct_row in direct_rows:
        source = dict(direct_row)
        diagnostic_hash = str(source.get("direct_diagnostic_route_hash", "")) or _direct_pairprob_route_hash(source)
        adoption = dict(source)
        adoption["method"] = str(adoption_method)
        adoption["method_role"] = "learned"
        adoption["adoption_eligible"] = 1
        adoption["diagnostic_only"] = 0
        adoption["diagnostic_only_reason"] = ""
        adoption["base_method"] = DIRECT_PAIRPROB_DIAGNOSTIC_METHOD
        adoption["adoption_feature_family"] = DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY
        adoption["direct_adoption_is_alias_of"] = DIRECT_PAIRPROB_DIAGNOSTIC_METHOD
        adoption["direct_diagnostic_route_hash"] = diagnostic_hash
        adoption_hash = _direct_pairprob_route_hash(adoption)
        adoption["direct_adoption_route_hash"] = adoption_hash
        metrics_match = all(str(adoption.get(field, "")) == str(source.get(field, "")) for field in _DIRECT_ALIAS_MATCH_FIELDS)
        source_evidence_ok = not bool(str(source.get("diagnostic_only_reason", "")).strip())
        same_route = bool(diagnostic_hash == adoption_hash and metrics_match)
        source_only_pass = bool(
            str(adoption.get("feature_set", "")) == DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY
            and str(adoption.get("pairprob_feature_set", "")) == DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY
            and str(adoption.get("adoption_feature_family", "")) == DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY
            and source_evidence_ok
        )
        target_pass = bool(
            int(float(adoption.get("routing_uses_eval_nelbo", 0) or 0)) == 0
            and int(float(adoption.get("routing_uses_eval_domain_statistics", 0) or 0)) == 0
        )
        adoption["direct_adoption_same_route_as_direct"] = int(same_route)
        adoption["source_only_audit_pass"] = int(source_only_pass)
        adoption["target_leakage_audit_pass"] = int(target_pass)
        adoption["excluded_from_sign_ci_selection"] = 0
        adoption["sign_ci_candidate"] = 1
        adoption["direct_vs_group_robust_primary_comparator"] = 1
        reason = _direct_adoption_audit_reason(
            diagnostic_hash=diagnostic_hash,
            adoption_hash=adoption_hash,
            metrics_match=metrics_match,
            source_only_audit_pass=source_only_pass,
            target_leakage_audit_pass=target_pass,
            feature_family=str(adoption.get("adoption_feature_family", "")),
        )
        adoption["direct_adoption_audit_failure_reason"] = reason
        if reason != "none":
            adoption["method_role"] = "diagnostic"
            adoption["adoption_eligible"] = 0
            adoption["diagnostic_only"] = 1
            adoption["excluded_from_sign_ci_selection"] = 1
            adoption["sign_ci_candidate"] = 0
            adoption["diagnostic_only_reason"] = reason
        cloned.append(adoption)
    return cloned


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


@dataclass(frozen=True)
class Top2RerankTrainingData:
    x: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    query_domains: np.ndarray
    total_active_rows: int
    dropped_near_tie: int
    positive_rows: int
    negative_rows: int
    kept_by_domain: Dict[int, int]
    switch_candidate_rate: float


@dataclass(frozen=True)
class Top2RerankModelBundle:
    feature_set: str
    ridge_l2: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    model: _LogisticRidgePairprob


@dataclass(frozen=True)
class Top2RerankCalibrationBlock:
    validation_domain: int
    query_domains: np.ndarray
    expert_domains: Tuple[int, ...]
    x_rows: np.ndarray
    prob_matrix: np.ndarray
    true_nelbo_matrix: np.ndarray
    global_true_nelbo_matrix: np.ndarray
    fold: FoldCandidateSet
    pairprob_direct_gap_pct: np.ndarray
    metadata_oracle_gap_pct: np.ndarray | None = None


@dataclass(frozen=True)
class Top2RerankSelection:
    method: str
    oracle_method: str
    base_method: str
    feature_set: str
    base_feature_set: str
    base_ridge_l2: float
    reranker_l2: float
    margin_threshold: float
    decision_threshold: float
    selected_by_inner_validation: bool
    diagnostic_only_reason: str = ""
    noop: bool = False
    guard_status: str = "selected"
    selection_stability_status: str = "stable"
    source_inner_validation_domains: int = 0
    source_inner_top2_rerank_rows: int = 0
    source_inner_top2_rerank_positive_rows: int = 0
    source_inner_top2_rerank_negative_rows: int = 0
    source_inner_top2_rerank_active_domains: int = 0
    source_inner_switch_candidate_rate: float = float("nan")
    source_inner_gap_reduction_abs_pct_points: float = float("nan")
    source_inner_high_regret_reduction: float = float("nan")
    source_inner_activation_rate: float = float("nan")
    source_inner_switch_rate: float = float("nan")
    source_inner_help_rate_active_only: float = float("nan")
    source_inner_harm_rate_active_only: float = float("nan")
    source_inner_mean_oracle_gap_pct: float = float("nan")
    source_inner_high_regret_rate: float = float("nan")
    source_inner_top1: float = float("nan")
    source_inner_spearman: float = float("nan")
    base_top2_margin_auc_for_high_regret: float = float("nan")
    base_top2_margin_spearman_with_oracle_gap: float = float("nan")
    overall_high_regret_rate_direct: float = float("nan")
    low_margin_active_high_regret_rate: float = float("nan")
    low_margin_high_regret_enrichment: float = float("nan")
    top2_rerank_auc_source_inner: float = float("nan")
    top2_rerank_brier_source_inner: float = float("nan")
    top2_rerank_calibration_status: str = ""
    oracle_top2_active_gap_reduction_pct: float = float("nan")
    oracle_top2_active_high_regret_reduction: float = float("nan")
    oracle_top2_recoverable_error_rate: float = float("nan")
    oracle_top2_recoverable_gap_mass_pct_points: float = float("nan")
    model: Top2RerankModelBundle | None = None


@dataclass(frozen=True)
class ConformalCalibrationBlock:
    validation_domain: int
    query_domains: np.ndarray
    expert_domains: Tuple[int, ...]
    prob_matrix: np.ndarray
    true_nelbo_matrix: np.ndarray
    global_true_nelbo_matrix: np.ndarray
    fold: FoldCandidateSet
    scalar_hard_oracle_gap_pct: np.ndarray


@dataclass(frozen=True)
class JackknifeCalibrationBlock:
    validation_domain: int
    query_domains: np.ndarray
    expert_domains: Tuple[int, ...]
    mean_win: np.ndarray
    std_win: np.ndarray
    n_models: int
    candidate_pool_consistent: bool
    true_nelbo_matrix: np.ndarray
    global_true_nelbo_matrix: np.ndarray
    fold: FoldCandidateSet
    pairprob_hard_win: np.ndarray
    pairprob_hard_selected_idx: np.ndarray
    pairprob_hard_oracle_gap_pct: np.ndarray
    metadata_oracle_gap_pct: np.ndarray | None = None


@dataclass(frozen=True)
class ConformalRegretSetSelection:
    method: str
    base_method: str
    feature_set: str
    ridge_l2: float
    alpha: float
    robust_lambda: float
    tau: float
    selected_by_inner_validation: bool
    diagnostic_only_reason: str = ""
    noop: bool = False
    conformal_calibration_n: int = 0
    conformal_quantile_k: int = 0
    conformal_quantile_clipped: int = 0
    quantile_clipped_rate: float = 0.0
    source_inner_validation_domains: int = 0
    source_inner_rows: int = 0
    source_inner_mean_oracle_gap_pct: float = float("nan")
    source_inner_worst_domain_oracle_gap_pct: float = float("nan")
    source_inner_relative_catastrophic_rate: float = float("nan")
    source_inner_absolute_high_regret_rate: float = float("nan")
    source_inner_top1: float = float("nan")
    source_inner_spearman: float = float("nan")
    mean_conformal_set_size: float = float("nan")
    set_size_gt1_rate: float = float("nan")
    set_size_gt3_rate: float = float("nan")
    oracle_in_conformal_set_rate: float = float("nan")
    primary_near_oracle_in_conformal_set_rate: float = float("nan")
    regret_set_override_rate: float = float("nan")
    regret_set_override_help_rate: float = float("nan")
    regret_set_override_harm_rate: float = float("nan")
    mean_override_delta_gap_pct: float = float("nan")
    mean_paired_gap_delta_vs_pairprob_hard: float = float("nan")
    median_paired_gap_delta_vs_pairprob_hard: float = float("nan")
    paired_improvement_rate_vs_pairprob_hard: float = float("nan")
    normalized_worst_regret_by_expert: Dict[int, float] | None = None
    mean_regret_by_expert: Dict[int, float] | None = None


@dataclass(frozen=True)
class JackknifeLCBSelection:
    method: str
    mean_method: str
    base_method: str
    feature_set: str
    ridge_l2: float
    jackknife_lambda: float
    selected_by_inner_validation: bool
    diagnostic_only_reason: str = ""
    noop: bool = False
    source_inner_validation_domains: int = 0
    source_inner_rows: int = 0
    source_inner_mean_oracle_gap_pct: float = float("nan")
    source_inner_worst_domain_oracle_gap_pct: float = float("nan")
    source_inner_relative_catastrophic_rate: float = float("nan")
    source_inner_absolute_high_regret_rate: float = float("nan")
    source_inner_top1: float = float("nan")
    source_inner_spearman: float = float("nan")
    jackknife_uncertainty_auc_for_pairprob_top1_error: float = float("nan")
    jackknife_uncertainty_auc_for_pairprob_high_regret: float = float("nan")
    uncertainty_error_spearman_source_inner: float = float("nan")
    lambda_stability_status: str = "stable_zero"
    candidate_pool_consistent: bool = True
    selected_lambda_is_zero_but_lcb_candidates_reported: bool = False
    jackknife_mean_vs_pairprob_hard_selection_change_rate: float = float("nan")
    mean_ensemble_override_rate_vs_pairprob_hard: float = float("nan")
    lcb_override_rate_vs_jackknife_mean: float = float("nan")
    lcb_override_rate_vs_pairprob_hard: float = float("nan")
    jackknife_override_help_rate: float = float("nan")
    jackknife_override_harm_rate: float = float("nan")
    total_override_help_gap_reduction: float = float("nan")
    total_override_harm_gap_increase: float = float("nan")
    mean_paired_gap_delta_vs_pairprob_hard: float = float("nan")
    paired_improvement_rate_vs_pairprob_hard: float = float("nan")


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


def top2_rerank_feature_names(*, embedding_dim: int, expert_feature_dim: int) -> Tuple[str, ...]:
    names: List[str] = []
    names.extend(f"query_embedding_{i}" for i in range(int(embedding_dim)))
    names.extend(f"top1_expert_identity_{i}" for i in range(int(expert_feature_dim)))
    names.extend(f"top2_expert_identity_{i}" for i in range(int(expert_feature_dim)))
    names.extend(f"query_by_top1_{i}_{j}" for i in range(int(embedding_dim)) for j in range(int(expert_feature_dim)))
    names.extend(f"query_by_top2_{i}_{j}" for i in range(int(embedding_dim)) for j in range(int(expert_feature_dim)))
    names.extend(f"top1_minus_top2_{i}" for i in range(int(expert_feature_dim)))
    names.extend(["base_top1_win", "base_top2_win", "top2_margin", "p_top1_beats_top2"])
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


def _top2_rerank_feature(
    row_top1: np.ndarray,
    row_top2: np.ndarray,
    *,
    embedding_dim: int,
    expert_feature_dim: int,
    base_top1_win: float,
    base_top2_win: float,
    top2_margin: float,
    p_top1_beats_top2: float,
) -> np.ndarray:
    query = np.asarray(row_top1[:embedding_dim], dtype=np.float64)
    top1 = np.asarray(row_top1[embedding_dim : embedding_dim + expert_feature_dim], dtype=np.float64)
    top2 = np.asarray(row_top2[embedding_dim : embedding_dim + expert_feature_dim], dtype=np.float64)
    interaction_top1 = (query[:, None] * top1[None, :]).reshape(-1)
    interaction_top2 = (query[:, None] * top2[None, :]).reshape(-1)
    return np.concatenate(
        [
            query,
            top1,
            top2,
            interaction_top1,
            interaction_top2,
            top1 - top2,
            np.asarray(
                [float(base_top1_win), float(base_top2_win), float(top2_margin), float(p_top1_beats_top2)],
                dtype=np.float64,
            ),
        ],
        axis=0,
    ).astype(np.float64, copy=False)


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


def fit_top2_rerank_model(
    *,
    train_data: Top2RerankTrainingData,
    ridge_l2: float,
    device: str,
) -> Top2RerankModelBundle:
    if train_data.x.shape[0] <= 0:
        raise ProtocolError("Cannot fit top-2 reranker without training rows")
    x_z, _x_unused = _zscore_features(train_data.x, train_data.x)
    mean = train_data.x.mean(axis=0)
    scale = train_data.x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    clf = _LogisticRidgePairprob(l2=float(ridge_l2), device=str(device))
    clf.fit(x_z, train_data.y, train_data.weight)
    return Top2RerankModelBundle(
        feature_set=TOP2_RERANK_FEATURE_SET,
        ridge_l2=float(ridge_l2),
        feature_mean=mean.astype(np.float64, copy=False),
        feature_scale=scale.astype(np.float64, copy=False),
        model=clf,
    )


def _concat_top2_training_data(parts: Sequence[Top2RerankTrainingData]) -> Top2RerankTrainingData:
    non_empty = [p for p in parts if p.x.shape[0] > 0]
    if not non_empty:
        return Top2RerankTrainingData(
            x=np.zeros((0, 0), dtype=np.float64),
            y=np.zeros((0,), dtype=np.float64),
            weight=np.zeros((0,), dtype=np.float64),
            query_domains=np.zeros((0,), dtype=np.int64),
            total_active_rows=int(sum(int(p.total_active_rows) for p in parts)),
            dropped_near_tie=int(sum(int(p.dropped_near_tie) for p in parts)),
            positive_rows=0,
            negative_rows=0,
            kept_by_domain={},
            switch_candidate_rate=float("nan"),
        )
    kept: Dict[int, int] = {}
    total_active = 0
    dropped = 0
    switch_rates_num = 0.0
    switch_rates_den = 0.0
    for part in parts:
        total_active += int(part.total_active_rows)
        dropped += int(part.dropped_near_tie)
        if np.isfinite(float(part.switch_candidate_rate)) and int(part.total_active_rows) > 0:
            switch_rates_num += float(part.switch_candidate_rate) * float(part.total_active_rows)
            switch_rates_den += float(part.total_active_rows)
        for domain, count in part.kept_by_domain.items():
            kept[int(domain)] = int(kept.get(int(domain), 0)) + int(count)
    y = np.concatenate([part.y for part in non_empty], axis=0)
    return Top2RerankTrainingData(
        x=np.vstack([part.x for part in non_empty]).astype(np.float64, copy=False),
        y=y.astype(np.float64, copy=False),
        weight=np.concatenate([part.weight for part in non_empty], axis=0).astype(np.float64, copy=False),
        query_domains=np.concatenate([part.query_domains for part in non_empty], axis=0).astype(np.int64, copy=False),
        total_active_rows=int(total_active),
        dropped_near_tie=int(dropped),
        positive_rows=int(np.sum(y >= 0.5)),
        negative_rows=int(np.sum(y < 0.5)),
        kept_by_domain=kept,
        switch_candidate_rate=float(switch_rates_num / switch_rates_den) if switch_rates_den > 0.0 else float("nan"),
    )


def _apply_pairprob_model(bundle: PairprobModelBundle, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    z = (arr - bundle.feature_mean) / bundle.feature_scale
    return bundle.model.predict_proba(z)


def _apply_top2_rerank_model(bundle: Top2RerankModelBundle, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[0] <= 0:
        return np.zeros((0,), dtype=np.float64)
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


def build_top2_rerank_training_data(
    *,
    x_rows: np.ndarray,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    prob_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    embedding_dim: int,
    expert_feature_dim: int,
    margin_threshold: float,
    near_tie_delta_pct: float,
    margin_weight_scale_pct: float,
    margin_weight_clip: Tuple[float, float],
) -> Top2RerankTrainingData:
    win, orders, margins = pairprob_order_and_margin(prob_matrix, expert_domains=expert_domains)
    k = int(win.shape[1])
    if k < 2:
        return Top2RerankTrainingData(
            x=np.zeros((0, 0), dtype=np.float64),
            y=np.zeros((0,), dtype=np.float64),
            weight=np.zeros((0,), dtype=np.float64),
            query_domains=np.zeros((0,), dtype=np.int64),
            total_active_rows=0,
            dropped_near_tie=0,
            positive_rows=0,
            negative_rows=0,
            kept_by_domain={},
            switch_candidate_rate=float("nan"),
        )
    if x_rows.shape[0] % k != 0:
        raise ProtocolError("Top-2 reranker feature rows are not divisible by candidate expert count")

    features: List[np.ndarray] = []
    labels: List[float] = []
    weights: List[float] = []
    domains: List[int] = []
    kept_by_domain: Dict[int, int] = {}
    total_active = 0
    dropped = 0
    switch_candidates = 0
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    low, high = float(margin_weight_clip[0]), float(margin_weight_clip[1])
    for row_idx in range(win.shape[0]):
        if float(margins[row_idx]) > float(margin_threshold):
            continue
        total_active += 1
        top1 = int(orders[row_idx, 0])
        top2 = int(orders[row_idx, 1])
        top1_nelbo = float(true[row_idx, top1])
        top2_nelbo = float(true[row_idx, top2])
        denom = max(abs(min(top1_nelbo, top2_nelbo)), 1e-12)
        delta_pct = 100.0 * abs(top1_nelbo - top2_nelbo) / denom
        if top2_nelbo < top1_nelbo:
            switch_candidates += 1
        if delta_pct < float(near_tie_delta_pct):
            dropped += 1
            continue
        base = int(row_idx * k)
        feature = _top2_rerank_feature(
            np.asarray(x_rows[base + top1], dtype=np.float64),
            np.asarray(x_rows[base + top2], dtype=np.float64),
            embedding_dim=int(embedding_dim),
            expert_feature_dim=int(expert_feature_dim),
            base_top1_win=float(win[row_idx, top1]),
            base_top2_win=float(win[row_idx, top2]),
            top2_margin=float(margins[row_idx]),
            p_top1_beats_top2=float(prob_matrix[row_idx, top1, top2]),
        )
        label = 1.0 if top1_nelbo < top2_nelbo else 0.0
        features.append(feature)
        labels.append(float(label))
        weights.append(float(np.clip(delta_pct / float(margin_weight_scale_pct), low, high)))
        domain = int(query_domains[row_idx])
        domains.append(domain)
        kept_by_domain[domain] = int(kept_by_domain.get(domain, 0)) + 1

    y = np.asarray(labels, dtype=np.float64)
    x = np.vstack(features).astype(np.float64, copy=False) if features else np.zeros((0, 0), dtype=np.float64)
    return Top2RerankTrainingData(
        x=x,
        y=y,
        weight=np.asarray(weights, dtype=np.float64),
        query_domains=np.asarray(domains, dtype=np.int64),
        total_active_rows=int(total_active),
        dropped_near_tie=int(dropped),
        positive_rows=int(np.sum(y >= 0.5)) if y.size else 0,
        negative_rows=int(np.sum(y < 0.5)) if y.size else 0,
        kept_by_domain=kept_by_domain,
        switch_candidate_rate=float(switch_candidates / total_active) if total_active > 0 else float("nan"),
    )


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


def conformal_quantile(values: Sequence[float], alpha: float) -> Tuple[float, int, int, int]:
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    n = int(vals.shape[0])
    if n <= 0:
        return 0.0, 0, 0, 1
    vals = np.sort(vals)
    k = int(np.ceil((float(n) + 1.0) * (1.0 - float(alpha))))
    clipped = int(k > n)
    k_eff = min(max(k, 1), n)
    return float(vals[k_eff - 1]), int(n), int(k), int(clipped)


def _stable_true_oracle_indices(true_nelbo_matrix: np.ndarray) -> np.ndarray:
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    out = np.zeros((true.shape[0],), dtype=np.int64)
    tie = np.arange(true.shape[1], dtype=np.int64)
    for i in range(true.shape[0]):
        out[i] = int(np.lexsort((tie, true[i, :]))[0])
    return out


def _oracle_gap_pct_matrix(true_nelbo_matrix: np.ndarray) -> np.ndarray:
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    oracle_idx = _stable_true_oracle_indices(true)
    oracle_nelbo = true[np.arange(true.shape[0]), oracle_idx]
    denom = np.maximum(np.abs(oracle_nelbo), 1e-12)
    return ((true - oracle_nelbo[:, None]) / denom[:, None]) * 100.0


def _conformal_mask(win: np.ndarray, tau: float) -> np.ndarray:
    wins = np.asarray(win, dtype=np.float64)
    top = np.max(wins, axis=1, keepdims=True)
    return (top - wins) <= (float(tau) + 1e-12)


def _near_oracle_key(threshold: float) -> str:
    value = float(threshold)
    if abs(value - round(value)) < 1e-9:
        suffix = str(int(round(value)))
    else:
        suffix = str(value).replace(".", "_")
    return f"near_oracle_in_conformal_set_gap_le_{suffix}"


def _top_indices_from_win(win: np.ndarray, expert_domains: Sequence[int]) -> np.ndarray:
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    out = np.zeros((win.shape[0],), dtype=np.int64)
    for i in range(win.shape[0]):
        out[i] = int(np.lexsort((experts, -win[i, :]))[0])
    return out


def pairprob_selected_indices(prob_matrix: np.ndarray, expert_domains: Sequence[int]) -> np.ndarray:
    return _top_indices_from_win(pairprob_win_scores(prob_matrix), expert_domains)


def _select_indices_from_conformal_set(
    *,
    win: np.ndarray,
    mask: np.ndarray,
    expert_domains: Sequence[int],
    selection: ConformalRegretSetSelection,
    true_nelbo_matrix: np.ndarray | None = None,
    oracle_mode: bool = False,
    topwin_mode: bool = False,
) -> np.ndarray:
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    top_idx = _top_indices_from_win(win, expert_domains)
    if topwin_mode or bool(selection.noop):
        return top_idx

    if oracle_mode:
        if true_nelbo_matrix is None:
            raise ProtocolError("oracle conformal diagnostic requires true_nelbo_matrix")
        true = np.asarray(true_nelbo_matrix, dtype=np.float64)
        out = np.zeros((true.shape[0],), dtype=np.int64)
        tie = np.arange(true.shape[1], dtype=np.int64)
        masked = np.where(mask, true, np.inf)
        for i in range(true.shape[0]):
            out[i] = int(np.lexsort((tie, masked[i, :]))[0])
        return out

    penalties_by_expert = selection.normalized_worst_regret_by_expert or {}
    mean_regret_by_expert = selection.mean_regret_by_expert or {}
    penalties = np.asarray([float(penalties_by_expert.get(int(e), 0.0)) for e in experts], dtype=np.float64)
    mean_regret = np.asarray([float(mean_regret_by_expert.get(int(e), 0.0)) for e in experts], dtype=np.float64)
    robust = win - (float(selection.robust_lambda) * penalties[None, :])
    robust = np.where(mask, robust, -np.inf)
    out = np.zeros((win.shape[0],), dtype=np.int64)
    for i in range(win.shape[0]):
        out[i] = int(np.lexsort((experts, mean_regret, -robust[i, :]))[0])
    return out


def _gap_pct_for_selected(true_nelbo_matrix: np.ndarray, selected_idx: np.ndarray) -> np.ndarray:
    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    oracle_idx = _stable_true_oracle_indices(true)
    oracle_nelbo = true[np.arange(true.shape[0]), oracle_idx]
    selected_nelbo = true[np.arange(true.shape[0]), np.asarray(selected_idx, dtype=np.int64)]
    return ((selected_nelbo - oracle_nelbo) / np.maximum(np.abs(oracle_nelbo), 1e-12)) * 100.0


def _conformal_set_fields_for_row(
    *,
    row_idx: int,
    mask: np.ndarray,
    expert_domains: Sequence[int],
    true_nelbo_matrix: np.ndarray,
    cfg: ConformalRegretSetConfig,
) -> Dict[str, Any]:
    experts = [int(v) for v in expert_domains]
    selected_set = [experts[j] for j, active in enumerate(mask[row_idx, :].tolist()) if bool(active)]
    oracle_idx = int(_stable_true_oracle_indices(true_nelbo_matrix[[row_idx], :])[0])
    gap_matrix = _oracle_gap_pct_matrix(true_nelbo_matrix[[row_idx], :])
    out: Dict[str, Any] = {
        "conformal_set_experts": "|".join(str(v) for v in selected_set),
        "conformal_set_size": int(len(selected_set)),
        "oracle_in_conformal_set": int(bool(mask[row_idx, oracle_idx])),
    }
    for threshold in cfg.near_oracle_gap_pct_values:
        key = _near_oracle_key(float(threshold))
        out[key] = int(bool(np.any(mask[row_idx, :] & (gap_matrix[0, :] <= float(threshold)))))
    primary_key = _near_oracle_key(float(cfg.primary_near_oracle_gap_pct))
    out["primary_near_oracle_in_conformal_set"] = int(out.get(primary_key, 0))
    return out


def conformal_pairprob_route_rows(
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
    selection: ConformalRegretSetSelection,
    cfg: ConformalRegretSetConfig,
    pairprob_baseline_gap_pct: np.ndarray | None,
    scalar_hard_oracle_gap_pct: np.ndarray | None,
    metadata_oracle_gap_pct: np.ndarray | None = None,
    topwin_diagnostic: bool = False,
    oracle_diagnostic: bool = False,
) -> List[Dict[str, Any]]:
    win, _orders, margins = pairprob_order_and_margin(prob_matrix, expert_domains=expert_domains)
    mask = _conformal_mask(win, float(selection.tau))
    top_idx = _top_indices_from_win(win, expert_domains)
    selected_idx = _select_indices_from_conformal_set(
        win=win,
        mask=mask,
        expert_domains=expert_domains,
        selection=selection,
        true_nelbo_matrix=true_nelbo_matrix,
        oracle_mode=bool(oracle_diagnostic),
        topwin_mode=bool(topwin_diagnostic),
    )
    ranking_score = -win
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
    pairprob_gap = (
        np.asarray(pairprob_baseline_gap_pct, dtype=np.float64)
        if pairprob_baseline_gap_pct is not None
        else _gap_pct_for_selected(true_nelbo_matrix, top_idx)
    )
    scalar_gap = (
        np.asarray(scalar_hard_oracle_gap_pct, dtype=np.float64)
        if scalar_hard_oracle_gap_pct is not None
        else np.full((len(rows),), float("nan"), dtype=np.float64)
    )
    metadata_gap = (
        np.asarray(metadata_oracle_gap_pct, dtype=np.float64)
        if metadata_oracle_gap_pct is not None
        else np.full((len(rows),), float("nan"), dtype=np.float64)
    )
    if pairprob_gap.shape[0] != len(rows):
        pairprob_gap = _gap_pct_for_selected(true_nelbo_matrix, top_idx)
    if scalar_gap.shape[0] != len(rows):
        scalar_gap = np.full((len(rows),), float("nan"), dtype=np.float64)
    if metadata_gap.shape[0] != len(rows):
        metadata_gap = np.full((len(rows),), float("nan"), dtype=np.float64)

    reason = str(selection.diagnostic_only_reason)
    if topwin_diagnostic:
        reason = "conformal_pairprob_topwin_set_diagnostic_v1"
    if oracle_diagnostic:
        reason = "oracle_conformal_regret_set_diagnostic"

    penalties_by_expert = selection.normalized_worst_regret_by_expert or {}
    for i, row in enumerate(rows):
        selected_col = int(selected_idx[i])
        selected_expert = int(expert_domains[selected_col])
        pair_diag = _pair_diagnostics_for_row(prob_matrix[i, :, :], true_nelbo_matrix[i, :])
        paired_delta = float(row["oracle_gap_pct"]) - float(pairprob_gap[i])
        paired_delta_metadata = (
            float(row["oracle_gap_pct"]) - float(metadata_gap[i])
            if np.isfinite(float(metadata_gap[i]))
            else float("nan")
        )
        override_active = int((not topwin_diagnostic) and (not oracle_diagnostic) and int(selected_idx[i]) != int(top_idx[i]))
        row.update(
            {
                "policy_name": str(policy_name),
                "base_method": str(selection.base_method),
                "feature_set": str(selection.feature_set),
                "selected_tau": float(selection.ridge_l2),
                "selected_by_inner_validation": int(bool(selection.selected_by_inner_validation)),
                "threshold_selection_policy": str(cfg.calibration_policy),
                "route_experts": str(selected_expert),
                "route_weights": "1",
                "route_size": 1,
                "route_mode": (
                    "oracle_conformal_regret_set_diagnostic"
                    if oracle_diagnostic
                    else "conformal_topwin_diagnostic"
                    if topwin_diagnostic
                    else "conformal_regret_set"
                ),
                "pairprob_predictor": "logistic_ridge_pairprob",
                "pairprob_probability_calibration": "none_v1",
                "pairprob_ridge_l2": float(selection.ridge_l2),
                "pairprob_feature_set": str(selection.feature_set),
                "pairprob_selection_policy": str(cfg.selection_rule),
                "pairprob_win_top1": float(win[i, int(top_idx[i])]),
                "top1_win_margin": float(margins[i]),
                "tournament_margin": float(margins[i]),
                "conformal_alpha": float(selection.alpha),
                "conformal_tau": float(selection.tau),
                "conformal_calibration_n": int(selection.conformal_calibration_n),
                "conformal_quantile_k": int(selection.conformal_quantile_k),
                "conformal_quantile_clipped": int(selection.conformal_quantile_clipped),
                "robust_lambda": float(selection.robust_lambda),
                "normalized_source_inner_worst_regret_selected": float(
                    penalties_by_expert.get(int(selected_expert), 0.0)
                ),
                "regret_set_override_active": int(override_active),
                "override_delta_gap_pct_vs_pairprob_top1": float(paired_delta) if override_active else 0.0,
                "paired_gap_delta_vs_pairprob_hard": float(paired_delta),
                "paired_gap_delta_vs_metadata": float(paired_delta_metadata),
                "absolute_high_regret_gap_gt_5": int(
                    float(row["oracle_gap_pct"]) > float(cfg.absolute_high_regret_gap_pct)
                ),
                "relative_catastrophic_regression_vs_pairprob_hard_gt_5": int(
                    float(paired_delta) > float(cfg.catastrophic_regression_vs_pairprob_hard_gap_pct)
                ),
                "relative_catastrophic_regression_vs_hard_gt_5": int(
                    np.isfinite(float(scalar_gap[i]))
                    and float(row["oracle_gap_pct"]) - float(scalar_gap[i])
                    > float(cfg.catastrophic_regression_vs_pairprob_hard_gap_pct)
                ),
                "hard_oracle_gap_pct": float(scalar_gap[i]),
                "pairprob_hard_oracle_gap_pct": float(pairprob_gap[i]),
                "metadata_oracle_gap_pct": float(metadata_gap[i]),
                "mean_conformal_set_size": float(selection.mean_conformal_set_size),
                "set_size_gt1_rate": float(selection.set_size_gt1_rate),
                "set_size_gt3_rate": float(selection.set_size_gt3_rate),
                "oracle_in_conformal_set_rate": float(selection.oracle_in_conformal_set_rate),
                "primary_near_oracle_in_conformal_set_rate": float(
                    selection.primary_near_oracle_in_conformal_set_rate
                ),
                "quantile_clipped_rate": float(selection.quantile_clipped_rate),
                "regret_set_override_rate": float(selection.regret_set_override_rate),
                "regret_set_override_help_rate": float(selection.regret_set_override_help_rate),
                "regret_set_override_harm_rate": float(selection.regret_set_override_harm_rate),
                "mean_override_delta_gap_pct": float(selection.mean_override_delta_gap_pct),
                "mean_paired_gap_delta_vs_pairprob_hard": float(
                    selection.mean_paired_gap_delta_vs_pairprob_hard
                ),
                "median_paired_gap_delta_vs_pairprob_hard": float(
                    selection.median_paired_gap_delta_vs_pairprob_hard
                ),
                "paired_improvement_rate_vs_pairprob_hard": float(
                    selection.paired_improvement_rate_vs_pairprob_hard
                ),
                "worst_inner_domain_oracle_gap_pct": float(selection.source_inner_worst_domain_oracle_gap_pct),
                "absolute_high_regret_rate_gap_gt_5": float(selection.source_inner_absolute_high_regret_rate),
                "relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate": float(
                    selection.source_inner_relative_catastrophic_rate
                ),
                "diagnostic_only_reason": str(reason),
                **_conformal_set_fields_for_row(
                    row_idx=i,
                    mask=mask,
                    expert_domains=expert_domains,
                    true_nelbo_matrix=true_nelbo_matrix,
                    cfg=cfg,
                ),
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


def summarize_conformal_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "validation_domains": 0.0,
            "mean_oracle_gap_pct": float("nan"),
            "worst_inner_domain_oracle_gap_pct": float("nan"),
            "relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate": float("nan"),
            "absolute_high_regret_rate_gap_gt_5": float("nan"),
            "top1_oracle_hit": float("nan"),
            "spearman": float("nan"),
            "mean_conformal_set_size": float("nan"),
            "set_size_gt1_rate": float("nan"),
            "set_size_gt3_rate": float("nan"),
            "oracle_in_conformal_set_rate": float("nan"),
            "primary_near_oracle_in_conformal_set_rate": float("nan"),
            "regret_set_override_rate": float("nan"),
            "regret_set_override_help_rate": float("nan"),
            "regret_set_override_harm_rate": float("nan"),
            "mean_override_delta_gap_pct": float("nan"),
            "mean_paired_gap_delta_vs_pairprob_hard": float("nan"),
            "median_paired_gap_delta_vs_pairprob_hard": float("nan"),
            "paired_improvement_rate_vs_pairprob_hard": float("nan"),
        }
    by_domain: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(int(row["query_domain"]), []).append(row)

    domain_gap = [
        float(np.mean([float(r["oracle_gap_pct"]) for r in domain_rows]))
        for domain_rows in by_domain.values()
    ]
    spearman_vals = [float(r["spearman"]) for r in rows if np.isfinite(float(r["spearman"]))]
    override_rows = [r for r in rows if int(float(r.get("regret_set_override_active", 0) or 0)) == 1]
    override_delta = [float(r.get("override_delta_gap_pct_vs_pairprob_top1", float("nan"))) for r in override_rows]
    paired_delta = [float(r.get("paired_gap_delta_vs_pairprob_hard", float("nan"))) for r in rows]
    return {
        "n_rows": float(len(rows)),
        "validation_domains": float(len(by_domain)),
        "mean_oracle_gap_pct": float(np.mean([float(r["oracle_gap_pct"]) for r in rows])),
        "worst_inner_domain_oracle_gap_pct": float(max(domain_gap)) if domain_gap else float("nan"),
        "relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate": float(
            np.mean([float(r.get("relative_catastrophic_regression_vs_pairprob_hard_gt_5", 0.0)) for r in rows])
        ),
        "absolute_high_regret_rate_gap_gt_5": float(
            np.mean([float(r.get("absolute_high_regret_gap_gt_5", 0.0)) for r in rows])
        ),
        "top1_oracle_hit": float(np.mean([float(r["top1_oracle_hit"]) for r in rows])),
        "spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "mean_conformal_set_size": float(np.mean([float(r.get("conformal_set_size", 0.0)) for r in rows])),
        "set_size_gt1_rate": float(np.mean([float(r.get("conformal_set_size", 0.0)) > 1.0 for r in rows])),
        "set_size_gt3_rate": float(np.mean([float(r.get("conformal_set_size", 0.0)) > 3.0 for r in rows])),
        "oracle_in_conformal_set_rate": float(np.mean([float(r.get("oracle_in_conformal_set", 0.0)) for r in rows])),
        "primary_near_oracle_in_conformal_set_rate": float(
            np.mean([float(r.get("primary_near_oracle_in_conformal_set", 0.0)) for r in rows])
        ),
        "regret_set_override_rate": float(np.mean([float(r.get("regret_set_override_active", 0.0)) for r in rows])),
        "regret_set_override_help_rate": float(
            np.mean([1.0 if float(v) < 0.0 else 0.0 for v in override_delta])
        )
        if override_delta
        else float("nan"),
        "regret_set_override_harm_rate": float(
            np.mean([1.0 if float(v) > 0.0 else 0.0 for v in override_delta])
        )
        if override_delta
        else float("nan"),
        "mean_override_delta_gap_pct": float(np.mean(override_delta)) if override_delta else float("nan"),
        "mean_paired_gap_delta_vs_pairprob_hard": float(np.mean(paired_delta)) if paired_delta else float("nan"),
        "median_paired_gap_delta_vs_pairprob_hard": float(np.median(paired_delta)) if paired_delta else float("nan"),
        "paired_improvement_rate_vs_pairprob_hard": float(
            np.mean([1.0 if float(v) < 0.0 else 0.0 for v in paired_delta])
        )
        if paired_delta
        else float("nan"),
    }


def _source_inner_regret_penalties(
    *,
    blocks: Sequence[ConformalCalibrationBlock],
    outer_candidate_experts: Sequence[int],
    min_rows_per_expert: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    regret_by_expert_domain: Dict[int, Dict[int, List[float]]] = {
        int(e): {} for e in outer_candidate_experts
    }
    count_by_expert: Dict[int, int] = {int(e): 0 for e in outer_candidate_experts}
    for block in blocks:
        gap_matrix = _oracle_gap_pct_matrix(block.true_nelbo_matrix)
        for col, expert in enumerate(block.expert_domains):
            expert_int = int(expert)
            regret_by_expert_domain.setdefault(expert_int, {}).setdefault(
                int(block.validation_domain),
                [],
            ).extend(float(v) for v in gap_matrix[:, col].tolist())
            count_by_expert[expert_int] = int(count_by_expert.get(expert_int, 0)) + int(gap_matrix.shape[0])

    raw_worst: Dict[int, float] = {}
    raw_mean: Dict[int, float] = {}
    for expert in [int(v) for v in outer_candidate_experts]:
        domain_vals = regret_by_expert_domain.get(expert, {})
        all_vals = [float(v) for vals in domain_vals.values() for v in vals]
        if int(count_by_expert.get(expert, 0)) < int(min_rows_per_expert) or not all_vals:
            raw_worst[expert] = float("nan")
            raw_mean[expert] = float("nan")
            continue
        raw_worst[expert] = float(max(np.mean(vals) for vals in domain_vals.values() if vals))
        raw_mean[expert] = float(np.mean(all_vals))

    finite_worst = [float(v) for v in raw_worst.values() if np.isfinite(float(v))]
    max_penalty = float(max(finite_worst)) if finite_worst else 0.0
    for expert in raw_worst:
        if not np.isfinite(float(raw_worst[expert])):
            raw_worst[expert] = max_penalty
        if not np.isfinite(float(raw_mean[expert])):
            raw_mean[expert] = max_penalty

    vals = np.asarray([float(raw_worst[int(e)]) for e in outer_candidate_experts], dtype=np.float64)
    if vals.size == 0 or float(np.max(vals) - np.min(vals)) < 1e-12:
        normalized = np.zeros_like(vals)
    else:
        normalized = (vals - float(np.min(vals))) / float(np.max(vals) - np.min(vals))
    return (
        {int(e): float(v) for e, v in zip(outer_candidate_experts, normalized.tolist())},
        {int(e): float(raw_mean[int(e)]) for e in outer_candidate_experts},
    )


def _baseline_gap_for_block(block: ConformalCalibrationBlock) -> np.ndarray:
    win = pairprob_win_scores(block.prob_matrix)
    top_idx = _top_indices_from_win(win, block.expert_domains)
    return _gap_pct_for_selected(block.true_nelbo_matrix, top_idx)


def _conformal_calibration_nonconformity(
    blocks: Sequence[ConformalCalibrationBlock],
) -> List[float]:
    values: List[float] = []
    for block in blocks:
        win = pairprob_win_scores(block.prob_matrix)
        top = np.max(win, axis=1)
        oracle_idx = _stable_true_oracle_indices(block.true_nelbo_matrix)
        for row_idx, oracle_col in enumerate(oracle_idx.tolist()):
            values.append(float(top[row_idx] - win[row_idx, int(oracle_col)]))
    return values


def _conformal_selection_reason(
    *,
    summary: Mapping[str, float],
    quantile_clipped_rate: float,
    cfg: ConformalRegretSetConfig,
    no_valid: bool,
) -> str:
    if no_valid:
        return "no_valid_alpha_lambda_candidate"
    if float(summary.get("mean_conformal_set_size", float("inf"))) > float(cfg.max_mean_set_size):
        return "excessive_set_size"
    if float(summary.get("set_size_gt3_rate", float("inf"))) > float(cfg.max_set_size_gt3_rate):
        return "excessive_set_size"
    if float(summary.get("oracle_in_conformal_set_rate", 0.0)) < float(cfg.min_oracle_in_set_rate):
        return "low_oracle_in_set_rate"
    if float(quantile_clipped_rate) > float(cfg.max_quantile_clipped_fold_rate):
        return "quantile_clipping_high"
    if float(summary.get("mean_paired_gap_delta_vs_pairprob_hard", 0.0)) > 0.0:
        return "worsens_pairprob_baseline"
    if float(summary.get("relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate", 0.0)) > 0.0:
        return "catastrophic_regression_persists"
    return ""


def select_conformal_regret_set_policy(
    *,
    blocks: Sequence[ConformalCalibrationBlock],
    base_selection: PairprobPolicySelection | None,
    outer_candidate_experts: Sequence[int],
    global_expert_domains: Sequence[int],
    cfg: ConformalRegretSetConfig,
) -> ConformalRegretSetSelection | None:
    if not bool(cfg.enabled):
        return None
    if base_selection is None:
        return ConformalRegretSetSelection(
            method=cfg.method_name,
            base_method=cfg.base_method,
            feature_set=cfg.feature_set,
            ridge_l2=float("nan"),
            alpha=float(cfg.alpha_values[0]),
            robust_lambda=0.0,
            tau=0.0,
            selected_by_inner_validation=False,
            diagnostic_only_reason="source_inner_evidence_insufficient",
            noop=True,
        )
    if not blocks:
        return ConformalRegretSetSelection(
            method=cfg.method_name,
            base_method=cfg.base_method,
            feature_set=base_selection.feature_set,
            ridge_l2=base_selection.ridge_l2,
            alpha=float(cfg.alpha_values[0]),
            robust_lambda=0.0,
            tau=0.0,
            selected_by_inner_validation=False,
            diagnostic_only_reason="source_inner_evidence_insufficient",
            noop=True,
        )

    penalties, mean_regret = _source_inner_regret_penalties(
        blocks=blocks,
        outer_candidate_experts=outer_candidate_experts,
        min_rows_per_expert=int(cfg.min_source_inner_regret_rows_per_expert),
    )
    nonconformity = _conformal_calibration_nonconformity(blocks)
    candidates: List[Tuple[Tuple[float, ...], float, float, float, int, int, Dict[str, float], List[Dict[str, Any]]]] = []
    invalid_candidates: List[Tuple[Tuple[float, ...], float, float, float, int, int, Dict[str, float], List[Dict[str, Any]]]] = []
    for alpha in cfg.alpha_values:
        tau, n, k, clipped = conformal_quantile(nonconformity, float(alpha))
        quantile_clipped_rate = float(clipped)
        for robust_lambda in cfg.robust_lambda_values:
            selection = ConformalRegretSetSelection(
                method=cfg.method_name,
                base_method=cfg.base_method,
                feature_set=base_selection.feature_set,
                ridge_l2=base_selection.ridge_l2,
                alpha=float(alpha),
                robust_lambda=float(robust_lambda),
                tau=float(tau),
                selected_by_inner_validation=True,
                conformal_calibration_n=int(n),
                conformal_quantile_k=int(k),
                conformal_quantile_clipped=int(clipped),
                quantile_clipped_rate=float(quantile_clipped_rate),
                normalized_worst_regret_by_expert=penalties,
                mean_regret_by_expert=mean_regret,
            )
            rows: List[Dict[str, Any]] = []
            for block in blocks:
                rows.extend(
                    conformal_pairprob_route_rows(
                        method=cfg.method_name,
                        fold=block.fold,
                        query_domains=block.query_domains,
                        expert_domains=block.expert_domains,
                        prob_matrix=block.prob_matrix,
                        true_nelbo_matrix=block.true_nelbo_matrix,
                        global_true_nelbo_matrix=block.global_true_nelbo_matrix,
                        global_expert_domains=global_expert_domains,
                        policy_name=cfg.method_name,
                        selection=selection,
                        cfg=cfg,
                        pairprob_baseline_gap_pct=_baseline_gap_for_block(block),
                        scalar_hard_oracle_gap_pct=block.scalar_hard_oracle_gap_pct,
                    )
                )
            summary = summarize_conformal_rows(rows)
            score = (
                -float(summary["worst_inner_domain_oracle_gap_pct"]),
                -float(summary["relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate"]),
                -float(summary["mean_oracle_gap_pct"]),
                -float(summary["mean_conformal_set_size"]),
                float(summary["top1_oracle_hit"]),
                float(summary["spearman"]) if np.isfinite(float(summary["spearman"])) else -1e9,
                -float(robust_lambda),
                -float(alpha),
            )
            item = (score, float(alpha), float(robust_lambda), float(tau), int(k), int(clipped), summary, rows)
            valid = (
                float(summary["mean_conformal_set_size"]) <= float(cfg.max_mean_set_size)
                and float(summary["set_size_gt3_rate"]) <= float(cfg.max_set_size_gt3_rate)
                and float(summary["oracle_in_conformal_set_rate"]) >= float(cfg.min_oracle_in_set_rate)
            )
            (candidates if valid else invalid_candidates).append(item)

    source_domains = sorted({int(block.validation_domain) for block in blocks})
    no_valid = not candidates
    pool = candidates if candidates else invalid_candidates
    if not pool:
        return ConformalRegretSetSelection(
            method=cfg.method_name,
            base_method=cfg.base_method,
            feature_set=base_selection.feature_set,
            ridge_l2=base_selection.ridge_l2,
            alpha=float(cfg.alpha_values[0]),
            robust_lambda=0.0,
            tau=0.0,
            selected_by_inner_validation=False,
            diagnostic_only_reason="source_inner_evidence_insufficient",
            noop=True,
            normalized_worst_regret_by_expert=penalties,
            mean_regret_by_expert=mean_regret,
        )
    _score, alpha, robust_lambda, tau, k, clipped, summary, _rows = sorted(
        pool,
        key=lambda item: item[0],
        reverse=True,
    )[0]
    reason = _conformal_selection_reason(
        summary=summary,
        quantile_clipped_rate=float(clipped),
        cfg=cfg,
        no_valid=bool(no_valid),
    )
    if str(base_selection.diagnostic_only_reason):
        reason = "|".join(
            part
            for part in dict.fromkeys([str(base_selection.diagnostic_only_reason), str(reason)])
            if part
        )
    return ConformalRegretSetSelection(
        method=cfg.method_name,
        base_method=cfg.base_method,
        feature_set=base_selection.feature_set,
        ridge_l2=base_selection.ridge_l2,
        alpha=float(alpha),
        robust_lambda=float(robust_lambda),
        tau=float(tau),
        selected_by_inner_validation=True,
        diagnostic_only_reason=str(reason),
        noop=bool(no_valid),
        conformal_calibration_n=len(nonconformity),
        conformal_quantile_k=int(k),
        conformal_quantile_clipped=int(clipped),
        quantile_clipped_rate=float(clipped),
        source_inner_validation_domains=len(source_domains),
        source_inner_rows=int(summary.get("n_rows", 0.0)),
        source_inner_mean_oracle_gap_pct=float(summary["mean_oracle_gap_pct"]),
        source_inner_worst_domain_oracle_gap_pct=float(summary["worst_inner_domain_oracle_gap_pct"]),
        source_inner_relative_catastrophic_rate=float(
            summary["relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate"]
        ),
        source_inner_absolute_high_regret_rate=float(summary["absolute_high_regret_rate_gap_gt_5"]),
        source_inner_top1=float(summary["top1_oracle_hit"]),
        source_inner_spearman=float(summary["spearman"]),
        mean_conformal_set_size=float(summary["mean_conformal_set_size"]),
        set_size_gt1_rate=float(summary["set_size_gt1_rate"]),
        set_size_gt3_rate=float(summary["set_size_gt3_rate"]),
        oracle_in_conformal_set_rate=float(summary["oracle_in_conformal_set_rate"]),
        primary_near_oracle_in_conformal_set_rate=float(
            summary["primary_near_oracle_in_conformal_set_rate"]
        ),
        regret_set_override_rate=float(summary["regret_set_override_rate"]),
        regret_set_override_help_rate=float(summary["regret_set_override_help_rate"]),
        regret_set_override_harm_rate=float(summary["regret_set_override_harm_rate"]),
        mean_override_delta_gap_pct=float(summary["mean_override_delta_gap_pct"]),
        mean_paired_gap_delta_vs_pairprob_hard=float(summary["mean_paired_gap_delta_vs_pairprob_hard"]),
        median_paired_gap_delta_vs_pairprob_hard=float(summary["median_paired_gap_delta_vs_pairprob_hard"]),
        paired_improvement_rate_vs_pairprob_hard=float(summary["paired_improvement_rate_vs_pairprob_hard"]),
        normalized_worst_regret_by_expert=penalties,
        mean_regret_by_expert=mean_regret,
    )


def _stable_argmax_indices(score: np.ndarray, expert_domains: Sequence[int]) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float64)
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    out = np.zeros((arr.shape[0],), dtype=np.int64)
    for i in range(arr.shape[0]):
        out[i] = int(np.lexsort((experts, -arr[i, :]))[0])
    return out


def _rank_of_indices_desc(score: np.ndarray, selected_idx: np.ndarray, expert_domains: Sequence[int]) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float64)
    experts = np.asarray([int(v) for v in expert_domains], dtype=np.int64)
    selected = np.asarray(selected_idx, dtype=np.int64)
    ranks = np.full((arr.shape[0],), float("nan"), dtype=np.float64)
    for i in range(arr.shape[0]):
        order = np.lexsort((experts, -arr[i, :]))
        inv = np.empty((arr.shape[1],), dtype=np.int64)
        inv[order] = np.arange(1, arr.shape[1] + 1, dtype=np.int64)
        ranks[i] = float(inv[int(selected[i])])
    return ranks


def _finite_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if np.isfinite(float(a)) and np.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if max(xs) - min(xs) < 1e-12 or max(ys) - min(ys) < 1e-12:
        return float("nan")
    return float(spearman_corr(xs, ys))


def jackknife_pairprob_route_rows(
    *,
    method: str,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    mean_win: np.ndarray,
    std_win: np.ndarray,
    n_models: int,
    candidate_pool_consistent: bool,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    policy_name: str,
    selection: JackknifeLCBSelection,
    pairprob_hard_win: np.ndarray,
    pairprob_hard_selected_idx: np.ndarray,
    pairprob_hard_oracle_gap_pct: np.ndarray,
    metadata_oracle_gap_pct: np.ndarray | None,
    cfg: JackknifeLCBTournamentConfig,
    force_lambda: float | None = None,
) -> List[Dict[str, Any]]:
    experts = [int(v) for v in expert_domains]
    mean = np.asarray(mean_win, dtype=np.float64)
    std = np.asarray(std_win, dtype=np.float64)
    if mean.shape != std.shape:
        raise ProtocolError("Jackknife mean/std win matrices must have matching shapes")
    lam = float(selection.jackknife_lambda if force_lambda is None else force_lambda)
    lcb = mean - (lam * std)
    selected_idx = _stable_argmax_indices(lcb, experts)
    mean_idx = _stable_argmax_indices(mean, experts)
    hard_idx = np.asarray(pairprob_hard_selected_idx, dtype=np.int64)
    hard_win = np.asarray(pairprob_hard_win, dtype=np.float64)
    if hard_idx.shape != selected_idx.shape:
        raise ProtocolError("Pairprob hard selected index shape mismatch for jackknife routing")
    if str(method) == str(selection.method) and (bool(selection.noop) or not bool(candidate_pool_consistent)):
        selected_idx = hard_idx.astype(np.int64, copy=True)

    ranking_score = -lcb
    _metrics, rows = _selection_metrics(
        method=method,
        query_domains=query_domains,
        expert_domains=experts,
        score_matrix=ranking_score,
        true_nelbo_matrix=true_nelbo_matrix,
        fold=fold,
        global_true_nelbo_matrix=global_true_nelbo_matrix,
        global_expert_domains=global_expert_domains,
        selected_idx_override=selected_idx,
        ranking_score_matrix=ranking_score,
    )
    hard_gap = np.asarray(pairprob_hard_oracle_gap_pct, dtype=np.float64)
    if hard_gap.shape[0] != len(rows):
        hard_gap = _gap_pct_for_selected(true_nelbo_matrix, hard_idx)
    metadata_gap = (
        np.asarray(metadata_oracle_gap_pct, dtype=np.float64)
        if metadata_oracle_gap_pct is not None
        else np.full((len(rows),), float("nan"), dtype=np.float64)
    )
    if metadata_gap.shape[0] != len(rows):
        metadata_gap = np.full((len(rows),), float("nan"), dtype=np.float64)

    mean_order = np.zeros_like(mean, dtype=np.int64)
    lcb_order = np.zeros_like(lcb, dtype=np.int64)
    for i in range(mean.shape[0]):
        mean_order[i, :] = np.lexsort((np.asarray(experts, dtype=np.int64), -mean[i, :]))
        lcb_order[i, :] = np.lexsort((np.asarray(experts, dtype=np.int64), -lcb[i, :]))
    mean_margin = (
        mean[np.arange(mean.shape[0]), mean_order[:, 0]] - mean[np.arange(mean.shape[0]), mean_order[:, 1]]
        if mean.shape[1] > 1
        else np.full((mean.shape[0],), float("inf"), dtype=np.float64)
    )
    lcb_margin = (
        lcb[np.arange(lcb.shape[0]), lcb_order[:, 0]] - lcb[np.arange(lcb.shape[0]), lcb_order[:, 1]]
        if lcb.shape[1] > 1
        else np.full((lcb.shape[0],), float("inf"), dtype=np.float64)
    )
    std_winner_minus_runnerup = (
        std[np.arange(std.shape[0]), mean_order[:, 0]] - std[np.arange(std.shape[0]), mean_order[:, 1]]
        if std.shape[1] > 1
        else np.full((std.shape[0],), float("nan"), dtype=np.float64)
    )
    std_rank = _rank_of_indices_desc(std, selected_idx, experts)
    hard_rank_of_selected = _rank_of_indices_desc(hard_win, selected_idx, experts)
    lcb_rank_of_hard = _rank_of_indices_desc(lcb, hard_idx, experts)
    oracle_idx = _stable_true_oracle_indices(true_nelbo_matrix)

    base_reason = str(selection.diagnostic_only_reason)
    if str(method) == str(selection.mean_method):
        base_reason = "|".join(
            part for part in dict.fromkeys([base_reason, "jackknife_mean_ensemble_diagnostic"]) if part
        )

    for i, row in enumerate(rows):
        selected_col = int(selected_idx[i])
        selected_expert = int(experts[selected_col])
        paired_delta = float(row["oracle_gap_pct"]) - float(hard_gap[i])
        paired_delta_metadata = (
            float(row["oracle_gap_pct"]) - float(metadata_gap[i])
            if np.isfinite(float(metadata_gap[i]))
            else float("nan")
        )
        override_vs_hard = int(int(selected_idx[i]) != int(hard_idx[i]))
        override_vs_mean = int(int(selected_idx[i]) != int(mean_idx[i]))
        mean_override_vs_hard = int(int(mean_idx[i]) != int(hard_idx[i]))
        hard_gap_i = float(hard_gap[i])
        row.update(
            {
                "policy_name": str(policy_name),
                "base_method": str(selection.base_method),
                "feature_set": str(selection.feature_set),
                "selected_tau": float(selection.ridge_l2),
                "selected_by_inner_validation": int(bool(selection.selected_by_inner_validation)),
                "threshold_selection_policy": str(cfg.calibration_policy),
                "route_experts": str(selected_expert),
                "route_weights": "1",
                "route_size": 1,
                "route_mode": "jackknife_lcb_top1" if str(method) == str(selection.method) else "jackknife_mean_top1",
                "pairprob_predictor": "logistic_ridge_pairprob",
                "pairprob_probability_calibration": "none_v1",
                "pairprob_ridge_l2": float(selection.ridge_l2),
                "pairprob_feature_set": str(selection.feature_set),
                "pairprob_selection_policy": str(cfg.calibration_policy),
                "adoption_feature_family": str(cfg.adoption_feature_family),
                "candidate_pool_consistent": int(bool(candidate_pool_consistent)),
                "selected_lambda_is_zero_but_lcb_candidates_reported": int(
                    bool(selection.selected_lambda_is_zero_but_lcb_candidates_reported)
                ),
                "lambda_stability_status": str(selection.lambda_stability_status),
                "jackknife_lambda": float(lam),
                "jackknife_n_models": int(n_models),
                "jackknife_mean_win_selected": float(mean[i, selected_col]),
                "jackknife_std_win_selected": float(std[i, selected_col]),
                "jackknife_std_pairprob_hard_selected": float(std[i, int(hard_idx[i])]),
                "jackknife_std_selected_rank": float(std_rank[i]),
                "jackknife_mean_win_margin_top1_top2": float(mean_margin[i]),
                "jackknife_std_winner_minus_runnerup": float(std_winner_minus_runnerup[i]),
                "jackknife_lcb_margin_top1_top2": float(lcb_margin[i]),
                "jackknife_override_active": int(override_vs_hard),
                "override_from_pairprob_hard_expert": int(experts[int(hard_idx[i])]) if override_vs_hard else "",
                "override_to_lcb_expert": int(selected_expert) if override_vs_hard else "",
                "pairprob_hard_rank_of_lcb_expert": float(hard_rank_of_selected[i]),
                "lcb_rank_of_pairprob_hard_expert": float(lcb_rank_of_hard[i]),
                "jackknife_mean_vs_pairprob_hard_selection_change": int(mean_override_vs_hard),
                "mean_ensemble_override_vs_pairprob_hard": int(mean_override_vs_hard),
                "lcb_override_vs_jackknife_mean": int(override_vs_mean),
                "lcb_override_vs_pairprob_hard": int(override_vs_hard),
                "paired_gap_delta_vs_pairprob_hard": float(paired_delta),
                "paired_gap_delta_vs_metadata": float(paired_delta_metadata),
                "pairprob_hard_oracle_gap_pct": float(hard_gap_i),
                "metadata_oracle_gap_pct": float(metadata_gap[i]),
                "pairprob_top1_error": int(int(hard_idx[i]) != int(oracle_idx[i])),
                "pairprob_high_regret_error": int(hard_gap_i > float(cfg.absolute_high_regret_gap_pct)),
                "absolute_high_regret_gap_gt_5": int(
                    float(row["oracle_gap_pct"]) > float(cfg.absolute_high_regret_gap_pct)
                ),
                "relative_catastrophic_regression_vs_pairprob_hard_gt_5": int(
                    float(paired_delta) > float(cfg.catastrophic_regression_vs_pairprob_hard_gap_pct)
                ),
                "diagnostic_only_reason": str(base_reason),
                "jackknife_uncertainty_auc_for_pairprob_top1_error": float(
                    selection.jackknife_uncertainty_auc_for_pairprob_top1_error
                ),
                "jackknife_uncertainty_auc_for_pairprob_high_regret": float(
                    selection.jackknife_uncertainty_auc_for_pairprob_high_regret
                ),
                "uncertainty_error_spearman_source_inner": float(
                    selection.uncertainty_error_spearman_source_inner
                ),
            }
        )
        if base_reason:
            row.update(
                {
                    "method_role": "diagnostic",
                    "adoption_eligible": 0,
                    "diagnostic_only": 1,
                }
            )
    return rows


def summarize_jackknife_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "validation_domains": 0.0,
            "mean_oracle_gap_pct": float("nan"),
            "worst_inner_domain_oracle_gap_pct": float("nan"),
            "relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate": float("nan"),
            "absolute_high_regret_rate_gap_gt_5": float("nan"),
            "top1_oracle_hit": float("nan"),
            "spearman": float("nan"),
            "jackknife_uncertainty_auc_for_pairprob_top1_error": float("nan"),
            "jackknife_uncertainty_auc_for_pairprob_high_regret": float("nan"),
            "uncertainty_error_spearman_source_inner": float("nan"),
            "jackknife_override_rate": float("nan"),
            "jackknife_override_help_rate": float("nan"),
            "jackknife_override_harm_rate": float("nan"),
            "total_override_help_gap_reduction": float("nan"),
            "total_override_harm_gap_increase": float("nan"),
            "mean_paired_gap_delta_vs_pairprob_hard": float("nan"),
            "paired_improvement_rate_vs_pairprob_hard": float("nan"),
            "jackknife_mean_vs_pairprob_hard_selection_change_rate": float("nan"),
            "mean_ensemble_override_rate_vs_pairprob_hard": float("nan"),
            "lcb_override_rate_vs_jackknife_mean": float("nan"),
            "lcb_override_rate_vs_pairprob_hard": float("nan"),
            "candidate_pool_consistent": 0.0,
        }
    by_domain: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(int(row["query_domain"]), []).append(row)
    domain_gap = [
        float(np.mean([float(r["oracle_gap_pct"]) for r in domain_rows]))
        for domain_rows in by_domain.values()
    ]
    spearman_vals = [float(r["spearman"]) for r in rows if np.isfinite(float(r["spearman"]))]
    paired_delta = [float(r.get("paired_gap_delta_vs_pairprob_hard", float("nan"))) for r in rows]
    override_rows = [r for r in rows if int(float(r.get("jackknife_override_active", 0) or 0)) == 1]
    override_delta = [float(r.get("paired_gap_delta_vs_pairprob_hard", float("nan"))) for r in override_rows]
    uncertainty = [float(r.get("jackknife_std_pairprob_hard_selected", float("nan"))) for r in rows]
    top1_error = [int(float(r.get("pairprob_top1_error", 0) or 0)) for r in rows]
    high_regret = [int(float(r.get("pairprob_high_regret_error", 0) or 0)) for r in rows]
    hard_gap = [float(r.get("pairprob_hard_oracle_gap_pct", float("nan"))) for r in rows]
    help_delta = [abs(float(v)) for v in override_delta if np.isfinite(float(v)) and float(v) < 0.0]
    harm_delta = [float(v) for v in override_delta if np.isfinite(float(v)) and float(v) > 0.0]
    return {
        "n_rows": float(len(rows)),
        "validation_domains": float(len(by_domain)),
        "mean_oracle_gap_pct": float(np.mean([float(r["oracle_gap_pct"]) for r in rows])),
        "worst_inner_domain_oracle_gap_pct": float(max(domain_gap)) if domain_gap else float("nan"),
        "relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate": float(
            np.mean([float(r.get("relative_catastrophic_regression_vs_pairprob_hard_gt_5", 0.0)) for r in rows])
        ),
        "absolute_high_regret_rate_gap_gt_5": float(
            np.mean([float(r.get("absolute_high_regret_gap_gt_5", 0.0)) for r in rows])
        ),
        "top1_oracle_hit": float(np.mean([float(r["top1_oracle_hit"]) for r in rows])),
        "spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "jackknife_uncertainty_auc_for_pairprob_top1_error": _binary_auc(uncertainty, top1_error),
        "jackknife_uncertainty_auc_for_pairprob_high_regret": _binary_auc(uncertainty, high_regret),
        "uncertainty_error_spearman_source_inner": _finite_spearman(uncertainty, hard_gap),
        "jackknife_override_rate": float(np.mean([float(r.get("jackknife_override_active", 0.0)) for r in rows])),
        "jackknife_override_help_rate": float(np.mean([1.0 if float(v) < 0.0 else 0.0 for v in override_delta]))
        if override_delta
        else float("nan"),
        "jackknife_override_harm_rate": float(np.mean([1.0 if float(v) > 0.0 else 0.0 for v in override_delta]))
        if override_delta
        else float("nan"),
        "total_override_help_gap_reduction": float(np.sum(help_delta)) if help_delta else 0.0,
        "total_override_harm_gap_increase": float(np.sum(harm_delta)) if harm_delta else 0.0,
        "mean_paired_gap_delta_vs_pairprob_hard": float(np.mean(paired_delta)) if paired_delta else float("nan"),
        "paired_improvement_rate_vs_pairprob_hard": float(
            np.mean([1.0 if float(v) < 0.0 else 0.0 for v in paired_delta])
        )
        if paired_delta
        else float("nan"),
        "jackknife_mean_vs_pairprob_hard_selection_change_rate": float(
            np.mean([float(r.get("jackknife_mean_vs_pairprob_hard_selection_change", 0.0)) for r in rows])
        ),
        "mean_ensemble_override_rate_vs_pairprob_hard": float(
            np.mean([float(r.get("mean_ensemble_override_vs_pairprob_hard", 0.0)) for r in rows])
        ),
        "lcb_override_rate_vs_jackknife_mean": float(
            np.mean([float(r.get("lcb_override_vs_jackknife_mean", 0.0)) for r in rows])
        ),
        "lcb_override_rate_vs_pairprob_hard": float(
            np.mean([float(r.get("lcb_override_vs_pairprob_hard", 0.0)) for r in rows])
        ),
        "candidate_pool_consistent": float(
            min(int(float(r.get("candidate_pool_consistent", 0) or 0)) for r in rows)
        ),
    }


def _lambda_domain_preferences(
    rows_by_lambda: Mapping[float, Sequence[Mapping[str, Any]]],
) -> Dict[int, float]:
    domains = sorted({int(row["query_domain"]) for rows in rows_by_lambda.values() for row in rows})
    out: Dict[int, float] = {}
    for domain in domains:
        scored: List[Tuple[float, float]] = []
        for lam, rows in rows_by_lambda.items():
            vals = [float(r["oracle_gap_pct"]) for r in rows if int(r["query_domain"]) == int(domain)]
            if vals:
                scored.append((float(np.mean(vals)), float(lam)))
        if scored:
            out[int(domain)] = sorted(scored, key=lambda item: (item[0], item[1]))[0][1]
    return out


def select_jackknife_lcb_policy(
    *,
    blocks: Sequence[JackknifeCalibrationBlock],
    base_selection: PairprobPolicySelection | None,
    global_expert_domains: Sequence[int],
    cfg: JackknifeLCBTournamentConfig,
) -> JackknifeLCBSelection | None:
    if not bool(cfg.enabled):
        return None
    if base_selection is None or not blocks:
        return JackknifeLCBSelection(
            method=cfg.method_name,
            mean_method=cfg.mean_method_name,
            base_method=cfg.base_method,
            feature_set=cfg.adoption_feature_family,
            ridge_l2=float("nan"),
            jackknife_lambda=0.0,
            selected_by_inner_validation=False,
            diagnostic_only_reason="source_inner_evidence_insufficient",
            noop=True,
            lambda_stability_status="forced_zero_uncertainty_failed",
            candidate_pool_consistent=False,
            selected_lambda_is_zero_but_lcb_candidates_reported=True,
        )

    if len({int(block.validation_domain) for block in blocks}) < int(cfg.min_source_inner_validation_domains):
        evidence_reason = "source_inner_evidence_insufficient"
    elif any(int(block.n_models) < int(cfg.min_jackknife_models) for block in blocks):
        evidence_reason = "source_inner_evidence_insufficient"
    elif not all(bool(block.candidate_pool_consistent) for block in blocks):
        evidence_reason = "candidate_pool_inconsistent"
    else:
        evidence_reason = ""

    rows_by_lambda: Dict[float, List[Dict[str, Any]]] = {}
    for lam in cfg.lambda_values:
        rows: List[Dict[str, Any]] = []
        selection = JackknifeLCBSelection(
            method=cfg.method_name,
            mean_method=cfg.mean_method_name,
            base_method=cfg.base_method,
            feature_set=base_selection.feature_set,
            ridge_l2=base_selection.ridge_l2,
            jackknife_lambda=float(lam),
            selected_by_inner_validation=True,
            candidate_pool_consistent=not bool(evidence_reason),
        )
        for block in blocks:
            rows.extend(
                jackknife_pairprob_route_rows(
                    method=cfg.method_name,
                    fold=block.fold,
                    query_domains=block.query_domains,
                    expert_domains=block.expert_domains,
                    mean_win=block.mean_win,
                    std_win=block.std_win,
                    n_models=block.n_models,
                    candidate_pool_consistent=block.candidate_pool_consistent,
                    true_nelbo_matrix=block.true_nelbo_matrix,
                    global_true_nelbo_matrix=block.global_true_nelbo_matrix,
                    global_expert_domains=global_expert_domains,
                    policy_name=cfg.method_name,
                    selection=selection,
                    pairprob_hard_win=block.pairprob_hard_win,
                    pairprob_hard_selected_idx=block.pairprob_hard_selected_idx,
                    pairprob_hard_oracle_gap_pct=block.pairprob_hard_oracle_gap_pct,
                    metadata_oracle_gap_pct=block.metadata_oracle_gap_pct,
                    cfg=cfg,
                )
            )
        rows_by_lambda[float(lam)] = rows

    zero_rows = rows_by_lambda.get(0.0, next(iter(rows_by_lambda.values()), []))
    zero_summary = summarize_jackknife_rows(zero_rows)
    auc_high = float(zero_summary["jackknife_uncertainty_auc_for_pairprob_high_regret"])
    spearman_uncertainty = float(zero_summary["uncertainty_error_spearman_source_inner"])
    allow_positive = (
        np.isfinite(auc_high)
        and auc_high >= float(cfg.allow_lcb_penalty_auc_min)
    ) or (
        np.isfinite(spearman_uncertainty)
        and spearman_uncertainty >= float(cfg.allow_lcb_penalty_spearman_min)
    )
    allowed_lambdas = [float(v) for v in cfg.lambda_values if float(v) == 0.0 or allow_positive]
    candidates: List[Tuple[Tuple[float, ...], float, Dict[str, float]]] = []
    for lam in allowed_lambdas:
        summary = summarize_jackknife_rows(rows_by_lambda.get(float(lam), []))
        if int(summary.get("n_rows", 0.0)) <= 0:
            continue
        candidates.append(
            (
                (
                    -float(summary["worst_inner_domain_oracle_gap_pct"]),
                    -float(summary["relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate"]),
                    -float(summary["mean_oracle_gap_pct"]),
                    float(summary["top1_oracle_hit"]),
                    float(summary["spearman"]) if np.isfinite(float(summary["spearman"])) else -1e9,
                    -float(summary["jackknife_override_rate"]),
                    -float(lam),
                ),
                float(lam),
                summary,
            )
        )
    if not candidates:
        return JackknifeLCBSelection(
            method=cfg.method_name,
            mean_method=cfg.mean_method_name,
            base_method=cfg.base_method,
            feature_set=base_selection.feature_set,
            ridge_l2=base_selection.ridge_l2,
            jackknife_lambda=0.0,
            selected_by_inner_validation=False,
            diagnostic_only_reason="source_inner_evidence_insufficient",
            noop=True,
            lambda_stability_status="forced_zero_uncertainty_failed",
            candidate_pool_consistent=False,
            selected_lambda_is_zero_but_lcb_candidates_reported=True,
        )

    _score, selected_lambda, selected_summary = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    domain_pref = _lambda_domain_preferences(rows_by_lambda)
    lambda_status = "stable_zero" if abs(float(selected_lambda)) < 1e-12 else "stable_nonzero"
    reason_parts = [evidence_reason]
    if not allow_positive and any(float(v) > 0.0 for v in cfg.lambda_values):
        if float(selected_lambda) > 0.0:
            selected_lambda = 0.0
            selected_summary = zero_summary
        lambda_status = "forced_zero_uncertainty_failed"
        reason_parts.append("forced_zero_uncertainty_failed")

    if float(selected_lambda) > 0.0:
        zero_by_domain: Dict[int, float] = {}
        selected_by_domain: Dict[int, float] = {}
        for domain in sorted({int(row["query_domain"]) for row in rows_by_lambda.get(0.0, [])}):
            zero_vals = [float(r["oracle_gap_pct"]) for r in rows_by_lambda.get(0.0, []) if int(r["query_domain"]) == domain]
            selected_vals = [
                float(r["oracle_gap_pct"])
                for r in rows_by_lambda.get(float(selected_lambda), [])
                if int(r["query_domain"]) == domain
            ]
            if zero_vals:
                zero_by_domain[int(domain)] = float(np.mean(zero_vals))
            if selected_vals:
                selected_by_domain[int(domain)] = float(np.mean(selected_vals))
        catastrophic_domain_harm = any(
            float(selected_by_domain[d]) - float(zero_by_domain[d])
            > float(cfg.catastrophic_regression_vs_pairprob_hard_gap_pct)
            for d in selected_by_domain
            if d in zero_by_domain
        )
        pref_values = set(float(v) for v in domain_pref.values())
        contradictory = 0.0 in pref_values and any(v > 0.0 for v in pref_values)
        if catastrophic_domain_harm or contradictory:
            selected_lambda = 0.0
            selected_summary = zero_summary
            lambda_status = "unstable"
            reason_parts.append("lambda_unstable_forced_zero")

    if abs(float(selected_lambda)) < 1e-12 and lambda_status == "stable_zero":
        reason_parts.append("selected_lambda_zero_mean_ensemble")
    if float(selected_summary.get("jackknife_override_rate", 0.0)) > float(cfg.max_override_rate):
        reason_parts.append("jackknife_override_rate_too_high")
    if float(selected_summary.get("candidate_pool_consistent", 0.0)) < 1.0:
        reason_parts.append("candidate_pool_inconsistent")

    reason = "|".join(part for part in dict.fromkeys(str(v) for v in reason_parts if str(v)) if part)
    return JackknifeLCBSelection(
        method=cfg.method_name,
        mean_method=cfg.mean_method_name,
        base_method=cfg.base_method,
        feature_set=base_selection.feature_set,
        ridge_l2=base_selection.ridge_l2,
        jackknife_lambda=float(selected_lambda),
        selected_by_inner_validation=True,
        diagnostic_only_reason=str(reason),
        noop=bool(evidence_reason),
        source_inner_validation_domains=int(selected_summary.get("validation_domains", 0.0)),
        source_inner_rows=int(selected_summary.get("n_rows", 0.0)),
        source_inner_mean_oracle_gap_pct=float(selected_summary["mean_oracle_gap_pct"]),
        source_inner_worst_domain_oracle_gap_pct=float(selected_summary["worst_inner_domain_oracle_gap_pct"]),
        source_inner_relative_catastrophic_rate=float(
            selected_summary["relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate"]
        ),
        source_inner_absolute_high_regret_rate=float(selected_summary["absolute_high_regret_rate_gap_gt_5"]),
        source_inner_top1=float(selected_summary["top1_oracle_hit"]),
        source_inner_spearman=float(selected_summary["spearman"]),
        jackknife_uncertainty_auc_for_pairprob_top1_error=float(
            zero_summary["jackknife_uncertainty_auc_for_pairprob_top1_error"]
        ),
        jackknife_uncertainty_auc_for_pairprob_high_regret=float(
            zero_summary["jackknife_uncertainty_auc_for_pairprob_high_regret"]
        ),
        uncertainty_error_spearman_source_inner=float(zero_summary["uncertainty_error_spearman_source_inner"]),
        lambda_stability_status=str(lambda_status),
        candidate_pool_consistent=bool(float(selected_summary.get("candidate_pool_consistent", 0.0)) >= 1.0),
        selected_lambda_is_zero_but_lcb_candidates_reported=bool(abs(float(selected_lambda)) < 1e-12),
        jackknife_mean_vs_pairprob_hard_selection_change_rate=float(
            selected_summary["jackknife_mean_vs_pairprob_hard_selection_change_rate"]
        ),
        mean_ensemble_override_rate_vs_pairprob_hard=float(
            selected_summary["mean_ensemble_override_rate_vs_pairprob_hard"]
        ),
        lcb_override_rate_vs_jackknife_mean=float(selected_summary["lcb_override_rate_vs_jackknife_mean"]),
        lcb_override_rate_vs_pairprob_hard=float(selected_summary["lcb_override_rate_vs_pairprob_hard"]),
        jackknife_override_help_rate=float(selected_summary["jackknife_override_help_rate"]),
        jackknife_override_harm_rate=float(selected_summary["jackknife_override_harm_rate"]),
        total_override_help_gap_reduction=float(selected_summary["total_override_help_gap_reduction"]),
        total_override_harm_gap_increase=float(selected_summary["total_override_harm_gap_increase"]),
        mean_paired_gap_delta_vs_pairprob_hard=float(selected_summary["mean_paired_gap_delta_vs_pairprob_hard"]),
        paired_improvement_rate_vs_pairprob_hard=float(selected_summary["paired_improvement_rate_vs_pairprob_hard"]),
    )


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
    selection_policy = _pairprob_selection_policy_for_method(method)
    is_direct_diagnostic = str(method) == DIRECT_PAIRPROB_DIAGNOSTIC_METHOD
    is_direct_adoption = str(method) == DIRECT_PAIRPROB_ADOPTION_METHOD
    is_combined_diagnostic = str(method) == COMBINED_PAIRPROB_DIAGNOSTIC_METHOD
    excluded_from_sign_ci = int(is_direct_diagnostic or is_combined_diagnostic)
    sign_ci_candidate = int(
        (not bool(excluded_from_sign_ci))
        and str(method) in {DIRECT_PAIRPROB_ADOPTION_METHOD, GROUP_ROBUST_PAIRPROB_METHOD}
    )

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
                "threshold_selection_policy": str(selection_policy),
                "route_experts": str(selected_expert),
                "route_weights": "1",
                "route_size": 1,
                "route_mode": "pairprob_hard_top1",
                "pairprob_predictor": "logistic_ridge_pairprob",
                "pairprob_probability_calibration": "none_v1",
                "pairprob_ridge_l2": float(selection.ridge_l2),
                "pairprob_feature_set": str(selection.feature_set),
                "pairprob_selection_policy": str(selection_policy),
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
                "excluded_from_sign_ci_selection": int(excluded_from_sign_ci),
                "sign_ci_candidate": int(sign_ci_candidate),
                "adoption_feature_family": (
                    DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY if is_direct_adoption else ""
                ),
                "direct_adoption_is_alias_of": (
                    DIRECT_PAIRPROB_DIAGNOSTIC_METHOD if is_direct_adoption else ""
                ),
                "direct_adoption_same_route_as_direct": int(0 if is_direct_adoption else 0),
                "direct_adoption_audit_failure_reason": (
                    "missing_diagnostic_direct_row" if is_direct_adoption else ""
                ),
                "source_only_audit_pass": int(
                    str(selection.feature_set) == DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY
                ),
                "target_leakage_audit_pass": 1,
                "direct_vs_group_robust_primary_comparator": int(is_direct_adoption),
                **pair_diag,
            }
        )
        route_hash = _direct_pairprob_route_hash(row)
        if is_direct_diagnostic:
            row["direct_diagnostic_route_hash"] = route_hash
            row["direct_adoption_route_hash"] = ""
        elif is_direct_adoption:
            row["direct_diagnostic_route_hash"] = ""
            row["direct_adoption_route_hash"] = route_hash
        else:
            row["direct_diagnostic_route_hash"] = ""
            row["direct_adoption_route_hash"] = ""
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


def _top2_rerank_features_for_rows(
    *,
    x_rows: np.ndarray,
    expert_domains: Sequence[int],
    prob_matrix: np.ndarray,
    embedding_dim: int,
    expert_feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win, orders, margins = pairprob_order_and_margin(prob_matrix, expert_domains=expert_domains)
    k = int(win.shape[1])
    if x_rows.shape[0] % k != 0:
        raise ProtocolError("Top-2 reranker feature rows are not divisible by candidate expert count")
    features: List[np.ndarray] = []
    top1 = orders[:, 0].astype(np.int64, copy=False)
    top2 = orders[:, 1].astype(np.int64, copy=False) if k > 1 else orders[:, 0].astype(np.int64, copy=False)
    p_top1_top2 = np.zeros((win.shape[0],), dtype=np.float64)
    for row_idx in range(win.shape[0]):
        base = int(row_idx * k)
        p = float(prob_matrix[row_idx, int(top1[row_idx]), int(top2[row_idx])]) if k > 1 else 0.5
        p_top1_top2[row_idx] = p
        features.append(
            _top2_rerank_feature(
                np.asarray(x_rows[base + int(top1[row_idx])], dtype=np.float64),
                np.asarray(x_rows[base + int(top2[row_idx])], dtype=np.float64),
                embedding_dim=int(embedding_dim),
                expert_feature_dim=int(expert_feature_dim),
                base_top1_win=float(win[row_idx, int(top1[row_idx])]),
                base_top2_win=float(win[row_idx, int(top2[row_idx])]),
                top2_margin=float(margins[row_idx]),
                p_top1_beats_top2=p,
            )
        )
    return (
        np.vstack(features).astype(np.float64, copy=False) if features else np.zeros((0, 0), dtype=np.float64),
        win,
        orders,
        margins,
        top1,
        top2,
    )


def top2_rerank_route_rows(
    *,
    method: str,
    fold: FoldCandidateSet,
    query_domains: np.ndarray,
    expert_domains: Sequence[int],
    x_rows: np.ndarray,
    prob_matrix: np.ndarray,
    true_nelbo_matrix: np.ndarray,
    global_true_nelbo_matrix: np.ndarray,
    global_expert_domains: Sequence[int],
    policy_name: str,
    selection: Top2RerankSelection,
    reranker_bundle: Top2RerankModelBundle | None,
    pairprob_direct_gap_pct: np.ndarray,
    metadata_oracle_gap_pct: np.ndarray | None,
    embedding_dim: int,
    expert_feature_dim: int,
    cfg: Top2MarginRerankerConfig,
    keep_prob_override: np.ndarray | None = None,
    oracle_diagnostic: bool = False,
) -> List[Dict[str, Any]]:
    features, win, orders, margins, top1_idx, top2_idx = _top2_rerank_features_for_rows(
        x_rows=x_rows,
        expert_domains=expert_domains,
        prob_matrix=prob_matrix,
        embedding_dim=int(embedding_dim),
        expert_feature_dim=int(expert_feature_dim),
    )
    n = int(win.shape[0])
    direct_gap = np.asarray(pairprob_direct_gap_pct, dtype=np.float64)
    if direct_gap.shape[0] != n:
        direct_gap = _gap_pct_for_selected(true_nelbo_matrix, top1_idx)
    metadata_gap = (
        np.asarray(metadata_oracle_gap_pct, dtype=np.float64)
        if metadata_oracle_gap_pct is not None
        else np.full((n,), float("nan"), dtype=np.float64)
    )
    if metadata_gap.shape[0] != n:
        metadata_gap = np.full((n,), float("nan"), dtype=np.float64)

    active = margins <= (float(selection.margin_threshold) + 1e-12)
    if bool(selection.noop) and not bool(oracle_diagnostic):
        active = np.zeros_like(active, dtype=bool)
    if keep_prob_override is not None:
        keep_prob = np.asarray(keep_prob_override, dtype=np.float64)
    elif reranker_bundle is not None:
        keep_prob = _apply_top2_rerank_model(reranker_bundle, features)
    else:
        keep_prob = np.ones((n,), dtype=np.float64)
    if keep_prob.shape[0] != n:
        keep_prob = np.ones((n,), dtype=np.float64)

    true = np.asarray(true_nelbo_matrix, dtype=np.float64)
    top2_better = true[np.arange(n), top2_idx] < true[np.arange(n), top1_idx]
    if bool(oracle_diagnostic):
        switched = active & top2_better
    else:
        switched = active & (keep_prob < float(selection.decision_threshold))
    selected_idx = np.where(switched, top2_idx, top1_idx).astype(np.int64, copy=False)
    ranking_score = -win
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
    reason = str(selection.diagnostic_only_reason)
    if bool(oracle_diagnostic):
        reason = str(selection.oracle_method)

    top2_gap = _gap_pct_for_selected(true_nelbo_matrix, top2_idx)
    for i, row in enumerate(rows):
        selected_col = int(selected_idx[i])
        selected_expert = int(expert_domains[selected_col])
        paired_delta = float(row["oracle_gap_pct"]) - float(direct_gap[i])
        paired_delta_metadata = (
            float(row["oracle_gap_pct"]) - float(metadata_gap[i])
            if np.isfinite(float(metadata_gap[i]))
            else float("nan")
        )
        pair_diag = _pair_diagnostics_for_row(prob_matrix[i, :, :], true_nelbo_matrix[i, :])
        row.update(
            {
                "policy_name": str(policy_name),
                "base_method": str(selection.base_method),
                "feature_set": str(selection.feature_set),
                "selected_tau": float(selection.base_ridge_l2),
                "selected_by_inner_validation": int(bool(selection.selected_by_inner_validation)),
                "threshold_selection_policy": str(cfg.calibration_policy),
                "route_experts": str(selected_expert),
                "route_weights": "1",
                "route_size": 1,
                "route_mode": "oracle_top2_margin_reranker_diagnostic" if oracle_diagnostic else "top2_margin_rerank",
                "pairprob_predictor": "logistic_ridge_pairprob",
                "pairprob_probability_calibration": "none_v1",
                "pairprob_ridge_l2": float(selection.base_ridge_l2),
                "pairprob_feature_set": str(selection.base_feature_set),
                "pairprob_selection_policy": str(cfg.calibration_policy),
                "adoption_feature_family": DIRECT_PAIRPROB_ADOPTION_FEATURE_FAMILY if not oracle_diagnostic else "",
                "base_direct_selected_expert": int(expert_domains[int(top1_idx[i])]),
                "base_direct_top2_expert": int(expert_domains[int(top2_idx[i])]),
                "base_direct_top1_win": float(win[i, int(top1_idx[i])]),
                "base_direct_top2_win": float(win[i, int(top2_idx[i])]),
                "base_direct_top2_margin": float(margins[i]),
                "top2_rerank_active": int(bool(active[i])),
                "top2_rerank_switched": int(bool(switched[i])),
                "top2_rerank_keep_top1_prob": float(keep_prob[i]),
                "top2_rerank_threshold": float(selection.margin_threshold),
                "top2_rerank_l2": float(selection.reranker_l2),
                "top2_rerank_delta_gap_pct_vs_direct": float(paired_delta),
                "top2_rerank_help": int(bool(switched[i]) and paired_delta < 0.0),
                "top2_rerank_harm": int(bool(switched[i]) and paired_delta > 0.0),
                "top2_rerank_candidate_delta_gap_pct_vs_direct": float(top2_gap[i] - direct_gap[i]),
                "top2_rerank_top2_better_than_base": int(bool(top2_better[i])),
                "top2_rerank_guard_status": str(selection.guard_status),
                "top2_rerank_diagnostic_only_reason": str(reason),
                "top2_rerank_selection_stability_status": str(selection.selection_stability_status),
                "source_inner_top2_rerank_gap_reduction_abs_pct_points": float(
                    selection.source_inner_gap_reduction_abs_pct_points
                ),
                "source_inner_top2_rerank_high_regret_reduction": float(
                    selection.source_inner_high_regret_reduction
                ),
                "source_inner_top2_rerank_rows": int(selection.source_inner_top2_rerank_rows),
                "source_inner_top2_rerank_positive_rows": int(selection.source_inner_top2_rerank_positive_rows),
                "source_inner_top2_rerank_negative_rows": int(selection.source_inner_top2_rerank_negative_rows),
                "source_inner_top2_rerank_active_domains": int(selection.source_inner_top2_rerank_active_domains),
                "source_inner_switch_candidate_rate": float(selection.source_inner_switch_candidate_rate),
                "reranker_selection_stability_status": str(selection.selection_stability_status),
                "base_top2_margin_auc_for_high_regret": float(selection.base_top2_margin_auc_for_high_regret),
                "base_top2_margin_spearman_with_oracle_gap": float(
                    selection.base_top2_margin_spearman_with_oracle_gap
                ),
                "overall_high_regret_rate_direct": float(selection.overall_high_regret_rate_direct),
                "low_margin_active_high_regret_rate": float(selection.low_margin_active_high_regret_rate),
                "low_margin_high_regret_enrichment": float(selection.low_margin_high_regret_enrichment),
                "top2_rerank_auc_source_inner": float(selection.top2_rerank_auc_source_inner),
                "top2_rerank_brier_source_inner": float(selection.top2_rerank_brier_source_inner),
                "top2_rerank_calibration_status": str(selection.top2_rerank_calibration_status),
                "oracle_top2_active_gap_reduction_pct": float(selection.oracle_top2_active_gap_reduction_pct),
                "oracle_top2_active_high_regret_reduction": float(
                    selection.oracle_top2_active_high_regret_reduction
                ),
                "oracle_top2_recoverable_error_rate": float(selection.oracle_top2_recoverable_error_rate),
                "oracle_top2_recoverable_gap_mass_pct_points": float(
                    selection.oracle_top2_recoverable_gap_mass_pct_points
                ),
                "paired_gap_delta_vs_pairprob_hard": float(paired_delta),
                "paired_gap_delta_vs_metadata": float(paired_delta_metadata),
                "pairprob_hard_oracle_gap_pct": float(direct_gap[i]),
                "metadata_oracle_gap_pct": float(metadata_gap[i]),
                "absolute_high_regret_gap_gt_5": int(float(row["oracle_gap_pct"]) > float(cfg.absolute_high_regret_gap_pct)),
                "relative_catastrophic_regression_vs_pairprob_hard_gt_5": int(
                    float(paired_delta) > float(cfg.catastrophic_regression_vs_direct_gap_pct)
                ),
                "diagnostic_only_reason": str(reason),
                **pair_diag,
            }
        )
        if reason:
            row.update({"method_role": "diagnostic", "adoption_eligible": 0, "diagnostic_only": 1})
    return rows


def summarize_top2_rerank_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "n_rows": 0.0,
            "validation_domains": 0.0,
            "mean_oracle_gap_pct": float("nan"),
            "high_regret_rate": float("nan"),
            "top1_oracle_hit": float("nan"),
            "spearman": float("nan"),
            "top2_rerank_activation_rate": float("nan"),
            "top2_rerank_switch_rate": float("nan"),
            "top2_rerank_help_rate_active_only": float("nan"),
            "top2_rerank_harm_rate_active_only": float("nan"),
            "mean_top2_rerank_delta_gap_pct_vs_direct": float("nan"),
            "median_top2_rerank_delta_gap_pct_vs_direct": float("nan"),
            "paired_improvement_rate_vs_direct_pairprob": float("nan"),
            "base_top2_margin_auc_for_high_regret": float("nan"),
            "base_top2_margin_spearman_with_oracle_gap": float("nan"),
            "overall_high_regret_rate_direct": float("nan"),
            "low_margin_active_high_regret_rate": float("nan"),
            "low_margin_high_regret_enrichment": float("nan"),
            "top2_rerank_auc_source_inner": float("nan"),
            "top2_rerank_brier_source_inner": float("nan"),
        }
    by_domain: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(int(row["query_domain"]), []).append(row)
    spearman_vals = [float(r["spearman"]) for r in rows if np.isfinite(float(r["spearman"]))]
    active_rows = [r for r in rows if int(float(r.get("top2_rerank_active", 0) or 0)) == 1]
    switched_rows = [r for r in rows if int(float(r.get("top2_rerank_switched", 0) or 0)) == 1]
    deltas = [float(r.get("top2_rerank_delta_gap_pct_vs_direct", float("nan"))) for r in rows]
    keep_probs = [
        float(r.get("top2_rerank_keep_top1_prob", float("nan")))
        for r in active_rows
        if np.isfinite(float(r.get("top2_rerank_keep_top1_prob", float("nan"))))
    ]
    keep_labels = [int(float(r.get("top2_rerank_top2_better_than_base", 0) or 0)) == 0 for r in active_rows]
    brier_vals = [
        (float(prob) - (1.0 if bool(label) else 0.0)) ** 2
        for prob, label in zip(keep_probs, keep_labels)
        if np.isfinite(float(prob))
    ]
    direct_gaps = [float(r.get("pairprob_hard_oracle_gap_pct", float("nan"))) for r in rows]
    margins = [float(r.get("base_direct_top2_margin", float("nan"))) for r in rows]
    high_regret_direct = [
        int(float(r.get("pairprob_hard_oracle_gap_pct", 0.0) or 0.0) > float(r.get("absolute_high_regret_gap_pct", 5.0)))
        for r in rows
    ]
    overall_high = float(np.mean(high_regret_direct)) if high_regret_direct else float("nan")
    active_high = float(
        np.mean(
            [
                1.0 if float(r.get("pairprob_hard_oracle_gap_pct", 0.0) or 0.0) > 5.0 else 0.0
                for r in active_rows
            ]
        )
    ) if active_rows else float("nan")
    return {
        "n_rows": float(len(rows)),
        "validation_domains": float(len(by_domain)),
        "mean_oracle_gap_pct": float(np.mean([float(r["oracle_gap_pct"]) for r in rows])),
        "high_regret_rate": float(np.mean([float(r.get("absolute_high_regret_gap_gt_5", 0.0)) for r in rows])),
        "top1_oracle_hit": float(np.mean([float(r["top1_oracle_hit"]) for r in rows])),
        "spearman": float(np.mean(spearman_vals)) if spearman_vals else float("nan"),
        "top2_rerank_activation_rate": float(np.mean([float(r.get("top2_rerank_active", 0.0)) for r in rows])),
        "top2_rerank_switch_rate": float(np.mean([float(r.get("top2_rerank_switched", 0.0)) for r in rows])),
        "top2_rerank_help_rate_active_only": float(
            np.mean([float(r.get("top2_rerank_help", 0.0)) for r in active_rows])
        ) if active_rows else float("nan"),
        "top2_rerank_harm_rate_active_only": float(
            np.mean([float(r.get("top2_rerank_harm", 0.0)) for r in active_rows])
        ) if active_rows else float("nan"),
        "top2_rerank_switch_harm_rate": float(
            np.mean([1.0 if float(r.get("top2_rerank_delta_gap_pct_vs_direct", 0.0)) > 0.0 else 0.0 for r in switched_rows])
        ) if switched_rows else 0.0,
        "mean_top2_rerank_delta_gap_pct_vs_direct": float(np.mean(deltas)) if deltas else float("nan"),
        "median_top2_rerank_delta_gap_pct_vs_direct": float(np.median(deltas)) if deltas else float("nan"),
        "paired_improvement_rate_vs_direct_pairprob": float(
            np.mean([1.0 if float(v) < 0.0 else 0.0 for v in deltas])
        ) if deltas else float("nan"),
        "base_top2_margin_auc_for_high_regret": _binary_auc([-m for m in margins], high_regret_direct),
        "base_top2_margin_spearman_with_oracle_gap": _finite_spearman([-m for m in margins], direct_gaps),
        "overall_high_regret_rate_direct": float(overall_high),
        "low_margin_active_high_regret_rate": float(active_high),
        "low_margin_high_regret_enrichment": (
            float(active_high / overall_high) if np.isfinite(active_high) and overall_high > 0.0 else float("nan")
        ),
        "top2_rerank_auc_source_inner": _binary_auc(keep_probs, [1 if label else 0 for label in keep_labels]),
        "top2_rerank_brier_source_inner": float(np.mean(brier_vals)) if brier_vals else float("nan"),
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


def top2_rerank_evidence_reason(
    *,
    train_data: Top2RerankTrainingData,
    validation_domains: int,
    cfg: Top2MarginRerankerConfig,
) -> str:
    if int(validation_domains) < int(cfg.min_source_inner_validation_domains):
        return "insufficient_source_inner_rerank_rows"
    if int(train_data.x.shape[0]) < int(cfg.min_source_inner_rerank_rows):
        return "insufficient_source_inner_rerank_rows"
    if int(train_data.positive_rows) < int(cfg.min_source_inner_positive_rows):
        return "insufficient_source_inner_positive_rows"
    if int(train_data.negative_rows) < int(cfg.min_source_inner_negative_rows):
        return "insufficient_source_inner_negative_rows"
    if len(train_data.kept_by_domain) < int(cfg.min_source_inner_active_domains):
        return "insufficient_source_inner_active_domains"
    return ""


def _top2_reason_from_summary(
    *,
    summary: Mapping[str, float],
    train_data: Top2RerankTrainingData,
    cfg: Top2MarginRerankerConfig,
    stability_status: str,
) -> str:
    if int(train_data.x.shape[0]) < int(cfg.min_source_inner_rerank_rows):
        return "insufficient_source_inner_rerank_rows"
    if int(train_data.positive_rows) < int(cfg.min_source_inner_positive_rows):
        return "insufficient_source_inner_positive_rows"
    if int(train_data.negative_rows) < int(cfg.min_source_inner_negative_rows):
        return "insufficient_source_inner_negative_rows"
    if len(train_data.kept_by_domain) < int(cfg.min_source_inner_active_domains):
        return "insufficient_source_inner_active_domains"
    enrichment = float(summary.get("low_margin_high_regret_enrichment", float("nan")))
    if not np.isfinite(enrichment) or enrichment < float(cfg.min_low_margin_high_regret_enrichment):
        return "low_margin_not_high_regret_enriched"
    if float(summary.get("top2_rerank_activation_rate", 0.0)) > float(cfg.max_rerank_activation_rate):
        return "activation_rate_too_high"
    if float(summary.get("top2_rerank_switch_rate", 0.0)) > float(cfg.max_rerank_switch_rate):
        return "switch_rate_too_high"
    if float(summary.get("top2_rerank_switch_harm_rate", 0.0)) > float(cfg.max_switch_harm_rate_active_only):
        return "harm_rate_too_high"
    if float(summary.get("source_inner_gap_reduction_abs_pct_points", 0.0)) < float(
        cfg.min_source_inner_gap_reduction_abs_pct_points
    ):
        return "insufficient_gap_reduction"
    if str(stability_status) == "unstable":
        return "unstable_source_inner_selection"
    auc = float(summary.get("top2_rerank_auc_source_inner", float("nan")))
    if np.isfinite(auc) and auc < 0.50:
        return "weak_reranker_auc_or_calibration"
    if float(summary.get("mean_top2_rerank_delta_gap_pct_vs_direct", 0.0)) > 0.0:
        return "worsens_direct_pairprob"
    return ""


def _source_inner_oracle_top2_headroom(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    active = [r for r in rows if int(float(r.get("top2_rerank_active", 0) or 0)) == 1]
    if not active:
        return {
            "oracle_top2_active_gap_reduction_pct": float("nan"),
            "oracle_top2_active_high_regret_reduction": float("nan"),
            "oracle_top2_recoverable_error_rate": float("nan"),
            "oracle_top2_recoverable_gap_mass_pct_points": 0.0,
        }
    direct_gap = [float(r.get("pairprob_hard_oracle_gap_pct", float("nan"))) for r in active]
    oracle_gap = [float(r.get("oracle_gap_pct", float("nan"))) for r in active]
    recoverable_delta = [
        max(0.0, float(r.get("pairprob_hard_oracle_gap_pct", 0.0)) - float(r.get("oracle_gap_pct", 0.0)))
        for r in active
    ]
    return {
        "oracle_top2_active_gap_reduction_pct": float(np.nanmean(direct_gap) - np.nanmean(oracle_gap)),
        "oracle_top2_active_high_regret_reduction": float(
            np.mean([1.0 if v > 5.0 else 0.0 for v in direct_gap])
            - np.mean([1.0 if v > 5.0 else 0.0 for v in oracle_gap])
        ),
        "oracle_top2_recoverable_error_rate": float(
            np.mean([1.0 if float(v) > 0.0 else 0.0 for v in recoverable_delta])
        ),
        "oracle_top2_recoverable_gap_mass_pct_points": float(np.mean(recoverable_delta)),
    }


def select_top2_margin_reranker_policy(
    *,
    blocks: Sequence[Top2RerankCalibrationBlock],
    base_selection: PairprobPolicySelection | None,
    global_expert_domains: Sequence[int],
    cfg: Top2MarginRerankerConfig,
    embedding_dim: int,
    expert_feature_dim: int,
    device: str,
) -> Top2RerankSelection | None:
    if not bool(cfg.enabled):
        return None
    if base_selection is None or not blocks:
        return Top2RerankSelection(
            method=cfg.method_name,
            oracle_method=cfg.diagnostic_oracle_method_name,
            base_method=cfg.base_method,
            feature_set=cfg.feature_set,
            base_feature_set=cfg.base_feature_set,
            base_ridge_l2=float("nan"),
            reranker_l2=float(cfg.reranker_l2_values[0]),
            margin_threshold=float(cfg.margin_thresholds[0]),
            decision_threshold=float(cfg.decision_threshold),
            selected_by_inner_validation=False,
            diagnostic_only_reason="insufficient_source_inner_rerank_rows",
            noop=True,
            guard_status="failed_guards_noop",
            selection_stability_status="forced_direct_pairprob",
        )

    candidates: List[Tuple[Tuple[float, ...], float, float, Dict[str, float], Top2RerankTrainingData, str, List[Dict[str, Any]]]] = []
    invalid: List[Tuple[Tuple[float, ...], float, float, Dict[str, float], Top2RerankTrainingData, str, List[Dict[str, Any]]]] = []
    source_domains = sorted({int(block.validation_domain) for block in blocks})
    for threshold in cfg.margin_thresholds:
        training_by_domain: Dict[int, Top2RerankTrainingData] = {}
        for block in blocks:
            training_by_domain[int(block.validation_domain)] = build_top2_rerank_training_data(
                x_rows=block.x_rows,
                query_domains=block.query_domains,
                expert_domains=block.expert_domains,
                prob_matrix=block.prob_matrix,
                true_nelbo_matrix=block.true_nelbo_matrix,
                embedding_dim=int(embedding_dim),
                expert_feature_dim=int(expert_feature_dim),
                margin_threshold=float(threshold),
                near_tie_delta_pct=float(cfg.near_tie_delta_pct),
                margin_weight_scale_pct=float(cfg.margin_weight_scale_pct),
                margin_weight_clip=cfg.margin_weight_clip,
            )
        full_train_data = _concat_top2_training_data(list(training_by_domain.values()))
        for l2 in cfg.reranker_l2_values:
            rows: List[Dict[str, Any]] = []
            keep_probs_all: List[float] = []
            keep_labels_all: List[int] = []
            for block in blocks:
                train_parts = [
                    data for domain, data in training_by_domain.items() if int(domain) != int(block.validation_domain)
                ]
                train_data = _concat_top2_training_data(train_parts)
                reason = top2_rerank_evidence_reason(
                    train_data=train_data,
                    validation_domains=len(train_parts),
                    cfg=cfg,
                )
                bundle: Top2RerankModelBundle | None = None
                if not reason:
                    bundle = fit_top2_rerank_model(train_data=train_data, ridge_l2=float(l2), device=str(device))
                validation_rows = top2_rerank_route_rows(
                    method=cfg.method_name,
                    fold=block.fold,
                    query_domains=block.query_domains,
                    expert_domains=block.expert_domains,
                    x_rows=block.x_rows,
                    prob_matrix=block.prob_matrix,
                    true_nelbo_matrix=block.true_nelbo_matrix,
                    global_true_nelbo_matrix=block.global_true_nelbo_matrix,
                    global_expert_domains=global_expert_domains,
                    policy_name=cfg.method_name,
                    selection=Top2RerankSelection(
                        method=cfg.method_name,
                        oracle_method=cfg.diagnostic_oracle_method_name,
                        base_method=cfg.base_method,
                        feature_set=cfg.feature_set,
                        base_feature_set=cfg.base_feature_set,
                        base_ridge_l2=base_selection.ridge_l2,
                        reranker_l2=float(l2),
                        margin_threshold=float(threshold),
                        decision_threshold=float(cfg.decision_threshold),
                        selected_by_inner_validation=True,
                        diagnostic_only_reason=str(reason),
                        noop=bool(reason),
                    ),
                    reranker_bundle=bundle,
                    pairprob_direct_gap_pct=block.pairprob_direct_gap_pct,
                    metadata_oracle_gap_pct=block.metadata_oracle_gap_pct,
                    embedding_dim=int(embedding_dim),
                    expert_feature_dim=int(expert_feature_dim),
                    cfg=cfg,
                )
                rows.extend(validation_rows)
                for row in validation_rows:
                    if int(float(row.get("top2_rerank_active", 0) or 0)) == 1:
                        keep_probs_all.append(float(row.get("top2_rerank_keep_top1_prob", float("nan"))))
                        keep_labels_all.append(0 if int(float(row.get("top2_rerank_top2_better_than_base", 0) or 0)) == 1 else 1)

            summary = summarize_top2_rerank_rows(rows)
            direct_mean_gap = float(np.mean([float(r.get("pairprob_hard_oracle_gap_pct", float("nan"))) for r in rows]))
            direct_high = float(
                np.mean([1.0 if float(r.get("pairprob_hard_oracle_gap_pct", 0.0)) > float(cfg.absolute_high_regret_gap_pct) else 0.0 for r in rows])
            ) if rows else float("nan")
            summary["source_inner_gap_reduction_abs_pct_points"] = float(direct_mean_gap - summary["mean_oracle_gap_pct"])
            summary["source_inner_high_regret_reduction"] = float(direct_high - summary["high_regret_rate"])
            if keep_probs_all and len(set(keep_labels_all)) == 2:
                summary["top2_rerank_auc_source_inner"] = _binary_auc(keep_probs_all, keep_labels_all)
                summary["top2_rerank_brier_source_inner"] = float(
                    np.mean([(float(p) - float(y)) ** 2 for p, y in zip(keep_probs_all, keep_labels_all)])
                )
            summary["top2_rerank_calibration_status"] = "ok" if float(summary.get("top2_rerank_auc_source_inner", 0.0)) >= 0.50 else "weak"

            domain_harm = False
            for domain in sorted({int(row["query_domain"]) for row in rows}):
                selected_vals = [float(r["oracle_gap_pct"]) for r in rows if int(r["query_domain"]) == domain]
                direct_vals = [float(r.get("pairprob_hard_oracle_gap_pct", float("nan"))) for r in rows if int(r["query_domain"]) == domain]
                if selected_vals and direct_vals and float(np.mean(selected_vals)) - float(np.mean(direct_vals)) > float(cfg.catastrophic_regression_vs_direct_gap_pct):
                    domain_harm = True
                    break
            stability = "unstable" if domain_harm else "stable"
            reason = _top2_reason_from_summary(
                summary=summary,
                train_data=full_train_data,
                cfg=cfg,
                stability_status=stability,
            )
            score = (
                -float(summary["mean_oracle_gap_pct"]),
                -float(summary["high_regret_rate"]),
                float(summary["source_inner_gap_reduction_abs_pct_points"]),
                float(summary["top1_oracle_hit"]),
                -float(summary["top2_rerank_harm_rate_active_only"])
                if np.isfinite(float(summary["top2_rerank_harm_rate_active_only"]))
                else -1e9,
                -float(summary["top2_rerank_activation_rate"]),
                -float(l2),
                -float(threshold),
            )
            item = (score, float(threshold), float(l2), summary, full_train_data, stability, rows)
            (invalid if reason else candidates).append(item)

    pool = candidates if candidates else invalid
    if not pool:
        return Top2RerankSelection(
            method=cfg.method_name,
            oracle_method=cfg.diagnostic_oracle_method_name,
            base_method=cfg.base_method,
            feature_set=cfg.feature_set,
            base_feature_set=cfg.base_feature_set,
            base_ridge_l2=base_selection.ridge_l2,
            reranker_l2=float(cfg.reranker_l2_values[0]),
            margin_threshold=float(cfg.margin_thresholds[0]),
            decision_threshold=float(cfg.decision_threshold),
            selected_by_inner_validation=False,
            diagnostic_only_reason="insufficient_source_inner_rerank_rows",
            noop=True,
            guard_status="failed_guards_noop",
            selection_stability_status="forced_direct_pairprob",
        )
    _score, threshold, l2, summary, train_data, stability, selected_rows = sorted(
        pool,
        key=lambda item: item[0],
        reverse=True,
    )[0]
    reason = _top2_reason_from_summary(summary=summary, train_data=train_data, cfg=cfg, stability_status=stability)
    if str(base_selection.diagnostic_only_reason):
        reason = "|".join(
            part for part in dict.fromkeys([str(base_selection.diagnostic_only_reason), str(reason)]) if part
        )
    model: Top2RerankModelBundle | None = None
    if not reason:
        model = fit_top2_rerank_model(train_data=train_data, ridge_l2=float(l2), device=str(device))
    headroom_rows: List[Dict[str, Any]] = []
    headroom_selection = Top2RerankSelection(
        method=cfg.method_name,
        oracle_method=cfg.diagnostic_oracle_method_name,
        base_method=cfg.base_method,
        feature_set=cfg.feature_set,
        base_feature_set=cfg.base_feature_set,
        base_ridge_l2=base_selection.ridge_l2,
        reranker_l2=float(l2),
        margin_threshold=float(threshold),
        decision_threshold=float(cfg.decision_threshold),
        selected_by_inner_validation=True,
    )
    for block in blocks:
        headroom_rows.extend(
            top2_rerank_route_rows(
                method=cfg.diagnostic_oracle_method_name,
                fold=block.fold,
                query_domains=block.query_domains,
                expert_domains=block.expert_domains,
                x_rows=block.x_rows,
                prob_matrix=block.prob_matrix,
                true_nelbo_matrix=block.true_nelbo_matrix,
                global_true_nelbo_matrix=block.global_true_nelbo_matrix,
                global_expert_domains=global_expert_domains,
                policy_name=cfg.method_name,
                selection=headroom_selection,
                reranker_bundle=None,
                pairprob_direct_gap_pct=block.pairprob_direct_gap_pct,
                metadata_oracle_gap_pct=block.metadata_oracle_gap_pct,
                embedding_dim=int(embedding_dim),
                expert_feature_dim=int(expert_feature_dim),
                cfg=cfg,
                oracle_diagnostic=True,
            )
        )
    headroom = _source_inner_oracle_top2_headroom(headroom_rows)
    guard_status = "selected" if not reason else "failed_guards_noop"
    return Top2RerankSelection(
        method=cfg.method_name,
        oracle_method=cfg.diagnostic_oracle_method_name,
        base_method=cfg.base_method,
        feature_set=cfg.feature_set,
        base_feature_set=cfg.base_feature_set,
        base_ridge_l2=base_selection.ridge_l2,
        reranker_l2=float(l2),
        margin_threshold=float(threshold),
        decision_threshold=float(cfg.decision_threshold),
        selected_by_inner_validation=True,
        diagnostic_only_reason=str(reason),
        noop=bool(reason),
        guard_status=str(guard_status),
        selection_stability_status=str(stability if not reason else "forced_direct_pairprob"),
        source_inner_validation_domains=len(source_domains),
        source_inner_top2_rerank_rows=int(train_data.x.shape[0]),
        source_inner_top2_rerank_positive_rows=int(train_data.positive_rows),
        source_inner_top2_rerank_negative_rows=int(train_data.negative_rows),
        source_inner_top2_rerank_active_domains=int(len(train_data.kept_by_domain)),
        source_inner_switch_candidate_rate=float(train_data.switch_candidate_rate),
        source_inner_gap_reduction_abs_pct_points=float(summary["source_inner_gap_reduction_abs_pct_points"]),
        source_inner_high_regret_reduction=float(summary["source_inner_high_regret_reduction"]),
        source_inner_activation_rate=float(summary["top2_rerank_activation_rate"]),
        source_inner_switch_rate=float(summary["top2_rerank_switch_rate"]),
        source_inner_help_rate_active_only=float(summary["top2_rerank_help_rate_active_only"]),
        source_inner_harm_rate_active_only=float(summary["top2_rerank_harm_rate_active_only"]),
        source_inner_mean_oracle_gap_pct=float(summary["mean_oracle_gap_pct"]),
        source_inner_high_regret_rate=float(summary["high_regret_rate"]),
        source_inner_top1=float(summary["top1_oracle_hit"]),
        source_inner_spearman=float(summary["spearman"]),
        base_top2_margin_auc_for_high_regret=float(summary["base_top2_margin_auc_for_high_regret"]),
        base_top2_margin_spearman_with_oracle_gap=float(summary["base_top2_margin_spearman_with_oracle_gap"]),
        overall_high_regret_rate_direct=float(summary["overall_high_regret_rate_direct"]),
        low_margin_active_high_regret_rate=float(summary["low_margin_active_high_regret_rate"]),
        low_margin_high_regret_enrichment=float(summary["low_margin_high_regret_enrichment"]),
        top2_rerank_auc_source_inner=float(summary["top2_rerank_auc_source_inner"]),
        top2_rerank_brier_source_inner=float(summary["top2_rerank_brier_source_inner"]),
        top2_rerank_calibration_status=str(summary["top2_rerank_calibration_status"]),
        oracle_top2_active_gap_reduction_pct=float(headroom["oracle_top2_active_gap_reduction_pct"]),
        oracle_top2_active_high_regret_reduction=float(headroom["oracle_top2_active_high_regret_reduction"]),
        oracle_top2_recoverable_error_rate=float(headroom["oracle_top2_recoverable_error_rate"]),
        oracle_top2_recoverable_gap_mass_pct_points=float(headroom["oracle_top2_recoverable_gap_mass_pct_points"]),
        model=model,
    )
