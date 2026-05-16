from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.eval.metrics import spearman_corr


_PROTOCOL_VERSION = "learned_utility_loqdo_candidate_exclusion_v2"
_CANDIDATE_POLICY = "exclude_outer_target_expert"
_CANDIDATE_EXPERT_ORDER = "ascending_by_domain_label"
_ORACLE_POLICY = "candidate_pool_excludes_target_expert"
_LEARNED_PAIR_POLICY = "exclude_outer_target_and_query_self_expert"
_METRIC_AGGREGATION_POLICY = "sample_micro_and_query_domain_macro"
_AGGREGATION_SOURCE = "learned_utility_sample_selections.csv"
_MIN_CANDIDATES_FOR_RANK_METRICS = 2
_SPEARMAN_NAN_POLICY = "skip"
_PAIRWISE_AUC_NAN_POLICY = "skip"
_DELTA_GATE_GUARD_REASON_PRIORITY = (
    "insufficient_validation_domains",
    "insufficient_active_rows",
    "insufficient_active_domains",
    "activation_rate_too_high",
    "harm_rate_too_high",
    "help_minus_harm_too_low",
    "insufficient_gap_reduction",
)


class ProtocolError(ValueError):
    """Raised when learned-utility LOQDO protocol invariants are violated."""


@dataclass(frozen=True)
class FoldCandidateSet:
    heldout_domain: int
    candidate_col_indices: Tuple[int, ...]
    candidate_expert_domains: Tuple[int, ...]
    excluded_expert_domain: int
    target_expert_excluded: bool

    @classmethod
    def for_heldout_domain(
        cls,
        *,
        heldout_domain: int,
        expert_domains: Sequence[int],
        excluded_domains: Sequence[int] | None = None,
    ) -> "FoldCandidateSet":
        if int(heldout_domain) not in {int(d) for d in expert_domains}:
            raise ProtocolError(f"Heldout domain {heldout_domain} has no matching expert checkpoint")
        excluded = {int(heldout_domain)}
        for domain in excluded_domains or ():
            excluded.add(int(domain))

        pairs = [
            (idx, int(domain))
            for idx, domain in enumerate(expert_domains)
            if int(domain) not in excluded
        ]
        pairs = sorted(pairs, key=lambda item: int(item[1]))
        candidate = cls(
            heldout_domain=int(heldout_domain),
            candidate_col_indices=tuple(int(idx) for idx, _domain in pairs),
            candidate_expert_domains=tuple(int(domain) for _idx, domain in pairs),
            excluded_expert_domain=int(heldout_domain),
            target_expert_excluded=int(heldout_domain) not in [int(domain) for _idx, domain in pairs],
        )
        candidate.assert_valid()
        return candidate

    def assert_valid(self) -> None:
        if not self.candidate_col_indices:
            raise ProtocolError(
                f"No candidate experts remain for heldout_domain={self.heldout_domain} "
                f"under candidate_policy={_CANDIDATE_POLICY}"
            )
        if len(self.candidate_col_indices) != len(self.candidate_expert_domains):
            raise ProtocolError("Candidate column/domain lengths do not match")
        if int(self.heldout_domain) in set(int(d) for d in self.candidate_expert_domains):
            raise ProtocolError(
                f"Heldout target expert {self.heldout_domain} is present in candidate experts"
            )
        if tuple(sorted(self.candidate_expert_domains)) != tuple(self.candidate_expert_domains):
            raise ProtocolError("Candidate experts must be ordered ascending by domain label")
        if not bool(self.target_expert_excluded):
            raise ProtocolError("target_expert_excluded must be true for LOQDO v2")

    def contains(self, expert_domain: int) -> bool:
        return int(expert_domain) in set(int(d) for d in self.candidate_expert_domains)

    def label(self) -> str:
        return "|".join(str(int(d)) for d in self.candidate_expert_domains)

    def slice_nelbo(self, nelbo_matrix: np.ndarray, row_indices: np.ndarray | None = None) -> np.ndarray:
        cols = list(self.candidate_col_indices)
        if row_indices is None:
            return nelbo_matrix[:, cols]
        return nelbo_matrix[np.asarray(row_indices, dtype=np.int64)][:, cols]


@dataclass(frozen=True)
class MethodProtocol:
    method_role: str
    adoption_eligible: int
    diagnostic_only: int
    routing_uses_query_features: int = 0
    routing_uses_eval_domain_statistics: int = 0
    routing_uses_eval_nelbo: int = 0


def _protocol_row_fields(
    *,
    fold: FoldCandidateSet,
    method_protocol: MethodProtocol,
    method: str,
) -> Dict[str, Any]:
    return {
        "protocol_version": _PROTOCOL_VERSION,
        "fold_query_domain": int(fold.heldout_domain),
        "candidate_policy": _CANDIDATE_POLICY,
        "candidate_expert_order": _CANDIDATE_EXPERT_ORDER,
        "oracle_policy": _ORACLE_POLICY,
        "learned_pair_policy": _LEARNED_PAIR_POLICY,
        "metric_aggregation_policy": _METRIC_AGGREGATION_POLICY,
        "aggregation_source": _AGGREGATION_SOURCE,
        "candidate_experts": fold.label(),
        "n_candidate_experts": int(len(fold.candidate_expert_domains)),
        "excluded_experts": str(int(fold.excluded_expert_domain)),
        "target_expert_excluded": int(fold.target_expert_excluded),
        "oracle_scope": _ORACLE_POLICY,
        "method_role": str(method_protocol.method_role),
        "adoption_eligible": int(method_protocol.adoption_eligible),
        "diagnostic_only": int(method_protocol.diagnostic_only),
        "routing_uses_query_features": int(method_protocol.routing_uses_query_features),
        "routing_uses_eval_domain_statistics": int(method_protocol.routing_uses_eval_domain_statistics),
        "routing_uses_eval_nelbo": int(method_protocol.routing_uses_eval_nelbo),
        "global_oracle_used_for_metrics": 0,
        "metrics_comparable_to_previous_protocol": 0,
        "previous_protocol_invalidated_by_target_candidate_leakage": 1,
        "method": str(method),
    }


def _method_protocol(method: str) -> MethodProtocol:
    name = str(method)
    if name in {
        "support_metadata_routing",
        "support_static_embedding_routing",
        "support_set_nelbo_top1",
        "support_set_nelbo_conservative",
    }:
        return MethodProtocol(
            method_role="baseline",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "source_global_prior_routing":
        return MethodProtocol(
            method_role="baseline",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=0,
        )
    if name == "support_response_pairwise_static_response_indirect":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "risk_constrained_response_routing":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "expert_id_only_pairwise" or name.startswith("support_response_pairwise_response_indirect_shuffled"):
        return MethodProtocol(
            method_role="control",
            adoption_eligible=0,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "source_leave_pseudo_domain_out_ranker_diagnostic":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name == "support_candidate_oracle":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_eval_nelbo=1,
        )
    if name == "support_random_expert_floor":
        return MethodProtocol(
            method_role="control",
            adoption_eligible=0,
            diagnostic_only=1,
        )
    if name == "metadata_routing":
        return MethodProtocol(
            method_role="baseline",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "candidate_oracle_routing":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_eval_nelbo=1,
        )
    if name in {
        "pairwise_tournament_hard",
        "pairwise_tournament_topk_uniform",
        "pairwise_tournament_inner_selected",
        "pairwise_tournament_delta_gated_sparse_mix_v1",
    }:
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_tournament_delta_gated_sparse_mix_combined_diagnostic_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name == "pairwise_group_robust_pairprob_tournament_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_direct_pairprob_adoption_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_direct_top2_margin_reranker_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_direct_precision_top2_delta_gate_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_direct_allpair_utility_delta_gate_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_direct_group_oof_hardpair_boosted_pairprob_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name in {
        "pairwise_direct_group_oof_hardpair_miss_boosted_pairprob_v1_diagnostic",
        "pairwise_direct_random_low_margin_boost_pairprob_v1_diagnostic",
    }:
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name == "oracle_top2_margin_reranker_diagnostic_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
            routing_uses_eval_nelbo=1,
        )
    if name == "oracle_top2_delta_gate_diagnostic_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
            routing_uses_eval_nelbo=1,
        )
    if name == "oracle_allpair_top2_delta_gate_diagnostic_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
            routing_uses_eval_nelbo=1,
        )
    if name == "pairwise_jackknife_lcb_pairprob_tournament_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name == "pairwise_jackknife_mean_pairprob_tournament_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name == "conformal_pairprob_regret_set_router_v1":
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name in {"pairwise_direct_pairprob_tournament_v1", "pairwise_pairprob_combined_diagnostic_v1"}:
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name == "conformal_pairprob_topwin_set_diagnostic_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name == "oracle_conformal_regret_set_diagnostic_v1":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
            routing_uses_eval_nelbo=1,
        )
    if name == "oracle_confidence_set_diagnostic":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
            routing_uses_eval_nelbo=1,
        )
    if name == "latent_wasserstein_routing" or name.startswith("hybrid_alpha_"):
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_eval_domain_statistics=1,
        )
    if name in {"random_rank_floor", "random_score_floor", "expert_label_permutation", "metadata_permutation"}:
        return MethodProtocol(method_role="control", adoption_eligible=0, diagnostic_only=0)
    if name == "unconstrained_learned_reference" or name == "metadata_residual_argmax":
        return MethodProtocol(
            method_role="diagnostic",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
        )
    if name in {
        "metadata_residual_thresholded",
        "metadata_residual_group_robust",
        "metadata_residual_thresholded_safe_v2",
        "metadata_residual_group_robust_safe_v2",
        "metadata_residual_inner_selected",
    }:
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name in {"linear_regressor", "mlp_regressor", "metadata_only_regressor"} or name.startswith("pairwise_ranker"):
        return MethodProtocol(
            method_role="learned",
            adoption_eligible=1,
            diagnostic_only=0,
            routing_uses_query_features=1,
        )
    if name in {"static_embedding_routing", "static_embedding_baseline"}:
        return MethodProtocol(
            method_role="baseline",
            adoption_eligible=0,
            diagnostic_only=1,
            routing_uses_query_features=1,
            routing_uses_eval_domain_statistics=1,
        )
    return MethodProtocol(method_role="diagnostic", adoption_eligible=0, diagnostic_only=1)


def _assert_method_eligibility(method: str, protocol: MethodProtocol) -> None:
    if int(protocol.adoption_eligible) == 1:
        if int(protocol.diagnostic_only) != 0:
            raise ProtocolError(f"adoption_eligible method {method} cannot be diagnostic_only")
        if int(protocol.routing_uses_eval_nelbo) != 0:
            raise ProtocolError(f"adoption_eligible method {method} cannot use evaluation NELBO")
        if int(protocol.routing_uses_eval_domain_statistics) != 0:
            raise ProtocolError(f"adoption_eligible method {method} cannot use eval-domain statistics")
    if str(method) == "candidate_oracle_routing" and int(protocol.adoption_eligible) == 1:
        raise ProtocolError("candidate_oracle_routing must not be adoption eligible")
    if str(method) == "support_candidate_oracle" and int(protocol.adoption_eligible) == 1:
        raise ProtocolError("support_candidate_oracle must not be adoption eligible")


def _parse_candidate_experts_label(value: object) -> List[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split("|") if str(part).strip()]


def _finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _finite_median(values: Sequence[float]) -> float:
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


def _pipe_tokens(value: object) -> List[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _join_tokens(values: Sequence[object]) -> str:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        for token in _pipe_tokens(value):
            if token not in seen:
                seen.add(token)
                out.append(token)
    return "|".join(out)


def _prioritized_reason(tokens: Sequence[str]) -> str:
    token_set = {str(token) for token in tokens if str(token)}
    for reason in _DELTA_GATE_GUARD_REASON_PRIORITY:
        if reason in token_set:
            return reason
    return sorted(token_set)[0] if token_set else ""


def _delta_gate_guard_failure_reason(method: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if "delta_gated_sparse_mix" not in str(method):
        return ""
    statuses = _join_tokens([row.get("delta_gate_selection_status", "") for row in rows])
    reasons = _join_tokens([row.get("delta_gate_diagnostic_only_reason", "") for row in rows])
    status_tokens = _pipe_tokens(statuses)
    reason_tokens = _pipe_tokens(reasons)
    non_selected_statuses = [status for status in status_tokens if status != "selected"]
    if reason_tokens:
        return _prioritized_reason(reason_tokens)
    if non_selected_statuses:
        return _prioritized_reason(non_selected_statuses) or "delta_gate_source_inner_guard_failed"
    return ""


def _validate_sample_rows_for_aggregation(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        method = str(row.get("method", ""))
        adoption_eligible = int(float(row.get("adoption_eligible", 0) or 0))
        diagnostic_only = int(float(row.get("diagnostic_only", 0) or 0))
        uses_eval_nelbo = int(float(row.get("routing_uses_eval_nelbo", 0) or 0))
        uses_eval_stats = int(float(row.get("routing_uses_eval_domain_statistics", 0) or 0))
        if method == "oracle_routing":
            raise ProtocolError("oracle_routing must not be emitted under learned utility LOQDO v2")
        if method in {"candidate_oracle_routing", "support_candidate_oracle"} and adoption_eligible == 1:
            raise ProtocolError(f"{method} must not be adoption eligible")
        if adoption_eligible == 1 and (diagnostic_only != 0 or uses_eval_nelbo != 0 or uses_eval_stats != 0):
            raise ProtocolError(f"Invalid adoption eligibility flags for method={method}")

        candidate_experts = set(_parse_candidate_experts_label(row.get("candidate_experts", "")))
        selected_expert = int(row.get("selected_expert", -10**9))
        candidate_oracle_expert = int(row.get("candidate_oracle_expert", row.get("oracle_expert", -10**9)))
        fold_query_domain = int(row.get("fold_query_domain", row.get("query_domain", -10**9)))
        if selected_expert not in candidate_experts:
            raise ProtocolError(f"selected_expert={selected_expert} is outside candidate_experts for {method}")
        if candidate_oracle_expert not in candidate_experts:
            raise ProtocolError(
                f"candidate_oracle_expert={candidate_oracle_expert} is outside candidate_experts for {method}"
            )
        if fold_query_domain in candidate_experts:
            raise ProtocolError(f"fold_query_domain={fold_query_domain} appears in candidate_experts for {method}")


def _aggregate_metrics_from_sample_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    _validate_sample_rows_for_aggregation(rows)
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    out: Dict[str, Dict[str, float]] = {}
    metric_cols = {
        "top1_oracle_hit": "top1_oracle_hit",
        "selected_rank": "selected_rank",
        "oracle_gap": "oracle_gap",
        "oracle_gap_pct": "oracle_gap_pct",
        "spearman": "spearman",
        "pairwise_auc": "pairwise_auc",
        "selected_nelbo": "selected_nelbo",
        "candidate_oracle_nelbo": "candidate_oracle_nelbo",
        "bottom_half_selection": "bottom_half_selection",
        "high_regret_selection": "high_regret_selection",
        "catastrophic_mistake": "catastrophic_mistake",
        "oracle_in_route_set": "oracle_in_route_set",
        "sparse_mix_active": "sparse_mix_active",
        "mean_nelbo_spread_in_route_set": "mean_nelbo_spread_in_route_set",
        "route_set_regret": "route_set_regret",
        "fallback_delta": "fallback_delta",
        "fallback_help": "fallback_help",
        "fallback_harm": "fallback_harm",
        "tournament_margin": "tournament_margin",
        "delta_gate_active": "delta_gate_active",
        "fallback_delta_pct_raw": "fallback_delta_pct_raw",
        "fallback_delta_pct_clipped_for_training": "fallback_delta_pct_clipped_for_training",
        "predicted_fallback_delta_pct": "predicted_fallback_delta_pct",
        "hard_oracle_gap_pct": "hard_oracle_gap_pct",
        "hard_high_regret_selection": "hard_high_regret_selection",
        "pairprob_win_top1": "pairprob_win_top1",
        "top1_win_margin": "top1_win_margin",
        "absolute_high_regret_gap_gt_5": "absolute_high_regret_gap_gt_5",
        "relative_catastrophic_regression_vs_hard_gt_5": "relative_catastrophic_regression_vs_hard_gt_5",
        "pairwise_cycle_rate": "pairwise_cycle_rate",
        "mean_pairwise_confidence": "mean_pairwise_confidence",
        "pairwise_calibration_brier": "pairwise_calibration_brier",
        "pairwise_auc_helpful_preferences": "pairwise_auc_helpful_preferences",
        "conformal_set_size": "conformal_set_size",
        "oracle_in_conformal_set": "oracle_in_conformal_set",
        "primary_near_oracle_in_conformal_set": "primary_near_oracle_in_conformal_set",
        "conformal_quantile_clipped": "conformal_quantile_clipped",
        "regret_set_override_active": "regret_set_override_active",
        "override_delta_gap_pct_vs_pairprob_top1": "override_delta_gap_pct_vs_pairprob_top1",
        "paired_gap_delta_vs_pairprob_hard": "paired_gap_delta_vs_pairprob_hard",
        "paired_gap_delta_vs_metadata": "paired_gap_delta_vs_metadata",
        "relative_catastrophic_regression_vs_pairprob_hard_gt_5": (
            "relative_catastrophic_regression_vs_pairprob_hard_gt_5"
        ),
        "jackknife_mean_win_selected": "jackknife_mean_win_selected",
        "jackknife_std_win_selected": "jackknife_std_win_selected",
        "jackknife_std_pairprob_hard_selected": "jackknife_std_pairprob_hard_selected",
        "jackknife_std_selected_rank": "jackknife_std_selected_rank",
        "jackknife_mean_win_margin_top1_top2": "jackknife_mean_win_margin_top1_top2",
        "jackknife_std_winner_minus_runnerup": "jackknife_std_winner_minus_runnerup",
        "jackknife_lcb_margin_top1_top2": "jackknife_lcb_margin_top1_top2",
        "jackknife_override_active": "jackknife_override_active",
        "jackknife_mean_vs_pairprob_hard_selection_change": (
            "jackknife_mean_vs_pairprob_hard_selection_change"
        ),
        "mean_ensemble_override_vs_pairprob_hard": "mean_ensemble_override_vs_pairprob_hard",
        "lcb_override_vs_jackknife_mean": "lcb_override_vs_jackknife_mean",
        "lcb_override_vs_pairprob_hard": "lcb_override_vs_pairprob_hard",
        "pairprob_top1_error": "pairprob_top1_error",
        "pairprob_high_regret_error": "pairprob_high_regret_error",
        "base_direct_top2_margin": "base_direct_top2_margin",
        "top2_rerank_active": "top2_rerank_active",
        "top2_rerank_switched": "top2_rerank_switched",
        "top2_rerank_keep_top1_prob": "top2_rerank_keep_top1_prob",
        "top2_rerank_delta_gap_pct_vs_direct": "top2_rerank_delta_gap_pct_vs_direct",
        "top2_rerank_help": "top2_rerank_help",
        "top2_rerank_harm": "top2_rerank_harm",
        "top2_rerank_candidate_delta_gap_pct_vs_direct": (
            "top2_rerank_candidate_delta_gap_pct_vs_direct"
        ),
        "top2_rerank_top2_better_than_base": "top2_rerank_top2_better_than_base",
        "top2_delta_gate_active": "top2_delta_gate_active",
        "top2_delta_gate_switched": "top2_delta_gate_switched",
        "top2_delta_gate_delta_gap_pct_vs_direct": "top2_delta_gate_delta_gap_pct_vs_direct",
        "top2_delta_gate_help": "top2_delta_gate_help",
        "top2_delta_gate_harm": "top2_delta_gate_harm",
        "top2_delta_gate_predicted_delta_gap_pct": "top2_delta_gate_predicted_delta_gap_pct",
        "top2_delta_gate_true_delta_gap_pct_top2_vs_top1": (
            "top2_delta_gate_true_delta_gap_pct_top2_vs_top1"
        ),
        "allpair_delta_gate_active": "allpair_delta_gate_active",
        "allpair_delta_gate_switched": "allpair_delta_gate_switched",
        "allpair_delta_gate_delta_gap_pct_vs_direct": "allpair_delta_gate_delta_gap_pct_vs_direct",
        "allpair_delta_gate_help": "allpair_delta_gate_help",
        "allpair_delta_gate_harm": "allpair_delta_gate_harm",
        "allpair_delta_gate_predicted_delta_gap_pct": "allpair_delta_gate_predicted_delta_gap_pct",
        "allpair_delta_gate_true_delta_gap_pct_top2_vs_top1": (
            "allpair_delta_gate_true_delta_gap_pct_top2_vs_top1"
        ),
        "heldout_mean_gap_delta_vs_direct": "heldout_mean_gap_delta_vs_direct",
        "heldout_high_regret_delta_vs_direct": "heldout_high_regret_delta_vs_direct",
        "heldout_top1_delta_vs_direct": "heldout_top1_delta_vs_direct",
        "heldout_spearman_delta_vs_direct": "heldout_spearman_delta_vs_direct",
        "heldout_mean_gap_delta_vs_metadata": "heldout_mean_gap_delta_vs_metadata",
        "boosted_selection_changed": "boosted_selection_changed",
        "boosted_to_base_top2": "boosted_to_base_top2",
        "boosted_delta_gap_pct_vs_direct_pairprob": "boosted_delta_gap_pct_vs_direct_pairprob",
        "boosted_help": "boosted_help",
        "boosted_harm": "boosted_harm",
        "hardpair_weighted_pair_fraction": "hardpair_weighted_pair_fraction",
        "hardpair_mean_pair_weight": "hardpair_mean_pair_weight",
    }
    for method, vals in sorted(by_method.items()):
        metrics: Dict[str, float] = {}
        for out_name, col in metric_cols.items():
            micro = _finite_mean([float(r.get(col, float("nan"))) for r in vals])
            metrics[f"micro_{out_name}"] = micro
            by_domain: Dict[int, List[float]] = {}
            for row in vals:
                by_domain.setdefault(int(row["query_domain"]), []).append(float(row.get(col, float("nan"))))
            domain_means = [_finite_mean(domain_vals) for domain_vals in by_domain.values()]
            metrics[f"macro_{out_name}_by_query_domain"] = _finite_mean(domain_means)

        # Backward-compatible aliases intentionally point at sample-micro values.
        metrics["top1_oracle_hit"] = metrics["micro_top1_oracle_hit"]
        metrics["mean_rank"] = metrics["micro_selected_rank"]
        metrics["mean_oracle_gap"] = metrics["micro_oracle_gap"]
        metrics["mean_oracle_gap_pct"] = metrics["micro_oracle_gap_pct"]
        oracle_gap_domain_means: List[float] = []
        for domain in sorted(set(int(r["query_domain"]) for r in vals)):
            oracle_gap_domain_means.append(
                _finite_mean([float(r.get("oracle_gap_pct", float("nan"))) for r in vals if int(r["query_domain"]) == domain])
            )
        metrics["worst_heldout_domain_oracle_gap_pct"] = (
            float(max(oracle_gap_domain_means)) if oracle_gap_domain_means else float("nan")
        )
        metrics["spearman"] = metrics["micro_spearman"]
        metrics["pairwise_auc"] = metrics["micro_pairwise_auc"]
        metrics["selected_nelbo"] = metrics["micro_selected_nelbo"]
        metrics["oracle_nelbo"] = metrics["micro_candidate_oracle_nelbo"]
        metrics["candidate_oracle_nelbo"] = metrics["micro_candidate_oracle_nelbo"]
        metrics["bottom_half_selection_rate"] = metrics["micro_bottom_half_selection"]
        metrics["high_regret_selection_rate"] = metrics["micro_high_regret_selection"]
        metrics["catastrophic_mistake_rate"] = metrics["micro_catastrophic_mistake"]
        metrics["oracle_in_route_set"] = metrics["micro_oracle_in_route_set"]
        metrics["sparse_mix_active_rate"] = metrics["micro_sparse_mix_active"]
        metrics["mean_nelbo_spread_in_route_set"] = metrics["micro_mean_nelbo_spread_in_route_set"]
        metrics["route_set_regret"] = metrics["micro_route_set_regret"]
        metrics["fallback_delta"] = metrics["micro_fallback_delta"]
        metrics["fallback_help_rate"] = metrics["micro_fallback_help"]
        metrics["fallback_harm_rate"] = metrics["micro_fallback_harm"]
        metrics["mean_tournament_margin"] = metrics["micro_tournament_margin"]
        metrics["delta_gate_active_rate"] = metrics["micro_delta_gate_active"]
        metrics["mean_fallback_delta_pct_raw"] = metrics["micro_fallback_delta_pct_raw"]
        metrics["mean_predicted_fallback_delta_pct"] = metrics["micro_predicted_fallback_delta_pct"]
        metrics["heldout_paired_gap_reduction_vs_hard"] = float(
            metrics["micro_hard_oracle_gap_pct"] - metrics["micro_oracle_gap_pct"]
        ) if np.isfinite(metrics["micro_hard_oracle_gap_pct"]) else float("nan")
        metrics["heldout_paired_high_regret_reduction_vs_hard"] = float(
            metrics["micro_hard_high_regret_selection"] - metrics["micro_high_regret_selection"]
        ) if np.isfinite(metrics["micro_hard_high_regret_selection"]) else float("nan")
        metrics["absolute_high_regret_rate_gap_gt_5"] = metrics["micro_absolute_high_regret_gap_gt_5"]
        metrics["relative_catastrophic_regression_vs_hard_gt_5_rate"] = (
            metrics["micro_relative_catastrophic_regression_vs_hard_gt_5"]
        )
        metrics["mean_pairprob_win_top1"] = metrics["micro_pairprob_win_top1"]
        metrics["top1_win_margin"] = metrics["micro_top1_win_margin"]
        metrics["pairwise_cycle_rate"] = metrics["micro_pairwise_cycle_rate"]
        metrics["mean_pairwise_confidence"] = metrics["micro_mean_pairwise_confidence"]
        metrics["pairwise_calibration_brier"] = metrics["micro_pairwise_calibration_brier"]
        metrics["pairwise_auc_helpful_preferences"] = metrics["micro_pairwise_auc_helpful_preferences"]
        metrics["mean_conformal_set_size"] = metrics["micro_conformal_set_size"]
        metrics["set_size_gt1_rate"] = float(
            np.mean([1.0 if float(r.get("conformal_set_size", 0.0)) > 1.0 else 0.0 for r in vals])
        ) if any("conformal_set_size" in r for r in vals) else float("nan")
        metrics["set_size_gt3_rate"] = float(
            np.mean([1.0 if float(r.get("conformal_set_size", 0.0)) > 3.0 else 0.0 for r in vals])
        ) if any("conformal_set_size" in r for r in vals) else float("nan")
        metrics["oracle_in_conformal_set_rate"] = metrics["micro_oracle_in_conformal_set"]
        metrics["primary_near_oracle_in_conformal_set_rate"] = (
            metrics["micro_primary_near_oracle_in_conformal_set"]
        )
        metrics["quantile_clipped_rate"] = metrics["micro_conformal_quantile_clipped"]
        metrics["regret_set_override_rate"] = metrics["micro_regret_set_override_active"]
        override_rows = [
            r for r in vals
            if int(float(r.get("regret_set_override_active", 0) or 0)) == 1
        ]
        metrics["regret_set_override_help_rate"] = _finite_mean(
            [
                1.0 if float(r.get("override_delta_gap_pct_vs_pairprob_top1", float("nan"))) < 0.0 else 0.0
                for r in override_rows
            ]
        )
        metrics["regret_set_override_harm_rate"] = _finite_mean(
            [
                1.0 if float(r.get("override_delta_gap_pct_vs_pairprob_top1", float("nan"))) > 0.0 else 0.0
                for r in override_rows
            ]
        )
        metrics["mean_override_delta_gap_pct"] = _finite_mean(
            [float(r.get("override_delta_gap_pct_vs_pairprob_top1", float("nan"))) for r in override_rows]
        )
        metrics["mean_paired_gap_delta_vs_pairprob_hard"] = metrics["micro_paired_gap_delta_vs_pairprob_hard"]
        metrics["median_paired_gap_delta_vs_pairprob_hard"] = _finite_median(
            [float(r.get("paired_gap_delta_vs_pairprob_hard", float("nan"))) for r in vals]
        )
        metrics["paired_improvement_rate_vs_pairprob_hard"] = float(
            np.mean([1.0 if float(r.get("paired_gap_delta_vs_pairprob_hard", float("nan"))) < 0.0 else 0.0 for r in vals])
        ) if any("paired_gap_delta_vs_pairprob_hard" in r for r in vals) else float("nan")
        metrics["mean_paired_gap_delta_vs_metadata"] = metrics["micro_paired_gap_delta_vs_metadata"]
        metrics["median_paired_gap_delta_vs_metadata"] = _finite_median(
            [float(r.get("paired_gap_delta_vs_metadata", float("nan"))) for r in vals]
        )
        metrics["paired_improvement_rate_vs_metadata"] = float(
            np.mean([1.0 if float(r.get("paired_gap_delta_vs_metadata", float("nan"))) < 0.0 else 0.0 for r in vals])
        ) if any("paired_gap_delta_vs_metadata" in r for r in vals) else float("nan")
        metrics["relative_catastrophic_regression_vs_pairprob_hard_gt_5_rate"] = (
            metrics["micro_relative_catastrophic_regression_vs_pairprob_hard_gt_5"]
        )
        metrics["jackknife_mean_win_selected"] = metrics["micro_jackknife_mean_win_selected"]
        metrics["jackknife_std_win_selected"] = metrics["micro_jackknife_std_win_selected"]
        metrics["jackknife_mean_win_margin_top1_top2"] = metrics[
            "micro_jackknife_mean_win_margin_top1_top2"
        ]
        metrics["jackknife_lcb_margin_top1_top2"] = metrics["micro_jackknife_lcb_margin_top1_top2"]
        metrics["jackknife_mean_vs_pairprob_hard_selection_change_rate"] = metrics[
            "micro_jackknife_mean_vs_pairprob_hard_selection_change"
        ]
        metrics["mean_ensemble_override_rate_vs_pairprob_hard"] = metrics[
            "micro_mean_ensemble_override_vs_pairprob_hard"
        ]
        metrics["lcb_override_rate_vs_jackknife_mean"] = metrics["micro_lcb_override_vs_jackknife_mean"]
        metrics["lcb_override_rate_vs_pairprob_hard"] = metrics["micro_lcb_override_vs_pairprob_hard"]
        metrics["jackknife_override_rate"] = metrics["micro_jackknife_override_active"]
        jackknife_override_rows = [
            r for r in vals
            if int(float(r.get("jackknife_override_active", 0) or 0)) == 1
        ]
        jackknife_override_delta = [
            float(r.get("paired_gap_delta_vs_pairprob_hard", float("nan")))
            for r in jackknife_override_rows
        ]
        metrics["jackknife_override_help_rate"] = _finite_mean(
            [1.0 if float(v) < 0.0 else 0.0 for v in jackknife_override_delta]
        )
        metrics["jackknife_override_harm_rate"] = _finite_mean(
            [1.0 if float(v) > 0.0 else 0.0 for v in jackknife_override_delta]
        )
        metrics["total_override_help_gap_reduction"] = float(
            np.sum([abs(float(v)) for v in jackknife_override_delta if np.isfinite(float(v)) and float(v) < 0.0])
        )
        metrics["total_override_harm_gap_increase"] = float(
            np.sum([float(v) for v in jackknife_override_delta if np.isfinite(float(v)) and float(v) > 0.0])
        )
        metrics["jackknife_uncertainty_auc_for_pairprob_top1_error"] = _binary_auc(
            [float(r.get("jackknife_std_pairprob_hard_selected", float("nan"))) for r in vals],
            [int(float(r.get("pairprob_top1_error", 0) or 0)) for r in vals],
        )
        metrics["jackknife_uncertainty_auc_for_pairprob_high_regret"] = _binary_auc(
            [float(r.get("jackknife_std_pairprob_hard_selected", float("nan"))) for r in vals],
            [int(float(r.get("pairprob_high_regret_error", 0) or 0)) for r in vals],
        )
        metrics["uncertainty_error_spearman_outer_eval"] = _finite_spearman(
            [float(r.get("jackknife_std_pairprob_hard_selected", float("nan"))) for r in vals],
            [float(r.get("pairprob_hard_oracle_gap_pct", float("nan"))) for r in vals],
        )
        metrics["top2_rerank_activation_rate"] = metrics["micro_top2_rerank_active"]
        metrics["top2_rerank_switch_rate"] = metrics["micro_top2_rerank_switched"]
        top2_active_rows = [
            r for r in vals
            if int(float(r.get("top2_rerank_active", 0) or 0)) == 1
        ]
        metrics["top2_rerank_help_rate_active_only"] = _finite_mean(
            [float(r.get("top2_rerank_help", float("nan"))) for r in top2_active_rows]
        )
        metrics["top2_rerank_harm_rate_active_only"] = _finite_mean(
            [float(r.get("top2_rerank_harm", float("nan"))) for r in top2_active_rows]
        )
        metrics["mean_top2_rerank_delta_gap_pct_vs_direct"] = (
            metrics["micro_top2_rerank_delta_gap_pct_vs_direct"]
        )
        metrics["median_top2_rerank_delta_gap_pct_vs_direct"] = _finite_median(
            [float(r.get("top2_rerank_delta_gap_pct_vs_direct", float("nan"))) for r in vals]
        )
        metrics["paired_improvement_rate_vs_direct_pairprob"] = float(
            np.mean(
                [
                    1.0 if float(r.get("top2_rerank_delta_gap_pct_vs_direct", float("nan"))) < 0.0 else 0.0
                    for r in vals
                    if "top2_rerank_delta_gap_pct_vs_direct" in r
                ]
            )
        ) if any("top2_rerank_delta_gap_pct_vs_direct" in r for r in vals) else float("nan")
        metrics["top2_delta_gate_activation_rate"] = metrics["micro_top2_delta_gate_active"]
        metrics["top2_delta_gate_switch_rate"] = metrics["micro_top2_delta_gate_switched"]
        top2_delta_switched = [
            r for r in vals
            if int(float(r.get("top2_delta_gate_switched", 0) or 0)) == 1
        ]
        metrics["top2_delta_gate_help_rate_changed_only"] = _finite_mean(
            [float(r.get("top2_delta_gate_help", float("nan"))) for r in top2_delta_switched]
        )
        metrics["top2_delta_gate_harm_rate_changed_only"] = _finite_mean(
            [float(r.get("top2_delta_gate_harm", float("nan"))) for r in top2_delta_switched]
        )
        metrics["mean_top2_delta_gate_delta_gap_pct_vs_direct"] = metrics[
            "micro_top2_delta_gate_delta_gap_pct_vs_direct"
        ]
        metrics["median_top2_delta_gate_delta_gap_pct_vs_direct"] = _finite_median(
            [float(r.get("top2_delta_gate_delta_gap_pct_vs_direct", float("nan"))) for r in vals]
        )
        metrics["allpair_delta_gate_activation_rate"] = metrics["micro_allpair_delta_gate_active"]
        metrics["allpair_delta_gate_switch_rate"] = metrics["micro_allpair_delta_gate_switched"]
        allpair_delta_switched = [
            r for r in vals
            if int(float(r.get("allpair_delta_gate_switched", 0) or 0)) == 1
        ]
        metrics["allpair_delta_gate_help_rate_changed_only"] = _finite_mean(
            [float(r.get("allpair_delta_gate_help", float("nan"))) for r in allpair_delta_switched]
        )
        metrics["allpair_delta_gate_harm_rate_changed_only"] = _finite_mean(
            [float(r.get("allpair_delta_gate_harm", float("nan"))) for r in allpair_delta_switched]
        )
        metrics["mean_allpair_delta_gate_delta_gap_pct_vs_direct"] = metrics[
            "micro_allpair_delta_gate_delta_gap_pct_vs_direct"
        ]
        metrics["median_allpair_delta_gate_delta_gap_pct_vs_direct"] = _finite_median(
            [float(r.get("allpair_delta_gate_delta_gap_pct_vs_direct", float("nan"))) for r in vals]
        )
        metrics["heldout_mean_gap_delta_vs_direct"] = metrics["micro_heldout_mean_gap_delta_vs_direct"]
        metrics["heldout_high_regret_delta_vs_direct"] = metrics["micro_heldout_high_regret_delta_vs_direct"]
        metrics["heldout_top1_delta_vs_direct"] = metrics["micro_heldout_top1_delta_vs_direct"]
        metrics["heldout_spearman_delta_vs_direct"] = metrics["micro_heldout_spearman_delta_vs_direct"]
        metrics["heldout_mean_gap_delta_vs_metadata"] = metrics["micro_heldout_mean_gap_delta_vs_metadata"]
        metrics["boosted_selection_change_rate"] = metrics["micro_boosted_selection_changed"]
        metrics["boosted_to_base_top2_rate"] = metrics["micro_boosted_to_base_top2"]
        boosted_changed_rows = [
            r for r in vals
            if int(float(r.get("boosted_selection_changed", 0) or 0)) == 1
        ]
        metrics["boosted_help_rate_changed_only"] = _finite_mean(
            [float(r.get("boosted_help", float("nan"))) for r in boosted_changed_rows]
        )
        metrics["boosted_harm_rate_changed_only"] = _finite_mean(
            [float(r.get("boosted_harm", float("nan"))) for r in boosted_changed_rows]
        )
        metrics["mean_paired_gap_delta_vs_direct_pairprob"] = metrics[
            "micro_boosted_delta_gap_pct_vs_direct_pairprob"
        ]
        metrics["hardpair_weighted_pair_fraction"] = metrics["micro_hardpair_weighted_pair_fraction"]
        metrics["hardpair_mean_pair_weight"] = metrics["micro_hardpair_mean_pair_weight"]

        if any("delta_gate_active" in r for r in vals):
            active_rows = [
                r for r in vals
                if int(float(r.get("delta_gate_active", 0) or 0)) == 1
            ]
            metrics["fallback_help_rate_active_only"] = _finite_mean(
                [float(r.get("fallback_help", float("nan"))) for r in active_rows]
            )
            metrics["fallback_harm_rate_active_only"] = _finite_mean(
                [float(r.get("fallback_harm", float("nan"))) for r in active_rows]
            )
            metrics["fallback_help_rate_all_rows"] = float(
                np.mean(
                    [
                        1.0
                        if int(float(r.get("delta_gate_active", 0) or 0)) == 1
                        and float(r.get("fallback_delta_pct_raw", float("nan"))) < 0.0
                        else 0.0
                        for r in vals
                    ]
                )
            )
            metrics["fallback_harm_rate_all_rows"] = float(
                np.mean(
                    [
                        1.0
                        if int(float(r.get("delta_gate_active", 0) or 0)) == 1
                        and float(r.get("fallback_delta_pct_raw", float("nan"))) > 0.0
                        else 0.0
                        for r in vals
                    ]
                )
            )
            metrics["mean_fallback_delta_pct_when_active"] = _finite_mean(
                [float(r.get("fallback_delta_pct_raw", float("nan"))) for r in active_rows]
            )
            metrics["median_fallback_delta_pct_when_active"] = _finite_median(
                [float(r.get("fallback_delta_pct_raw", float("nan"))) for r in active_rows]
            )

        correct_margins = [
            float(r.get("tournament_margin", float("nan")))
            for r in vals
            if int(float(r.get("top1_oracle_hit", 0) or 0)) == 1
        ]
        wrong_margins = [
            float(r.get("tournament_margin", float("nan")))
            for r in vals
            if int(float(r.get("top1_oracle_hit", 0) or 0)) == 0
        ]
        metrics["mean_margin_when_top1_correct"] = _finite_mean(correct_margins)
        metrics["mean_margin_when_top1_wrong"] = _finite_mean(wrong_margins)
        margins_for_auc = [
            (
                float(r.get("tournament_margin", float("nan"))),
                int(float(r.get("top1_oracle_hit", 0) or 0)),
            )
            for r in vals
            if np.isfinite(float(r.get("tournament_margin", float("nan"))))
        ]
        positives = [m for m, y in margins_for_auc if y == 1]
        negatives = [m for m, y in margins_for_auc if y == 0]
        if positives and negatives:
            total = 0.0
            correct = 0.0
            for p in positives:
                for n in negatives:
                    total += 1.0
                    if p > n:
                        correct += 1.0
                    elif abs(p - n) < 1e-12:
                        correct += 0.5
            metrics["margin_auc_for_oracle_hit"] = float(correct / total)
        else:
            metrics["margin_auc_for_oracle_hit"] = float("nan")

        query_domains = sorted(set(int(r["query_domain"]) for r in vals))
        metrics["n_samples_micro"] = float(len(vals))
        metrics["n_query_domains_macro"] = float(len(query_domains))
        metrics["n_valid_spearman_samples"] = float(
            sum(1 for r in vals if np.isfinite(float(r.get("spearman", float("nan")))))
        )
        metrics["n_valid_auc_samples"] = float(
            sum(1 for r in vals if np.isfinite(float(r.get("pairwise_auc", float("nan")))))
        )
        method_protocol = _method_protocol(method)
        first = vals[0]
        metrics["protocol_version"] = str(first.get("protocol_version", _PROTOCOL_VERSION))
        metrics["method_role"] = str(first.get("method_role", method_protocol.method_role))
        metrics["adoption_eligible"] = float(first.get("adoption_eligible", method_protocol.adoption_eligible))
        metrics["diagnostic_only"] = float(first.get("diagnostic_only", method_protocol.diagnostic_only))
        metrics["routing_uses_query_features"] = float(
            first.get("routing_uses_query_features", method_protocol.routing_uses_query_features)
        )
        metrics["routing_uses_eval_nelbo"] = float(
            first.get("routing_uses_eval_nelbo", method_protocol.routing_uses_eval_nelbo)
        )
        metrics["routing_uses_eval_domain_statistics"] = float(
            first.get(
                "routing_uses_eval_domain_statistics",
                method_protocol.routing_uses_eval_domain_statistics,
            )
        )
        for key in [
            "decision_policy_version",
            "residual_policy_version",
            "threshold_selection_policy",
            "feature_set",
            "residual_variant",
            "selected_tau",
            "tau_margin",
            "tau_regret",
            "alpha",
            "alpha_grid",
            "alpha_selection_policy",
            "selection_source",
            "policy_name",
            "adoption_selected_method",
            "harmful_override_max",
            "allow_calibrated_adoption",
            "fallback_used",
            "fallback_to_alpha0",
            "n_aggregation_units",
            "top1_tolerance_abs",
            "base_method",
            "sparse_mix_topk",
            "score_temperature",
            "temperature_policy",
            "route_mode",
            "diagnostic_only_reason",
            "source_inner_rows",
            "source_inner_gap_pct",
            "source_inner_high_regret_rate",
            "source_inner_oracle_in_route_set",
            "source_inner_top1",
            "source_inner_sparse_mix_rate",
            "delta_gate_selection_status",
            "delta_gate_threshold",
            "delta_gate_feature_set",
            "delta_gate_source_inner_gap_pct",
            "delta_gate_source_inner_paired_gap_reduction_vs_hard",
            "delta_gate_source_inner_activation_rate",
            "delta_gate_source_inner_harm_rate_active_only",
            "delta_gate_source_inner_help_rate_active_only",
            "delta_gate_spearman_pred_vs_true_delta_source_inner",
            "delta_gate_auc_help_vs_harm_source_inner",
            "delta_gate_diagnostic_only_reason",
            "pairprob_predictor",
            "pairprob_probability_calibration",
            "pairprob_feature_set",
            "pairprob_selection_policy",
            "pairwise_near_tie_drop_rate",
            "pairwise_train_pairs_after_filter",
            "pairwise_validation_pairs_after_filter",
            "pairwise_train_domains_after_filter",
            "worst_inner_domain_oracle_gap_pct",
            "std_oracle_gap_pct_across_inner_domains",
            "std_top1_across_inner_domains",
            "max_minus_min_oracle_gap_pct_across_inner_domains",
            "conformal_alpha",
            "conformal_tau",
            "conformal_calibration_n",
            "conformal_quantile_k",
            "robust_lambda",
            "normalized_source_inner_worst_regret_selected",
            "adoption_feature_family",
            "direct_adoption_is_alias_of",
            "direct_adoption_same_route_as_direct",
            "direct_adoption_audit_failure_reason",
            "excluded_from_sign_ci_selection",
            "sign_ci_candidate",
            "source_only_audit_pass",
            "target_leakage_audit_pass",
            "direct_vs_group_robust_primary_comparator",
            "mean_gap_delta_vs_group_robust_pairprob",
            "worst_domain_gap_delta_vs_group_robust_pairprob",
            "high_regret_delta_vs_group_robust_pairprob",
            "top1_delta_vs_group_robust_pairprob",
            "spearman_delta_vs_group_robust_pairprob",
            "candidate_pool_consistent",
            "selected_lambda_is_zero_but_lcb_candidates_reported",
            "lambda_stability_status",
            "jackknife_lambda",
            "jackknife_n_models",
            "uncertainty_error_spearman_source_inner",
            "top2_rerank_threshold",
            "top2_rerank_l2",
            "top2_rerank_guard_status",
            "top2_rerank_diagnostic_only_reason",
            "top2_rerank_selection_stability_status",
            "source_inner_top2_rerank_gap_reduction_abs_pct_points",
            "source_inner_top2_rerank_high_regret_reduction",
            "source_inner_top2_rerank_rows",
            "source_inner_top2_rerank_positive_rows",
            "source_inner_top2_rerank_negative_rows",
            "source_inner_top2_rerank_active_domains",
            "source_inner_switch_candidate_rate",
            "reranker_selection_stability_status",
            "base_top2_margin_auc_for_high_regret",
            "base_top2_margin_spearman_with_oracle_gap",
            "overall_high_regret_rate_direct",
            "low_margin_active_high_regret_rate",
            "low_margin_high_regret_enrichment",
            "top2_rerank_auc_source_inner",
            "top2_rerank_brier_source_inner",
            "top2_rerank_calibration_status",
            "oracle_top2_active_gap_reduction_pct",
            "oracle_top2_active_high_regret_reduction",
            "oracle_top2_recoverable_error_rate",
            "oracle_top2_recoverable_gap_mass_pct_points",
            "top2_delta_gate_threshold",
            "top2_delta_gate_predicted_delta_threshold",
            "top2_delta_gate_l2",
            "top2_delta_gate_guard_status",
            "top2_delta_gate_diagnostic_only_reason",
            "top2_delta_gate_selection_stability_status",
            "source_inner_top2_delta_gate_gap_reduction_abs_pct_points",
            "source_inner_top2_delta_gate_high_regret_reduction",
            "source_inner_top2_delta_gate_rows",
            "source_inner_top2_delta_gate_switch_rows",
            "source_inner_top2_delta_gate_keep_rows",
            "source_inner_top2_delta_gate_helpful_switch_rows",
            "source_inner_top2_delta_gate_harmful_switch_rows",
            "source_inner_top2_delta_gate_active_domains",
            "delta_gate_spearman_pred_vs_true_source_inner",
            "delta_gate_auc_switch_help_source_inner",
            "delta_gate_mae_source_inner",
            "active_low_margin_oracle_is_top2_rate",
            "active_low_margin_oracle_in_top2_rate",
            "active_low_margin_high_regret_oracle_is_top2_rate",
            "selected_margin_threshold",
            "selected_predicted_delta_threshold",
            "selected_l2",
            "selected_guard_status",
            "selected_reason",
            "source_inner_selected_config_mean_gap",
            "source_inner_selected_config_high_regret_rate",
            "source_inner_selected_config_top1_delta_vs_direct",
            "source_inner_selected_config_spearman_delta_vs_direct",
            "source_inner_top2_candidate_rows",
            "source_inner_top2_switch_rows",
            "source_inner_top2_help_rate_changed_only",
            "source_inner_top2_harm_rate_changed_only",
            "source_inner_top2_mean_delta_vs_direct",
            "source_inner_top2_high_regret_delta_vs_direct",
            "source_inner_top2_top1_delta_vs_direct",
            "source_inner_top2_spearman_delta_vs_direct",
            "source_inner_allpair_delta_rows",
            "source_inner_allpair_unique_queries",
            "source_inner_allpair_unique_query_domains",
            "source_inner_allpair_unique_base_top2_events",
            "source_inner_allpair_helpful_pair_rows",
            "source_inner_allpair_harmful_pair_rows",
            "allpair_delta_spearman_pred_vs_true_source_inner",
            "allpair_delta_auc_switch_help_source_inner",
            "allpair_delta_mae_source_inner",
            "allpair_delta_gate_guard_status",
            "allpair_delta_gate_diagnostic_only_reason",
            "hardpair_boost_margin_threshold",
            "hardpair_miss_boost_weight",
            "hardpair_confirm_boost_weight",
            "hardpair_boost_guard_status",
            "hardpair_boost_diagnostic_only_reason",
            "group_oof_grouping_level",
            "group_oof_grouping_warning",
            "group_oof_unique_groups",
            "group_oof_min_groups_per_fold",
            "group_oof_folds_used",
            "group_oof_train_domains_per_fold_min",
            "group_oof_candidate_experts_per_fold_min",
            "group_oof_same_slide_leakage_rate",
            "hardpair_oof_low_margin_rows",
            "hardpair_oof_switch_rows",
            "hardpair_oof_keep_rows",
            "hardpair_oof_active_domains",
            "low_margin_high_regret_rows",
            "low_margin_high_regret_oracle_in_base_top2_rate",
            "low_margin_high_regret_oracle_is_base_top2_rate",
            "low_margin_high_regret_oracle_is_not_base_top2_rate",
            "source_inner_boost_gap_reduction_abs_pct_points",
            "worst_source_inner_domain_regression_abs_pct_points",
            "median_source_inner_domain_delta_gap",
            "source_inner_domains_regressed_gt_threshold",
        ]:
            vals_for_key = sorted(set(str(r.get(key, "")) for r in vals if str(r.get(key, "")) != ""))
            if vals_for_key:
                metrics[key] = vals_for_key[0] if len(vals_for_key) == 1 else "|".join(vals_for_key)
        if any("selected_by_inner_validation" in r for r in vals):
            metrics["selected_by_inner_validation"] = float(
                max(int(float(r.get("selected_by_inner_validation", 0) or 0)) for r in vals)
            )
        if any(str(r.get("direct_adoption_route_hash", "")) for r in vals):
            joined = "|".join(str(r.get("direct_adoption_route_hash", "")) for r in vals)
            metrics["direct_adoption_route_hash"] = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
        if any(str(r.get("direct_diagnostic_route_hash", "")) for r in vals):
            joined = "|".join(str(r.get("direct_diagnostic_route_hash", "")) for r in vals)
            metrics["direct_diagnostic_route_hash"] = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
        guard_reason = _delta_gate_guard_failure_reason(method, vals)
        metrics["delta_gate_source_inner_guard_pass"] = float(0 if guard_reason else 1)
        if guard_reason:
            metrics["method_role"] = "diagnostic"
            metrics["adoption_eligible"] = 0.0
            metrics["diagnostic_only"] = 1.0
            existing_diagnostic_reason = str(metrics.get("diagnostic_only_reason", ""))
            existing_gate_reason = str(metrics.get("delta_gate_diagnostic_only_reason", ""))
            metrics["diagnostic_only_reason"] = _join_tokens(
                [existing_diagnostic_reason, guard_reason]
            )
            metrics["delta_gate_diagnostic_only_reason"] = _join_tokens(
                [existing_gate_reason, guard_reason]
            )
        out[method] = metrics
    return out


def _domain_breakdown_rows(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _validate_sample_rows_for_aggregation(sample_rows)
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in sample_rows:
        grouped.setdefault((str(row["method"]), int(row["query_domain"])), []).append(row)

    domain_rows: List[Dict[str, Any]] = []
    for (method, query_domain), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        base = rows[0]
        domain_rows.append(
            {
                "protocol_version": str(base.get("protocol_version", _PROTOCOL_VERSION)),
                "method": method,
                "query_domain": int(query_domain),
                "fold_query_domain": int(query_domain),
                "candidate_experts": str(base.get("candidate_experts", "")),
                "n_candidate_experts": int(base.get("n_candidate_experts", 0)),
                "target_expert_excluded": int(base.get("target_expert_excluded", 0)),
                "method_role": str(base.get("method_role", "")),
                "adoption_eligible": int(base.get("adoption_eligible", 0)),
                "diagnostic_only": int(base.get("diagnostic_only", 0)),
                "decision_policy_version": str(base.get("decision_policy_version", "")),
                "residual_policy_version": str(base.get("residual_policy_version", "")),
                "threshold_selection_policy": str(base.get("threshold_selection_policy", "")),
                "feature_set": str(base.get("feature_set", "")),
                "residual_variant": str(base.get("residual_variant", "")),
                "selected_tau": str(base.get("selected_tau", "")),
                "tau_margin": str(base.get("tau_margin", "")),
                "tau_regret": str(base.get("tau_regret", "")),
                "alpha": str(base.get("alpha", "")),
                "alpha_grid": str(base.get("alpha_grid", "")),
                "alpha_selection_policy": str(base.get("alpha_selection_policy", "")),
                "selection_source": str(base.get("selection_source", "")),
                "policy_name": str(base.get("policy_name", "")),
                "selected_by_inner_validation": int(base.get("selected_by_inner_validation", 0) or 0),
                "adoption_selected_method": str(base.get("adoption_selected_method", "")),
                "harmful_override_max": str(base.get("harmful_override_max", "")),
                "allow_calibrated_adoption": str(base.get("allow_calibrated_adoption", "")),
                "fallback_used": str(base.get("fallback_used", "")),
                "fallback_to_alpha0": str(base.get("fallback_to_alpha0", "")),
                "n_aggregation_units": str(base.get("n_aggregation_units", "")),
                "top1_tolerance_abs": str(base.get("top1_tolerance_abs", "")),
                "n_samples": int(len(rows)),
                "top1_oracle_hit": _finite_mean([float(r["top1_oracle_hit"]) for r in rows]),
                "mean_rank": _finite_mean([float(r["selected_rank"]) for r in rows]),
                "mean_oracle_gap": _finite_mean([float(r["oracle_gap"]) for r in rows]),
                "mean_oracle_gap_pct": _finite_mean([float(r["oracle_gap_pct"]) for r in rows]),
                "pairwise_auc": _finite_mean([float(r["pairwise_auc"]) for r in rows]),
                "spearman": _finite_mean([float(r["spearman"]) for r in rows]),
                "bottom_half_selection_rate": _finite_mean(
                    [float(r.get("bottom_half_selection", 0.0)) for r in rows]
                ),
                "high_regret_selection_rate": _finite_mean(
                    [float(r.get("high_regret_selection", 0.0)) for r in rows]
                ),
                "catastrophic_mistake_rate": _finite_mean(
                    [float(r.get("catastrophic_mistake", 0.0)) for r in rows]
                ),
                "oracle_in_route_set": _finite_mean(
                    [float(r.get("oracle_in_route_set", float("nan"))) for r in rows]
                ),
                "sparse_mix_active_rate": _finite_mean(
                    [float(r.get("sparse_mix_active", float("nan"))) for r in rows]
                ),
                "fallback_help_rate": _finite_mean(
                    [float(r.get("fallback_help", float("nan"))) for r in rows]
                ),
                "fallback_harm_rate": _finite_mean(
                    [float(r.get("fallback_harm", float("nan"))) for r in rows]
                ),
                "mean_tournament_margin": _finite_mean(
                    [float(r.get("tournament_margin", float("nan"))) for r in rows]
                ),
                "pairprob_win_top1": _finite_mean(
                    [float(r.get("pairprob_win_top1", float("nan"))) for r in rows]
                ),
                "top1_win_margin": _finite_mean(
                    [float(r.get("top1_win_margin", float("nan"))) for r in rows]
                ),
                "absolute_high_regret_rate_gap_gt_5": _finite_mean(
                    [float(r.get("absolute_high_regret_gap_gt_5", float("nan"))) for r in rows]
                ),
                "relative_catastrophic_regression_vs_hard_gt_5_rate": _finite_mean(
                    [float(r.get("relative_catastrophic_regression_vs_hard_gt_5", float("nan"))) for r in rows]
                ),
                "pairwise_cycle_rate": _finite_mean(
                    [float(r.get("pairwise_cycle_rate", float("nan"))) for r in rows]
                ),
                "mean_pairwise_confidence": _finite_mean(
                    [float(r.get("mean_pairwise_confidence", float("nan"))) for r in rows]
                ),
                "pairwise_calibration_brier": _finite_mean(
                    [float(r.get("pairwise_calibration_brier", float("nan"))) for r in rows]
                ),
                "pairwise_auc_helpful_preferences": _finite_mean(
                    [float(r.get("pairwise_auc_helpful_preferences", float("nan"))) for r in rows]
                ),
                "delta_gate_selection_status": str(base.get("delta_gate_selection_status", "")),
                "delta_gate_threshold": str(base.get("delta_gate_threshold", "")),
                "delta_gate_feature_set": str(base.get("delta_gate_feature_set", "")),
                "delta_gate_diagnostic_only_reason": str(base.get("delta_gate_diagnostic_only_reason", "")),
                "delta_gate_active_rate": _finite_mean(
                    [float(r.get("delta_gate_active", float("nan"))) for r in rows]
                ),
                "fallback_help_rate_active_only": _finite_mean(
                    [
                        float(r.get("fallback_help", float("nan")))
                        for r in rows
                        if int(float(r.get("delta_gate_active", 0) or 0)) == 1
                    ]
                ),
                "fallback_harm_rate_active_only": _finite_mean(
                    [
                        float(r.get("fallback_harm", float("nan")))
                        for r in rows
                        if int(float(r.get("delta_gate_active", 0) or 0)) == 1
                    ]
                ),
                "heldout_paired_gap_reduction_vs_hard": _finite_mean(
                    [
                        float(r.get("hard_oracle_gap_pct", float("nan"))) - float(r.get("oracle_gap_pct", float("nan")))
                        for r in rows
                    ]
                ),
                "heldout_paired_high_regret_reduction_vs_hard": _finite_mean(
                    [
                        float(r.get("hard_high_regret_selection", float("nan")))
                        - float(r.get("high_regret_selection", float("nan")))
                        for r in rows
                    ]
                ),
            }
        )
    return domain_rows
