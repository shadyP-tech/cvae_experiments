from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


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
    if name in {"metadata_residual_thresholded", "metadata_residual_group_robust", "metadata_residual_inner_selected"}:
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


def _parse_candidate_experts_label(value: object) -> List[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split("|") if str(part).strip()]


def _finite_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _validate_sample_rows_for_aggregation(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        method = str(row.get("method", ""))
        adoption_eligible = int(float(row.get("adoption_eligible", 0) or 0))
        diagnostic_only = int(float(row.get("diagnostic_only", 0) or 0))
        uses_eval_nelbo = int(float(row.get("routing_uses_eval_nelbo", 0) or 0))
        uses_eval_stats = int(float(row.get("routing_uses_eval_domain_statistics", 0) or 0))
        if method == "oracle_routing":
            raise ProtocolError("oracle_routing must not be emitted under learned utility LOQDO v2")
        if method == "candidate_oracle_routing" and adoption_eligible == 1:
            raise ProtocolError("candidate_oracle_routing must not be adoption eligible")
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
        metrics["spearman"] = metrics["micro_spearman"]
        metrics["pairwise_auc"] = metrics["micro_pairwise_auc"]
        metrics["selected_nelbo"] = metrics["micro_selected_nelbo"]
        metrics["oracle_nelbo"] = metrics["micro_candidate_oracle_nelbo"]
        metrics["candidate_oracle_nelbo"] = metrics["micro_candidate_oracle_nelbo"]

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
        metrics["protocol_version"] = _PROTOCOL_VERSION
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
            "adoption_selected_method",
        ]:
            vals_for_key = sorted(set(str(r.get(key, "")) for r in vals if str(r.get(key, "")) != ""))
            if vals_for_key:
                metrics[key] = vals_for_key[0] if len(vals_for_key) == 1 else "|".join(vals_for_key)
        if any("selected_by_inner_validation" in r for r in vals):
            metrics["selected_by_inner_validation"] = float(
                max(int(float(r.get("selected_by_inner_validation", 0) or 0)) for r in vals)
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
                "protocol_version": _PROTOCOL_VERSION,
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
                "selected_by_inner_validation": int(base.get("selected_by_inner_validation", 0) or 0),
                "adoption_selected_method": str(base.get("adoption_selected_method", "")),
                "n_samples": int(len(rows)),
                "top1_oracle_hit": _finite_mean([float(r["top1_oracle_hit"]) for r in rows]),
                "mean_rank": _finite_mean([float(r["selected_rank"]) for r in rows]),
                "mean_oracle_gap": _finite_mean([float(r["oracle_gap"]) for r in rows]),
                "mean_oracle_gap_pct": _finite_mean([float(r["oracle_gap_pct"]) for r in rows]),
                "pairwise_auc": _finite_mean([float(r["pairwise_auc"]) for r in rows]),
                "spearman": _finite_mean([float(r["spearman"]) for r in rows]),
            }
        )
    return domain_rows
